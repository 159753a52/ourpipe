"""
1F1B (One Forward One Backward) 调度器

实现经典的 1F1B 流水线调度策略：
- Phase 1 (Warmup):   连续执行若干前向传播填充流水线
- Phase 2 (Steady):   每做一次前向紧接一次反向，保持流水线满载
- Phase 3 (Cooldown): 排空剩余的反向传播

相比 GPipe 的 all-forward-then-all-backward，1F1B 显著降低了
峰值内存占用（activation 不需要全部缓存）。

通信方式：使用 batch_isend_irecv 批量提交 P2P 操作，避免 NCCL 死锁。
"""

from typing import List, TYPE_CHECKING

import torch
from torch.profiler import record_function

from .base import BaseScheduler, SCHEDULER_REGISTRY
from ..comm import execute_p2p_ops

if TYPE_CHECKING:
    from ..stage import BaseStage
    from ..config import PipelineConfig


@SCHEDULER_REGISTRY.register("1f1b")
class OneFOneBScheduler(BaseScheduler):
    """1F1B 调度器

    调度模式 (以 4 stage, 8 microbatch 为例, pipeline_rank=0):
    ```
    Warmup:  F0 F1 F2
    Steady:  F3 B0  F4 B1  F5 B2  F6 B3  F7 B4
    Cooldown:       B5 B6 B7
    ```

    示例:
        scheduler = OneFOneBScheduler(config)
        scheduler.run_iteration(stage, micro_inputs, micro_labels, iteration, current_stage)
    """

    def __init__(self, config: 'PipelineConfig'):
        super().__init__(config)

    def run_iteration(
        self,
        stage: 'BaseStage',
        micro_inputs: List[torch.Tensor],
        micro_labels: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行一次完整的 1F1B 训练迭代"""
        stage.reset_cache()
        stage.zero_grad()

        pipeline_rank = current_stage - 1
        num_warmup = min(self.num_microbatches,
                         self.model_parallel_size - pipeline_rank - 1)
        is_first = (current_stage == 1)
        is_last = (current_stage == self.model_parallel_size)

        fwd_step = 0
        bwd_step = 0
        recv_fwd_bufs = {}
        recv_bwd_bufs = {}
        send_bufs = []  # 保持 send buffer 引用，防止 GC

        # ======= Phase 1: Warmup (only forward) =======
        for _ in range(num_warmup):
            with record_function(f"1F1B Warmup Fwd mb_{fwd_step}"):
                # recv
                if not is_first:
                    op, buf = stage.p2p_forward_recv(fwd_step)
                    execute_p2p_ops([op])
                    recv_fwd_bufs[fwd_step] = buf

                # compute
                if is_first:
                    y = stage.forward_compute(micro_inputs[fwd_step], fwd_step)
                else:
                    y = stage.forward_compute(recv_fwd_bufs.pop(fwd_step), fwd_step)

                # send
                if not is_last:
                    op, sbuf = stage.p2p_forward_send(y, fwd_step)
                    send_bufs.append(sbuf)
                    execute_p2p_ops([op])

            fwd_step += 1

        # ======= Phase 2: 1F1B Steady State =======
        while fwd_step < self.num_microbatches:
            # --- Forward ---
            with record_function(f"1F1B Steady Fwd mb_{fwd_step}"):
                if not is_first:
                    op, buf = stage.p2p_forward_recv(fwd_step)
                    execute_p2p_ops([op])
                    recv_fwd_bufs[fwd_step] = buf

                if is_first:
                    y = stage.forward_compute(micro_inputs[fwd_step], fwd_step)
                else:
                    y = stage.forward_compute(recv_fwd_bufs.pop(fwd_step), fwd_step)

            # send_fwd + recv_bwd 同时提交（防止死锁的关键）
            ops = []
            if not is_last:
                op, sbuf = stage.p2p_forward_send(y, fwd_step)
                send_bufs.append(sbuf)
                ops.append(op)

                op, rbuf = stage.p2p_backward_recv(bwd_step)
                recv_bwd_bufs[bwd_step] = rbuf
                ops.append(op)
            execute_p2p_ops(ops)

            fwd_step += 1

            # --- Backward ---
            with record_function(f"1F1B Steady Bwd mb_{bwd_step}"):
                if is_last:
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, micro_labels[bwd_step], is_last_stage=True)
                else:
                    grad = recv_bwd_bufs.pop(bwd_step)
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, grad, is_last_stage=False)

                # send backward grad
                if not is_first and input_grad is not None:
                    op, sbuf = stage.p2p_backward_send(input_grad, bwd_step)
                    send_bufs.append(sbuf)
                    execute_p2p_ops([op])

            bwd_step += 1

        # ======= Phase 3: Cooldown (only backward) =======
        while bwd_step < self.num_microbatches:
            with record_function(f"1F1B Cooldown Bwd mb_{bwd_step}"):
                # recv
                if not is_last:
                    op, rbuf = stage.p2p_backward_recv(bwd_step)
                    recv_bwd_bufs[bwd_step] = rbuf
                    execute_p2p_ops([op])

                # compute
                if is_last:
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, micro_labels[bwd_step], is_last_stage=True)
                else:
                    grad = recv_bwd_bufs.pop(bwd_step)
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, grad, is_last_stage=False)

                # send
                if not is_first and input_grad is not None:
                    op, sbuf = stage.p2p_backward_send(input_grad, bwd_step)
                    send_bufs.append(sbuf)
                    execute_p2p_ops([op])

            bwd_step += 1

        # ======= Update =======
        self.run_update(stage)

    def run_forward(self, stage, micro_inputs, iteration, current_stage):
        """1F1B 不单独使用此方法，由 run_iteration 统一调度"""
        pass

    def run_backward(self, stage, micro_labels, iteration, current_stage):
        """1F1B 不单独使用此方法，由 run_iteration 统一调度"""
        pass
