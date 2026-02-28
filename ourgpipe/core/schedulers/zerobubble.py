"""
ZeroBubble (ZB-H1) 调度器

核心思想：将反向传播拆分为 B (input gradient) 和 W (weight gradient)，
W 延迟执行以填充 pipeline bubble，从而接近零气泡。

前提条件：模型中的 nn.Linear 需要替换为 BubbleLinear（通过 convert_to_bubble_model）。

调度模式 (以 4 stage 为例):
```
Stage 0: F F F F F F F F B B B B B B B B W W W W W W W W
Stage 1: . F F F F F F F F B B B B B B B B W W W W W W W W
Stage 2: . . F F F F F F F F B(split) B B B B B B B W W W ...
Stage 3: . . . F F F F F F F F B(split) B(split) B B B B B B W W ...
```

通信方式：使用 batch_isend_irecv 批量提交 P2P 操作。
"""

from typing import List, TYPE_CHECKING

import torch
from torch.profiler import record_function

from .base import BaseScheduler, SCHEDULER_REGISTRY
from ..comm import execute_p2p_ops
from ..weight_grad_store import WeightGradStore

if TYPE_CHECKING:
    from ..stage import BaseStage
    from ..config import PipelineConfig


@SCHEDULER_REGISTRY.register("zerobubble")
class ZeroBubbleScheduler(BaseScheduler):
    """ZeroBubble (ZB-H1) 调度器

    特点:
    - 基于 1F1B 框架，额外实现 B/W 分离
    - 通过 WeightGradStore 延迟 W 计算
    - 在 pipeline bubble 时间执行 W，减少空闲
    - 需要配合 BubbleLinear 使用

    示例:
        scheduler = ZeroBubbleScheduler(config)
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
        """执行一次完整的 ZeroBubble 训练迭代"""
        stage.reset_cache()
        stage.zero_grad()
        WeightGradStore.clear()

        pipeline_rank = current_stage - 1
        is_first = (current_stage == 1)
        is_last = (current_stage == self.model_parallel_size)

        num_warmup = min(self.num_microbatches,
                         self.model_parallel_size - pipeline_rank - 1)
        num_remaining = self.num_microbatches - num_warmup

        recv_fwd_bufs = {}
        recv_bwd_bufs = {}
        send_bufs = []

        fwd_step = 0
        bwd_step = 0

        # === Phase 1: Warmup (only forward) ===
        for _ in range(num_warmup):
            with record_function(f"ZB Warmup Fwd mb_{fwd_step}"):
                if not is_first:
                    op, buf = stage.p2p_forward_recv(fwd_step)
                    execute_p2p_ops([op])
                    recv_fwd_bufs[fwd_step] = buf

                if is_first:
                    y = stage.forward_compute(micro_inputs[fwd_step], fwd_step)
                else:
                    y = stage.forward_compute(
                        recv_fwd_bufs.pop(fwd_step), fwd_step)

                if not is_last:
                    op, sbuf = stage.p2p_forward_send(y, fwd_step)
                    send_bufs.append(sbuf)
                    execute_p2p_ops([op])

            fwd_step += 1

        # === Phase 2: Steady (1F1B with B/W split) ===
        for i in range(num_remaining):
            last_iteration = (i == num_remaining - 1)

            # --- Forward ---
            with record_function(f"ZB Steady Fwd mb_{fwd_step}"):
                if not is_first:
                    op, buf = stage.p2p_forward_recv(fwd_step)
                    execute_p2p_ops([op])
                    recv_fwd_bufs[fwd_step] = buf

                if is_first:
                    y = stage.forward_compute(micro_inputs[fwd_step], fwd_step)
                else:
                    y = stage.forward_compute(
                        recv_fwd_bufs.pop(fwd_step), fwd_step)

            # send_fwd + recv_bwd 同时提交
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

            # --- Backward (B only, W deferred) ---
            # 控制 split_bw 开关：
            # - pipeline_rank == 0 不需要 split（没有 bubble 可填）
            # - 前 pipeline_rank 个 steady step 和最后一个 step 需要 split
            WeightGradStore.split_bw = (
                (i < pipeline_rank or last_iteration)
                and pipeline_rank > 0
            )

            with record_function(f"ZB Steady Bwd mb_{bwd_step}"):
                if is_last:
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, micro_labels[bwd_step], is_last_stage=True)
                else:
                    grad = recv_bwd_bufs.pop(bwd_step)
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, grad, is_last_stage=False)

            # flush W 闭包到队列
            if WeightGradStore.split_bw:
                WeightGradStore.flush()

            # send backward grad
            if not is_first and input_grad is not None:
                op, sbuf = stage.p2p_backward_send(input_grad, bwd_step)
                send_bufs.append(sbuf)
                execute_p2p_ops([op])

            # 在 bubble 时间执行 W
            if last_iteration:
                if i >= pipeline_rank > 0:
                    WeightGradStore.pop()

            bwd_step += 1

        # === Phase 3: Cooldown (only backward, with B/W split) ===
        for i in range(num_warmup):
            with record_function(f"ZB Cooldown Bwd mb_{bwd_step}"):
                if not is_last:
                    op, rbuf = stage.p2p_backward_recv(bwd_step)
                    recv_bwd_bufs[bwd_step] = rbuf
                    execute_p2p_ops([op])

                WeightGradStore.split_bw = pipeline_rank > 0

                if is_last:
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, micro_labels[bwd_step], is_last_stage=True)
                else:
                    grad = recv_bwd_bufs.pop(bwd_step)
                    loss_val, input_grad = stage.backward_compute(
                        bwd_step, grad, is_last_stage=False)

                if not is_first and input_grad is not None:
                    op, sbuf = stage.p2p_backward_send(input_grad, bwd_step)
                    send_bufs.append(sbuf)
                    execute_p2p_ops([op])

                if WeightGradStore.split_bw:
                    WeightGradStore.flush()
                    if num_remaining + i >= pipeline_rank:
                        WeightGradStore.pop()

            bwd_step += 1

        # 执行所有剩余的 W 计算
        WeightGradStore.pop_all()

        # ======= Update =======
        self.run_update(stage)

    def run_forward(self, stage, micro_inputs, iteration, current_stage):
        """ZeroBubble 不单独使用此方法，由 run_iteration 统一调度"""
        pass

    def run_backward(self, stage, micro_labels, iteration, current_stage):
        """ZeroBubble 不单独使用此方法，由 run_iteration 统一调度"""
        pass
