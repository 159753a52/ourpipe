"""
Hanayo 双向流水线调度器

实现 Hanayo 双向流水线调度策略：
- Wave A: 正向流水线 (stage1 -> stage2 -> ... -> stageN)
- Wave B: 反向流水线 (stageN -> ... -> stage2 -> stage1)
- decoder 层在两个 wave 间共享，有效利用 pipeline bubble

调度模式:
  Phase 1: Wave A warmup forward
  Phase 2: 剩余 Wave A fwd 与 Wave B fwd 交错
  Phase 3: 剩余 Wave B fwd 与 Wave B bwd 交错
  Phase 4: Wave A bwd cooldown + Wave B bwd

通信方式：使用 batch_isend_irecv 批量提交 P2P 操作。
"""

from typing import List, TYPE_CHECKING
import os
import time

import torch
from torch.profiler import record_function

from .base import BaseScheduler, SCHEDULER_REGISTRY
from ..comm import execute_p2p_ops

if TYPE_CHECKING:
    from ..stage import BaseStage
    from ..config import PipelineConfig


def build_hanayo_schedule(pipeline_rank: int, num_stages: int, num_mb: int):
    """构建 Hanayo 调度表

    Args:
        pipeline_rank: 当前 pipeline rank (0-indexed)
        num_stages: 总阶段数
        num_mb: micro-batch 数量

    Returns:
        调度表列表，每个元素为 (action, mb_idx)
        action: 'fA' (forward A), 'fB' (forward B), 'bA' (backward A), 'bB' (backward B)
    """
    schedule = []
    num_warmup_A = min(num_mb, num_stages - 1 - pipeline_rank)

    fA = fB = bA = bB = 0

    # Phase 1: Wave A warmup
    for _ in range(num_warmup_A):
        schedule.append(('fA', fA))
        fA += 1

    # Phase 2: 剩余 fA 与 fB 交错
    while fA < num_mb:
        schedule.append(('fA', fA))
        fA += 1
        if fB < num_mb:
            schedule.append(('fB', fB))
            fB += 1

    # Phase 3: 剩余 fB 与 bB 交错
    while fB < num_mb:
        if bB < num_mb:
            schedule.append(('bB', bB))
            bB += 1
        if fB < num_mb:
            schedule.append(('fB', fB))
            fB += 1
            if fB == num_mb:
                schedule.append(('bB', bB))
                bB += 1

    # Phase 4: bA cooldown + 剩余 bB
    while bA < num_mb or bB < num_mb:
        if bA < num_mb:
            schedule.append(('bA', bA))
            bA += 1
        if bB < num_mb:
            schedule.append(('bB', bB))
            bB += 1

    return schedule


def _hanayo_debug_enabled() -> bool:
    """是否启用 Hanayo 调试追踪日志"""
    return os.environ.get("HANAYO_DEBUG_TRACE", "0") == "1"



def _hanayo_debug_iter() -> int:
    """读取要追踪的 iteration（默认 0）"""
    try:
        return int(os.environ.get("HANAYO_DEBUG_ITER", "0"))
    except ValueError:
        return 0



def _trace_hanayo(debug: bool, message: str) -> None:
    """按需打印 Hanayo 追踪日志"""
    if debug:
        print(f"[HANAYO][{time.time():.6f}] {message}", flush=True)



def _get_comm_meta(global_rank: int, action: str, mb_idx: int):
    """返回 (recv_peer, send_peer, tag)"""
    if action == 'fA':
        return global_rank - 1, global_rank + 1, mb_idx
    if action == 'fB':
        return global_rank + 1, global_rank - 1, 2000 + mb_idx
    if action == 'bA':
        return global_rank + 1, global_rank - 1, 1000 + mb_idx
    if action == 'bB':
        return global_rank - 1, global_rank + 1, 3000 + mb_idx
    raise ValueError(f"Unknown hanayo action: {action}")


@SCHEDULER_REGISTRY.register("hanayo")
class HanayoScheduler(BaseScheduler):
    """Hanayo 双向流水线调度器

    特点:
    - 双向流水线，Wave A 和 Wave B 交错执行
    - decoder 层共享，减少参数量
    - 需要配合 GPTHanayoStage 使用
    - batch 会被拆成 2 * num_microbatches 份

    示例:
        scheduler = HanayoScheduler(config)
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
        """执行一次完整的 Hanayo 训练迭代

        注意：micro_inputs/micro_labels 应该已经是 2*num_microbatches 份，
        前半为 Wave A，后半为 Wave B。调用方需要负责拆分。
        """
        stage.zero_grad()

        pipeline_rank = current_stage - 1

        # 拆分 A/B 数据
        num_mb = self.num_microbatches
        micro_inputs_A = micro_inputs[:num_mb]
        micro_labels_A = micro_labels[:num_mb]
        micro_inputs_B = micro_inputs[num_mb:]
        micro_labels_B = micro_labels[num_mb:]

        schedule = build_hanayo_schedule(
            pipeline_rank, self.model_parallel_size, num_mb)

        debug_this_iter = (
            _hanayo_debug_enabled() and iteration == _hanayo_debug_iter())

        _trace_hanayo(
            debug_this_iter,
            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
            f"pipeline_rank={pipeline_rank} schedule={schedule}"
        )

        recv_bufs = {'fA': {}, 'fB': {}, 'bA': {}, 'bB': {}}

        for action, mb_idx in schedule:
            wave = action[1]  # 'A' or 'B'
            is_fwd = action[0] == 'f'
            is_entry = stage.is_entry(wave)
            is_exit = stage.is_exit(wave)
            recv_peer, send_peer, tag = _get_comm_meta(
                stage.global_rank, action, mb_idx)

            _trace_hanayo(
                debug_this_iter,
                f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                f"action={action} mb={mb_idx} entry={is_entry} exit={is_exit} "
                f"recv_peer={recv_peer} send_peer={send_peer} tag={tag}"
            )

            if is_fwd:
                # --- Forward ---
                with record_function(f"{action}_{mb_idx}"):
                    if not is_entry:
                        # 根据 wave 方向选择通信方法
                        if wave == 'A':
                            op, buf = stage.p2p_forward_recv(mb_idx)
                        else:
                            op, buf = stage.p2p_forward_recv_B(mb_idx)
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} RECV_START peer={recv_peer} "
                            f"tag={tag} expected_shape={stage.get_activation_shape()}"
                        )
                        execute_p2p_ops([op])
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} RECV_DONE peer={recv_peer} "
                            f"tag={tag} recv_shape={tuple(buf.shape)}"
                        )
                        if tuple(buf.shape) != stage.get_activation_shape():
                            raise ValueError(
                                "[P2P] forward_recv shape mismatch: "
                                f"stage_id={stage.stage_id} rank={stage.global_rank} mb_idx={mb_idx} wave={wave} "
                                f"recv_buf.shape={tuple(buf.shape)} expected={stage.get_activation_shape()}"
                            )
                        recv_bufs[action][mb_idx] = buf

                    inputs = micro_inputs_A if wave == 'A' else micro_inputs_B
                    _trace_hanayo(
                        debug_this_iter,
                        f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                        f"action={action} mb={mb_idx} FORWARD_START entry={is_entry}"
                    )
                    if is_entry:
                        y = stage.forward_wave(wave, mb_idx,
                                               micro_input=inputs[mb_idx])
                    else:
                        y = stage.forward_wave(wave, mb_idx,
                                       recv_buf=recv_bufs[action].pop(mb_idx))
                    _trace_hanayo(
                        debug_this_iter,
                        f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                        f"action={action} mb={mb_idx} FORWARD_DONE output_shape={tuple(y.shape)}"
                    )

                    if not is_exit:
                        if wave == 'A':
                            op, sbuf = stage.p2p_forward_send(y, mb_idx)
                        else:
                            op, sbuf = stage.p2p_forward_send_B(y, mb_idx)
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} SEND_START peer={send_peer} "
                            f"tag={tag} send_shape={tuple(sbuf.shape)}"
                        )
                        execute_p2p_ops([op])
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} SEND_DONE peer={send_peer}"
                        )
            else:
                # --- Backward ---
                with record_function(f"{action}_{mb_idx}"):
                    if not is_exit:
                        if wave == 'A':
                            op, rbuf = stage.p2p_backward_recv(mb_idx)
                        else:
                            op, rbuf = stage.p2p_backward_recv_B(mb_idx)
                        recv_bufs[action][mb_idx] = rbuf
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} RECV_START peer={recv_peer} "
                            f"tag={tag} expected_shape={stage.get_activation_shape()}"
                        )
                        execute_p2p_ops([op])
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} RECV_DONE peer={recv_peer} "
                            f"tag={tag} recv_shape={tuple(rbuf.shape)}"
                        )
                        if tuple(rbuf.shape) != stage.get_activation_shape():
                            raise ValueError(
                                "[P2P] backward_recv shape mismatch: "
                                f"stage_id={stage.stage_id} rank={stage.global_rank} mb_idx={mb_idx} wave={wave} "
                                f"recv_buf.shape={tuple(rbuf.shape)} expected={stage.get_activation_shape()}"
                            )

                    labels = micro_labels_A if wave == 'A' else micro_labels_B
                    if is_exit:
                        loss_val, input_grad = stage.backward_wave(
                            wave, mb_idx, labels[mb_idx])
                    else:
                        grad = recv_bufs[action].pop(mb_idx)
                        loss_val, input_grad = stage.backward_wave(
                            wave, mb_idx, grad)

                    if not is_entry and input_grad is not None:
                        if wave == 'A':
                            op, sbuf = stage.p2p_backward_send(input_grad, mb_idx)
                        else:
                            op, sbuf = stage.p2p_backward_send_B(input_grad, mb_idx)
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} SEND_START peer={send_peer} "
                            f"tag={tag} send_shape={tuple(sbuf.shape)}"
                        )
                        execute_p2p_ops([op])
                        _trace_hanayo(
                            debug_this_iter,
                            f"iter={iteration} rank={stage.global_rank} stage={stage.stage_id} "
                            f"action={action} mb={mb_idx} SEND_DONE peer={send_peer}"
                        )

        # ======= Update =======
        self.run_update(stage)

    def run_forward(self, stage, micro_inputs, iteration, current_stage):
        """Hanayo 不单独使用此方法，由 run_iteration 统一调度"""
        pass

    def run_backward(self, stage, micro_labels, iteration, current_stage):
        """Hanayo 不单独使用此方法，由 run_iteration 统一调度"""
        pass
