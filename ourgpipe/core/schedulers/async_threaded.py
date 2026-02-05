"""
异步多线程调度器

实现高效的异步 GPipe 调度：
- 多线程并行执行 micro-batch
- 使用异步 isend/irecv 进行通信
- 独立的计算流和通信流
- 支持 Orion 多优先级调度

这是高性能实现，适合生产环境使用。
"""

from typing import List, Dict, TYPE_CHECKING
import threading

import torch
import torch.distributed as dist
from torch.profiler import record_function

from .base import BaseScheduler, SCHEDULER_REGISTRY

if TYPE_CHECKING:
    from ..stage import BaseStage
    from ..config import PipelineConfig


@SCHEDULER_REGISTRY.register("async_threaded")
class AsyncThreadedScheduler(BaseScheduler):
    """异步多线程调度器
    
    特点:
    - 异步非阻塞通信 (dist.isend/irecv)
    - 多线程并行执行 micro-batch
    - 独立的计算流和通信流
    - 支持 Orion 多优先级 GPU 调度
    
    调度模式:
    ```
    Forward:  F0, F1, F2, F3 并行执行（通过多线程）
    Backward: B3, B2, B1, B0 并行执行（通过多线程）
    ```
    
    示例:
        scheduler = AsyncThreadedScheduler(config)
        scheduler.run_iteration(stage, micro_inputs, micro_labels, iteration, current_stage)
    """
    
    def __init__(self, config: 'PipelineConfig'):
        """初始化异步多线程调度器
        
        Args:
            config: 流水线配置
        """
        super().__init__(config)
        
        # 通信句柄存储
        self.fwd_recv_handles: Dict[int, any] = {}
        self.fwd_send_handles: Dict[int, any] = {}
        self.bwd_recv_handles: Dict[int, any] = {}
        self.bwd_send_handles: Dict[int, any] = {}
    
    def run_iteration(
        self,
        stage: 'BaseStage',
        micro_inputs: List[torch.Tensor],
        micro_labels: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行一次完整的训练迭代
        
        Args:
            stage: 当前阶段对象
            micro_inputs: micro-batch 输入列表
            micro_labels: micro-batch 标签列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        # 重置句柄
        self.fwd_recv_handles.clear()
        self.fwd_send_handles.clear()
        self.bwd_recv_handles.clear()
        self.bwd_send_handles.clear()
        
        # 重置缓存
        stage.reset_cache()
        
        # 前向传播
        self.run_forward(stage, micro_inputs, iteration, current_stage)
        
        # 反向传播
        self.run_backward(stage, micro_labels, iteration, current_stage)
        
        # 同步所有流和通信
        self._synchronize(stage)
        
        # 梯度累加（非最后阶段）
        if current_stage < self.model_parallel_size:
            with record_function("Gradient Accumulation"):
                stage.accumulate_gradients()
        
        # 参数更新
        self.run_update(stage)
    
    def run_forward(
        self,
        stage: 'BaseStage',
        micro_inputs: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行前向传播阶段（异步多线程版本）
        
        Args:
            stage: 当前阶段对象
            micro_inputs: micro-batch 输入列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        # 步骤 1：所有接收方提交 irecv
        if current_stage > 1:
            for mb_idx in range(self.num_microbatches):
                with record_function(f"Fwd Post Irecv mb_{mb_idx}"):
                    handle = stage.forward_irecv(mb_idx, iteration)
                    self.fwd_recv_handles[mb_idx] = handle
        
        # 步骤 2：多线程并行计算
        def fwd_compute_mb(mb_idx: int):
            torch.cuda.set_device(stage.device)
            
            # Orion 调度器设置
            if stage.orion_scheduler is not None:
                stage.orion_scheduler.set_client_idx(mb_idx)
            
            current_comp_stream = stage.comp_streams[mb_idx]
            
            with torch.cuda.stream(current_comp_stream):
                with record_function(f"Fwd Compute mb_{mb_idx}"):
                    if current_stage > 1:
                        self.fwd_recv_handles[mb_idx].wait()
                        input_tensor = stage.out_x_buffers[mb_idx]
                    else:
                        input_tensor = micro_inputs[mb_idx]
                    
                    output_tensor = stage.forward(input_tensor, mb_idx)
            
            # 发送
            if current_stage < self.model_parallel_size:
                with stage.comm_stream_lock:
                    with torch.cuda.stream(stage.comm_stream):
                        stage.comm_stream.wait_stream(current_comp_stream)
                        with record_function(f"Fwd Send mb_{mb_idx}"):
                            handle = stage.forward_isend(
                                stage.fwd_cache[mb_idx], mb_idx, iteration
                            )
                            self.fwd_send_handles[mb_idx] = handle
        
        # 启动所有前向线程
        fwd_threads = []
        for mb_idx in range(self.num_microbatches):
            t = threading.Thread(target=fwd_compute_mb, args=(mb_idx,))
            t.start()
            fwd_threads.append(t)
        
        # 等待所有前向线程完成
        for t in fwd_threads:
            t.join()
    
    def run_backward(
        self,
        stage: 'BaseStage',
        micro_labels: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行反向传播阶段（异步多线程版本）
        
        Args:
            stage: 当前阶段对象
            micro_labels: micro-batch 标签列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        # 步骤 1：所有非最后阶段提交 irecv
        if current_stage < self.model_parallel_size:
            for mb_idx in reversed(range(self.num_microbatches)):
                with record_function(f"Bwd Post Irecv mb_{mb_idx}"):
                    handle = stage.backward_irecv(mb_idx, iteration)
                    self.bwd_recv_handles[mb_idx] = handle
        
        # 步骤 2：多线程并行计算
        def bwd_compute_mb(mb_idx: int):
            torch.cuda.set_device(stage.device)
            
            # Orion 调度器设置
            if stage.orion_scheduler is not None:
                stage.orion_scheduler.set_client_idx(mb_idx)
            
            current_comp_stream = stage.comp_streams[mb_idx]
            
            with torch.cuda.stream(current_comp_stream):
                with record_function(f"Bwd Compute mb_{mb_idx}"):
                    grad_tensor = None
                    if current_stage < self.model_parallel_size:
                        self.bwd_recv_handles[mb_idx].wait()
                        grad_tensor = stage.grad_y_buffers[mb_idx]
                    
                    if current_stage == self.model_parallel_size:
                        # 最后阶段：计算损失
                        loss = stage.compute_loss(
                            stage.fwd_cache[mb_idx], micro_labels[mb_idx]
                        )
                        loss.backward()
                    else:
                        # 其他阶段：只计算梯度
                        stage.compute_grad_only(mb_idx, grad_tensor)
            
            # 发送
            if current_stage > 1:
                with stage.comm_stream_lock:
                    with torch.cuda.stream(stage.comm_stream):
                        stage.comm_stream.wait_stream(current_comp_stream)
                        with record_function(f"Bwd Send mb_{mb_idx}"):
                            handle = stage.backward_isend(mb_idx, iteration)
                            self.bwd_send_handles[mb_idx] = handle
        
        # 启动所有反向线程
        bwd_threads = []
        for mb_idx in reversed(range(self.num_microbatches)):
            t = threading.Thread(target=bwd_compute_mb, args=(mb_idx,))
            t.start()
            bwd_threads.append(t)
        
        # 等待所有反向线程完成
        for t in bwd_threads:
            t.join()
    
    def _synchronize(self, stage: 'BaseStage') -> None:
        """同步所有流和通信句柄
        
        Args:
            stage: 阶段对象
        """
        # 同步所有计算流
        if stage.comp_streams:
            for s in stage.comp_streams:
                torch.cuda.current_stream().wait_stream(s)
        
        # 同步通信流
        if stage.comm_stream:
            torch.cuda.current_stream().wait_stream(stage.comm_stream)
        
        # 等待所有 P2P 通信完成
        for handle in self.fwd_send_handles.values():
            handle.wait()
        for handle in self.bwd_send_handles.values():
            handle.wait()
