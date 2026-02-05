"""
Naive 同步调度器

实现简单的同步阻塞式 GPipe 调度：
- 所有 micro-batch 串行执行
- 使用同步 send/recv 进行通信
- 先完成所有前向传播，再执行所有反向传播

这是最简单的实现，易于理解和调试，但效率较低。
"""

from typing import List, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import record_function

from .base import BaseScheduler, SCHEDULER_REGISTRY

if TYPE_CHECKING:
    from ..stage import BaseStage
    from ..config import PipelineConfig


@SCHEDULER_REGISTRY.register("naive")
class NaiveScheduler(BaseScheduler):
    """Naive 同步调度器
    
    特点:
    - 同步阻塞通信 (dist.send/recv)
    - 串行执行所有 micro-batch
    - 简单直观，适合调试和基准测试
    
    调度模式:
    ```
    Forward:  F0 -> F1 -> F2 -> F3 (所有 micro-batch 依次前向)
    Backward: B3 -> B2 -> B1 -> B0 (所有 micro-batch 依次反向)
    ```
    
    示例:
        scheduler = NaiveScheduler(config)
        scheduler.run_iteration(stage, micro_inputs, micro_labels, iteration, current_stage)
    """
    
    def __init__(self, config: 'PipelineConfig'):
        """初始化 Naive 调度器
        
        Args:
            config: 流水线配置
        """
        super().__init__(config)
    
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
        # 重置缓存
        stage.reset_cache()
        
        # 前向传播
        self.run_forward(stage, micro_inputs, iteration, current_stage)
        
        # 反向传播
        self.run_backward(stage, micro_labels, iteration, current_stage)
        
        # 参数更新
        self.run_update(stage)
    
    def run_forward(
        self,
        stage: 'BaseStage',
        micro_inputs: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行前向传播阶段（同步版本）
        
        所有 micro-batch 串行执行前向传播。
        
        Args:
            stage: 当前阶段对象
            micro_inputs: micro-batch 输入列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        for mb_idx in range(self.num_microbatches):
            with record_function(f"Fwd mb_{mb_idx}"):
                if current_stage == 1:
                    # 第一阶段：直接使用输入
                    self._forward_first_stage(stage, micro_inputs[mb_idx], mb_idx)
                elif current_stage == self.model_parallel_size:
                    # 最后阶段：接收后计算
                    self._forward_last_stage(stage, mb_idx)
                else:
                    # 中间阶段：接收、计算、发送
                    self._forward_middle_stage(stage, mb_idx)
    
    def run_backward(
        self,
        stage: 'BaseStage',
        micro_labels: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行反向传播阶段（同步版本）
        
        所有 micro-batch 逆序串行执行反向传播。
        
        Args:
            stage: 当前阶段对象
            micro_labels: micro-batch 标签列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        for mb_idx in reversed(range(self.num_microbatches)):
            with record_function(f"Bwd mb_{mb_idx}"):
                if current_stage == self.model_parallel_size:
                    # 最后阶段：计算损失并反向传播
                    self._backward_last_stage(stage, micro_labels[mb_idx], mb_idx)
                elif current_stage == 1:
                    # 第一阶段：接收梯度并反向传播
                    self._backward_first_stage(stage, mb_idx)
                else:
                    # 中间阶段：接收梯度、反向传播、发送梯度
                    self._backward_middle_stage(stage, mb_idx)
    
    # ==================== 前向传播辅助方法 ====================
    
    def _forward_first_stage(
        self,
        stage: 'BaseStage',
        input_tensor: torch.Tensor,
        mb_idx: int
    ) -> None:
        """第一阶段前向传播
        
        Args:
            stage: 阶段对象
            input_tensor: 输入张量
            mb_idx: micro-batch 索引
        """
        with record_function(f"Fwd Compute mb_{mb_idx}"):
            output = stage.forward(input_tensor, mb_idx)
        
        with record_function(f"Fwd Send mb_{mb_idx}"):
            stage.forward_send(output)
    
    def _forward_last_stage(
        self,
        stage: 'BaseStage',
        mb_idx: int
    ) -> None:
        """最后阶段前向传播
        
        Args:
            stage: 阶段对象
            mb_idx: micro-batch 索引
        """
        with record_function(f"Fwd Recv mb_{mb_idx}"):
            stage.forward_recv(mb_idx)
        
        with record_function(f"Fwd Compute mb_{mb_idx}"):
            input_tensor = stage.out_x_buffers[mb_idx]
            stage.forward(input_tensor, mb_idx)
    
    def _forward_middle_stage(
        self,
        stage: 'BaseStage',
        mb_idx: int
    ) -> None:
        """中间阶段前向传播
        
        Args:
            stage: 阶段对象
            mb_idx: micro-batch 索引
        """
        with record_function(f"Fwd Recv mb_{mb_idx}"):
            stage.forward_recv(mb_idx)
        
        with record_function(f"Fwd Compute mb_{mb_idx}"):
            input_tensor = stage.out_x_buffers[mb_idx]
            output = stage.forward(input_tensor, mb_idx)
        
        with record_function(f"Fwd Send mb_{mb_idx}"):
            stage.forward_send(output)
    
    # ==================== 反向传播辅助方法 ====================
    
    def _backward_last_stage(
        self,
        stage: 'BaseStage',
        labels: torch.Tensor,
        mb_idx: int
    ) -> None:
        """最后阶段反向传播
        
        Args:
            stage: 阶段对象
            labels: 标签张量
            mb_idx: micro-batch 索引
        """
        with record_function(f"Bwd Loss mb_{mb_idx}"):
            # 计算损失
            output = stage.fwd_cache[mb_idx]
            loss = stage.compute_loss(output, labels)
        
        with record_function(f"Bwd Compute mb_{mb_idx}"):
            # 反向传播
            loss.backward()
        
        with record_function(f"Bwd Send mb_{mb_idx}"):
            # 发送梯度给上一阶段
            stage.backward_send(mb_idx)
    
    def _backward_first_stage(
        self,
        stage: 'BaseStage',
        mb_idx: int
    ) -> None:
        """第一阶段反向传播
        
        Args:
            stage: 阶段对象
            mb_idx: micro-batch 索引
        """
        with record_function(f"Bwd Recv mb_{mb_idx}"):
            stage.backward_recv(mb_idx)
        
        with record_function(f"Bwd Compute mb_{mb_idx}"):
            grad_tensor = stage.grad_y_buffers[mb_idx]
            stage.backward(mb_idx, grad_tensor)
    
    def _backward_middle_stage(
        self,
        stage: 'BaseStage',
        mb_idx: int
    ) -> None:
        """中间阶段反向传播
        
        Args:
            stage: 阶段对象
            mb_idx: micro-batch 索引
        """
        with record_function(f"Bwd Recv mb_{mb_idx}"):
            stage.backward_recv(mb_idx)
        
        with record_function(f"Bwd Compute mb_{mb_idx}"):
            grad_tensor = stage.grad_y_buffers[mb_idx]
            stage.backward(mb_idx, grad_tensor)
        
        with record_function(f"Bwd Send mb_{mb_idx}"):
            stage.backward_send(mb_idx)
