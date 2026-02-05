"""
GPipe 流水线调度器基类

定义调度器的抽象接口，所有具体调度策略都需要继承此基类。
"""

from abc import ABC, abstractmethod
from typing import List, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..stage import BaseStage
    from ..config import PipelineConfig


class Registry:
    """调度器注册表"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry = {}
    
    def register(self, name: str, force: bool = False):
        """装饰器：注册一个调度器类"""
        def decorator(cls):
            if name in self._registry and not force:
                raise ValueError(f"'{name}' is already registered in {self.name}")
            self._registry[name] = cls
            cls._registry_name = name
            return cls
        return decorator
    
    def get(self, name: str):
        """获取注册的调度器类"""
        return self._registry.get(name)
    
    def create(self, name: str, *args, **kwargs):
        """创建调度器实例"""
        cls = self.get(name)
        if cls is None:
            available = list(self._registry.keys())
            raise ValueError(f"'{name}' not found in {self.name}. Available: {available}")
        return cls(*args, **kwargs)
    
    def list_registered(self) -> List[str]:
        """列出所有注册的调度器"""
        return list(self._registry.keys())


# 全局调度器注册表
SCHEDULER_REGISTRY = Registry("schedulers")


class BaseScheduler(ABC):
    """流水线调度器基类
    
    调度器负责协调流水线各阶段的前向传播、反向传播和通信。
    不同的调度策略（如 Naive、1F1B、Interleaved）通过继承此类实现。
    
    主要职责:
    - 协调 micro-batch 的执行顺序
    - 管理前向/反向传播的调度
    - 处理阶段间的通信时序
    
    示例:
        scheduler = NaiveScheduler(config)
        scheduler.run_iteration(stage, micro_inputs, micro_labels, iteration)
    """
    
    def __init__(self, config: 'PipelineConfig'):
        """初始化调度器
        
        Args:
            config: 流水线配置
        """
        self.config = config
        self.num_microbatches = config.training.num_microbatches
        self.model_parallel_size = config.parallel.model_parallel_size
        self.data_parallel_size = config.parallel.data_parallel_size
    
    @abstractmethod
    def run_iteration(
        self,
        stage: 'BaseStage',
        micro_inputs: List[torch.Tensor],
        micro_labels: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行一次完整的训练迭代
        
        包括所有 micro-batch 的前向传播、反向传播和参数更新。
        
        Args:
            stage: 当前阶段对象
            micro_inputs: micro-batch 输入列表
            micro_labels: micro-batch 标签列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID (1-indexed)
        """
        pass
    
    @abstractmethod
    def run_forward(
        self,
        stage: 'BaseStage',
        micro_inputs: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行前向传播阶段
        
        Args:
            stage: 当前阶段对象
            micro_inputs: micro-batch 输入列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        pass
    
    @abstractmethod
    def run_backward(
        self,
        stage: 'BaseStage',
        micro_labels: List[torch.Tensor],
        iteration: int,
        current_stage: int
    ) -> None:
        """执行反向传播阶段
        
        Args:
            stage: 当前阶段对象
            micro_labels: micro-batch 标签列表
            iteration: 当前迭代编号
            current_stage: 当前阶段 ID
        """
        pass
    
    def run_update(
        self,
        stage: 'BaseStage'
    ) -> None:
        """执行参数更新
        
        默认实现：AllReduce 梯度 + 优化器步骤
        
        Args:
            stage: 当前阶段对象
        """
        if stage.is_training:
            # 数据并行梯度同步
            if self.data_parallel_size > 1:
                stage.all_reduce_gradients()
            
            # 优化器步骤
            stage.step()
            stage.zero_grad()
    
    def get_name(self) -> str:
        """获取调度器名称"""
        return getattr(self, '_registry_name', self.__class__.__name__)
    
    def __repr__(self) -> str:
        return f"{self.get_name()}(microbatches={self.num_microbatches}, mp_size={self.model_parallel_size})"
