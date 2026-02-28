"""
GPipe 流水线框架核心模块

提供模型无关的流水线并行训练基础设施。
"""

from .interfaces import ModelInterface, StageInterface, DatasetInterface
from .config import PipelineConfig, ModelConfig, DatasetConfig, TrainingConfig, ParallelConfig
from .registry import MODEL_REGISTRY, DATASET_REGISTRY, STAGE_REGISTRY
from .stage import BaseStage
from .schedulers import (
    BaseScheduler, SCHEDULER_REGISTRY,
    NaiveScheduler, AsyncThreadedScheduler,
    OneFOneBScheduler, HanayoScheduler, ZeroBubbleScheduler,
)
from .metrics import MetricsTracker, collect_model_params, get_gpu_peak_flops
from .comm import execute_p2p_ops, make_p2p_send, make_p2p_recv
from .weight_grad_store import WeightGradStore
from .bubble_linear import BubbleLinear, convert_to_bubble_model

__all__ = [
    # 接口
    'ModelInterface',
    'StageInterface', 
    'DatasetInterface',
    # 配置
    'PipelineConfig',
    'ModelConfig',
    'DatasetConfig',
    'TrainingConfig',
    'ParallelConfig',
    # 注册表
    'MODEL_REGISTRY',
    'DATASET_REGISTRY',
    'STAGE_REGISTRY',
    'SCHEDULER_REGISTRY',
    # 基类
    'BaseStage',
    'BaseScheduler',
    # 调度器
    'NaiveScheduler',
    'AsyncThreadedScheduler',
    'OneFOneBScheduler',
    'HanayoScheduler',
    'ZeroBubbleScheduler',
    # P2P 通信
    'execute_p2p_ops',
    'make_p2p_send',
    'make_p2p_recv',
    # ZeroBubble 组件
    'WeightGradStore',
    'BubbleLinear',
    'convert_to_bubble_model',
    # 指标追踪
    'MetricsTracker',
    'collect_model_params',
    'get_gpu_peak_flops',
]
