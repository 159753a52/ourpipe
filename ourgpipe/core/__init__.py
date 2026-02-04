"""
GPipe 流水线框架核心模块

提供模型无关的流水线并行训练基础设施。
"""

from .interfaces import ModelInterface, StageInterface, DatasetInterface
from .config import PipelineConfig, ModelConfig, DatasetConfig, TrainingConfig, ParallelConfig
from .registry import MODEL_REGISTRY, DATASET_REGISTRY, STAGE_REGISTRY
from .stage import BaseStage

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
    # 基类
    'BaseStage',
]