"""
LLaMA 模型适配器

提供 LLaMA 模型在 GPipe 流水线中的实现。
"""

from .model import LlamaModel
from .stage import LlamaStage

__all__ = [
    'LlamaModel',
    'LlamaStage',
]
