"""
GPipe 流水线框架模型实现

提供各种模型的适配器实现。
"""

# 导入所有模型以触发注册
from . import gpt

__all__ = ['gpt']