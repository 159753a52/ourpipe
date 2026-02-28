"""
GPipe 流水线调度器模块

提供不同的流水线调度策略：
- NaiveScheduler: 同步阻塞调度（简单但效率较低）
- AsyncThreadedScheduler: 异步多线程调度（高效但复杂）
- OneFOneBScheduler: 1F1B 调度（降低峰值内存）
- HanayoScheduler: 双向流水线调度（利用 bubble）
- ZeroBubbleScheduler: ZeroBubble 调度（B/W 分离，接近零气泡）
"""

from .base import BaseScheduler, SCHEDULER_REGISTRY
from .naive import NaiveScheduler
from .async_threaded import AsyncThreadedScheduler
from .one_f_one_b import OneFOneBScheduler
from .hanayo import HanayoScheduler
from .zerobubble import ZeroBubbleScheduler

__all__ = [
    'BaseScheduler',
    'SCHEDULER_REGISTRY',
    'NaiveScheduler',
    'AsyncThreadedScheduler',
    'OneFOneBScheduler',
    'HanayoScheduler',
    'ZeroBubbleScheduler',
]
