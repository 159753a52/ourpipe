"""
Orion 调度器绑定模块 - 为 GPipe 提供兼容接口

这个模块提供了 gpipe_thread-stream.py 期望的接口：
- get_scheduler(): 获取调度器单例
- SchedulingMode: 调度模式枚举
"""

from enum import Enum
from typing import Optional
import threading

# 导入实际的调度器实现
from orion_gpipe_wrapper import OrionScheduler as _OrionScheduler


class SchedulingMode(Enum):
    """调度模式枚举"""
    FIFO = 0              # 先进先出
    MULTI_PRIORITY = 1    # 多优先级（HP/BE）
    ROUND_ROBIN = 2       # 轮询


class OrionSchedulerWrapper:
    """
    Orion 调度器包装类，提供 GPipe 期望的接口
    
    主要功能：
    - 单例模式
    - 调度模式设置
    - 客户端索引管理
    - 统计信息收集
    """
    
    _instance: Optional['OrionSchedulerWrapper'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._scheduler: Optional[_OrionScheduler] = None
        self._mode = SchedulingMode.FIFO
        self._running = False
        self._num_clients = 0
        self._stats = {}  # client_idx -> (scheduled_count, waiting_count)
        self._initialized = True
    
    def _ensure_scheduler(self):
        """确保调度器已初始化"""
        if self._scheduler is None:
            try:
                self._scheduler = _OrionScheduler()
            except FileNotFoundError:
                # 如果库文件不存在，使用模拟模式
                self._scheduler = None
                print("[Orion] Warning: libgpu_scheduler.so not found, using mock mode")
    
    def set_scheduling_mode(self, mode: SchedulingMode):
        """
        设置调度模式
        
        Args:
            mode: SchedulingMode 枚举值
        """
        self._mode = mode
    
    def start(self, num_clients: int, device_id: int = 0) -> int:
        """
        启动调度器
        
        Args:
            num_clients: 客户端数量（通常等于 micro-batch 数量）
            device_id: GPU 设备 ID（默认为 0）
            
        Returns:
            0 表示成功
        """
        self._ensure_scheduler()
        self._num_clients = num_clients
        self._running = True
        
        # 初始化统计信息
        for i in range(num_clients):
            self._stats[i] = (0, 0)
        
        if self._scheduler is not None:
            return self._scheduler.start(num_clients, device_id)
        return 0
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler is not None:
            self._scheduler.stop()
    
    def is_running(self) -> bool:
        """
        检查调度器是否正在运行
        
        Returns:
            True 表示正在运行
        """
        return self._running
    
    def set_client_idx(self, idx: int):
        """
        设置当前线程的 client index
        
        Args:
            idx: 客户端索引，0 表示 HP，>0 表示 BE
        """
        if self._scheduler is not None:
            self._scheduler.set_client_idx(idx)
        
        # 更新统计信息
        if idx in self._stats:
            scheduled, waiting = self._stats[idx]
            self._stats[idx] = (scheduled + 1, waiting)
    
    def get_client_idx(self) -> int:
        """
        获取当前线程的 client index
        
        Returns:
            客户端索引
        """
        if self._scheduler is not None:
            return self._scheduler.get_client_idx()
        return -1
    
    def sync_stream(self, client_idx: int):
        """
        同步特定客户端的 CUDA stream
        
        Args:
            client_idx: 要同步的客户端索引
        """
        if self._scheduler is not None:
            self._scheduler.sync_stream(client_idx)
    
    def set_sm_threshold(self, threshold: int):
        """
        设置 SM 阈值
        
        Args:
            threshold: SM 阈值
        """
        if self._scheduler is not None:
            self._scheduler.set_sm_threshold(threshold)
    
    def get_sm_threshold(self) -> int:
        """
        获取当前 SM 阈值
        
        Returns:
            当前 SM 阈值
        """
        if self._scheduler is not None:
            return self._scheduler.get_sm_threshold()
        return 0
    
    def get_stats(self, client_idx: int) -> tuple:
        """
        获取指定客户端的统计信息
        
        Args:
            client_idx: 客户端索引
            
        Returns:
            (scheduled_count, waiting_count) 元组
        """
        return self._stats.get(client_idx, (0, 0))
    
    def reset_state(self):
        """重置调度器状态"""
        if self._scheduler is not None:
            self._scheduler.reset_state()
        self._stats = {i: (0, 0) for i in range(self._num_clients)}


# 全局调度器实例
_global_scheduler: Optional[OrionSchedulerWrapper] = None


def get_scheduler() -> OrionSchedulerWrapper:
    """
    获取调度器单例
    
    Returns:
        OrionSchedulerWrapper 实例
    """
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = OrionSchedulerWrapper()
    return _global_scheduler