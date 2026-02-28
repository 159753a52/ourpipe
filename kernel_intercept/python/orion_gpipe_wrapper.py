"""
Orion 调度器的 Python 封装，用于 GPipe 集成

使用方法：
    from orion_gpipe_wrapper import OrionScheduler

    scheduler = OrionScheduler()
    scheduler.start(num_microbatches)
    scheduler.set_sm_threshold(30)

    # 在每个 micro-batch 的计算线程中
    scheduler.set_client_idx(mb_idx)  # mb_idx=0 是 HP，其他是 BE

运行测试：
    srun --partition=vip_gpu_scx9kvs --gpus=1 bash -c '
    module load compilers/cuda/12.1 compilers/gcc/11.3.0
    export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH
    cd /home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/python
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
        pytest test_orion_gpipe_wrapper.py -v --cov=orion_gpipe_wrapper --cov-report=term-missing
    '
"""

import ctypes
import os
from typing import Optional


class OrionScheduler:
    """
    Orion 调度器封装类

    用于在 GPipe 流水线中实现多优先级调度：
    - client_idx=0: HP (High-Priority) 任务，通常是 micro-batch 0
    - client_idx>0: BE (Best-Effort) 任务，通常是 micro-batch 1,2,3,...

    调度策略：
    - HP kernel 直接执行
    - BE kernel 在 HP 队列为空时执行
    - 内存操作直接执行
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        初始化 Orion 调度器

        Args:
            lib_path: libgpu_scheduler.so 的路径，默认为相对于本文件的 ../build/libgpu_scheduler.so

        Raises:
            FileNotFoundError: 如果库文件不存在
        """
        if lib_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(script_dir, "..", "build", "libgpu_scheduler.so")

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        self._setup_functions()
        self._started = False

    def _setup_functions(self):
        """设置 C 函数签名"""
        # 调度器生命周期
        self.lib.orion_start_scheduler.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.orion_start_scheduler.restype = ctypes.c_int
        self.lib.orion_stop_scheduler.restype = None

        # 客户端索引（线程局部）
        self.lib.orion_set_client_idx.argtypes = [ctypes.c_int]
        self.lib.orion_set_client_idx.restype = None
        self.lib.orion_get_client_idx.argtypes = []
        self.lib.orion_get_client_idx.restype = ctypes.c_int

        # 流同步
        self.lib.orion_sync_client_stream.argtypes = [ctypes.c_int]
        self.lib.orion_sync_client_stream.restype = None

        # SM 阈值
        self.lib.orion_set_sm_threshold.argtypes = [ctypes.c_int]
        self.lib.orion_set_sm_threshold.restype = None
        self.lib.orion_get_sm_threshold.argtypes = []
        self.lib.orion_get_sm_threshold.restype = ctypes.c_int

        # 状态重置
        self.lib.orion_reset_state.restype = None

    def start(self, num_clients: int, device_id: int = 0) -> int:
        """
        启动调度器

        Args:
            num_clients: 客户端数量（通常等于 micro-batch 数量）
            device_id: GPU 设备 ID（默认为 0）

        Returns:
            0 表示成功，非 0 表示失败
        """
        ret = self.lib.orion_start_scheduler(num_clients, device_id)
        if ret == 0:
            self._started = True
        return ret

    def stop(self):
        """停止调度器"""
        if self._started:
            self.lib.orion_stop_scheduler()
            self._started = False

    def set_client_idx(self, idx: int):
        """
        设置当前线程的 client index

        Args:
            idx: 客户端索引，0 表示 HP，>0 表示 BE

        注意：这是线程局部的，每个线程需要单独设置
        """
        self.lib.orion_set_client_idx(idx)

    def get_client_idx(self) -> int:
        """
        获取当前线程的 client index

        Returns:
            客户端索引，未设置时返回 -1
        """
        return self.lib.orion_get_client_idx()

    def sync_stream(self, client_idx: int):
        """
        同步特定客户端的 CUDA stream

        Args:
            client_idx: 要同步的客户端索引
        """
        self.lib.orion_sync_client_stream(client_idx)

    def set_sm_threshold(self, threshold: int):
        """
        设置 SM 阈值

        Args:
            threshold: SM 阈值，用于控制 BE 任务的 SM 使用量
        """
        self.lib.orion_set_sm_threshold(threshold)

    def get_sm_threshold(self) -> int:
        """
        获取当前 SM 阈值

        Returns:
            当前 SM 阈值
        """
        return self.lib.orion_get_sm_threshold()

    def reset_state(self):
        """重置调度器状态"""
        self.lib.orion_reset_state()

    @property
    def is_started(self) -> bool:
        """
        检查调度器是否已启动

        Returns:
            True 表示已启动，False 表示未启动
        """
        return self._started
