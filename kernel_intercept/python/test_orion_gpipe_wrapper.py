#!/usr/bin/env python3
"""
Orion GPipe Wrapper 测试文件

TDD 测试用例，覆盖率目标 >= 90%

运行方式:
    srun --partition=vip_gpu_scx9kvs --gpus=1 bash -c '
    module load compilers/cuda/12.1 compilers/gcc/11.3.0
    export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH
    cd /home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/python
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
        pytest test_orion_gpipe_wrapper.py -v --cov=orion_gpipe_wrapper --cov-report=term-missing
    '
"""

import pytest
import os
import sys
import threading

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def lib_path():
    """返回 libgpu_scheduler.so 的路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "build", "libgpu_scheduler.so")


@pytest.fixture
def scheduler(lib_path):
    """创建并返回 OrionScheduler 实例，测试后自动清理"""
    from orion_gpipe_wrapper import OrionScheduler
    sched = OrionScheduler(lib_path)
    yield sched
    # 清理：确保调度器停止
    if sched.is_started:
        sched.stop()


# ============================================================================
# 测试类 1：OrionScheduler 初始化测试
# ============================================================================

class TestOrionSchedulerInit:
    """测试 OrionScheduler 初始化"""

    def test_init_with_default_path(self):
        """测试使用默认路径初始化"""
        from orion_gpipe_wrapper import OrionScheduler
        # 默认路径应该能找到库
        scheduler = OrionScheduler()
        assert scheduler.lib is not None
        assert scheduler.is_started is False

    def test_init_with_custom_path(self, lib_path):
        """测试使用自定义路径初始化"""
        from orion_gpipe_wrapper import OrionScheduler
        scheduler = OrionScheduler(lib_path)
        assert scheduler.lib is not None
        assert scheduler.is_started is False

    def test_init_with_invalid_path(self):
        """测试使用无效路径时抛出 FileNotFoundError"""
        from orion_gpipe_wrapper import OrionScheduler
        with pytest.raises(FileNotFoundError) as excinfo:
            OrionScheduler("/nonexistent/path/to/lib.so")
        assert "Library not found" in str(excinfo.value)


# ============================================================================
# 测试类 2：调度器生命周期测试
# ============================================================================

class TestSchedulerLifecycle:
    """测试调度器生命周期管理"""

    def test_start_scheduler(self, scheduler):
        """测试启动调度器"""
        ret = scheduler.start(num_clients=2)
        assert ret == 0
        assert scheduler.is_started is True

    def test_start_scheduler_multiple_clients(self, scheduler):
        """测试启动多客户端调度器"""
        ret = scheduler.start(num_clients=4)
        assert ret == 0
        assert scheduler.is_started is True

    def test_stop_scheduler(self, scheduler):
        """测试停止调度器"""
        scheduler.start(num_clients=2)
        assert scheduler.is_started is True
        scheduler.stop()
        assert scheduler.is_started is False

    def test_stop_without_start(self, scheduler):
        """测试未启动时停止不会出错"""
        assert scheduler.is_started is False
        scheduler.stop()  # 不应该抛出异常
        assert scheduler.is_started is False

    def test_is_started_property(self, scheduler):
        """测试 is_started 属性"""
        assert scheduler.is_started is False
        scheduler.start(num_clients=2)
        assert scheduler.is_started is True
        scheduler.stop()
        assert scheduler.is_started is False


# ============================================================================
# 测试类 3：客户端索引测试
# ============================================================================

class TestClientIndex:
    """测试客户端索引设置和获取"""

    def test_set_client_idx_hp(self, scheduler):
        """测试设置 HP 客户端 (idx=0)"""
        scheduler.start(num_clients=2)
        scheduler.set_client_idx(0)
        assert scheduler.get_client_idx() == 0

    def test_set_client_idx_be(self, scheduler):
        """测试设置 BE 客户端 (idx>0)"""
        scheduler.start(num_clients=4)
        scheduler.set_client_idx(1)
        assert scheduler.get_client_idx() == 1
        scheduler.set_client_idx(2)
        assert scheduler.get_client_idx() == 2
        scheduler.set_client_idx(3)
        assert scheduler.get_client_idx() == 3

    def test_get_client_idx_after_set(self, scheduler):
        """测试设置后获取客户端索引"""
        # client_idx 是线程局部变量，设置后应该能获取到
        scheduler.start(num_clients=2)
        scheduler.set_client_idx(0)
        assert scheduler.get_client_idx() == 0

    def test_client_idx_thread_local(self, scheduler):
        """测试 client_idx 是线程局部的"""
        scheduler.start(num_clients=4)
        results = {}

        def worker(idx):
            scheduler.set_client_idx(idx)
            # 短暂等待，确保其他线程也设置了
            import time
            time.sleep(0.01)
            results[idx] = scheduler.get_client_idx()

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 每个线程应该获取到自己设置的值
        for i in range(4):
            assert results[i] == i


# ============================================================================
# 测试类 4：SM 阈值测试
# ============================================================================

class TestSMThreshold:
    """测试 SM 阈值设置和获取"""

    def test_set_sm_threshold(self, scheduler):
        """测试设置 SM 阈值"""
        scheduler.start(num_clients=2)
        scheduler.set_sm_threshold(30)
        # 设置后应该能获取到
        assert scheduler.get_sm_threshold() == 30

    def test_get_sm_threshold(self, scheduler):
        """测试获取 SM 阈值"""
        scheduler.start(num_clients=2)
        # 默认阈值是 num_sms / 2，A100 有 108 个 SM，所以默认是 54
        threshold = scheduler.get_sm_threshold()
        assert threshold > 0

    def test_set_different_thresholds(self, scheduler):
        """测试设置不同的 SM 阈值"""
        scheduler.start(num_clients=2)
        for threshold in [10, 30, 50, 80]:
            scheduler.set_sm_threshold(threshold)
            assert scheduler.get_sm_threshold() == threshold


# ============================================================================
# 测试类 5：流同步测试
# ============================================================================

class TestStreamSync:
    """测试流同步功能"""

    def test_sync_hp_stream(self, scheduler):
        """测试同步 HP 流"""
        scheduler.start(num_clients=2)
        # 同步 HP 流（client_idx=0）不应该抛出异常
        scheduler.sync_stream(0)

    def test_sync_be_stream(self, scheduler):
        """测试同步 BE 流"""
        scheduler.start(num_clients=4)
        # 同步 BE 流不应该抛出异常
        scheduler.sync_stream(1)
        scheduler.sync_stream(2)
        scheduler.sync_stream(3)

    def test_sync_all_streams(self, scheduler):
        """测试同步所有流"""
        num_clients = 4
        scheduler.start(num_clients=num_clients)
        for i in range(num_clients):
            scheduler.sync_stream(i)


# ============================================================================
# 测试类 6：状态重置测试
# ============================================================================

class TestResetState:
    """测试状态重置功能"""

    def test_reset_state(self, scheduler):
        """测试重置调度器状态"""
        scheduler.start(num_clients=2)
        scheduler.set_sm_threshold(30)
        # 重置状态不应该抛出异常
        scheduler.reset_state()


# ============================================================================
# 测试类 7：集成测试（需要 GPU）
# ============================================================================

@pytest.mark.gpu
class TestGPipeIntegration:
    """GPipe 集成测试（需要 GPU 环境）"""

    def test_microbatch_priority_mapping(self, scheduler):
        """测试 micro-batch 到优先级的映射"""
        num_microbatches = 4
        scheduler.start(num_clients=num_microbatches)
        scheduler.set_sm_threshold(30)

        # 模拟 GPipe 的 micro-batch 映射
        # mb_idx=0 → HP, mb_idx=1,2,3 → BE
        for mb_idx in range(num_microbatches):
            scheduler.set_client_idx(mb_idx)
            assert scheduler.get_client_idx() == mb_idx

    def test_multi_thread_scheduling(self, scheduler):
        """测试多线程调度场景"""
        num_clients = 4
        scheduler.start(num_clients=num_clients)
        scheduler.set_sm_threshold(30)

        errors = []

        def worker(idx):
            try:
                scheduler.set_client_idx(idx)
                # 验证设置成功
                if scheduler.get_client_idx() != idx:
                    errors.append(f"Thread {idx}: expected {idx}, got {scheduler.get_client_idx()}")
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = []
        for i in range(num_clients):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=orion_gpipe_wrapper", "--cov-report=term-missing"])
