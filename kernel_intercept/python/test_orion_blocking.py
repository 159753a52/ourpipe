#!/usr/bin/env python3
"""
Orion 调度测试脚本（重构后版本）

测试 HP (High-Priority) 和多个 BE (Best-Effort) 客户端的并发执行。

=== Orion 调度核心概念 ===
1. HP (High-Priority): 高优先级客户端，通常是延迟敏感的推理任务
2. BE (Best-Effort): 尽力而为客户端，通常是训练任务，可以被延迟
3. SM (Streaming Multiprocessor): GPU 的计算单元，Orion 通过控制 SM 分配来调度

=== 调度策略（简化版）===
- HP kernel 直接执行
- BE kernel 在 HP 队列为空时执行
- 内存操作直接执行

=== 编译方法 ===
在登录节点上编译 libgpu_scheduler.so:

    cd /home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept
    module load compilers/cuda/12.1 compilers/gcc/11.3.0
    make clean
    make CUDA_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1 \
         CUDNN_PATH=/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x \
         LDFLAGS="-shared -fPIC \
                  -L/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64 \
                  -L/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64/stubs \
                  -lcudart -lcuda -lnvToolsExt \
                  -L/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64 \
                  -l:libcudnn.so.8 -lcublas -lcublasLt -ldl -lpthread"

=== 在 GPU 节点上运行（使用 srun）===
使用 VIP 分区申请 GPU 节点并运行测试:

    srun --partition=vip_gpu_scx9kvs --gpus=1 bash -c '
    module load compilers/cuda/12.1 compilers/gcc/11.3.0
    export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH
    cd /home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/python
    export PYTHONPATH=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/python:$PYTHONPATH
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so python3 test_orion_blocking.py --num-be 1 --num-iters 2
    '

或者使用普通 GPU 分区:

    srun --gpus=1 bash -c '
    module load compilers/cuda/12.1 compilers/gcc/11.3.0
    export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH
    cd /home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/python
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so python3 test_orion_blocking.py --num-be 1 --num-iters 2
    '

=== 简单使用方法（在 GPU 节点上）===
    # 1 HP + 1 BE (默认)
    LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py

    # 1 HP + 2 BE
    LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py --num-be 2

    # 1 HP + 3 BE, 4 iterations
    LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py --num-be 3 --num-iters 4
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import ctypes
import sys
import threading
import time
import os
import json
import argparse

sys.path.insert(0, ".")
import GPT as gpt_module

# ============================================================================
# 模型配置
# ============================================================================
EMB_SIZE = 4096
HEAD_SIZE = 8
N_LAYER = 1
SEQUENCE_LEN = 1024
BATCH_SIZE = 8
VOCAB_SIZE = 65

gpt_module.emb_size = EMB_SIZE
gpt_module.head_size = HEAD_SIZE
gpt_module.n_layer = N_LAYER
gpt_module.sequence_len = SEQUENCE_LEN

from GPT import CharGPT


def save_results_tsv(result, num_be, output_file=None):
    """保存结果到TSV文件"""
    if output_file is None:
        output_file = f"profiles/orion_1hp_{num_be}be_results.tsv"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("method\thp_time_ms\ttotal_time_ms\n")
        f.write(f"orion_1hp_{num_be}be\t{result['hp']:.2f}\t{result['total']:.2f}\n")
    print(f"Results saved to {output_file}")


def add_thread_names_to_trace(trace_file, thread_ids, num_clients):
    """后处理 Chrome trace 文件，添加线程名称标记"""
    with open(trace_file, 'r') as f:
        trace = json.load(f)

    events = trace.get('traceEvents', [])
    if not events:
        return

    pid = None
    for e in events:
        if e.get('pid') and isinstance(e.get('pid'), int):
            pid = e.get('pid')
            break
    if pid is None:
        return

    scheduler_tids = {}
    client_tid_set = set(thread_ids.values())
    for e in events:
        tid = e.get('tid')
        cat = e.get('cat', '')
        if tid and isinstance(tid, int) and tid > 100000 and tid not in client_tid_set:
            if 'cuda_runtime' in cat and tid not in scheduler_tids:
                scheduler_tids[tid] = len(scheduler_tids)

    stream_tids = {}
    for e in events:
        tid = e.get('tid')
        cat = e.get('cat', '')
        if tid and isinstance(tid, int) and tid < 100 and 'kernel' in cat:
            if tid not in stream_tids:
                stream_tids[tid] = len(stream_tids)

    tids_to_rename = set(thread_ids.values()) | set(scheduler_tids.keys()) | set(stream_tids.keys())
    new_events = []
    for e in events:
        if e.get('name') == 'thread_name' and e.get('tid') in tids_to_rename:
            continue
        new_events.append(e)

    for client_idx, tid in thread_ids.items():
        name = "HP-Client" if client_idx == 0 else f"BE{client_idx}-Client"
        new_events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": name}})

    for i, sched_tid in enumerate(sorted(scheduler_tids.keys())):
        name = "Scheduler"
        new_events.append({"ph": "M", "pid": pid, "tid": sched_tid, "name": "thread_name", "args": {"name": name}})

    for i, stream_tid in enumerate(sorted(stream_tids.keys())):
        name = "GPU-Stream-HP" if i == 0 else f"GPU-Stream-BE{i}"
        new_events.append({"ph": "M", "pid": 0, "tid": stream_tid, "name": "thread_name", "args": {"name": name}})

    trace['traceEvents'] = new_events
    with open(trace_file, 'w') as f:
        json.dump(trace, f)


def load_library():
    """加载 Orion 调度器的共享库"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    lib_path = os.path.join(project_root, "build", "libgpu_scheduler.so")

    if not os.path.exists(lib_path):
        print(f"ERROR: Library not found at {lib_path}")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

    lib.orion_start_scheduler.argtypes = [ctypes.c_int]
    lib.orion_start_scheduler.restype = ctypes.c_int
    lib.orion_stop_scheduler.restype = None
    lib.orion_set_client_idx.argtypes = [ctypes.c_int]
    lib.orion_set_client_idx.restype = None
    lib.orion_load_kernel_info.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.orion_load_kernel_info.restype = ctypes.c_int
    lib.orion_set_client_kernels.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.orion_set_client_kernels.restype = None
    lib.orion_set_sm_threshold.argtypes = [ctypes.c_int]
    lib.orion_set_sm_threshold.restype = None
    lib.orion_get_sm_threshold.argtypes = []
    lib.orion_get_sm_threshold.restype = ctypes.c_int
    lib.orion_sync_client_stream.argtypes = [ctypes.c_int]
    lib.orion_sync_client_stream.restype = None
    lib.orion_reset_state.restype = None

    print("Library loaded successfully")
    return lib


def run_test(lib, num_be, num_iters, kernel_info_path, trace_file, sm_threshold=None):
    """
    运行 Orion 调度测试

    Args:
        lib: 调度器共享库
        num_be: BE 任务数量
        num_iters: 每个任务的迭代次数
        kernel_info_path: kernel profile CSV 文件路径
        trace_file: 输出的 Chrome trace 文件路径
        sm_threshold: SM 阈值
    """
    num_clients = 1 + num_be  # 1 HP + num_be BE

    print(f"\n{'='*60}")
    print(f"Orion Scheduling Test")
    print(f"{'='*60}")
    print(f"Configuration: 1 HP + {num_be} BE, {num_iters} iterations each")

    # 启动调度器
    ret = lib.orion_start_scheduler(num_clients)
    if ret != 0:
        print("ERROR: Failed to start scheduler")
        return None
    print(f"Scheduler started with {num_clients} clients")

    if sm_threshold is not None:
        lib.orion_set_sm_threshold(sm_threshold)
    current_threshold = lib.orion_get_sm_threshold()
    print(f"SM threshold: {current_threshold}")

    # 创建模型
    print("\nCreating model...")
    model = CharGPT(vs=VOCAB_SIZE).to("cuda")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M")
    print(f"  Config: emb={EMB_SIZE}, heads={HEAD_SIZE}, layers={N_LAYER}")
    print(f"  Input: batch={BATCH_SIZE}, seq_len={SEQUENCE_LEN}")

    # 为每个客户端创建输入数据
    inputs = [torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LEN), device="cuda")
              for _ in range(num_clients)]
    torch.cuda.synchronize()

    # Warmup
    print("\nWarmup...")
    lib.orion_set_client_idx(0)
    with torch.no_grad():
        _ = model(inputs[0])
    torch.cuda.synchronize()

    # 加载 kernel profile
    if kernel_info_path and os.path.exists(kernel_info_path):
        print(f"\nLoading kernel profiles from {kernel_info_path}...")
        for i in range(num_clients):
            ret = lib.orion_load_kernel_info(i, kernel_info_path.encode())
            print(f"  Client {i}: loaded {ret} kernel profiles")
            lib.orion_set_client_kernels(i, ret if ret > 0 else 100)
    else:
        print("\nNo kernel profile file, using default scheduling")

    # ========================================================================
    # 时间测量说明
    # ========================================================================
    # 使用 orion_sync_client_stream() 只等待特定客户端的 stream 完成
    # - HP (client 0) 的操作在 hp_stream_ 上执行
    # - BE (client 1+) 的操作在 be_streams_[client_idx-1] 上执行
    # - orion_sync_client_stream(idx) 只同步该客户端的 stream
    # ========================================================================
    streams = [torch.cuda.Stream() for _ in range(num_clients)]
    barrier = threading.Barrier(num_clients)
    start = threading.Event()
    done = [threading.Event() for _ in range(num_clients)]
    client_times = [0.0] * num_clients
    client_start_times = [0.0] * num_clients
    client_end_times = [0.0] * num_clients
    thread_ids = {}

    def worker(idx):
        """客户端工作线程"""
        libc = ctypes.CDLL('libc.so.6')
        SYS_gettid = 186
        tid = libc.syscall(SYS_gettid)
        thread_ids[idx] = tid

        lib.orion_set_client_idx(idx)
        client_type = "HP" if idx == 0 else f"BE{idx}"

        # 等待所有线程同时开始
        start.wait()
        barrier.wait()

        # 开始计时
        t0 = time.time()
        client_start_times[idx] = t0

        # 执行模型推理
        # 算子被拦截并提交到调度器队列，在调度器的 stream 上执行
        with torch.cuda.stream(streams[idx]):
            with torch.no_grad():
                for i in range(num_iters):
                    _ = model(inputs[idx])

        # 只同步该客户端的 stream（不是全局同步）
        lib.orion_sync_client_stream(idx)

        # 结束计时
        t1 = time.time()
        client_end_times[idx] = t1
        client_times[idx] = (t1 - t0) * 1000

        print(f"  {client_type}: {num_iters} iters in {client_times[idx]:.2f} ms")
        done[idx].set()

    # 创建并启动工作线程
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_clients)]
    for t in threads:
        t.start()

    time.sleep(0.2)

    os.makedirs(os.path.dirname(trace_file) if os.path.dirname(trace_file) else ".", exist_ok=True)

    print(f"\nRunning {num_iters} iterations per client...")

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        start.set()

        for d in done:
            d.wait()

        for t in threads:
            t.join()

        torch.cuda.synchronize()
        time.sleep(0.3)

    # 计算总时间：所有流上第一个算子开始到最后一个算子结束
    first_start = min(client_start_times)
    last_end = max(client_end_times)
    total_time = (last_end - first_start) * 1000

    prof.export_chrome_trace(trace_file)
    add_thread_names_to_trace(trace_file, thread_ids, num_clients)
    print(f"\nTrace saved to {trace_file}")

    # 输出结果
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"HP (client 0): {client_times[0]:.2f} ms")
    for i in range(1, num_clients):
        print(f"BE{i} (client {i}): {client_times[i]:.2f} ms")

    lib.orion_stop_scheduler()

    return {
        'total': total_time,
        'hp': client_times[0],
        'be': client_times[1:],
        'sm_threshold': current_threshold
    }


def main():
    parser = argparse.ArgumentParser(description='Orion Scheduling Test')
    parser.add_argument('--num-be', type=int, default=1, help='Number of BE tasks (default: 1)')
    parser.add_argument('--num-iters', type=int, default=2, help='Iterations per task (default: 2)')
    parser.add_argument('--sm-threshold', type=int, default=30, help='SM threshold (default: 30)')
    parser.add_argument('--output', type=str, default=None, help='Output trace file path')
    args = parser.parse_args()

    lib = load_library()

    kernel_info_paths = [
        "/lihongliang/fangzl/kernel_intercept/profiles/kernel_info.csv",
        "/lihongliang/fangzl/kernel_intercept/profiles/kernel_info_high_sm.csv",
    ]

    kernel_info = None
    for path in kernel_info_paths:
        if os.path.exists(path):
            kernel_info = path
            break

    if kernel_info:
        print(f"Using kernel profile: {kernel_info}")
    else:
        print("No kernel profile found, using default scheduling")

    trace_file = args.output or f"profiles/orion_1hp_{args.num_be}be_trace.json"
    result = run_test(lib, args.num_be, args.num_iters, kernel_info, trace_file, args.sm_threshold)

    if result:
        print(f"\n{'='*60}")
        print("TEST COMPLETE")
        print(f"{'='*60}")
        print(f"\nView trace in Chrome: chrome://tracing -> Load {trace_file}")

        # 保存TSV结果
        save_results_tsv(result, args.num_be)


if __name__ == "__main__":
    main()
