# Orion-Style GPU Scheduler

基于 Orion 论文的算子级 GPU 调度器，支持多 DNN 任务共享 GPU 并发执行。

## 核心特性

- **LD_PRELOAD 透明拦截** - 无需修改用户代码
- **算子级调度** - 以 kernel 为粒度，而非整个请求
- **多客户端支持** - HP (高优先级) + BE (尽力而为) 调度模型
- **Per-client Stream** - 不同客户端使用不同 CUDA stream 实现并发

## 快速开始

```bash
# 编译
make

# 运行测试
LD_PRELOAD=./build/libgpu_scheduler.so python3 test_gpt_intercept.py
```

## 目录结构

```
├── include/
│   ├── common.h          # 日志、类型定义
│   ├── gpu_capture.h     # 操作捕获接口
│   ├── scheduler.h       # 调度器接口
│   └── kernel_profile.h  # Kernel 分类
├── src/
│   ├── cuda_intercept.cpp    # CUDA API 拦截
│   ├── cublas_intercept.cpp  # cuBLAS/cuBLASLt 拦截
│   ├── cudnn_intercept.cpp   # cuDNN 拦截
│   ├── gpu_capture.cpp       # 操作队列管理
│   └── scheduler.cpp         # 调度器实现
├── docs/                     # 文档
├── GPT.py                    # 测试用 CharGPT 模型
├── test_complementary_scheduling.py  # 并发调度测试
├── profile_chargpt_full.py   # 完整 PyTorch Profiler 分析
└── Makefile
```

## 使用方法

### 基本用法

```bash
# 设置日志级别 (1=ERROR, 2=WARN, 3=INFO, 4=DEBUG)
export ORION_LOG_LEVEL=3

# 使用 LD_PRELOAD 运行
LD_PRELOAD=./build/libgpu_scheduler.so python3 your_script.py
```

### Python API

```python
import ctypes
import threading

# 加载调度器
lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)

# 启动调度器 (2 个客户端)
lib.orion_start_scheduler(2)

# 客户端 0: 高优先级任务
def hp_client():
    lib.orion_set_client_idx(0)
    model(input)
    torch.cuda.synchronize()

# 客户端 1: 尽力而为任务
def be_client():
    lib.orion_set_client_idx(1)
    model(input)
    torch.cuda.synchronize()

# 并发执行
t1 = threading.Thread(target=hp_client)
t2 = threading.Thread(target=be_client)
t1.start(); t2.start()
t1.join(); t2.join()

# 停止调度器
lib.orion_stop_scheduler()
```

## 编译选项

```bash
make                          # 默认编译
make CUDA_PATH=/usr/local/cuda-12.0  # 指定 CUDA 路径
make DEBUG=1                  # 调试版本
make clean                    # 清理
```

**依赖：**
- CUDA Toolkit 11.0+
- GCC 7+ (C++17)
- Python 3.8+, PyTorch 2.0+

## 调度策略

```
Client 0 (HP) ──→ Stream 0 (高优先级)
Client 1 (BE) ──→ Stream 1
Client 2 (BE) ──→ Stream 2
...
```

**调度规则：**
1. HP 操作立即执行
2. BE 操作在资源互补时并发 (compute + memory)
3. SM 占用 < 阈值时允许 BE 并发
