#!/bin/bash
# GPipe with Orion Scheduler
# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 启用 Orion 调度器
export USE_ORION_SCHEDULER=1

# 注意：LD_PRELOAD 方式的 CUDA 拦截与 PyTorch/cuBLAS 有兼容性问题
# 暂时禁用 LD_PRELOAD，使用 Python 模拟的调度器
# 如果需要启用真正的 CUDA 拦截，取消下面的注释并解决兼容性问题

# ORION_LIB="${SCRIPT_DIR}/../kernel_intercept/build/libgpu_scheduler.so"
# if [ -f "$ORION_LIB" ]; then
#     echo "Loading Orion scheduler library: $ORION_LIB"
#     export LD_PRELOAD="$ORION_LIB"
#     export LD_LIBRARY_PATH="/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib:$LD_LIBRARY_PATH"
# fi

echo "Running GPipe with Orion multi-priority scheduler (Python mock)..."

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr="127.0.0.1" \
    --master_port=1232 \
    gpipe_thread-stream.py