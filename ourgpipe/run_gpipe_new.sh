#!/bin/bash
# GPipe 流水线训练运行脚本（新架构）
#
# 使用方法:
#   # 基本运行（不启用 Orion）
#   ./run_gpipe_new.sh configs/gpt_small.yaml
#
#   # 启用 Orion 调度器
#   ./run_gpipe_new.sh configs/gpt_small_orion.yaml --orion

set -e

# 默认配置
CONFIG_FILE="${1:-configs/gpt_small.yaml}"
NPROC_PER_NODE=4
MASTER_PORT=29500

# 检查是否启用 Orion
USE_ORION=0
if [[ "$2" == "--orion" ]] || [[ "$CONFIG_FILE" == *"orion"* ]]; then
    USE_ORION=1
fi

# 设置环境
cd "$(dirname "$0")"

echo "=========================================="
echo "GPipe Pipeline Training (New Architecture)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Processes per node: $NPROC_PER_NODE"
echo "Orion scheduler: $USE_ORION"
echo "=========================================="

# 设置环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ "$USE_ORION" -eq 1 ]; then
    export USE_ORION_SCHEDULER=1
    
    # Orion 库路径
    ORION_LIB="../kernel_intercept/build/libgpu_scheduler.so"
    
    if [ ! -f "$ORION_LIB" ]; then
        echo "Error: Orion library not found at $ORION_LIB"
        echo "Please build it first: cd ../kernel_intercept && make"
        exit 1
    fi
    
    echo "Running with Orion scheduler..."
    LD_PRELOAD="$ORION_LIB" \
        torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
        gpipe_runner.py --config "$CONFIG_FILE"
else
    echo "Running without Orion scheduler..."
    torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
        gpipe_runner.py --config "$CONFIG_FILE"
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="