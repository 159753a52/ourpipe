#!/bin/bash
# 运行 GPipe Naive 调度器测试
#
# 使用方法:
#   bash run_gpipe_naive.sh
#
# 这个脚本使用 Naive 同步调度器，适合调试和基准测试

set -e

# 环境设置
source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron

# 设置环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 运行参数
NPROC_PER_NODE=4
MASTER_PORT=29500
CONFIG_FILE="configs/gpt_small_naive.yaml"

echo "=========================================="
echo "Running GPipe with Naive Scheduler"
echo "Config: ${CONFIG_FILE}"
echo "Processes per node: ${NPROC_PER_NODE}"
echo "=========================================="

# 运行训练
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    gpipe_runner.py \
    --config ${CONFIG_FILE}

echo "=========================================="
echo "Training completed!"
echo "=========================================="
