#!/bin/bash
# ============================================================================
# GPipe 8-Stage 交互式多节点运行脚本
#
# 使用方法:
#   1. 申请 2 个节点的资源:
#      salloc --nodes=2 --gpus-per-node=4 --time=01:00:00
#
#   2. 运行此脚本:
#      bash run_gpipe_2nodes_interactive.sh
#
# 架构:
#   Node 0: Stage 1-4 (GPU 0-3)
#   Node 1: Stage 5-8 (GPU 0-3)
# ============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 环境设置
source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron

# 检查是否在 SLURM 分配中
if [ -z "$SLURM_JOB_ID" ]; then
    echo "错误: 请先使用 salloc 申请资源"
    echo "示例: salloc --nodes=2 --gpus-per-node=4 --time=01:00:00"
    exit 1
fi

# 获取节点信息
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
MASTER_ADDR=$(echo $NODES | awk '{print $1}')
MASTER_PORT=29500
NNODES=$(echo $NODES | wc -w)
GPUS_PER_NODE=4

# NCCL 优化设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # 启用 InfiniBand（如果可用）
export NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA

# PyTorch 分布式设置
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "=========================================="
echo "GPipe 8-Stage Interactive Training"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $NODES"
echo "Num Nodes:    $NNODES"
echo "GPUs/Node:    $GPUS_PER_NODE"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "World Size:   $((NNODES * GPUS_PER_NODE))"
echo "=========================================="

# 检查节点数量
if [ "$NNODES" -lt 2 ]; then
    echo "警告: 只有 $NNODES 个节点，建议使用 2 个节点"
    echo "继续运行..."
fi

# 使用 srun 启动分布式训练
srun --nodes=$NNODES --ntasks-per-node=$GPUS_PER_NODE \
    python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    gpipe_runner.py --config configs/gpt_8stage.yaml

echo "=========================================="
echo "Training completed!"
echo "=========================================="
