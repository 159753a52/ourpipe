#!/bin/bash
#SBATCH --job-name=gpipe_4stage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpipe_4stage_%j.out
#SBATCH --error=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpipe_4stage_%j.err

# ============================================================================
# GPipe 4-Stage 单节点训练脚本
# 
# 使用方法:
#   mkdir -p logs
#   sbatch submit_gpipe_1node.sh
#
# 架构:
#   Node 0: Stage 1-4 (GPU 0-3)
# ============================================================================

# 使用绝对路径
SCRIPT_DIR="/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p logs

# 环境设置
source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron

# 获取主节点信息
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# 每个节点的 GPU 数量
GPUS_PER_NODE=4

# 计算总进程数
export WORLD_SIZE=$GPUS_PER_NODE

# 禁用 Python 输出缓冲
export PYTHONUNBUFFERED=1

# NCCL 优化设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # 单节点禁用 InfiniBand

# PyTorch 分布式设置
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "=========================================="
echo "GPipe 4-Stage Single-Node Training"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $GPUS_PER_NODE"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "World Size:   $WORLD_SIZE"
echo "Script Dir:   $SCRIPT_DIR"
echo "=========================================="

# 使用 srun 启动分布式训练
# 单节点上启动 4 个进程
srun --kill-on-bad-exit=1 \
    python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    gpipe_runner.py --config configs/gpt_large_1node.yaml

echo "=========================================="
echo "Training completed!"
echo "=========================================="
