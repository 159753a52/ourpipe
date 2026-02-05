#!/bin/bash
#SBATCH --job-name=gpipe_8stage
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpipe_8stage_%j.out
#SBATCH --error=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpipe_8stage_%j.err

# ============================================================================
# GPipe 8-Stage 多节点训练脚本
# 
# 使用方法:
#   sbatch submit_gpipe_2nodes.sh
#
# 架构:
#   Node 0: Stage 1-4 (GPU 0-3)
#   Node 1: Stage 5-8 (GPU 0-3)
# ============================================================================

# 使用绝对路径
SCRIPT_DIR="/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe"
cd "$SCRIPT_DIR"

# 环境设置
source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron

# 获取主节点信息
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# 每个节点的 GPU 数量
GPUS_PER_NODE=4

# 计算总进程数
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# NCCL 优化设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # 启用 InfiniBand（如果可用）
export NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3        # InfiniBand GID 索引

# PyTorch 分布式设置
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "=========================================="
echo "GPipe 8-Stage Multi-Node Training"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Num Nodes:    $SLURM_NNODES"
echo "GPUs/Node:    $GPUS_PER_NODE"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "World Size:   $WORLD_SIZE"
echo "Script Dir:   $SCRIPT_DIR"
echo "=========================================="

# 使用 srun 启动分布式训练
# 每个节点启动 1 个 srun 任务，torchrun 在每个节点上启动 4 个进程
srun --kill-on-bad-exit=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    gpipe_runner.py --config configs/gpt_8stage.yaml

echo "=========================================="
echo "Training completed!"
echo "=========================================="
