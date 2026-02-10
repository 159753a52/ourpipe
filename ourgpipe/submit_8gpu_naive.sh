#!/bin/bash
#SBATCH --job-name=gpt_8gpu_naive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpt_8gpu_naive_%j.out
#SBATCH --error=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpt_8gpu_naive_%j.err

# ============================================================================
# GPT 8-GPU Naive 调度器训练脚本
# 
# 使用方法:
#   mkdir -p logs
#   sbatch submit_8gpu_naive.sh
#
# 架构:
#   Node 0: Stage 1-4 (GPU 0-3)
#   Node 1: Stage 5-8 (GPU 0-3)
#
# 调度器: Naive (同步阻塞)
# TF32: 关闭（为了公平对比）
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

# 禁用 Python 输出缓冲
export PYTHONUNBUFFERED=1

# PyTorch 分布式设置
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "=========================================="
echo "GPT 8-GPU Naive Scheduler Training"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Num Nodes:    $SLURM_NNODES"
echo "GPUs/Node:    $GPUS_PER_NODE"
echo "Total GPUs:   $WORLD_SIZE"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "Scheduler:    Naive (Sync Blocking)"
echo "TF32:         Disabled"
echo "Config:       configs/gpt_8gpu_naive.yaml"
echo "Script Dir:   $SCRIPT_DIR"
echo "=========================================="

# 使用 srun 启动分布式训练
srun --kill-on-bad-exit=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    gpipe_runner.py --config configs/gpt_8gpu_naive.yaml

echo "=========================================="
echo "Training completed!"
echo "Check logs at: logs/gpt_8gpu_naive_${SLURM_JOB_ID}.out"
echo "=========================================="
