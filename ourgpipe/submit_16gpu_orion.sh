#!/bin/bash
#SBATCH --job-name=gpt_16gpu_orion
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpt_16gpu_orion_%j.out
#SBATCH --error=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpt_16gpu_orion_%j.err

# ============================================================================
# GPT 16-GPU Orion 调度器训练脚本
# 
# 使用方法:
#   mkdir -p logs
#   sbatch submit_16gpu_orion.sh
#
# 架构:
#   Node 0: Stage 1-4   (GPU 0-3)
#   Node 1: Stage 5-8   (GPU 0-3)
#   Node 2: Stage 9-12  (GPU 0-3)
#   Node 3: Stage 13-16 (GPU 0-3)
#
# 调度器: Async Threaded + Orion 多优先级调度
# TF32: 开启（提升性能）
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

# 启用 Orion 调度器
# export USE_ORION_SCHEDULER=1

# # Orion 库路径
# export LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so

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
echo "GPT 16-GPU Orion Scheduler Training"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Num Nodes:    $SLURM_NNODES"
echo "GPUs/Node:    $GPUS_PER_NODE"
echo "Total GPUs:   $WORLD_SIZE"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "Scheduler:    Async Threaded + Orion"
echo "TF32:         Enabled"
echo "Config:       configs/gpt_16gpu_orion.yaml"
echo "Orion Lib:    $LD_PRELOAD"
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
    gpipe_runner.py --config configs/gpt_16gpu_orion.yaml

echo "=========================================="
echo "Training completed!"
echo "Check logs at: logs/gpt_16gpu_orion_${SLURM_JOB_ID}.out"
echo "=========================================="
