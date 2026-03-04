#!/bin/bash
# ============================================================================
# 2 节点 8 GPU 通用提交脚本
#
# 支持所有 5 种 pipeline 调度器：
#   naive, async_threaded, 1f1b, zerobubble, hanayo
#
# 使用方法 (通过参数指定配置文件):
#   mkdir -p logs
#
#   # GPipe Naive (同步阻塞)
#   sbatch submit_2nodes_all.sh configs/gpt_8gpu_naive.yaml
#
#   # 1F1B (One Forward One Backward)
#   sbatch submit_2nodes_all.sh configs/gpt_8gpu_1f1b.yaml
#
#   # ZeroBubble (B/W 分离，延迟 W 计算填充 bubble)
#   sbatch submit_2nodes_all.sh configs/gpt_8gpu_zerobubble.yaml
#
#   # Hanayo (双向流水线，Wave A + Wave B)
#   sbatch submit_2nodes_all.sh configs/gpt_8gpu_hanayo.yaml
#
#   # Async Threaded (异步多线程 GPipe)
#   sbatch submit_2nodes_all.sh configs/gpt_8gpu_async.yaml
#
#   # 也可以用命令行覆盖调度器类型:
#   sbatch submit_2nodes_all.sh configs/gpt_8gpu_naive.yaml --scheduler 1f1b
#
# 一键提交所有调度器 (用于性能对比):
#   for cfg in naive 1f1b zerobubble hanayo async; do
#       sbatch submit_2nodes_all.sh configs/gpt_8gpu_${cfg}.yaml
#       sleep 5
#   done
#
# 查看日志:
#   tail -f logs/pipeline_8gpu_<JOB_ID>.out
#   squeue -u $USER
# ============================================================================

#SBATCH --job-name=pipeline_8gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/pipeline_8gpu_%j.out
#SBATCH --error=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/pipeline_8gpu_%j.err
#SBATCH --exclude=paraai-n32-h-01-agent-74,paraai-n32-h-01-agent-75,paraai-n32-h-01-agent-198,paraai-n32-h-01-agent-199,paraai-n32-h-01-agent-227,paraai-n32-h-01-agent-228

# 获取配置文件参数（默认使用 1f1b）
CONFIG=${1:-configs/gpt_8gpu_1f1b.yaml}
# 额外参数（如 --scheduler 1f1b）
EXTRA_ARGS="${@:2}"

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

# 验证 SLURM 分配的 GPU 数量是否满足要求
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    ACTUAL_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    if [ "$ACTUAL_GPUS" -lt "$GPUS_PER_NODE" ]; then
        echo "ERROR: SLURM only allocated $ACTUAL_GPUS GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES), but $GPUS_PER_NODE are required."
        echo "Please ensure the target nodes have enough GPUs, or reduce GPUS_PER_NODE."
        exit 1
    fi
fi

# 计算总进程数
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# 禁用 Python 输出缓冲
export PYTHONUNBUFFERED=1

# NCCL 优化设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # 启用 InfiniBand（如果可用）
export NCCL_NET_GDR_LEVEL=0       # 降低 GDR 要求，避免不支持 GPUDirect RDMA 时 P2P 失败
export NCCL_IB_GID_INDEX=3        # InfiniBand GID 索引
export NCCL_P2P_DISABLE=0         # 确保 P2P 未被禁用
export NCCL_P2P_LEVEL=SYS         # 允许跨节点 P2P
export NCCL_SOCKET_IFNAME=eth0,bond0  # TCP socket 回退（多网卡兼容）

# PyTorch 分布式设置
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# GPU 诊断：打印每个节点实际可见的 GPU 数量
echo "=========================================="
echo "Pipeline Parallel Training (2 Nodes)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Num Nodes:    $SLURM_NNODES"
echo "GPUs/Node:    $GPUS_PER_NODE"
echo "Total GPUs:   $WORLD_SIZE"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "Config:       $CONFIG"
echo "Extra Args:   $EXTRA_ARGS"
echo "Script Dir:   $SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

# 在每个节点上打印 GPU 信息
srun --ntasks-per-node=1 bash -c 'echo "Node $(hostname): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}, nvidia-smi GPU count: $(nvidia-smi -L 2>/dev/null | wc -l)"'

# 使用 srun 启动分布式训练
srun --kill-on-bad-exit=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    pipeline_runner.py --config $CONFIG $EXTRA_ARGS

echo "=========================================="
echo "Training completed!"
echo "Check logs at: logs/pipeline_8gpu_${SLURM_JOB_ID}.out"
echo "=========================================="
