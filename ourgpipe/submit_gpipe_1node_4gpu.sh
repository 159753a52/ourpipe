#!/bin/bash
#SBATCH --job-name=gpipe_orion
#SBATCH --partition=gpu_mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpipe_%j.out
#SBATCH --error=/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe/logs/gpipe_%j.err

# ============================================================
# GPipe 流水线并行训练 - 单节点4GPU配置（推荐）
#
# 使用方式：
#   mkdir -p logs
#   sbatch submit_gpipe_1node_4gpu.sh
#
# 启用 Orion 调度器：
#   USE_ORION_SCHEDULER=1 sbatch submit_gpipe_1node_4gpu.sh
#
# 查看作业状态：
#   squeue -u $USER
#
# 取消作业：
#   scancel <job_id>
# ============================================================

# 使用绝对路径
SCRIPT_DIR="/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
echo "Start Time: $(date)"
echo "============================================"

# 加载必要的模块
module load compilers/cuda/12.1 compilers/gcc/11.3.0
export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH

# 是否启用 Orion 调度器（设置为1启用，0禁用）
export USE_ORION_SCHEDULER=${USE_ORION_SCHEDULER:-0}

if [ "$USE_ORION_SCHEDULER" == "1" ]; then
    echo "Orion Scheduler: ENABLED"
    ORION_LIB="/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so"
    if [ -f "$ORION_LIB" ]; then
        export LD_PRELOAD="$ORION_LIB"
        echo "Loaded Orion library: $ORION_LIB"
    else
        echo "Warning: Orion library not found at $ORION_LIB"
    fi
else
    echo "Orion Scheduler: DISABLED (baseline mode)"
fi

# 使用 torchrun 启动分布式训练
# 单节点4GPU配置，使用绝对路径
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    "$SCRIPT_DIR/gpipe_thread-stream.py"

echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"