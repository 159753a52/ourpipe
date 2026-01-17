#!/bin/bash
#SBATCH --job-name=gpipe_orion
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/gpipe_%j.out
#SBATCH --error=logs/gpipe_%j.err

# ============================================================
# GPipe 流水线并行训练 - 4节点配置（每节点1个GPU）
# 
# 使用方式：
#   mkdir -p logs
#   sbatch submit_gpipe_4nodes.sh
#
# 查看作业状态：
#   squeue -u $USER
#
# 取消作业：
#   scancel <job_id>
# ============================================================

# 创建日志目录
mkdir -p logs

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: 1"
echo "Start Time: $(date)"
echo "============================================"

# 加载必要的模块
module load compilers/cuda/12.1 compilers/gcc/11.3.0
export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH

# 获取主节点地址
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# 设置分布式训练环境变量
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=1

# 是否启用 Orion 调度器（设置为1启用，0禁用）
export USE_ORION_SCHEDULER=${USE_ORION_SCHEDULER:-0}

if [ "$USE_ORION_SCHEDULER" == "1" ]; then
    echo "Orion Scheduler: ENABLED"
    ORION_LIB="${SCRIPT_DIR}/../kernel_intercept/build/libgpu_scheduler.so"
    if [ -f "$ORION_LIB" ]; then
        export LD_PRELOAD="$ORION_LIB"
        echo "Loaded Orion library: $ORION_LIB"
    else
        echo "Warning: Orion library not found at $ORION_LIB"
    fi
else
    echo "Orion Scheduler: DISABLED (baseline mode)"
fi

# 使用 srun 启动分布式训练
# 每个节点运行一个任务，每个任务使用1个GPU
srun --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=0
    echo "Node: $(hostname), Rank: $RANK, Local Rank: $LOCAL_RANK"
    python gpipe_thread-stream.py
'

echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"