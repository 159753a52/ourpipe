#!/bin/bash
# ============================================================================
# 快速提交所有 GPT 性能对比实验
# 
# 使用方法:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# 或者单独运行某个实验:
#   ./run_all_experiments.sh 8gpu_naive
#   ./run_all_experiments.sh 8gpu_orion
#   ./run_all_experiments.sh 16gpu_naive
#   ./run_all_experiments.sh 16gpu_orion
# ============================================================================

SCRIPT_DIR="/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p logs

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 提交单个实验
submit_experiment() {
    local exp_name=$1
    local script_name=$2
    
    print_info "Submitting experiment: $exp_name"
    
    if [ ! -f "$script_name" ]; then
        print_error "Script not found: $script_name"
        return 1
    fi
    
    job_id=$(sbatch "$script_name" | awk '{print $4}')
    
    if [ -z "$job_id" ]; then
        print_error "Failed to submit $exp_name"
        return 1
    fi
    
    print_success "Submitted $exp_name with Job ID: $job_id"
    echo "  - Log: logs/${exp_name}_${job_id}.out"
    echo ""
    
    return 0
}

# 主函数
main() {
    echo "=========================================="
    echo "GPT Performance Comparison Experiments"
    echo "=========================================="
    echo ""
    
    # 如果提供了参数，只运行指定的实验
    if [ $# -gt 0 ]; then
        case $1 in
            8gpu_naive)
                submit_experiment "gpt_8gpu_naive" "submit_8gpu_naive.sh"
                ;;
            8gpu_orion)
                submit_experiment "gpt_8gpu_orion" "submit_8gpu_orion.sh"
                ;;
            16gpu_naive)
                submit_experiment "gpt_16gpu_naive" "submit_16gpu_naive.sh"
                ;;
            16gpu_orion)
                submit_experiment "gpt_16gpu_orion" "submit_16gpu_orion.sh"
                ;;
            *)
                print_error "Unknown experiment: $1"
                echo "Available experiments:"
                echo "  - 8gpu_naive"
                echo "  - 8gpu_orion"
                echo "  - 16gpu_naive"
                echo "  - 16gpu_orion"
                exit 1
                ;;
        esac
    else
        # 运行所有实验
        print_info "Submitting all experiments..."
        echo ""
        
        submit_experiment "gpt_8gpu_naive" "submit_8gpu_naive.sh"
        sleep 2
        
        submit_experiment "gpt_8gpu_orion" "submit_8gpu_orion.sh"
        sleep 2
        
        submit_experiment "gpt_16gpu_naive" "submit_16gpu_naive.sh"
        sleep 2
        
        submit_experiment "gpt_16gpu_orion" "submit_16gpu_orion.sh"
    fi
    
    echo "=========================================="
    print_success "All experiments submitted!"
    echo "=========================================="
    echo ""
    print_info "Check job status with: squeue -u \$USER"
    print_info "View logs in: $SCRIPT_DIR/logs/"
    print_info "Monitor progress: tail -f logs/gpt_*_<JOB_ID>.out"
    echo ""
}

# 运行主函数
main "$@"
