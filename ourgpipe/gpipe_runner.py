#!/usr/bin/env python3
"""
GPipe 流水线并行训练 - 模型无关的主入口

使用方法:
    cd orion-docker/ourgpipe
    source /home/bingxing2/home/scx9kvs/mxy/env.sh
    conda activate pai-megatron
    
    # custom.yaml
    torchrun --nproc_per_node=4 --master_port=29500 gpipe_runner.py --config configs/gpt_custom.yaml
    
    # small.yaml
    torchrun --nproc_per_node=4 --master_port=29500 gpipe_runner.py --config configs/gpt_small.yaml
    
    # 启用 Orion 调度器
    export USE_ORION_SCHEDULER=1
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
        torchrun --nproc_per_node=4 --master_port=29500 gpipe_runner.py --config configs/gpt_small_orion.yaml
"""

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.profiler

import argparse
import datetime
import os
import sys
import contextlib
from tqdm import tqdm

# 导入核心框架
from core.config import PipelineConfig
from core.registry import MODEL_REGISTRY, DATASET_REGISTRY, STAGE_REGISTRY
from core.schedulers import SCHEDULER_REGISTRY
from core.metrics import MetricsTracker, collect_model_params

# 导入模型实现（触发注册）
import models


def get_device():
    """获取当前进程应该使用的设备"""
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        return f'cuda:{local_rank}'
    return 'cpu'


def setup_distributed():
    """初始化分布式训练环境"""
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    
    dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=300))
    
    global_rank = int(os.environ["RANK"])
    if global_rank == 0:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    
    return global_rank


def setup_parallel_groups(world_size: int, model_parallel_size: int):
    """设置 2D 并行组
    
    Args:
        world_size: 总进程数
        model_parallel_size: 模型并行大小（流水线阶段数）
        
    Returns:
        model_parallel_groups: 模型并行组列表
        data_parallel_groups: 数据并行组列表
    """
    data_parallel_size = world_size // model_parallel_size
    
    # 创建模型并行组
    model_parallel_groups = [
        dist.new_group(list(range(i * model_parallel_size, (i + 1) * model_parallel_size)))
        for i in range(data_parallel_size)
    ]
    
    # 创建数据并行组
    data_parallel_groups = [
        dist.new_group(list(range(i, world_size, model_parallel_size)))
        for i in range(model_parallel_size)
    ]
    
    return model_parallel_groups, data_parallel_groups


def main():
    parser = argparse.ArgumentParser(description='GPipe Pipeline Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--scheduler', type=str, default=None, 
                        help='Override scheduler type (naive or async_threaded)')
    args = parser.parse_args()
    
    # 加载配置
    config = PipelineConfig.from_yaml(args.config)
    
    # 命令行参数覆盖
    if args.scheduler:
        config.parallel.scheduler = args.scheduler
    
    # 检查环境变量覆盖
    if os.environ.get('USE_ORION_SCHEDULER', '0') == '1':
        config.parallel.use_orion_scheduler = True
    
    # 验证配置
    config.validate()
    
    # 根据调度器类型设置 TF32
    if config.parallel.scheduler == 'naive':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("TF32 disabled for Naive scheduler (fair comparison)")
    elif config.parallel.scheduler == 'async_threaded' and config.parallel.use_orion_scheduler:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for Async Threaded + Orion scheduler (performance boost)")
    else:
        # 默认关闭 TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("TF32 disabled by default")
    
    # 初始化分布式
    global_rank = setup_distributed()
    DEVICE = get_device()
    
    if global_rank == 0:
        print(f"Config: {config}")
        print(f"Using scheduler: {config.parallel.scheduler}")
    
    world_size = dist.get_world_size()
    model_parallel_size = config.parallel.model_parallel_size
    data_parallel_size = world_size // model_parallel_size
    
    # 更新配置中的 data_parallel_size
    config.parallel.data_parallel_size = data_parallel_size
    
    # 设置并行组
    model_parallel_groups, data_parallel_groups = setup_parallel_groups(
        world_size, model_parallel_size
    )
    model_parallel_group = model_parallel_groups[global_rank // model_parallel_size]
    data_parallel_group = data_parallel_groups[global_rank % model_parallel_size]
    
    # 确定当前阶段
    rank_to_stage = {}
    for dp in range(data_parallel_size):
        for mp in range(model_parallel_size):
            rank_to_stage[dp * model_parallel_size + mp] = mp + 1
    current_stage = rank_to_stage[global_rank]
    
    print(f"Rank {global_rank}: device={DEVICE}, stage={current_stage}")
    
    # 创建数据集
    dataset_cls = DATASET_REGISTRY.get(config.dataset.name)
    if dataset_cls is None:
        raise ValueError(f"Dataset '{config.dataset.name}' not found. Available: {DATASET_REGISTRY.list_registered()}")
    
    dataset = dataset_cls(config.dataset, config.model.sequence_length)
    
    # 更新 vocab_size
    if config.model.vocab_size == -1:
        config.model.vocab_size = dataset.get_vocab_size()
    
    # 加载数据（rank 0 先处理）
    if global_rank == 0:
        train_data, test_data = dataset.get_train_test_split()
    dist.barrier()
    if global_rank != 0:
        train_data, test_data = dataset.get_train_test_split()
    dist.barrier()
    
    train_data.set_format(type='torch', device=DEVICE)
    test_data.set_format(type='torch', device=DEVICE)
    
    if global_rank == 0:
        print(f"Train dataset size: {len(train_data)}")
    
    # 创建 DataLoader
    train_sampler = DistributedSampler(
        dataset=train_data,
        num_replicas=data_parallel_size,
        rank=global_rank // model_parallel_size,
        shuffle=True
    ) if data_parallel_size > 1 else None
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.dataset.batch_size,
        sampler=train_sampler,
        num_workers=config.dataset.num_workers
    )
    
    if global_rank == 0:
        print(f"Training steps per epoch: {len(train_loader)}")
    
    # 创建模型适配器
    model_cls = MODEL_REGISTRY.get(config.model.name)
    if model_cls is None:
        raise ValueError(f"Model '{config.model.name}' not found. Available: {MODEL_REGISTRY.list_registered()}")
    
    model_adapter = model_cls(config.model)
    
    # 获取模型配置（延迟初始化，不创建层实例）
    model_config = model_adapter.init_model()
    
    # 获取阶段划分
    partitions = model_adapter.get_stage_partition(model_parallel_size)
    layer_indices = partitions[current_stage - 1]
    
    if global_rank == 0:
        print(f"Stage partitions: {partitions}")
        print(f"Using lazy initialization: each stage will create only its required layers")
    
    # 创建 Stage（传递适配器和配置，而非完整模型）
    stage_cls = STAGE_REGISTRY.get(config.model.name)
    if stage_cls is None:
        raise ValueError(f"Stage '{config.model.name}' not found. Available: {STAGE_REGISTRY.list_registered()}")
    
    my_stage = stage_cls(
        stage_id=current_stage,
        model_adapter=model_adapter,
        model_config=model_config,
        layer_indices=layer_indices,
        config=config,
        device=torch.device(DEVICE),
        model_parallel_group=model_parallel_group,
        data_parallel_group=data_parallel_group,
        global_rank=global_rank,
        model_parallel_size=model_parallel_size,
        data_parallel_size=data_parallel_size
    )
    my_stage.to(DEVICE)
    
    # 创建调度器
    scheduler = SCHEDULER_REGISTRY.create(config.parallel.scheduler, config)
    if global_rank == 0:
        print(f"Scheduler: {scheduler}", flush=True)
    
    dist.barrier()
    
    # 收集模型总参数量
    total_model_params = collect_model_params(
        my_stage, model_parallel_group, torch.device(DEVICE)
    )
    
    # 获取 FLOPs per token
    flops_per_token = model_adapter.get_flops_per_token(total_model_params)
    
    if global_rank == 0:
        print(f"Total Model Parameters: {total_model_params:,}", flush=True)
        print(f"FLOPs per token: {flops_per_token:,}", flush=True)
    
    # 创建指标追踪器
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    metrics_tracker = MetricsTracker(
        batch_size=config.dataset.batch_size,
        sequence_length=config.model.sequence_length,
        num_gpus=world_size,
        device_name=device_name,
        warmup_iterations=10,
        dtype='tf32'  # 默认使用 TF32
    )
    
    dist.barrier()
    
    # 训练循环
    num_microbatches = config.training.num_microbatches
    total_iterations = 0
    
    # 开始追踪
    metrics_tracker.start()
    
    for epoch in range(config.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for i, data in tqdm(enumerate(train_loader, 0), disable=(global_rank != 0)):
            # Profiler 设置
            profile_iter = config.training.profile_iteration
            scheduler_name = config.parallel.scheduler
            orion_suffix = "_orion" if config.parallel.use_orion_scheduler else ""
            profiler_dir = f'./profiler_{scheduler_name}{orion_suffix}/rank{global_rank}'
            
            profiler_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                # record_shapes=True,
                # with_stack=True,
                # profile_memory=True,
                record_shapes=False,     # 关闭形状记录
                with_stack=False,        # 关闭调用栈记录（节省大量内存）
                profile_memory=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir)
            ) if i == profile_iter else contextlib.nullcontext()
            
            # 准备数据
            inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
            micro_inputs = list(torch.chunk(inputs, num_microbatches))
            micro_labels = list(torch.chunk(labels, num_microbatches))
            
            if global_rank == 0 and i % 50 == 0:
                # 输出当前进度和吞吐量
                current_throughput = metrics_tracker.get_throughput()
                print(f"\n################## iteration {i} | throughput: {current_throughput:,.0f} tokens/s ##################", flush=True)
            
            dist.barrier()
            
            with profiler_context:
                # 使用调度器执行训练迭代
                scheduler.run_iteration(
                    stage=my_stage,
                    micro_inputs=micro_inputs,
                    micro_labels=micro_labels,
                    iteration=i,
                    current_stage=current_stage
                )
            
            dist.barrier()
            torch.cuda.synchronize()
            
            # 记录迭代完成
            metrics_tracker.step()
            total_iterations += 1
            
            # 检查是否退出
            max_iter = config.training.max_iterations
            if max_iter > 0 and i >= max_iter:
                break
            
            if config.training.exit_after_profile and i == profile_iter + 10:
                print(f"Profiling finished at iteration {i}. Exiting.", flush=True)
                # 停止追踪并输出汇总
                metrics_tracker.stop()
                if global_rank == 0:
                    metrics_tracker.print_summary(
                        model_params=total_model_params,
                        flops_per_token=flops_per_token
                    )
                my_stage.stop_orion_scheduler()
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)
    
    # 停止追踪
    metrics_tracker.stop()
    
    # 输出训练汇总（只在 rank 0）
    if global_rank == 0:
        metrics_tracker.print_summary(
            model_params=total_model_params,
            flops_per_token=flops_per_token
        )
    
    # 清理
    my_stage.stop_orion_scheduler()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
