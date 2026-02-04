#!/usr/bin/env python3
"""
GPipe 流水线并行训练 - 模型无关的主入口

使用方法:
    source /home/bingxing2/home/scx9kvs/mxy/env.sh
    conda activate pai-megatron
    # 基本运行
    torchrun --nproc_per_node=4 --master_port=29500 gpipe_runner.py --config configs/gpt_small.yaml
    
    # 启用 Orion 调度器
    export USE_ORION_SCHEDULER=1
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
        torchrun --nproc_per_node=4 --master_port=29500 gpipe_runner.py --config configs/gpt_small_orion.yaml
"""

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.profiler
from torch.profiler import record_function

import argparse
import datetime
import time
import os
import sys
import threading
import contextlib
from tqdm import tqdm

# 导入核心框架
from core.config import PipelineConfig
from core.registry import MODEL_REGISTRY, DATASET_REGISTRY, STAGE_REGISTRY

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


def train_iteration(
    my_stage,
    micro_inputs,
    micro_labels,
    iteration: int,
    config: PipelineConfig,
    current_stage: int,
    model_parallel_size: int,
    num_microbatches: int
):
    """执行一次训练迭代
    
    Args:
        my_stage: 当前阶段对象
        micro_inputs: micro-batch 输入列表
        micro_labels: micro-batch 标签列表
        iteration: 当前迭代编号
        config: 流水线配置
        current_stage: 当前阶段 ID
        model_parallel_size: 模型并行大小
        num_microbatches: micro-batch 数量
    """
    import torch.nn.functional as F
    
    # 通信句柄字典
    fwd_recv_handles = {}
    fwd_send_handles = {}
    bwd_recv_handles = {}
    bwd_send_handles = {}
    
    # 重置缓存
    my_stage.reset_cache()
    
    # ==================== 前向传播 ====================
    
    # 步骤 1：所有接收方提交 irecv
    if current_stage > 1:
        for mb_idx in range(num_microbatches):
            with record_function(f"Fwd Post Irecv mb_{mb_idx}"):
                handle = my_stage.forward_irecv(mb_idx, iteration)
                fwd_recv_handles[mb_idx] = handle
    
    # 步骤 2：多线程并行计算
    def fwd_compute_mb(mb_idx):
        torch.cuda.set_device(my_stage.device)
        
        # Orion 调度器设置
        if my_stage.orion_scheduler is not None:
            my_stage.orion_scheduler.set_client_idx(mb_idx)
        
        current_comp_stream = my_stage.comp_streams[mb_idx]
        
        with torch.cuda.stream(current_comp_stream):
            with record_function(f"Fwd Compute mb_{mb_idx}"):
                if current_stage > 1:
                    fwd_recv_handles[mb_idx].wait()
                    input_tensor = my_stage.out_x_buffers[mb_idx]
                else:
                    input_tensor = micro_inputs[mb_idx]
                
                output_tensor = my_stage.forward(input_tensor, mb_idx)
        
        # 发送
        if current_stage < model_parallel_size:
            with my_stage.comm_stream_lock:
                with torch.cuda.stream(my_stage.comm_stream):
                    my_stage.comm_stream.wait_stream(current_comp_stream)
                    with record_function(f"Fwd Send mb_{mb_idx}"):
                        handle = my_stage.forward_isend(my_stage.fwd_cache[mb_idx], mb_idx, iteration)
                        fwd_send_handles[mb_idx] = handle
    
    fwd_threads = []
    for mb_idx in range(num_microbatches):
        t = threading.Thread(target=fwd_compute_mb, args=(mb_idx,))
        t.start()
        fwd_threads.append(t)
    
    for t in fwd_threads:
        t.join()
    
    # ==================== 反向传播 ====================
    
    # 步骤 1：所有非最后阶段提交 irecv
    if current_stage < model_parallel_size:
        for mb_idx in reversed(range(num_microbatches)):
            with record_function(f"Bwd Post Irecv mb_{mb_idx}"):
                handle = my_stage.backward_irecv(mb_idx, iteration)
                bwd_recv_handles[mb_idx] = handle
    
    # 步骤 2：多线程并行计算
    def bwd_compute_mb(mb_idx):
        torch.cuda.set_device(my_stage.device)
        
        if my_stage.orion_scheduler is not None:
            my_stage.orion_scheduler.set_client_idx(mb_idx)
        
        current_comp_stream = my_stage.comp_streams[mb_idx]
        
        with torch.cuda.stream(current_comp_stream):
            with record_function(f"Bwd Compute mb_{mb_idx}"):
                grad_tensor = None
                if current_stage < model_parallel_size:
                    bwd_recv_handles[mb_idx].wait()
                    grad_tensor = my_stage.grad_y_buffers[mb_idx]
                
                if current_stage == model_parallel_size:
                    # 最后阶段：计算损失
                    loss = my_stage.compute_loss(my_stage.fwd_cache[mb_idx], micro_labels[mb_idx])
                    loss.backward()
                else:
                    my_stage.compute_grad_only(mb_idx, grad_tensor)
        
        # 发送
        if current_stage > 1:
            with my_stage.comm_stream_lock:
                with torch.cuda.stream(my_stage.comm_stream):
                    my_stage.comm_stream.wait_stream(current_comp_stream)
                    with record_function(f"Bwd Send mb_{mb_idx}"):
                        handle = my_stage.backward_isend(mb_idx, iteration)
                        bwd_send_handles[mb_idx] = handle
    
    bwd_threads = []
    for mb_idx in reversed(range(num_microbatches)):
        t = threading.Thread(target=bwd_compute_mb, args=(mb_idx,))
        t.start()
        bwd_threads.append(t)
    
    for t in bwd_threads:
        t.join()
    
    # ==================== 同步和更新 ====================
    
    # 同步所有流
    if my_stage.comp_streams:
        for s in my_stage.comp_streams:
            torch.cuda.current_stream().wait_stream(s)
    if my_stage.comm_stream:
        torch.cuda.current_stream().wait_stream(my_stage.comm_stream)
    
    # 等待所有通信完成
    for handle in fwd_send_handles.values():
        handle.wait()
    for handle in bwd_send_handles.values():
        handle.wait()
    
    # 累加梯度（非最后阶段）
    if current_stage < model_parallel_size:
        with record_function("Gradient Accumulation"):
            my_stage.accumulate_gradients()
    
    # 参数更新
    if my_stage.is_training:
        if config.parallel.data_parallel_size > 1:
            my_stage.all_reduce_gradients()
        my_stage.step()
        my_stage.zero_grad()


def main():
    parser = argparse.ArgumentParser(description='GPipe Pipeline Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # 加载配置
    config = PipelineConfig.from_yaml(args.config)
    config.validate()
    
    # 检查环境变量覆盖
    if os.environ.get('USE_ORION_SCHEDULER', '0') == '1':
        config.parallel.use_orion_scheduler = True
    
    # 初始化分布式
    global_rank = setup_distributed()
    DEVICE = get_device()
    
    print(f"Rank {global_rank}: device={DEVICE}, config={config}")
    
    world_size = dist.get_world_size()
    model_parallel_size = config.parallel.model_parallel_size
    data_parallel_size = world_size // model_parallel_size
    
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
    
    print(f"Rank {global_rank}: stage={current_stage}")
    
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
    
    # 创建模型
    model_cls = MODEL_REGISTRY.get(config.model.name)
    if model_cls is None:
        raise ValueError(f"Model '{config.model.name}' not found. Available: {MODEL_REGISTRY.list_registered()}")
    
    model_adapter = model_cls(config.model)
    model_layers = model_adapter.init_model()
    
    # 获取阶段划分
    partitions = model_adapter.get_stage_partition(model_parallel_size)
    layer_indices = partitions[current_stage - 1]
    
    if global_rank == 0:
        print(f"Stage partitions: {partitions}")
    
    # 创建 Stage
    stage_cls = STAGE_REGISTRY.get(config.model.name)
    if stage_cls is None:
        raise ValueError(f"Stage '{config.model.name}' not found. Available: {STAGE_REGISTRY.list_registered()}")
    
    my_stage = stage_cls(
        stage_id=current_stage,
        model_layers=model_layers,
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
    
    dist.barrier()
    
    # 训练循环
    num_microbatches = config.training.num_microbatches
    
    for epoch in range(config.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for i, data in tqdm(enumerate(train_loader, 0), disable=(global_rank != 0)):
            # Profiler 设置
            profile_iter = config.training.profile_iteration
            profiler_dir = f'./profiler_{"orion" if config.parallel.use_orion_scheduler else "baseline"}/rank{global_rank}'
            
            profiler_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir)
            ) if i == profile_iter else contextlib.nullcontext()
            
            # 准备数据
            inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
            micro_inputs = torch.chunk(inputs, num_microbatches)
            micro_labels = torch.chunk(labels, num_microbatches)
            
            if global_rank == 0 and i % 50 == 0:
                print(f"\n################## iteration {i} ##################")
            
            dist.barrier()
            
            with profiler_context:
                train_iteration(
                    my_stage=my_stage,
                    micro_inputs=micro_inputs,
                    micro_labels=micro_labels,
                    iteration=i,
                    config=config,
                    current_stage=current_stage,
                    model_parallel_size=model_parallel_size,
                    num_microbatches=num_microbatches
                )
            
            dist.barrier()
            torch.cuda.synchronize()
            
            # 检查是否退出
            max_iter = config.training.max_iterations
            if max_iter > 0 and i >= max_iter:
                break
            
            if config.training.exit_after_profile and i == profile_iter + 10:
                print(f"Profiling finished at iteration {i}. Exiting.")
                my_stage.stop_orion_scheduler()
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)
    
    # 清理
    my_stage.stop_orion_scheduler()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()