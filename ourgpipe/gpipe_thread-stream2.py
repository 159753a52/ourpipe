import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
"""
    cd orion-docker/ourgpipe/
    source /home/bingxing2/home/scx9kvs/mxy/env.sh
    conda activate pai-megatron
    torchrun --nproc_per_node=4 --master_port=29500 gpipe_thread-stream2.py
    
    === 启用 Orion 调度器 ===
    export USE_ORION_SCHEDULER=1
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
        torchrun --nproc_per_node=4 --master_port=29500 gpipe_thread-stream2.py

        
        """
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import datetime
import time
import random
import numpy as np
import gc
import threading  # 导入 threading

import sys
from GPT import * # 导入 GPT.py 中的所有定义 (Block, tok, process, datasets 等)
import torch.profiler
from torch.profiler import record_function
import contextlib

# ==================== Orion 调度器集成 ====================
# 通过环境变量 USE_ORION_SCHEDULER 控制是否启用 Orion 多优先级调度
USE_ORION_SCHEDULER = os.environ.get('USE_ORION_SCHEDULER', '0') == '1'

if USE_ORION_SCHEDULER:
    try:
        # 添加 kernel_intercept/python 到路径
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kernel_intercept', 'python'))
        from orion_binding import get_scheduler, SchedulingMode
        ORION_AVAILABLE = True
        print("[Orion] Orion scheduler binding loaded successfully")
    except ImportError as e:
        print(f"[Orion] Warning: Failed to import orion_binding: {e}")
        ORION_AVAILABLE = False
        USE_ORION_SCHEDULER = False
else:
    ORION_AVAILABLE = False

# 使用你之前设置的 emb_size
emb_size = 512 
head_size = 32 
n_layer = 16
sequence_len = 128
learning_rate = 1e-4
eval_iters = 20
batch_size = 4
epochs = 2
num_microbatches = 4  # 保持 num_microbatches 为 4

def init_model(model_config):
    # model=[]
    model = nn.ModuleList()
    for i,j in model_config.items():
        if i == "em_tokn":
            mdl = nn.Embedding(j[0], j[1])
        elif i == "em_pos":
            mdl = nn.Embedding(j[0], j[1])
        elif i == "ln":
            mdl = nn.LayerNorm(j[0])
        elif i == "lm_head":
            mdl = nn.Linear(j[0],j[1])
        elif (re.search("decoder",i)): 
            mdl = Block(j[0],j[1]) 
        model.append(mdl)
    return model

# ... (dist.init_process_group, get_device, 2D DDP/MP group setup... 保持不变) ...
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=300))
global_rank = int(os.environ["RANK"])
print(f"global_rank is {global_rank}")
if global_rank == 0:
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
def get_device():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        return f'cuda:{local_rank}'
    return 'cpu'
DEVICE = get_device()
print("The current rank is {}".format(DEVICE))
world_size = dist.get_world_size()
model_parallel_size = 4
data_parallel_size = world_size // model_parallel_size
model_parallel_groups = [
    dist.new_group(list(range(i*model_parallel_size, (i+1)*model_parallel_size))) 
    for i in range(data_parallel_size)
]
data_parallel_groups = [
    dist.new_group(list(range(i, world_size, model_parallel_size)))
    for i in range(model_parallel_size)
]
model_parallel_group = model_parallel_groups[global_rank // model_parallel_size]
data_parallel_group = data_parallel_groups[global_rank % model_parallel_size]


class Stage:
    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        self.stage_ID = ID
        self.device = torch.device(device) 
        self.model_idx = model_idx
        self.is_training = True
        self.micro_batch_size = batch_size // num_microbatches

        self.model_parallel_group = model_parallel_group
        self.data_parallel_group = data_parallel_group
        self.local_rank = global_rank % model_parallel_size

        if self.stage_ID == 1:
            self.token_embedding = nn.Embedding(vs, emb_size)
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            self.sub_model = nn.Sequential(*[model[i] for i in range(2,len(self.model_idx))])
        else:
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])

        if self.stage_ID == 1:
            self.sub_model_opt = nn.Sequential(*[model[i] for i in model_idx])
            self.optimizer = optim.Adam(self.sub_model_opt.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.sub_model.parameters(), lr=learning_rate)
        
        self.out_x_buffers = [
            torch.zeros(self.micro_batch_size, sequence_len, emb_size, device=self.device)
            for _ in range(num_microbatches)
        ]
        self.grad_y_buffers = [
            torch.zeros(self.micro_batch_size, sequence_len, emb_size, device=self.device)
            for _ in range(num_microbatches)
        ]
        
        self.lossi = []    
        # === [不变] fwd_cache 初始化 ===
        self.fwd_cache = [None] * num_microbatches  # 缓存 Stage 的 *输出* (Y)
        self.input_cache = [None] * num_microbatches # 缓存 Stage 的 *输入* (X)

        # === [不变] 创建多个计算流 (每个 micro-batch 一个) ===
        self.num_comp_streams = num_microbatches 
        if self.device.type == 'cuda':
            self.comp_streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_comp_streams)]
            self.comm_stream = torch.cuda.Stream(device=self.device)
        else:
            self.comp_streams = None
            self.comm_stream = None
            
        # === [新添加] 用于保护共享 comm_stream 的线程锁 ===
        self.comm_stream_lock = threading.Lock()

        self.grad_buffers = [None] * num_microbatches

        # ==================== Orion 调度器初始化 ====================
        self.orion_scheduler = None
        if USE_ORION_SCHEDULER and ORION_AVAILABLE:
            try:
                # 在启动调度器之前，先初始化 CUDA 上下文
                if self.device.type == 'cuda':
                    torch.cuda.set_device(self.device)
                    torch.cuda.init()
                    torch.cuda.synchronize()
                    print(f"[Orion] Stage {ID}: CUDA context initialized on device={self.device} before scheduler start")
                
                # 获取 Orion 调度器单例
                self.orion_scheduler = get_scheduler()
                # 设置为多优先级调度模式
                self.orion_scheduler.set_scheduling_mode(SchedulingMode.MULTI_PRIORITY)
                # 启动调度器（如果尚未启动）
                if not self.orion_scheduler.is_running():
                    device_id = self.device.index if self.device.type == 'cuda' else 0
                    self.orion_scheduler.start(num_clients=num_microbatches, device_id=device_id)
                print(f"[Orion] Stage {ID}: Orion scheduler initialized with {num_microbatches} clients (multi-priority mode, device={device_id})")
            except Exception as e:
                print(f"[Orion] Stage {ID}: Failed to initialize Orion scheduler: {e}")
                self.orion_scheduler = None

    def to(self, device):
        device = torch.device(device) 
        if self.stage_ID == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        self.sub_model.to(device)
        
        self.out_x_buffers = [b.to(device) for b in self.out_x_buffers]
        self.grad_y_buffers = [b.to(device) for b in self.grad_y_buffers]
    
    # ... (eval, train, zero_grad 保持不变) ...
    def eval(self):
        self.sub_model.eval()
    def train(self):
        self.sub_model.train()
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    # === [不变] forward 接收 mb_idx 以便线程安全地写入 cache ===
    def forward(self, x, mb_idx):
        if self.stage_ID == 1:
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
            # (它不需要 requires_grad_() 因为它是图的根)
            self.input_cache[mb_idx] = x 
            y = self.sub_model(x)
        else:
            # 1. 启用输入的 grad，以便 autograd 跟踪它
            # 2. 保存输入，以便在 backward 时检索 .grad
            x.requires_grad_()
            self.input_cache[mb_idx] = x
            y = self.sub_model(x)
        
        y.retain_grad() # 我们仍然需要保留 *输出* 的梯度
        self.fwd_cache[mb_idx] = y  # 保存 *输出*
        return y
    def compute_grad_only(self, mb_idx, grad_tensor):
        cached_output = self.fwd_cache[mb_idx]
        cached_input = self.input_cache[mb_idx]
        
        # 1. 准备需要求导的对象列表：包括模型参数 + 输入张量
        # 我们必须显式告诉 autograd 我们需要 input 的梯度
        params = list(self.sub_model.parameters())
        
        # 如果不是第一阶段，我们需要计算输入的梯度以便传给上一级
        # (第一阶段没有上一级，所以不需要算 input 的梯度)
        inputs_to_diff = params
        need_input_grad = (self.stage_ID > 1) and (cached_input.requires_grad)
        
        if need_input_grad:
            inputs_to_diff = params + [cached_input]

        # 2. 执行无锁的梯度计算
        grads = torch.autograd.grad(
            outputs=cached_output,
            inputs=inputs_to_diff,
            grad_outputs=grad_tensor,
            retain_graph=False
        )
        
        # 3. 分离结果并存储
        
        # A. 提取权重的梯度并存入 buffer (供稍后累加)
        # 权重梯度是列表的前半部分
        weight_grads = grads[:len(params)]
        self.grad_buffers[mb_idx] = weight_grads
        
        # B. 提取输入的梯度并手动赋值 (供 backward_isend 使用)
        if need_input_grad:
            # 输入梯度是列表的最后一个元素
            input_grad = grads[-1]
            # 【关键修复】手动赋值给 .grad，这样 backward_isend 就能读到了
            cached_input.grad = input_grad

    # ... (isend, irecv 保持不变) ...
    def forward_isend(self, out, mb_idx, iter_num):
        dst = global_rank + 1
        direction = 0
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (self.stage_ID - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              mb_idx
        return dist.isend(tensor=out, dst=dst, tag=tag, group=self.model_parallel_group)
    def forward_irecv(self, mb_idx, iter_num):
        src = global_rank - 1
        sender_stage_id = self.stage_ID - 1
        direction = 0
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (sender_stage_id - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              mb_idx
        tensor_to_recv = self.out_x_buffers[mb_idx]
        return dist.irecv(tensor=tensor_to_recv, src=src, tag=tag, group=self.model_parallel_group)
    
    # === [不变] backward 和 backward_isend 已经是按索引读，是线程安全的 ===
    def backward(self, microbatch_idx, grad_tensor):
        cached_output = self.fwd_cache[microbatch_idx]
        cached_output.backward(grad_tensor)
    def backward_isend(self, microbatch_idx, iter_num):
        # 发送 *输入* 的梯度 (dL/dX)
        cached_input = self.input_cache[microbatch_idx]
        grad = cached_input.grad # 这才是我们需要的 dL/dX   
        dst = global_rank - 1
        direction = 1
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (self.stage_ID - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              microbatch_idx
        return dist.isend(grad, dst=dst, tag=tag, group=self.model_parallel_group)
    def backward_irecv(self, microbatch_idx, iter_num):
        src = global_rank + 1
        sender_stage_id = self.stage_ID + 1
        direction = 1
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (sender_stage_id - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              microbatch_idx
        tensor_to_recv = self.grad_y_buffers[microbatch_idx]
        return dist.irecv(tensor_to_recv, src=src, tag=tag, group=self.model_parallel_group)
    def all_reduce_gradients(self):
        if self.stage_ID == 1:
            params_to_reduce = self.sub_model_opt.parameters()
        else:
            params_to_reduce = self.sub_model.parameters()
        for param in params_to_reduce:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                param.grad /= data_parallel_size

datasets, tok = init_datasets()
# ... (数据加载, tokenized, Dataloader setup 保持不变) ...
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
if global_rank != 0:
    dist.barrier()
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
if global_rank == 0:
    dist.barrier()
tokenized.set_format(type='torch', device=DEVICE)
if global_rank == 0: 
    print("Train dataset inputs and labels shape are {} and {}".format(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape))
train_sampler = DistributedSampler(
    dataset=tokenized['train'],
    num_replicas=data_parallel_size,
    rank=global_rank // model_parallel_size,
    shuffle=True
) if data_parallel_size > 1 else None
train_loader = DataLoader(
    dataset=tokenized['train'],
    batch_size=batch_size,
    sampler=train_sampler, 
    num_workers=0
)
test_sampler = DistributedSampler(
    tokenized['test'],
    num_replicas=data_parallel_size,
    rank=global_rank // model_parallel_size,
    shuffle=False
) if data_parallel_size > 1 else None
test_loader = DataLoader(
    tokenized['test'],
    batch_size=batch_size,
    sampler=test_sampler,
    num_workers=0
)
if global_rank == 0: 
    print("training step num is {}".format(len(train_loader)))
    print("test step num is {}".format(len(test_loader)))

vs = len(tok.char2ind)

model_config = {
    "em_tokn": [vs, emb_size],
    "em_pos": [sequence_len, emb_size],
    **{f"decoder{i}": [emb_size, head_size] for i in range(1, n_layer+1)},
    "ln": [emb_size],
    "lm_head": [emb_size, vs]
}

gpt = init_model(model_config=model_config)

# model_idx1=[0,1,2]
# model_idx2=[3,4,5]
# model_idx3=[6,7,8]
# model_idx4=[9,10,11] 
model_idx1=[0,1,2,3,4,5]
model_idx2=[6,7,8,9]
model_idx3=[10,11,12,13]
model_idx4=[14,15,16,17,18,19]

rank_to_stage = {
    0: 1, 1: 2, 2: 3, 3: 4,
    4: 1, 5: 2, 6: 3, 7: 4,
}
current_stage = rank_to_stage[global_rank]
print("current rank is {}, current stage is {}".format(global_rank, current_stage))

if current_stage == 1:
    my_stage = Stage(1, gpt, model_idx1, learning_rate, DEVICE, batch_size)
elif current_stage == 2:
    my_stage = Stage(2, gpt, model_idx2, learning_rate, DEVICE, batch_size)
elif current_stage == 3:
    my_stage = Stage(3, gpt, model_idx3, learning_rate, DEVICE, batch_size)
elif current_stage == 4:
    my_stage = Stage(4, gpt, model_idx4, learning_rate, DEVICE, batch_size)
my_stage.to(DEVICE)


loss_list = []
dist.barrier()

def nvtx_push(now_iter, target_iter, context):
    if now_iter == target_iter:
        nvtx.range_push(context)
def nvtx_pop(now_iter, target_iter):
    if now_iter == target_iter:
        nvtx.range_pop()

ck_interval = 1000
loss = 0
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    iter_start = time.time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        # nvtx_push(i, 30, f"Iteration {i}")      
        
        # 根据是否启用 Orion 调度器来区分输出目录
        profiler_dir = f'./profiler_orion/rank{global_rank}' if USE_ORION_SCHEDULER else f'./profiler_thread-stream_512_2/rank{global_rank}'
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir)
        ) if i == 100 else contextlib.nullcontext()

        train_start = time.time()
        inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)
        
        fwd_recv_handles = {}
        fwd_send_handles = {}
        bwd_recv_handles = {}
        bwd_send_handles = {}
        
        # === [不变] 重置 cache ===
        my_stage.fwd_cache = [None] * num_microbatches 
        
        if global_rank == 0:
            print("\n################## iteration {} ##################".format(i))
        dist.barrier() 
        with profiler_context:
                 

            # === Forward ===

            # --- 步骤 1：所有接收方 (Rank > 0) 立即提交所有 irecv ---
            if current_stage > 1:
                for mb_idx in range(num_microbatches):
                    # nvtx_push(i, 30, f"Fwd Post Irecv mb_{mb_idx}")
                    with record_function(f"Fwd Post Irecv mb_{mb_idx}"):
                        handle = my_stage.forward_irecv(mb_idx, i)
                        fwd_recv_handles[mb_idx] = handle
                    # nvtx_pop(i, 30)

            # --- 步骤 2：[修改] 定义 fwd 线程函数 (包含 compute 和 send) ---
            def fwd_compute_mb(mb_idx):
                torch.cuda.set_device(DEVICE)
                
                # 【Orion】设置当前线程的 client index（用于多优先级调度）
                if my_stage.orion_scheduler is not None:
                    my_stage.orion_scheduler.set_client_idx(mb_idx)
                
                current_comp_stream = my_stage.comp_streams[mb_idx]
                
                # 1. Compute (在自己的 stream)
                # nvtx_push(i, 30, f"Fwd Compute mb_{mb_idx}")
                with torch.cuda.stream(current_comp_stream):
                    with torch.random.fork_rng(devices=[global_rank]):
                        with record_function(f"Fwd Compute mb_{mb_idx}"):
                            if current_stage > 1:
                                fwd_recv_handles[mb_idx].wait()
                                input_tensor = my_stage.out_x_buffers[mb_idx].to(DEVICE)
                            else:
                                input_tensor = micro_inputs[mb_idx]
                            
                            # 调用 forward，它会填充 my_stage.fwd_cache[mb_idx]
                            output_tensor = my_stage.forward(input_tensor, mb_idx)
                # nvtx_pop(i, 30)
                
                # 2. 立刻发送（不要等其他 micro）- 每个 micro 算完就发
                if current_stage < model_parallel_size:
                    with my_stage.comm_stream_lock:
                        with torch.cuda.stream(my_stage.comm_stream):
                            my_stage.comm_stream.wait_stream(current_comp_stream)
                            with record_function(f"Fwd Send mb_{mb_idx}"):
                                handle = my_stage.forward_isend(my_stage.fwd_cache[mb_idx], mb_idx, i)
                                fwd_send_handles[mb_idx] = handle
            fwd_threads = []
            for mb_idx in range(num_microbatches):
                t = threading.Thread(target=fwd_compute_mb, args=(mb_idx,))
                t.start()
                fwd_threads.append(t)
            
            # 等待所有 forward (compute + send) 线程完成 (确保所有 kernel 已 enqueue)
            for t in fwd_threads:
                t.join()

            # === Backward ===

            # --- 步骤 1：所有非最后阶段的 Rank 提交 Bwd irecv ---
            if current_stage < model_parallel_size:
                for mb_idx in reversed(range(num_microbatches)):
                    # nvtx_push(i, 30, f"Bwd Post Irecv mb_{mb_idx}")
                    with record_function(f"Bwd Post Irecv mb_{mb_idx}"):
                            handle = my_stage.backward_irecv(mb_idx, i)
                            bwd_recv_handles[mb_idx] = handle
                    # nvtx_pop(i, 30)

            # --- 步骤 2：[修改] 定义 bwd 线程函数 (包含 compute 和 send) ---
            def bwd_compute_mb(mb_idx):
                torch.cuda.set_device(DEVICE)
                
                # 【Orion】设置当前线程的 client index（用于多优先级调度）
                if my_stage.orion_scheduler is not None:
                    my_stage.orion_scheduler.set_client_idx(mb_idx)
                
                current_comp_stream = my_stage.comp_streams[mb_idx]
                
                # 1. Compute (在自己的 stream)
                # nvtx_push(i, 30, f"Bwd Compute mb_{mb_idx}")
                with torch.cuda.stream(current_comp_stream):
                    with record_function(f"Bwd Compute mb_{mb_idx}"):
                        grad_tensor = None
                        if current_stage < model_parallel_size:
                            bwd_recv_handles[mb_idx].wait()
                            grad_tensor = my_stage.grad_y_buffers[mb_idx].to(DEVICE)
                        
                        if current_stage == model_parallel_size:
                            logits = my_stage.fwd_cache[mb_idx].transpose(-2, -1)
                            loss = F.cross_entropy(logits, micro_labels[mb_idx])
                            loss.backward()
                        else:
                            my_stage.compute_grad_only(mb_idx, grad_tensor)
                # nvtx_pop(i, 30)
                
                # 2. 立刻发送（不要等其他 micro）- 每个 micro 算完就发
                if current_stage > 1:
                    with my_stage.comm_stream_lock:
                        with torch.cuda.stream(my_stage.comm_stream):
                            my_stage.comm_stream.wait_stream(current_comp_stream)
                            with record_function(f"Bwd Send mb_{mb_idx}"):
                                handle = my_stage.backward_isend(mb_idx, i)
                                bwd_send_handles[mb_idx] = handle
            bwd_threads = []
            for mb_idx in reversed(range(num_microbatches)):
                t = threading.Thread(target=bwd_compute_mb, args=(mb_idx,))
                t.start()
                bwd_threads.append(t)
            
            # 等待所有 backward (compute + send) 线程完成 (确保所有 kernel 已 enqueue)
            for t in bwd_threads:
                t.join()

            # === 同步和更新 === (保持不变)
            
            # 1. 同步所有流
            if my_stage.comp_streams:
                for s in my_stage.comp_streams:
                    torch.cuda.current_stream().wait_stream(s)
            if my_stage.comm_stream:
                torch.cuda.current_stream().wait_stream(my_stage.comm_stream)

            # 2. 确保所有 P2P 通信已完成 (CPU 句柄)
            for handle in fwd_send_handles.values():
                handle.wait()
            for handle in bwd_send_handles.values():
                handle.wait()

            if current_stage < model_parallel_size: # 最后一级如果直接用了 .backward() 则不需要这步
                with record_function("Gradient Accumulation"):
                    params = list(my_stage.sub_model.parameters())
                    # 遍历每一个参数
                    for param_idx, param in enumerate(params):
                        # 遍历每一个 micro-batch 的梯度
                        for mb_idx in range(num_microbatches):
                            grads = my_stage.grad_buffers[mb_idx]
                            if grads is not None:
                                mb_grad = grads[param_idx]
                                # 累加到 param.grad
                                if param.grad is None:
                                    param.grad = mb_grad
                                else:
                                    param.grad += mb_grad
                        
                    # 清空 buffer 以备下一次迭代
                    my_stage.grad_buffers = [None] * num_microbatches            

            # === Update ===
            if my_stage.is_training:
                if data_parallel_size > 1:
                    my_stage.all_reduce_gradients()
                my_stage.optimizer.step()
                my_stage.optimizer.zero_grad()

            train_end = time.time()
            
            if i % 50 == 0 and i != 0:
                iter_end = time.time()
            dist.barrier() 
        # nvtx_pop(i, 30)
        torch.cuda.synchronize()
        if i == 110:
            # nvtx.range_pop()
            print(f"Epoch {i} profiling finished. Exiting.")
            
            # 【Orion】停止调度器并打印统计信息
            if my_stage.orion_scheduler is not None:
                try:
                    # 打印每个客户端的统计信息
                    for client_idx in range(num_microbatches):
                        scheduled, waiting = my_stage.orion_scheduler.get_stats(client_idx)
                        print(f"[Orion] Stage {current_stage} client {client_idx}: scheduled={scheduled}, waiting={waiting}")
                    my_stage.orion_scheduler.stop()
                    print(f"[Orion] Stage {current_stage}: Orion scheduler stopped")
                except Exception as e:
                    print(f"[Orion] Stage {current_stage}: Error stopping scheduler: {e}")
            
            dist.barrier()
            sys.exit(0)

# 反向传播使用naive形式
# 模型参数（大小）的调整尽可能发挥我们的性能