import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader 
import datetime
import time
import random
import numpy as np
import gc
# import threading # <-- 已移除

import sys
from GPT import * # 导入 GPT.py 中的所有定义 (Block, tok, process, datasets 等)
import torch.profiler
from torch.profiler import record_function
import contextlib

# 使用你之前设置的 emb_size
emb_size = 4096 
head_size = 128 
n_layer = 16
sequence_len = 128
learning_rate = 1e-4
eval_iters = 20
batch_size = 32
epochs = 2
num_microbatches = 4 # 保持 num_microbatches 为 4

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
        # 【修复 A】: 我们需要两个缓存  
        self.fwd_cache = []  # 缓存 Stage 的 *输出* (Y)
        self.input_cache = [] # 缓存 Stage 的 *输入* (X)

        # === [新增修改] 创建多个计算流 (每个 micro-batch 一个) ===
        self.num_comp_streams = num_microbatches 
        if self.device.type == 'cuda':
            # 创建一个计算流的 *列表*
            self.comp_streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_comp_streams)]
            # 通信流仍然是 1 个
            self.comm_stream = torch.cuda.Stream(device=self.device)
        else:
            self.comp_streams = None
            self.comm_stream = None

    def to(self, device):
        device = torch.device(device) 
        if self.stage_ID == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        self.sub_model.to(device)
        
        self.out_x_buffers = [b.to(device) for b in self.out_x_buffers]
        self.grad_y_buffers = [b.to(device) for b in self.grad_y_buffers]
    
    # ... (eval, train, zero_grad, forward, backward 保持不变) ...
    def eval(self):
        self.sub_model.eval()
    def train(self):
        self.sub_model.train()
    def zero_grad(self):
        self.optimizer.zero_grad()
    def forward(self, x):
        if self.stage_ID == 1:
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
            # 【修复 B】: 保存 Stage 1 的输入
            # (它不需要 requires_grad_() 因为它是图的根)
            self.input_cache.append(x) 
            y = self.sub_model(x)
        else:
            # 【修复 C】: 
            # 1. 启用输入的 grad，以便 autograd 跟踪它
            # 2. 保存输入，以便在 backward 时检索 .grad
            x.requires_grad_()
            self.input_cache.append(x)
            y = self.sub_model(x)
        
        y.retain_grad() # 我们仍然需要保留 *输出* 的梯度
        self.fwd_cache.append(y)  # 保存 *输出*
        return y
        
    # ... (isend, irecv, backward, all_reduce_gradients 保持不变) ...
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
    def backward(self, microbatch_idx, grad_tensor):
        cached_output = self.fwd_cache[microbatch_idx]
        cached_output.backward(grad_tensor)
    def backward_isend(self, microbatch_idx, iter_num):
        # 【修复 D】: 发送 *输入* 的梯度 (dL/dX)
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


# ... (数据加载, tokenized, Dataloader setup 保持不变) ...
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# if global_rank != 0:
#     dist.barrier()
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
# if global_rank == 0:
#     dist.barrier()
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

ck_interval = 1000
loss = 0
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    iter_start = time.time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        if i == 1000:
            nvtx.range_push(f"Epoch {epoch}")        
        
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_stream_4096/rank{global_rank}')
        ) if i == 20 else contextlib.nullcontext()

        train_start = time.time()
        inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)
        
        fwd_recv_handles = {}
        fwd_send_handles = {}
        bwd_recv_handles = {}
        bwd_send_handles = {}
        my_stage.fwd_cache.clear()
        if global_rank == 0:
            print("\n################## iteration {} ##################".format(i))
        dist.barrier() 
        with profiler_context:
                 

            # === Forward ===

            # --- 步骤 1：所有接收方 (Rank > 0) 立即提交所有 irecv ---
            if current_stage > 1:
                for mb_idx in range(num_microbatches):
                    with record_function(f"Fwd Post Irecv mb_{mb_idx}"):
                        handle = my_stage.forward_irecv(mb_idx, i)
                        fwd_recv_handles[mb_idx] = handle

            # --- 步骤 2：所有 Rank 提交 Compute 和 Send ---
            for mb_idx in range(num_microbatches):
                
                # === [修改] 选取一个特定的计算流 ===
                # Rank 0 会使用 S0, S1, S2, S3
                # Rank > 0 也会使用 (虽然它们是串行的，但这无害)
                current_comp_stream = my_stage.comp_streams[mb_idx]
                
                # 1. 计算
                with record_function(f"Fwd Compute mb_{mb_idx}"):
                    # 使用特定的 micro-batch 计算流
                    with torch.cuda.stream(current_comp_stream): 
                        if current_stage > 1:
                            fwd_recv_handles[mb_idx].wait()
                            input_tensor = my_stage.out_x_buffers[mb_idx].to(DEVICE)
                        else:
                            # Rank 0: 在其自己的流 (S0, S1...) 上排队
                            # 这将实现计算的并行执行
                            input_tensor = micro_inputs[mb_idx]

                        output_tensor = my_stage.forward(input_tensor)
                
                # 2. 发送
                if current_stage < model_parallel_size:
                    with record_function(f"Fwd Send mb_{mb_idx}"):
                        with torch.cuda.stream(my_stage.comm_stream):
                            # === [修改] 通信流等待 *正确的* 计算流 ===
                            my_stage.comm_stream.wait_stream(current_comp_stream) 
                            handle = my_stage.forward_isend(output_tensor, mb_idx, i)
                            fwd_send_handles[mb_idx] = handle

            # === Backward ===

            # --- 步骤 1：所有非最后阶段的 Rank 提交 Bwd irecv ---
            if current_stage < model_parallel_size:
                for mb_idx in reversed(range(num_microbatches)):
                    with record_function(f"Bwd Post Irecv mb_{mb_idx}"):
                        handle = my_stage.backward_irecv(mb_idx, i)
                        bwd_recv_handles[mb_idx] = handle

            # --- 步骤 2：所有 Rank 提交 Bwd Compute 和 Send ---
            for mb_idx in reversed(range(num_microbatches)):
                
                # === [修改] 选取一个特定的计算流 ===
                current_comp_stream = my_stage.comp_streams[mb_idx]
                
                # 1. 计算
                with record_function(f"Bwd Compute mb_{mb_idx}"):
                    # 使用特定的 micro-batch 计算流
                    with torch.cuda.stream(current_comp_stream):
                        grad_tensor = None
                        if current_stage < model_parallel_size:
                            bwd_recv_handles[mb_idx].wait()
                            grad_tensor = my_stage.grad_y_buffers[mb_idx].to(DEVICE)
                        
                        if current_stage == model_parallel_size:
                            logits = my_stage.fwd_cache[mb_idx].transpose(-2, -1)
                            loss = F.cross_entropy(logits, micro_labels[mb_idx])
                            loss.backward()
                        else:
                            my_stage.backward(mb_idx, grad_tensor)

                # 2. 发送
                if current_stage > 1:
                    with record_function(f"Bwd Send mb_{mb_idx}"):
                        with torch.cuda.stream(my_stage.comm_stream):
                            # === [修改] 通信流等待 *正确的* 计算流 ===
                            my_stage.comm_stream.wait_stream(current_comp_stream)
                            handle = my_stage.backward_isend(mb_idx, i)
                            bwd_send_handles[mb_idx] = handle


            # === 同步和更新 ===
            
            # 1. 同步所有流
            if my_stage.comp_streams:
                # === [修改] 等待 *所有* 计算流 ===
                for s in my_stage.comp_streams:
                    torch.cuda.current_stream().wait_stream(s)
            if my_stage.comm_stream:
                torch.cuda.current_stream().wait_stream(my_stage.comm_stream)

            # 2. 确保所有 P2P 通信已完成 (CPU 句柄)
            for handle in fwd_send_handles.values():
                handle.wait()
            for handle in bwd_send_handles.values():
                handle.wait()
            
            # dist.barrier() 

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
        if i == 25:
            nvtx.range_pop()
            print(f"Epoch {i} profiling finished. Exiting.")
            dist.barrier()
            sys.exit(0)