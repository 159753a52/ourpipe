import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import datetime
import time
import random
import numpy as np
import gc
import threading

import sys
from GPT import *
import torch.profiler
from torch.profiler import record_function
import contextlib

# GPT-3.35B
emb_size = 4096
head_size = 128
# The n_layer of GPT-3.35B should be 16
n_layer = 16
sequence_len = 128
learning_rate = 1e-4
eval_iters = 20
batch_size = 32
epochs = 2

# # GPT-2 medium
# emb_size = 1024
# head_size = 64
# n_layer = 24
# sequence_len = 256
# learning_rate = 1e-4
# eval_iters = 20
# batch_size = 12
# epochs = 2

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
        elif (re.search("decoder",i)).group() == "decoder": # 可能会报错
            mdl = Block(j[0],j[1])
        model.append(mdl)
    return model

num_microbatches = 4  # 每个batch分成4个micro-batch

# 通信域创建
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=300)) # gloo
global_rank = int(os.environ["RANK"])
print(f"global_rank is {global_rank}")
if global_rank == 0:
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
base_time=time.time()

# if global_rank == 0:
#     DEVICE = 'cuda:0'
# elif global_rank == 1:
#     DEVICE = 'cuda:1'
# elif global_rank==2:
#     DEVICE = 'cuda:2'
# elif global_rank==3:
#     DEVICE = 'cuda:3'

# 替换原有的DEVICE分配逻
def get_device():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        return f'cuda:{local_rank}'
    return 'cpu'

DEVICE = get_device()

print("The current rank is {}".format(DEVICE))

#===================== 2D start =====================
# 在初始化分布式环境后添加
world_size = dist.get_world_size()
model_parallel_size = 4  # 每个节点4个GPU做模型并行
data_parallel_size = world_size // model_parallel_size

# 创建模型并行组（节点内通信）
model_parallel_groups = [
    dist.new_group(list(range(i*model_parallel_size, (i+1)*model_parallel_size))) 
    for i in range(data_parallel_size)
]

# 创建数据并行组（跨节点通信）
data_parallel_groups = [
    dist.new_group(list(range(i, world_size, model_parallel_size)))
    for i in range(model_parallel_size)
]

# 为当前rank分配组
model_parallel_group = model_parallel_groups[global_rank // model_parallel_size]
data_parallel_group = data_parallel_groups[global_rank % model_parallel_size]
#===================== 2D end =====================

class Stage:
    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        self.stage_ID = ID
        self.device = device
        self.model_idx = model_idx
        self.is_training = True
        self.micro_batch_size = batch_size // num_microbatches

        #===================== 2D start =====================
        # 添加通信组
        self.model_parallel_group = model_parallel_group
        self.data_parallel_group = data_parallel_group
        self.local_rank = global_rank % model_parallel_size  # 节点内局部rank
        #===================== 2D end =====================

        if self.stage_ID == 1:
            # 文字嵌入层
            self.token_embedding = nn.Embedding(vs, emb_size)
            # 位置嵌入层
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            self.sub_model = nn.Sequential(*[model[i] for i in range(2,len(self.model_idx))])
        else:
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])
        # self.sub_model_save = nn.Sequential(*[model[i] for i in model_idx])

        # 优化器现在只需要一个，因为模型已经被Sequential包装
        if self.stage_ID == 1:
            self.sub_model_opt = nn.Sequential(*[model[i] for i in model_idx])
            self.optimizer = optim.Adam(self.sub_model_opt.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.sub_model.parameters(), lr=learning_rate)
        # self.optimizer_list= [optim.Adam(model[i].parameters(), lr=learning_rate) for i in model_idx]

        self.out_x = torch.zeros(self.micro_batch_size, sequence_len, emb_size).to(device)
        self.grad_y = torch.zeros(self.micro_batch_size, sequence_len, emb_size).to(device)
        
        self.lossi = [] 
        # 【修复 A】: 我们需要两个缓存   
        self.fwd_cache = []  # 缓存 Stage 的 *输出* (Y)
        self.input_cache = [] # 缓存 Stage 的 *输入* (X)
        
    def to(self,device):
        if self.stage_ID == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        self.sub_model.to(device)
    
    def eval(self):
        self.sub_model.eval()
    
    def train(self):
        self.sub_model.train()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def forward(self, x):
        if self.stage_ID == 1:
            B, T = x.shape
            # 定义词元的位置，形状为(T)
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)

            # 词元语义特征
            tok_emb = self.token_embedding(x)       # (B, T,  C)
            # 位置特征
            pos_emb = self.position_embedding(pos)  # (   T,  C)

            x_emb = tok_emb + pos_emb
            # 【修复 B】: 保存 Stage 1 的输入
            # (它不需要 requires_grad_() 因为它是图的根)
            self.input_cache.append(x_emb) 
            y = self.sub_model(x_emb)
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
        
    def forward_send(self, out):
        # dist.send(tensor=out, dst=self.stage_ID, tag=self.stage_ID)
        """只在模型并行组内发送数据"""
        # dst = (global_rank - self.local_rank) + ((self.local_rank + 1) % model_parallel_size) #global_rank+1
        dst = global_rank + 1
        dist.send(tensor=out, dst=dst, tag=self.stage_ID, group=self.model_parallel_group)


    def forward_recv(self):
        # dist.recv(tensor=self.out_x, src=self.stage_ID-2, tag=self.stage_ID-1)
        # self.out_x.to(self.device)
        """只在模型并行组内接收数据"""
        # src = (global_rank - self.local_rank) + ((self.local_rank - 1) % model_parallel_size) #global_rank-1
        src = global_rank - 1
        dist.recv(tensor=self.out_x, src=src, tag=self.stage_ID-1, group=self.model_parallel_group)
        self.out_x = self.out_x.to(self.device)

    def backward(self, microbatch_idx=None):
        cached_output = self.fwd_cache[microbatch_idx]
        cached_output.backward(self.grad_y)
            
    def backward_send(self, microbatch_idx):
        # dist.send(tensor=self.fwd_cache[microbatch_idx].grad, dst=self.stage_ID-2, tag=self.stage_ID)
        # 【修复 D】: 发送 *输入* 的梯度 (dL/dX)
        cached_input = self.input_cache[microbatch_idx]
        grad_x = cached_input.grad # 这才是我们需要的 dL/dX
        # dst = (global_rank - self.local_rank) + ((self.local_rank - 1) % model_parallel_size) #global_rank-1
        dst = global_rank - 1
        dist.send(grad_x, dst=dst, group=self.model_parallel_group)

    def backward_recv(self):
        # dist.recv(tensor=self.grad_y, src=self.stage_ID, tag=self.stage_ID+1)
        # self.grad_y.to(self.device)
        # src = (global_rank - self.local_rank) + ((self.local_rank + 1) % model_parallel_size) #global_rank+1
        src = global_rank + 1
        dist.recv(self.grad_y, src=src, group=self.model_parallel_group)
        self.grad_y = self.grad_y.to(self.device)

    def all_reduce_gradients(self):
        for param in self.optimizer.param_groups[0]['params']: 
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                param.grad /= data_parallel_size

    # def all_reduce_gradients(self):
    #     """只在数据并行组内同步梯度"""
    #     if dist.get_world_size(self.data_parallel_group) > 1:  # 仅在需要时同步
    #         for param in self.sub_model.parameters():
    #             if param.grad is not None:
    #                 dist.all_reduce(
    #                     param.grad, 
    #                     op=dist.ReduceOp.SUM,
    #                     group=self.data_parallel_group
    #                 )
    #                 param.grad /= dist.get_world_size(self.data_parallel_group)


# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels

# 使用 barrier 来确保只有一个进程执行数据处理和缓存写入，避免竞态条件
if global_rank != 0:
    dist.barrier()

tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)

if global_rank == 0:
    dist.barrier()

tokenized.set_format(type='torch', device=DEVICE)
if global_rank == 0: 
    print("Train dataset inputs and labels shape are {} and {}".format(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape))
# 构建数据读取器

train_sampler = DistributedSampler(
    dataset=tokenized['train'],
    num_replicas=data_parallel_size,  # 数据并行组大小
    rank=global_rank // model_parallel_size,  # 数据并行组内的rank
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
    num_replicas=data_parallel_size,  # 必须与train_sampler相同
    rank=global_rank // model_parallel_size,
    shuffle=False  # 测试集通常不shuffle！
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
# # 获取一个批量的数据
# next(iter(test_loader))

'''
    emb_size = 4096
    head_size = 128
    n_layer = 16
    sequence_len = 256
    learning_rate = 1e-3
    eval_iters = 20
    batch_size= 12
'''
vs = len(tok.char2ind)
# GPT-3.35B
'''
    model_config={"em_tokn":[vs,emb_size],"em_pos":[sequence_len,emb_size],"decoder1":[emb_size,head_size],"decoder2":[emb_size,head_size],"decoder3":[emb_size,head_size],
              "decoder4":[emb_size,head_size],"decoder5":[emb_size,head_size],"decoder6":[emb_size,head_size],"decoder7":[emb_size,head_size],"decoder8":[emb_size,head_size],
              "decoder9":[emb_size,head_size],"decoder10":[emb_size,head_size],"decoder11":[emb_size,head_size],"decoder12":[emb_size,head_size],"decoder13":[emb_size,head_size],
              "decoder14":[emb_size,head_size],"decoder15":[emb_size,head_size],"decoder16":[emb_size,head_size],"ln":[emb_size],"lm_head":[emb_size,vs]}
'''

# GPT-2 medium
model_config = {
    "em_tokn": [vs, emb_size],
    "em_pos": [sequence_len, emb_size],
    **{f"decoder{i}": [emb_size, head_size] for i in range(1, n_layer+1)},
    "ln": [emb_size],
    "lm_head": [emb_size, vs]
}

gpt = init_model(model_config=model_config)
# # GPT2-MEDIUM 4stage 2dp
# model_idx1=[0,1,2,3,4,5,6,7]
# model_idx2=[8,9,10,11,12,13]
# model_idx3=[14,15,16,17,18,19]
# model_idx4=[20,21,22,23,24,25,26,27]

# GPT3-3.35B 4stage 2dp
model_idx1=[0,1,2,3,4,5]
model_idx2=[6,7,8,9]
model_idx3=[10,11,12,13]
model_idx4=[14,15,16,17,18,19]
# model_idx1=[0,1,2]
# model_idx2=[3,4,5]
# model_idx3=[6,7,8]
# model_idx4=[9,10,11] 

# # GPT3-3.35B 3stage 1dp
# model_idx1=[0,1,2,3,4,5,6]
# model_idx2=[7,8,9,10,11,12]
# model_idx3=[13,14,15,16,17,18,19]

# model_idx1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
# model_idx2 = [14,15,16,17,18,19,20,21,22,23,24,25,26,27]

rank_to_stage = {
    0: 1,  # rank0 -> stage1
    1: 2,  # rank1 -> stage2
    2: 3,  # rank2 -> stage3
    3: 4,  # rank3 -> stage4
    4: 1,  # rank0 -> stage1
    5: 2,  # rank1 -> stage2
    6: 3,  # rank2 -> stage3
    7: 4,  # rank3 -> stage4
}

# 获取当前rank对应的stage
current_stage = rank_to_stage[global_rank]
print("current rank is {}, current stage is {}".format(global_rank, current_stage))

Stage_list = []
for rank in range(world_size):
    if rank_to_stage[rank] == 1:
        Stage_list.append(Stage(1, gpt, model_idx1, learning_rate, DEVICE, batch_size))
    if rank_to_stage[rank] == 2:
        Stage_list.append(Stage(2, gpt, model_idx2, learning_rate, DEVICE, batch_size))
    if rank_to_stage[rank] == 3:
        Stage_list.append(Stage(3, gpt, model_idx3, learning_rate, DEVICE, batch_size))
    if rank_to_stage[rank] == 4:
        Stage_list.append(Stage(4, gpt, model_idx4, learning_rate, DEVICE, batch_size))

# 每个rank只初始化自己的stage
my_stage = Stage_list[global_rank]

for i in range(len(Stage_list)):
    if i == global_rank:
        Stage_list[i].to(DEVICE)

loss_list = []

ck_interval = 1000
loss = 0
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # if recovery and epoch < epoch_r:
    #     continue

    iter_start = time.time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        # if recovery and epoch == epoch_r and i <= i_r:
        #     continue
        train_start = time.time()
        inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)

        if global_rank == 0:
            print("\n################## iteration {} ##################".format(i))

        def Forward_First(mb_idx,current_stage,my_stage):
            # f_start = time.time()
            # print(f"[{f_start-base_time:.4f}] {i}iteration: Forward_START (micro_batch={mb_idx}, partition={current_stage})")
            out = my_stage.forward(micro_inputs[mb_idx])
            # f_end = time.time()
            # print(f"[{f_end-base_time:.4f}] {i}iteration: Forward_END (micro_batch={mb_idx}, partition={current_stage}), Duration:{f_end-f_start:.4f}")
            my_stage.forward_send(out)

        def Forward_Last(mb_idx,current_stage,my_stage):
            my_stage.forward_recv()
            # f_start = time.time()
            # print(f"[{f_start-base_time:.4f}] {i}iteration: Forward_START (micro_batch={mb_idx}, partition={current_stage})")
            out = my_stage.forward(my_stage.out_x)
            # f_end = time.time()
            # print(f"[{f_end-base_time:.4f}] {i}iteration: Forward_END (micro_batch={mb_idx}, partition={current_stage}), Duration:{f_end-f_start:.4f}")

        def Forward_Middle(mb_idx,current_stage,my_stage):
            my_stage.forward_recv()
            # f_start = time.time()
            # print(f"[{f_start-base_time:.4f}] {i}iteration: Forward_START (micro_batch={mb_idx}, partition={current_stage})")
            out = my_stage.forward(my_stage.out_x)
            # f_end = time.time()
            # print(f"[{f_end-base_time:.4f}] {i}iteration: Forward_END (micro_batch={mb_idx}, partition={current_stage}), Duration:{f_end-f_start:.4f}")
            my_stage.forward_send(out)   
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_naive_4096/rank{global_rank}')
        ) if i == 20 else contextlib.nullcontext()        
        dist.barrier() 
        with profiler_context:
            # === Forward ===
            for mb_idx in range(num_microbatches):
                # First stage
                if current_stage == 1:  
                    # f_start = time.time()
                    # print(f"[{f_start-base_time:.4f}] {i}iteration: Forward_START (micro_batch={mb_idx}, partition={current_stage})")
                    # out = my_stage.forward(micro_inputs[mb_idx])
                    # f_end = time.time()
                    # print(f"[{f_end-base_time:.4f}] {i}iteration: Forward_END (micro_batch={mb_idx}, partition={current_stage}), Duration:{f_end-f_start:.4f}")
                    # my_stage.forward_send(out)
                    Forward_First(mb_idx,current_stage,my_stage)
                    

                
                # Last stage
                elif current_stage == model_parallel_size:
                    # my_stage.forward_recv()
                    # f_start = time.time()
                    # print(f"[{f_start-base_time:.4f}] {i}iteration: Forward_START (micro_batch={mb_idx}, partition={current_stage})")
                    # out = my_stage.forward(my_stage.out_x)
                    # f_end = time.time()
                    # print(f"[{f_end-base_time:.4f}] {i}iteration: Forward_END (micro_batch={mb_idx}, partition={current_stage}), Duration:{f_end-f_start:.4f}")
                    Forward_Last(mb_idx,current_stage,my_stage)
                # Middle stage   
                else:  
                    # my_stage.forward_recv()
                    # f_start = time.time()
                    # print(f"[{f_start-base_time:.4f}] {i}iteration: Forward_START (micro_batch={mb_idx}, partition={current_stage})")
                    # out = my_stage.forward(my_stage.out_x)
                    # f_end = time.time()
                    # print(f"[{f_end-base_time:.4f}] {i}iteration: Forward_END (micro_batch={mb_idx}, partition={current_stage}), Duration:{f_end-f_start:.4f}")
                    # my_stage.forward_send(out)
                    Forward_Middle(mb_idx,current_stage,my_stage)

            # === Backward ===
            for mb_idx in reversed(range(num_microbatches)):
                # Last stage
                if current_stage == model_parallel_size:  
                    logits = my_stage.fwd_cache[mb_idx].transpose(-2, -1)
                    loss = F.cross_entropy(logits, micro_labels[mb_idx])
                    loss.backward()
                    my_stage.backward_send(mb_idx)

                    # if i % 5 == 0 and global_rank == world_size - 1:
                    #     print(f"epoch: {epoch}, iter: {i}, loss: {loss.item()}")
                
                # First stage
                elif current_stage == 1:  
                    my_stage.backward_recv()
                    # b_start = time.time()
                    my_stage.backward(mb_idx)
                    # b_end = time.time()
                    # print(f"backward time is {b_end-b_start}s")

                # Middle stage  
                else:  
                    my_stage.backward_recv()
                    my_stage.backward(mb_idx)
                    my_stage.backward_send(mb_idx)   
            # === Update ===
            if my_stage.is_training:
                if data_parallel_size > 1:
                    # comm_start = time.time()
                    my_stage.all_reduce_gradients()
                    # comm_end = time.time()
                    # print("allreduce cost is {}s".format(comm_end - comm_start))

                my_stage.optimizer.step()
                my_stage.optimizer.zero_grad()
                my_stage.fwd_cache.clear()
                my_stage.input_cache.clear()
                # u_e_time=time.time()
                # print(f"[{u_e_time-base_time:.4f}] {i}iteration: Update Finish!!!!!!!")
            dist.barrier() 
            if i == 25:
                sys.exit(0)
        # train_end = time.time()
        # print("train cost is {}s".format(train_end - train_start))
        

        # if i % 50 == 0 and i != 0:
        #     iter_end = time.time()
        #     print("50 iter end time is {}s".format(iter_end - iter_start))
        
