import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
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
import sys
from GPT import *
import contextlib

# ================= Configuration =================
emb_size = 4096
head_size = 128
n_layer = 16
sequence_len = 128
learning_rate = 1e-4
eval_iters = 20
batch_size = 32
epochs = 2
num_microbatches = 4

def init_model(model_config):
    model = nn.ModuleList()
    for i, j in model_config.items():
        if i == "em_tokn":
            mdl = nn.Embedding(j[0], j[1])
        elif i == "em_pos":
            mdl = nn.Embedding(j[0], j[1])
        elif i == "ln":
            mdl = nn.LayerNorm(j[0])
        elif i == "lm_head":
            mdl = nn.Linear(j[0], j[1])
        elif re.search("decoder", i):
            mdl = Block(j[0], j[1])
        model.append(mdl)
    return model

# ================= Distributed Setup =================
env_dict = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
}
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=300))
global_rank = int(os.environ["RANK"])
if global_rank == 0:
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

def get_device():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return f'cuda:{local_rank}'
    return 'cpu'

DEVICE = get_device()

# ================= 2D Parallelism Groups =================
world_size = dist.get_world_size()
model_parallel_size = 4
data_parallel_size = world_size // model_parallel_size

model_parallel_groups = [
    dist.new_group(list(range(i * model_parallel_size, (i + 1) * model_parallel_size)))
    for i in range(data_parallel_size)
]
data_parallel_groups = [
    dist.new_group(list(range(i, world_size, model_parallel_size)))
    for i in range(model_parallel_size)
]

model_parallel_group = model_parallel_groups[global_rank // model_parallel_size]
data_parallel_group = data_parallel_groups[global_rank % model_parallel_size]

# ================= P2P Communication Helpers =================
def p2p_send_forward(tensor, mb_idx):
    """发送 activation 到下一个 pipeline stage"""
    dst = global_rank + 1
    tag = mb_idx
    send_buf = tensor.detach().clone().contiguous()
    op = dist.P2POp(dist.isend, send_buf, dst, group=model_parallel_group, tag=tag)
    return op, send_buf

def p2p_recv_forward(mb_idx, device):
    """接收来自上一个 pipeline stage 的 activation"""
    src = global_rank - 1
    tag = mb_idx
    buf = torch.zeros(
        batch_size // num_microbatches, sequence_len, emb_size,
        device=device, dtype=torch.float32
    )
    op = dist.P2POp(dist.irecv, buf, src, group=model_parallel_group, tag=tag)
    return op, buf

def p2p_send_backward(tensor, mb_idx):
    """发送梯度到上一个 pipeline stage"""
    dst = global_rank - 1
    tag = 1000 + mb_idx
    send_buf = tensor.detach().clone().contiguous()
    op = dist.P2POp(dist.isend, send_buf, dst, group=model_parallel_group, tag=tag)
    return op, send_buf

def p2p_recv_backward(mb_idx, device):
    """接收来自下一个 pipeline stage 的梯度"""
    src = global_rank + 1
    tag = 1000 + mb_idx
    buf = torch.zeros(
        batch_size // num_microbatches, sequence_len, emb_size,
        device=device, dtype=torch.float32
    )
    op = dist.P2POp(dist.irecv, buf, src, group=model_parallel_group, tag=tag)
    return op, buf

def execute_p2p_ops(ops_list):
    """
    批量提交所有 P2P 操作，然后统一 wait。
    这是解决死锁的关键：NCCL 要求 send 和 recv 在一个 batch 中成对出现。
    """
    if not ops_list:
        return
    reqs = dist.batch_isend_irecv(ops_list)
    for req in reqs:
        req.wait()

# ================= Stage Class =================
class Stage:
    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        self.stage_ID = ID
        self.rank_in_pipeline = ID - 1
        self.device = device
        self.model_idx = model_idx
        self.is_training = True
        self.micro_batch_size = batch_size // num_microbatches

        if self.stage_ID == 1:
            self.token_embedding = nn.Embedding(vs, emb_size)
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            self.sub_model = nn.Sequential(*[model[i] for i in range(2, len(self.model_idx))])
            self.sub_model_opt = nn.Sequential(*[model[i] for i in model_idx])
            self.optimizer = optim.Adam(self.sub_model_opt.parameters(), lr=learning_rate)
        else:
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])
            self.optimizer = optim.Adam(self.sub_model.parameters(), lr=learning_rate)

        self.fwd_cache = {}
        self.input_cache = {}
        # 保存 send buffer 引用，防止被 GC
        self._send_bufs = []

    def to(self, device):
        if self.stage_ID == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        self.sub_model.to(device)

    def train(self):
        self.sub_model.train()

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.fwd_cache.clear()
        self.input_cache.clear()
        self._send_bufs.clear()

    # ---- Forward: 计算 + 返回需要执行的 P2P ops ----

    def forward_compute(self, mb_idx, micro_input=None):
        """
        只做本地计算，返回 (output, recv_input_or_None)
        不在这里做通信！
        """
        if self.stage_ID == 1:
            x = micro_input
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            x_emb = tok_emb + pos_emb
            self.input_cache[mb_idx] = x_emb
            y = self.sub_model(x_emb)
            y.retain_grad()
            self.fwd_cache[mb_idx] = y
            return y

        else:
            # 需要先 recv，交给调用方处理
            return None

    def forward_compute_with_input(self, mb_idx, recv_buf):
        """用接收到的 activation 做前向计算"""
        x = recv_buf.requires_grad_(True)
        self.input_cache[mb_idx] = x
        y = self.sub_model(x)
        y.retain_grad()
        self.fwd_cache[mb_idx] = y
        return y

    def backward_compute(self, mb_idx, grad_or_label):
        """
        做反向计算。
        - 最后一个 stage：grad_or_label 是 label，计算 loss
        - 其他 stage：grad_or_label 是从后面收到的梯度
        返回 (loss_value_or_None, input_grad_or_None)
        """
        cached_output = self.fwd_cache.pop(mb_idx)

        if self.stage_ID == model_parallel_size:
            logits = cached_output.transpose(-2, -1)
            loss = F.cross_entropy(logits, grad_or_label)
            loss.backward()
            cached_input = self.input_cache.pop(mb_idx)
            return loss.item(), cached_input.grad

        elif self.stage_ID == 1:
            cached_output.backward(grad_or_label)
            self.input_cache.pop(mb_idx, None)
            return None, None

        else:
            cached_output.backward(grad_or_label)
            cached_input = self.input_cache.pop(mb_idx)
            return None, cached_input.grad

    def all_reduce_gradients(self):
        if data_parallel_size > 1:
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM,
                                    group=data_parallel_group)
                    param.grad /= data_parallel_size


# ================= 1F1B Schedule (通信和计算分离) =================

def run_1f1b_iteration(stage, micro_inputs, micro_labels):
    """
    核心思想：每一步的通信（send/recv）都通过 batch_isend_irecv 同时提交，
    避免单独的 send 阻塞后续操作导致死锁。

    流程：
    1. 先做本地计算
    2. 收集本步需要的所有 P2P 操作（send_fwd, recv_fwd, send_bwd, recv_bwd）
    3. 用 batch_isend_irecv 一次性提交并 wait
    """

    pipeline_rank = stage.rank_in_pipeline
    num_warmup = min(num_microbatches, model_parallel_size - pipeline_rank - 1)
    is_first = (stage.stage_ID == 1)
    is_last = (stage.stage_ID == model_parallel_size)

    stage.zero_grad()

    fwd_step = 0
    bwd_step = 0

    # 存储从 recv 得到的 buffer，供后续 forward_compute_with_input 使用
    recv_fwd_bufs = {}
    recv_bwd_bufs = {}

    # ======= Phase 1: Warmup (only forward) =======
    for _ in range(num_warmup):
        # Step A: 如果不是第一个 stage，需要先 recv
        if not is_first:
            ops = []
            op, buf = p2p_recv_forward(fwd_step, stage.device)
            ops.append(op)
            execute_p2p_ops(ops)
            recv_fwd_bufs[fwd_step] = buf

        # Step B: 本地计算
        if is_first:
            y = stage.forward_compute(fwd_step, micro_inputs[fwd_step])
        else:
            y = stage.forward_compute_with_input(fwd_step, recv_fwd_bufs.pop(fwd_step))

        # Step C: 如果不是最后一个 stage，需要 send
        if not is_last:
            ops = []
            op, sbuf = p2p_send_forward(y, fwd_step)
            stage._send_bufs.append(sbuf)
            ops.append(op)
            execute_p2p_ops(ops)

        fwd_step += 1

    # ======= Phase 2: 1F1B Steady State =======
    while fwd_step < num_microbatches:
        # ---------- Forward part ----------
        # recv_fwd (如果需要)
        if not is_first:
            ops = []
            op, buf = p2p_recv_forward(fwd_step, stage.device)
            ops.append(op)
            execute_p2p_ops(ops)
            recv_fwd_bufs[fwd_step] = buf

        # 本地 forward 计算
        if is_first:
            y = stage.forward_compute(fwd_step, micro_inputs[fwd_step])
        else:
            y = stage.forward_compute_with_input(fwd_step, recv_fwd_bufs.pop(fwd_step))

        # send_fwd + recv_bwd 同时提交！
        # 这是防止死锁的关键：send 和 recv 在同一个 batch 中
        ops = []
        send_buf_ref = None

        if not is_last:
            op, sbuf = p2p_send_forward(y, fwd_step)
            stage._send_bufs.append(sbuf)
            ops.append(op)

        if not is_last:
            op, rbuf = p2p_recv_backward(bwd_step, stage.device)
            recv_bwd_bufs[bwd_step] = rbuf
            ops.append(op)

        execute_p2p_ops(ops)

        fwd_step += 1

        # ---------- Backward part ----------
        if is_last:
            # 最后一个 stage 直接用 label 算 loss
            loss_val, input_grad = stage.backward_compute(bwd_step, micro_labels[bwd_step])
            if loss_val is not None and global_rank == world_size - 1:
                print(f"MB {bwd_step} Loss: {loss_val:.4f}", flush=True)

            # send_bwd
            if input_grad is not None:
                ops = []
                op, sbuf = p2p_send_backward(input_grad, bwd_step)
                stage._send_bufs.append(sbuf)
                ops.append(op)
                execute_p2p_ops(ops)
        else:
            # 中间/第一个 stage：用收到的梯度做 backward
            grad = recv_bwd_bufs.pop(bwd_step)
            loss_val, input_grad = stage.backward_compute(bwd_step, grad)

            # send_bwd (如果不是第一个 stage)
            if not is_first and input_grad is not None:
                ops = []
                op, sbuf = p2p_send_backward(input_grad, bwd_step)
                stage._send_bufs.append(sbuf)
                ops.append(op)
                execute_p2p_ops(ops)

        bwd_step += 1

    # ======= Phase 3: Cooldown (only backward) =======
    while bwd_step < num_microbatches:
        # recv_bwd (如果需要)
        if not is_last:
            ops = []
            op, rbuf = p2p_recv_backward(bwd_step, stage.device)
            recv_bwd_bufs[bwd_step] = rbuf
            ops.append(op)
            execute_p2p_ops(ops)

        # backward 计算
        if is_last:
            loss_val, input_grad = stage.backward_compute(bwd_step, micro_labels[bwd_step])
            if loss_val is not None and global_rank == world_size - 1:
                print(f"MB {bwd_step} Loss: {loss_val:.4f}", flush=True)

            if input_grad is not None:
                ops = []
                op, sbuf = p2p_send_backward(input_grad, bwd_step)
                stage._send_bufs.append(sbuf)
                ops.append(op)
                execute_p2p_ops(ops)
        else:
            grad = recv_bwd_bufs.pop(bwd_step)
            loss_val, input_grad = stage.backward_compute(bwd_step, grad)

            if not is_first and input_grad is not None:
                ops = []
                op, sbuf = p2p_send_backward(input_grad, bwd_step)
                stage._send_bufs.append(sbuf)
                ops.append(op)
                execute_p2p_ops(ops)

        bwd_step += 1


# ================= Data Setup =================
raw_datasets = load_dataset('code_search_net', 'python')
datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])
tok = char_tokenizer(datasets['whole_func_string'])
vs = len(tok.char2ind)

model_config = {
    "em_tokn": [vs, emb_size],
    "em_pos": [sequence_len, emb_size],
    **{f"decoder{i}": [emb_size, head_size] for i in range(1, n_layer + 1)},
    "ln": [emb_size],
    "lm_head": [emb_size, vs]
}

gpt = init_model(model_config=model_config)

model_idx1 = [0, 1, 2, 3, 4, 5]
model_idx2 = [6, 7, 8, 9]
model_idx3 = [10, 11, 12, 13]
model_idx4 = [14, 15, 16, 17, 18, 19]

rank_to_stage = {
    0: 1, 1: 2, 2: 3, 3: 4,
    4: 1, 5: 2, 6: 3, 7: 4,
}

stage_id = rank_to_stage[global_rank]
if stage_id == 1:
    m_idx = model_idx1
elif stage_id == 2:
    m_idx = model_idx2
elif stage_id == 3:
    m_idx = model_idx3
else:
    m_idx = model_idx4

my_stage = Stage(stage_id, gpt, m_idx, learning_rate, DEVICE, batch_size)
my_stage.to(DEVICE)
del gpt
gc.collect()

# Data Loader
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
if global_rank != 0:
    dist.barrier()
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
if global_rank == 0:
    dist.barrier()
tokenized.set_format(type='torch', device=DEVICE)

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

# ================= Training Loop =================
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    iterator = enumerate(train_loader, 0)
    if global_rank == 0:
        iterator = tqdm(iterator, total=len(train_loader))

    for i, data in iterator:
        dist.barrier()
        inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)

        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)

        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_1f1b_4096/rank{global_rank}')
        ) if i == 30 else contextlib.nullcontext()     

        with profiler_context:
            run_1f1b_iteration(my_stage, micro_inputs, micro_labels)

            if my_stage.is_training:
                my_stage.all_reduce_gradients()
                my_stage.optimizer.step()

            dist.barrier()