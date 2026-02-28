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
import sys
import queue
from GPT import *
import contextlib
from torch.profiler import record_function

# ================= Configuration =================
emb_size = 512
head_size = 128
n_layer = 16
sequence_len = 128
learning_rate = 1e-4
eval_iters = 20
batch_size = 32
epochs = 2
num_microbatches = 8

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


# ================= WeightGradStore & Custom Autograd =================

class WeightGradStore:
    """
    存储延迟执行的 W (Weight Gradient) 计算任务。
    """
    cache = []  # 当前微批次的 W 任务列表
    weight_grad_queue = queue.Queue() # 等待执行的 W 任务队列
    split_bw = False

    @classmethod
    def put(cls, compute_func):
        """
        存入一个计算闭包 (Closure)。
        如果 split_bw=False，直接执行（退化为标准 1F1B）。
        """
        if not cls.split_bw:
            compute_func()
            return
        cls.cache.append(compute_func)

    @classmethod
    def flush(cls):
        """将当前 cache 打包推入队列"""
        if cls.cache:
            cls.weight_grad_queue.put(list(cls.cache))
            cls.cache = []

    @classmethod
    def pop(cls):
        """取出并执行一整组 W 计算任务"""
        if cls.weight_grad_queue.qsize() == 0:
            return
        # 使用 record_function 标记，方便在 Profiler 中看到 W 阶段
        with record_function("ZeroBubble_W_Step"):
            stored_funcs = cls.weight_grad_queue.get()
            for func in stored_funcs:
                func()

    @classmethod
    def pop_all(cls):
        while cls.weight_grad_queue.qsize() > 0:
            cls.pop()

    @classmethod
    def clear(cls):
        cls.cache = []
        cls.weight_grad_queue = queue.Queue()
        cls.split_bw = False


class BubbleLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # 1. 计算 B (Input Gradient) - 这是下一层反向传播必须的，立即计算
        if ctx.needs_input_grad[0]:
            with record_function("Bubble_Backward_B"):
                grad_input = grad_output.matmul(weight)

        # 2. 定义 W (Weight Gradient) 的计算任务
        def compute_w_grad():
            # 确保在计算 W 时不记录梯度图
            with torch.no_grad():
                # Fix: Flatten batch and sequence dimensions to perform valid matrix multiplication
                # input: (Batch, Seq, In) -> (Batch*Seq, In)
                # grad_output: (Batch, Seq, Out) -> (Batch*Seq, Out)
                input_reshaped = input.reshape(-1, input.shape[-1])
                grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

                if ctx.needs_input_grad[1]:
                    # gw: (Out, Batch*Seq) @ (Batch*Seq, In) -> (Out, In)
                    gw = grad_output_reshaped.t().matmul(input_reshaped)
                    
                    # 累加到 parameter.grad
                    if weight.grad is None:
                        weight.grad = gw
                    else:
                        weight.grad += gw
                
                if bias is not None and ctx.needs_input_grad[2]:
                    # Sum over all dimensions except the last one (feature dim)
                    gb = grad_output_reshaped.sum(dim=0)
                    if bias.grad is None:
                        bias.grad = gb
                    else:
                        bias.grad += gb

        # 3. 根据策略决定是立即执行 W 还是放入 Store
        WeightGradStore.put(compute_w_grad)

        # 返回 grad_input，其余为 None (因为 weight/bias 的梯度通过 side-effect 更新)
        return grad_input, None, None

class BubbleLinear(nn.Module):
    """
    替换 nn.Linear。使用自定义的 Function 来实现 B/W 分离。
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return BubbleLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# 辅助函数：将模型中的 nn.Linear 替换为 BubbleLinear
import math
def convert_to_bubble_model(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 获取旧层的参数
            new_layer = BubbleLinear(child.in_features, child.out_features, child.bias is not None)
            new_layer.weight.data = child.weight.data
            if child.bias is not None:
                new_layer.bias.data = child.bias.data
            # 替换
            setattr(module, name, new_layer)
        else:
            convert_to_bubble_model(child)

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


# ================= P2P Communication =================

def p2p_send_forward(tensor, mb_idx):
    dst = global_rank + 1
    tag = mb_idx
    send_buf = tensor.detach().clone().contiguous()
    op = dist.P2POp(dist.isend, send_buf, dst, group=model_parallel_group, tag=tag)
    return op, send_buf

def p2p_recv_forward(mb_idx, device):
    src = global_rank - 1
    tag = mb_idx
    buf = torch.zeros(
        batch_size // num_microbatches, sequence_len, emb_size,
        device=device, dtype=torch.float32
    )
    op = dist.P2POp(dist.irecv, buf, src, group=model_parallel_group, tag=tag)
    return op, buf

def p2p_send_backward(tensor, mb_idx):
    dst = global_rank - 1
    tag = 1000 + mb_idx
    send_buf = tensor.detach().clone().contiguous()
    op = dist.P2POp(dist.isend, send_buf, dst, group=model_parallel_group, tag=tag)
    return op, send_buf

def p2p_recv_backward(mb_idx, device):
    src = global_rank + 1
    tag = 1000 + mb_idx
    buf = torch.zeros(
        batch_size // num_microbatches, sequence_len, emb_size,
        device=device, dtype=torch.float32
    )
    op = dist.P2POp(dist.irecv, buf, src, group=model_parallel_group, tag=tag)
    return op, buf

def execute_p2p_ops(ops_list):
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

        # 1. 构建子模型
        if self.stage_ID == 1:
            self.token_embedding = nn.Embedding(vs, emb_size)
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            self.sub_model = nn.Sequential(*[model[i] for i in range(2, len(self.model_idx))])
            self.sub_model_opt = nn.Sequential(*[model[i] for i in model_idx]) # 用于优化器注册
            
            # 重要：将子模型中的 nn.Linear 替换为 BubbleLinear
            convert_to_bubble_model(self.sub_model)
            convert_to_bubble_model(self.sub_model_opt)
            
            self.optimizer = optim.Adam(self.sub_model_opt.parameters(), lr=learning_rate)
        else:
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])
            
            # 重要：将子模型中的 nn.Linear 替换为 BubbleLinear
            convert_to_bubble_model(self.sub_model)
            
            self.optimizer = optim.Adam(self.sub_model.parameters(), lr=learning_rate)

        self.fwd_cache = {}
        self.input_cache = {}
        self._send_bufs = []
        
        # 移除了 Hook 注册，因为 BubbleLinear 内置了逻辑

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

    def forward_compute(self, mb_idx, micro_input=None):
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
            return None

    def forward_compute_with_input(self, mb_idx, recv_buf):
        x = recv_buf.requires_grad_(True)
        self.input_cache[mb_idx] = x
        y = self.sub_model(x)
        y.retain_grad()
        self.fwd_cache[mb_idx] = y
        return y

    def backward_compute(self, mb_idx, grad_or_label):
        cached_output = self.fwd_cache.pop(mb_idx)

        # 这里的 backward() 会触发 BubbleLinear.backward
        # 从而产生 B (立即计算) 和 W (推入 WeightGradStore)
        if self.stage_ID == model_parallel_size:
            logits = cached_output.transpose(-2, -1)
            loss = F.cross_entropy(logits, grad_or_label)
            loss.backward()
            cached_input = self.input_cache.pop(mb_idx)
            return loss.item(), cached_input.grad if cached_input.grad is not None else None

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


# ================= ZB-H1 Schedule (Same Logic) =================
# 调度逻辑本身不需要大幅修改，只需要确保 Store 的行为正确
def run_zb1p_iteration(stage, micro_inputs, micro_labels):
    pipeline_rank = stage.rank_in_pipeline
    is_first = (stage.stage_ID == 1)
    is_last = (stage.stage_ID == model_parallel_size)

    stage.zero_grad()
    WeightGradStore.clear()

    num_warmup = min(num_microbatches, model_parallel_size - pipeline_rank - 1)
    num_microbatches_remaining = num_microbatches - num_warmup

    recv_fwd_bufs = {}
    recv_bwd_bufs = {}

    fwd_step = 0
    bwd_step = 0

    # === Phase 1: Warmup ===
    for _ in range(num_warmup):
        if not is_first:
            ops = []
            op, buf = p2p_recv_forward(fwd_step, stage.device)
            ops.append(op)
            execute_p2p_ops(ops)
            recv_fwd_bufs[fwd_step] = buf

        if is_first:
            y = stage.forward_compute(fwd_step, micro_inputs[fwd_step])
        else:
            y = stage.forward_compute_with_input(fwd_step, recv_fwd_bufs.pop(fwd_step))

        if not is_last:
            ops = []
            op, sbuf = p2p_send_forward(y, fwd_step)
            stage._send_bufs.append(sbuf)
            ops.append(op)
            execute_p2p_ops(ops)

        fwd_step += 1

    # === Phase 2: Steady ===
    for i in range(num_microbatches_remaining):
        last_iteration = (i == num_microbatches_remaining - 1)

        # --- Forward ---
        if not is_first:
            ops = []
            op, buf = p2p_recv_forward(fwd_step, stage.device)
            ops.append(op)
            execute_p2p_ops(ops)
            recv_fwd_bufs[fwd_step] = buf

        if is_first:
            y = stage.forward_compute(fwd_step, micro_inputs[fwd_step])
        else:
            y = stage.forward_compute_with_input(fwd_step, recv_fwd_bufs.pop(fwd_step))

        ops = []
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

        # --- Backward (B) ---
        # 这里的 split_bw 开关现在控制 BubbleLinear 是否生成闭包
        WeightGradStore.split_bw = (i < pipeline_rank or last_iteration) and pipeline_rank > 0

        if is_last:
            loss_val, input_grad = stage.backward_compute(bwd_step, micro_labels[bwd_step])
            if loss_val is not None and global_rank == world_size - 1:
                print(f"  MB {bwd_step} Loss: {loss_val:.4f}", flush=True)
        else:
            grad = recv_bwd_bufs.pop(bwd_step)
            loss_val, input_grad = stage.backward_compute(bwd_step, grad)

        # 将生成的 W 闭包刷入队列
        if WeightGradStore.split_bw:
            WeightGradStore.flush()

        # --- Communicate B & Execute W ---
        if last_iteration:
            if not is_first and input_grad is not None:
                ops = []
                op, sbuf = p2p_send_backward(input_grad, bwd_step)
                stage._send_bufs.append(sbuf)
                ops.append(op)
                execute_p2p_ops(ops)

            # 执行 W 
            if i >= pipeline_rank > 0:
                WeightGradStore.pop()
        else:
            if not is_first and input_grad is not None:
                ops = []
                op, sbuf = p2p_send_backward(input_grad, bwd_step)
                stage._send_bufs.append(sbuf)
                ops.append(op)
                execute_p2p_ops(ops)

            # if i >= pipeline_rank > 0:
            #     WeightGradStore.pop()

        bwd_step += 1

    # === Phase 3: Cooldown ===
    for i in range(num_warmup):
        if not is_last:
            ops = []
            op, rbuf = p2p_recv_backward(bwd_step, stage.device)
            recv_bwd_bufs[bwd_step] = rbuf
            ops.append(op)
            execute_p2p_ops(ops)

        WeightGradStore.split_bw = pipeline_rank > 0

        if is_last:
            loss_val, input_grad = stage.backward_compute(bwd_step, micro_labels[bwd_step])
            if loss_val is not None and global_rank == world_size - 1:
                print(f"  MB {bwd_step} Loss: {loss_val:.4f}", flush=True)
        else:
            grad = recv_bwd_bufs.pop(bwd_step)
            loss_val, input_grad = stage.backward_compute(bwd_step, grad)

        if not is_first and input_grad is not None:
            ops = []
            op, sbuf = p2p_send_backward(input_grad, bwd_step)
            stage._send_bufs.append(sbuf)
            ops.append(op)
            execute_p2p_ops(ops)

        if WeightGradStore.split_bw:
            WeightGradStore.flush()
            if num_microbatches_remaining + i >= pipeline_rank:
                WeightGradStore.pop()

        bwd_step += 1

    WeightGradStore.pop_all()
    
# ================= Data Setup =================
# 保持原样，仅在初始化时使用修改后的 GPT 初始化逻辑
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

if global_rank == 0:
    print(f"ZB-H1 (Real Split) initialized: {model_parallel_size} stages, {num_microbatches} microbatches")
    for r in range(model_parallel_size):
        nw = min(num_microbatches, model_parallel_size - r - 1)
        nr = num_microbatches - nw
        if r == 0:
            print(f"  Pipeline rank {r}: warmup={nw}, no B/W split (rank 0)")
        else:
            split_count = min(r, nr - 1) + 1 + nw
            print(f"  Pipeline rank {r}: warmup={nw}, ~{split_count} B/W splits, W delayed by {r} steps")

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

        # 仅在第30步进行 Profiling
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_zb1p_real/rank{global_rank}')
        ) if i == 30 else contextlib.nullcontext()     

        with profiler_context:
            run_zb1p_iteration(my_stage, micro_inputs, micro_labels)

            if my_stage.is_training:
                my_stage.all_reduce_gradients()
                my_stage.optimizer.step()

            dist.barrier()