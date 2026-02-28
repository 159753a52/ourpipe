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
import gc
from GPT import *
import contextlib

# ================= Configuration =================
emb_size = 512
head_size = 32
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
# Wave A 方向: fwd rank->rank+1, bwd rank->rank-1
# Wave B 方向: fwd rank->rank-1, bwd rank->rank+1

def _make_p2p(op_type, tensor_or_shape, peer, tag, device=None):
    if op_type == 'send':
        buf = tensor_or_shape.detach().clone().contiguous()
        op = dist.P2POp(dist.isend, buf, peer, group=model_parallel_group, tag=tag)
    else:
        buf = torch.zeros(*tensor_or_shape, device=device, dtype=torch.float32)
        op = dist.P2POp(dist.irecv, buf, peer, group=model_parallel_group, tag=tag)
    return op, buf

def act_shape():
    return (batch_size // (num_microbatches * 2), sequence_len, emb_size)

# Wave A
def p2p_send_forward_A(tensor, mb_idx):
    return _make_p2p('send', tensor, global_rank + 1, mb_idx)

def p2p_recv_forward_A(mb_idx, device):
    return _make_p2p('recv', act_shape(), global_rank - 1, mb_idx, device)

def p2p_send_backward_A(tensor, mb_idx):
    return _make_p2p('send', tensor, global_rank - 1, 1000 + mb_idx)

def p2p_recv_backward_A(mb_idx, device):
    return _make_p2p('recv', act_shape(), global_rank + 1, 1000 + mb_idx, device)

# Wave B (方向反转)
def p2p_send_forward_B(tensor, mb_idx):
    return _make_p2p('send', tensor, global_rank - 1, 2000 + mb_idx)

def p2p_recv_forward_B(mb_idx, device):
    return _make_p2p('recv', act_shape(), global_rank + 1, 2000 + mb_idx, device)

def p2p_send_backward_B(tensor, mb_idx):
    return _make_p2p('send', tensor, global_rank + 1, 3000 + mb_idx)

def p2p_recv_backward_B(mb_idx, device):
    return _make_p2p('recv', act_shape(), global_rank - 1, 3000 + mb_idx, device)

def execute_p2p_ops(ops_list):
    if not ops_list:
        return
    reqs = dist.batch_isend_irecv(ops_list)
    for req in reqs:
        req.wait()

# ================= Hanayo Stage =================
class HanayoStage:
    """
    双向 pipeline stage:
      Wave A: P0(emb+dec) -> P1(dec) -> P2(dec) -> P3(dec+ln+lm_head) -> loss
      Wave B: P3(emb_B+dec) -> P2(dec) -> P1(dec) -> P0(dec+ln_B+lm_head_B) -> loss
    decoder layers 在两个 wave 间共享。
    """

    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        self.stage_ID = ID
        self.rank_in_pipeline = ID - 1
        self.device = device
        self.is_training = True
        self.micro_batch_size = batch_size // (num_microbatches * 2)

        if self.stage_ID == 1:
            self.token_embedding_A = model[0]
            self.position_embedding_A = model[1]
            self.decoder_layers = nn.Sequential(*[model[i] for i in range(2, len(model_idx))])
            self.ln_B = nn.LayerNorm(emb_size)
            self.lm_head_B = nn.Linear(emb_size, vs)
        elif self.stage_ID == model_parallel_size:
            self.decoder_layers = nn.Sequential(*[model[i] for i in model_idx[:-2]])
            self.ln_A = model[model_idx[-2]]
            self.lm_head_A = model[model_idx[-1]]
            self.token_embedding_B = nn.Embedding(vs, emb_size)
            self.position_embedding_B = nn.Embedding(sequence_len, emb_size)
        else:
            self.decoder_layers = nn.Sequential(*[model[i] for i in model_idx])

        all_params = list(self.decoder_layers.parameters())
        if self.stage_ID == 1:
            all_params += list(self.token_embedding_A.parameters())
            all_params += list(self.position_embedding_A.parameters())
            all_params += list(self.ln_B.parameters())
            all_params += list(self.lm_head_B.parameters())
        elif self.stage_ID == model_parallel_size:
            all_params += list(self.ln_A.parameters())
            all_params += list(self.lm_head_A.parameters())
            all_params += list(self.token_embedding_B.parameters())
            all_params += list(self.position_embedding_B.parameters())

        self.optimizer = optim.Adam(all_params, lr=learning_rate)
        self.fwd_cache = {'A': {}, 'B': {}}
        self.input_cache = {'A': {}, 'B': {}}
        self._send_bufs = []

    def to(self, device):
        self.decoder_layers.to(device)
        if self.stage_ID == 1:
            self.token_embedding_A.to(device)
            self.position_embedding_A.to(device)
            self.ln_B.to(device)
            self.lm_head_B.to(device)
        elif self.stage_ID == model_parallel_size:
            self.ln_A.to(device)
            self.lm_head_A.to(device)
            self.token_embedding_B.to(device)
            self.position_embedding_B.to(device)

    def train(self):
        self.decoder_layers.train()
        if self.stage_ID == 1:
            self.token_embedding_A.train(); self.position_embedding_A.train()
            self.ln_B.train(); self.lm_head_B.train()
        elif self.stage_ID == model_parallel_size:
            self.ln_A.train(); self.lm_head_A.train()
            self.token_embedding_B.train(); self.position_embedding_B.train()

    def zero_grad(self):
        self.optimizer.zero_grad()
        for w in ('A', 'B'):
            self.fwd_cache[w].clear()
            self.input_cache[w].clear()
        self._send_bufs.clear()

    def _embed(self, x, wave):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        if wave == 'A':
            return self.token_embedding_A(x) + self.position_embedding_A(pos)
        else:
            return self.token_embedding_B(x) + self.position_embedding_B(pos)

    def _head(self, y, wave):
        if wave == 'A':
            return self.lm_head_A(self.ln_A(y))
        else:
            return self.lm_head_B(self.ln_B(y))

    def _is_entry(self, wave):
        return (wave == 'A' and self.stage_ID == 1) or \
               (wave == 'B' and self.stage_ID == model_parallel_size)

    def _is_exit(self, wave):
        return (wave == 'A' and self.stage_ID == model_parallel_size) or \
               (wave == 'B' and self.stage_ID == 1)

    def forward(self, wave, mb_idx, micro_input=None, recv_buf=None):
        if self._is_entry(wave):
            x_emb = self._embed(micro_input, wave)
            self.input_cache[wave][mb_idx] = x_emb
            y = self.decoder_layers(x_emb)
        elif self._is_exit(wave):
            x = recv_buf.requires_grad_(True)
            self.input_cache[wave][mb_idx] = x
            y = self._head(self.decoder_layers(x), wave)
        else:
            x = recv_buf.requires_grad_(True)
            self.input_cache[wave][mb_idx] = x
            y = self.decoder_layers(x)
        y.retain_grad()
        self.fwd_cache[wave][mb_idx] = y
        return y

    def backward(self, wave, mb_idx, grad_or_label):
        cached_output = self.fwd_cache[wave].pop(mb_idx)

        if self._is_exit(wave):
            logits = cached_output.transpose(-2, -1)
            loss = F.cross_entropy(logits, grad_or_label)
            loss.backward()
            cached_input = self.input_cache[wave].pop(mb_idx)
            return loss.item(), cached_input.grad
        elif self._is_entry(wave):
            cached_output.backward(grad_or_label)
            self.input_cache[wave].pop(mb_idx, None)
            return None, None
        else:
            cached_output.backward(grad_or_label)
            cached_input = self.input_cache[wave].pop(mb_idx)
            return None, cached_input.grad

    def all_reduce_gradients(self):
        if data_parallel_size > 1:
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM,
                                    group=data_parallel_group)
                    param.grad /= data_parallel_size

# ================= Hanayo Schedule =================
def build_hanayo_schedule(pipeline_rank, num_stages, num_mb):
    """
    Phase 1: Wave A warmup forward
    Phase 2: 剩余 Wave A fwd 与 Wave B fwd 交错
    Phase 3: 剩余 Wave B fwd 与 Wave A bwd 交错
    Phase 4: Wave A bwd cooldown + Wave B bwd
    """
    schedule = []
    num_warmup_A = min(num_mb, num_stages - 1 - pipeline_rank)

    fA = fB = bA = bB = 0
    
    # Phase 1: Wave A warmup
    for _ in range(num_warmup_A):
        schedule.append(('fA', fA)); fA += 1

    # Phase 2: 剩余 fA 与 fB 交错
    while fA < num_mb:
        schedule.append(('fA', fA)); fA += 1
        if fB < num_mb:
            schedule.append(('fB', fB)); fB += 1

    # Phase 3: 剩余 fB 与 bB 交错
    while fB < num_mb:
        if bB < num_mb:
            schedule.append(('bB', bB)); bB += 1
        if fB < num_mb:
            schedule.append(('fB', fB)); fB += 1
            if fB == num_mb:
                schedule.append(('bB', bB)); bB += 1

    # Phase 4: bA cooldown
    while bA < num_mb or bB < num_mb:
        if bA < num_mb:
            schedule.append(('bA', bA)); bA += 1
        if bB < num_mb:
            schedule.append(('bB', bB)); bB += 1

    return schedule

# ================= Communication Dispatch =================
COMM = {
    'fA': {'send': p2p_send_forward_A,  'recv': p2p_recv_forward_A},
    'fB': {'send': p2p_send_forward_B,  'recv': p2p_recv_forward_B},
    'bA': {'send': p2p_send_backward_A, 'recv': p2p_recv_backward_A},
    'bB': {'send': p2p_send_backward_B, 'recv': p2p_recv_backward_B},
}

def run_hanayo_iteration(stage, micro_inputs_A, micro_labels_A, micro_inputs_B, micro_labels_B):
    pipeline_rank = stage.rank_in_pipeline
    stage.zero_grad()

    schedule = build_hanayo_schedule(pipeline_rank, model_parallel_size, num_microbatches)
    recv_bufs = {'fA': {}, 'fB': {}, 'bA': {}, 'bB': {}}

    for action, mb_idx in schedule:
        wave = action[1]              # 'A' or 'B'
        is_fwd = action[0] == 'f'
        is_entry = stage._is_entry(wave)
        is_exit = stage._is_exit(wave)

        if is_fwd:
            # --- Forward ---
            if not is_entry:
                op, buf = COMM[action]['recv'](mb_idx, stage.device)
                execute_p2p_ops([op])
                recv_bufs[action][mb_idx] = buf

            with torch.profiler.record_function(f"{action}_{mb_idx}"):
                inputs = micro_inputs_A if wave == 'A' else micro_inputs_B
                if is_entry:
                    y = stage.forward(wave, mb_idx, micro_input=inputs[mb_idx])
                else:
                    y = stage.forward(wave, mb_idx, recv_buf=recv_bufs[action].pop(mb_idx))

            if not is_exit:
                op, sbuf = COMM[action]['send'](y, mb_idx)
                stage._send_bufs.append(sbuf)
                execute_p2p_ops([op])
        else:
            # --- Backward ---
            if not is_exit:
                op, rbuf = COMM[action]['recv'](mb_idx, stage.device)
                recv_bufs[action][mb_idx] = rbuf
                execute_p2p_ops([op])

            with torch.profiler.record_function(f"{action}_{mb_idx}"):
                labels = micro_labels_A if wave == 'A' else micro_labels_B
                if is_exit:
                    loss_val, input_grad = stage.backward(wave, mb_idx, labels[mb_idx])
                    if loss_val is not None:
                        print(f"Wave {wave} MB {mb_idx} Loss: {loss_val:.4f}", flush=True)
                else:
                    grad = recv_bufs[action].pop(mb_idx)
                    loss_val, input_grad = stage.backward(wave, mb_idx, grad)

            if not is_entry and input_grad is not None:
                op, sbuf = COMM[action]['send'](input_grad, mb_idx)
                stage._send_bufs.append(sbuf)
                execute_p2p_ops([op])

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

rank_to_stage = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1, 5: 2, 6: 3, 7: 4}
stage_id = rank_to_stage[global_rank]
m_idx = [model_idx1, model_idx2, model_idx3, model_idx4][stage_id - 1]

my_stage = HanayoStage(stage_id, gpt, m_idx, learning_rate, DEVICE, batch_size)
my_stage.to(DEVICE)
del gpt; gc.collect()

# ================= Data Loader =================
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

        # 拆成 2*num_microbatches 份，前半 Wave A，后半 Wave B
        all_micro_inputs = torch.chunk(inputs, num_microbatches * 2)
        all_micro_labels = torch.chunk(labels, num_microbatches * 2)
        micro_inputs_A = list(all_micro_inputs[:num_microbatches])
        micro_labels_A = list(all_micro_labels[:num_microbatches])
        micro_inputs_B = list(all_micro_inputs[num_microbatches:])
        micro_labels_B = list(all_micro_labels[num_microbatches:])

        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=True, profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_hanayo_512/rank{global_rank}')
        ) if i == 30 else contextlib.nullcontext()

        with profiler_context:
            run_hanayo_iteration(my_stage, micro_inputs_A, micro_labels_A,
                                micro_inputs_B, micro_labels_B)
            if my_stage.is_training:
                my_stage.all_reduce_gradients()
                my_stage.optimizer.step()
            dist.barrier()