"""
GPipe 流水线并行训练实现 - 多线程多流版本

本文件实现了一个基于 GPipe 的流水线并行训练框架，主要特点：
1. 使用多线程并行提交不同 micro-batch 的计算任务到不同的 CUDA 流
2. 使用专用的通信流来处理跨 GPU 的数据传输
3. 支持 2D 并行（数据并行 + 模型并行）
4. 实现了前向传播和反向传播的流水线调度
5. 支持 Orion 多优先级调度（micro-batch 0 为 HP，其他为 BE）

=== 运行方式 ===

在 GPU 节点上运行：

    cd /home/bingxing2/home/scx9kvs/orion-docker/ourgpipe

    module load compilers/cuda/12.1 compilers/gcc/11.3.0
    export LD_LIBRARY_PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.1/lib64:/home/bingxing2/apps/cudnn/8.9.4.25_cuda12.x/lib64:$LD_LIBRARY_PATH

    # 不启用 Orion 调度器（baseline）
    torchrun --nproc_per_node=4 --master_port=29500 gpipe_thread-stream.py

    # 启用 Orion 调度器（micro-batch 0 为 HP，其他为 BE）
    export USE_ORION_SCHEDULER=1
    LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
        torchrun --nproc_per_node=4 --master_port=29500 gpipe_thread-stream.py

Profiler trace 文件保存位置：
    - 启用 Orion: ./profiler_orion/rank{0,1,2,3}/
    - 不启用 Orion: ./profiler_baseline/rank{0,1,2,3}/

查看 trace 文件：在 Chrome 浏览器中打开 chrome://tracing -> Load
"""

# ==================== 导入必要的库 ====================

import torch.multiprocessing as mp
# 设置多进程启动方式为 'spawn'，这对于 CUDA 多进程是必需的
# 'spawn' 会创建全新的 Python 解释器进程，避免 fork 导致的 CUDA 上下文问题
mp.set_start_method('spawn', force=True)

import torch
import torch.cuda.nvtx as nvtx  # NVIDIA 工具扩展，用于性能分析标记
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re  # 正则表达式，用于解析模型配置
import os
import torch.distributed as dist  # PyTorch 分布式通信库
from torch.utils.data.distributed import DistributedSampler  # 分布式数据采样器
from torch.utils.data import DataLoader 
import datetime
import time
import random
import numpy as np
import gc  # 垃圾回收
import threading  # 多线程支持，用于并行提交计算任务

import sys
# 从 GPT.py 导入所有定义，包括 Block（Transformer块）、tok（分词器）、process（数据处理函数）、datasets 等
from GPT import *
import torch.profiler
from torch.profiler import record_function, ProfilerActivity, _ExperimentalConfig

import contextlib  # 上下文管理器工具

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

# ==================== 超参数配置 ====================

# 模型架构参数
emb_size = 4096       # 嵌入维度大小（词向量维度）
head_size = 128       # 注意力头的维度
n_layer = 16          # Transformer 层数
sequence_len = 128    # 序列长度（每个样本的 token 数量）

# 训练参数
learning_rate = 1e-4  # 学习率
eval_iters = 20       # 评估迭代次数
batch_size = 32       # 批次大小
epochs = 2            # 训练轮数
num_microbatches = 4  # 微批次数量（GPipe 的核心参数，将一个 batch 分成多个 micro-batch）

# FFN 低秩分解参数（用于减少 FFN 层的参数量）
ff_low_rank = emb_size // 2


def init_model(model_config):
    """
    根据配置字典初始化模型的各个组件
    
    参数:
        model_config: dict, 模型配置字典，包含各层的参数
        
    返回:
        model: nn.ModuleList, 包含所有模型组件的列表
        
    配置字典的键值对说明:
        - "em_tokn": [vocab_size, emb_size] - Token 嵌入层
        - "em_pos": [seq_len, emb_size] - 位置嵌入层
        - "ln": [emb_size] - LayerNorm 层
        - "lm_head": [emb_size, vocab_size] - 语言模型头（输出层）
        - "decoder{i}": [emb_size, head_size] - 第 i 个 Transformer 解码器块
    """
    model = nn.ModuleList()  # 使用 ModuleList 以便正确注册子模块
    
    for i, j in model_config.items():
        if i == "em_tokn":
            # Token 嵌入层：将 token ID 映射到嵌入向量
            mdl = nn.Embedding(j[0], j[1])
        elif i == "em_pos":
            # 位置嵌入层：为每个位置学习一个嵌入向量
            mdl = nn.Embedding(j[0], j[1])
        elif i == "ln":
            # 层归一化：稳定训练过程
            mdl = nn.LayerNorm(j[0])
        elif i == "lm_head":
            # 语言模型头：将隐藏状态映射到词表大小的 logits
            mdl = nn.Linear(j[0], j[1])
        elif (re.search("decoder", i)): 
            # Transformer 解码器块：包含自注意力和前馈网络
            mdl = Block(j[0], j[1], ff_low_rank=ff_low_rank) 
        model.append(mdl)
    return model


# ==================== 分布式训练初始化 ====================

# 从环境变量获取分布式训练配置
env_dict = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
}

# 初始化进程组，使用 NCCL 后端（针对 GPU 优化的通信库）
# timeout 设置为 300 秒，防止初始化超时
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=300))

# 获取当前进程的全局排名
global_rank = int(os.environ["RANK"])
print(f"global_rank is {global_rank}")

# 只在 rank 0 打印初始化信息，避免重复输出
if global_rank == 0:
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")


def get_device():
    """
    获取当前进程应该使用的设备
    
    返回:
        str: 设备字符串，如 'cuda:0' 或 'cpu'
        
    说明:
        LOCAL_RANK 环境变量指定了当前进程在本机上的排名，
        用于确定使用哪个 GPU
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        return f'cuda:{local_rank}'
    return 'cpu'


DEVICE = get_device()
print("The current rank is {}".format(DEVICE))

# ==================== 2D 并行组设置 ====================
# 2D 并行 = 数据并行 (DP) + 模型并行 (MP/流水线并行)

world_size = dist.get_world_size()  # 总进程数
model_parallel_size = 4  # 模型并行大小（流水线的阶段数）
data_parallel_size = world_size // model_parallel_size  # 数据并行大小

# 创建模型并行组（同一流水线内的进程）
# 例如：如果 world_size=8, model_parallel_size=4
# 则有两个模型并行组：[0,1,2,3] 和 [4,5,6,7]
model_parallel_groups = [
    dist.new_group(list(range(i*model_parallel_size, (i+1)*model_parallel_size))) 
    for i in range(data_parallel_size)
]

# 创建数据并行组（处理相同流水线阶段的进程）
# 例如：[0,4], [1,5], [2,6], [3,7]
# 这些进程处理相同的模型层，但处理不同的数据
data_parallel_groups = [
    dist.new_group(list(range(i, world_size, model_parallel_size)))
    for i in range(model_parallel_size)
]

# 获取当前进程所属的组
model_parallel_group = model_parallel_groups[global_rank // model_parallel_size]
data_parallel_group = data_parallel_groups[global_rank % model_parallel_size]


# ==================== Stage 类定义 ====================

class Stage:
    """
    流水线阶段类，封装了一个流水线阶段的所有功能
    
    每个 Stage 包含：
    - 模型的一部分（若干 Transformer 层）
    - 优化器
    - 通信缓冲区
    - CUDA 流（用于计算和通信的重叠）
    
    主要方法：
    - forward: 前向传播
    - backward: 反向传播
    - forward_isend/irecv: 前向传播的异步发送/接收
    - backward_isend/irecv: 反向传播的异步发送/接收
    """
    
    def __init__(self, ID, model, model_idx, learning_rate, device, batch_size):
        """
        初始化流水线阶段
        
        参数:
            ID: int, 阶段 ID（1-4）
            model: nn.ModuleList, 完整模型
            model_idx: list, 该阶段包含的模型层索引
            learning_rate: float, 学习率
            device: str, 设备字符串
            batch_size: int, 批次大小
        """
        self.stage_ID = ID
        self.device = torch.device(device) 
        self.model_idx = model_idx
        self.is_training = True  # 训练模式标志
        self.micro_batch_size = batch_size // num_microbatches  # 每个微批次的大小

        # 保存并行组引用
        self.model_parallel_group = model_parallel_group
        self.data_parallel_group = data_parallel_group
        self.local_rank = global_rank % model_parallel_size  # 在模型并行组内的排名

        # 根据阶段 ID 初始化模型
        if self.stage_ID == 1:
            # 第一阶段包含嵌入层
            self.token_embedding = nn.Embedding(vs, emb_size)
            self.position_embedding = nn.Embedding(sequence_len, emb_size)
            # sub_model 只包含 Transformer 块（跳过嵌入层索引 0,1）
            self.sub_model = nn.Sequential(*[model[i] for i in range(2, len(self.model_idx))])
        else:
            # 其他阶段只包含 Transformer 块
            self.sub_model = nn.Sequential(*[model[i] for i in model_idx])

        # 初始化优化器
        if self.stage_ID == 1:
            # 第一阶段需要优化嵌入层和 Transformer 块
            self.sub_model_opt = nn.Sequential(*[model[i] for i in model_idx])
            self.optimizer = optim.Adam(self.sub_model_opt.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.sub_model.parameters(), lr=learning_rate)
        
        # ==================== 通信缓冲区 ====================
        # 预分配缓冲区以避免动态内存分配
        
        # 前向传播输出缓冲区：用于接收上一阶段的激活值
        self.out_x_buffers = [
            torch.zeros(self.micro_batch_size, sequence_len, emb_size, device=self.device)
            for _ in range(num_microbatches)
        ]
        
        # 反向传播梯度缓冲区：用于接收下一阶段的梯度
        self.grad_y_buffers = [
            torch.zeros(self.micro_batch_size, sequence_len, emb_size, device=self.device)
            for _ in range(num_microbatches)
        ]
        
        self.lossi = []  # 损失记录列表
        
        # ==================== 缓存初始化 ====================
        # 【修改 1/3】fwd_cache 初始化为固定长度，以保证线程安全
        # 【修复 A】: 我们需要两个缓存
        self.fwd_cache = [None] * num_microbatches    # 缓存 Stage 的 *输出* (Y)
        self.input_cache = [None] * num_microbatches  # 缓存 Stage 的 *输入* (X)

        # ==================== CUDA 流初始化 ====================
        # 【不变】创建多个计算流（每个 micro-batch 一个）
        # 这允许不同 micro-batch 的计算并行执行
        self.num_comp_streams = num_microbatches
        if self.device.type == 'cuda':
            # 计算流：每个 micro-batch 使用独立的流
            self.comp_streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_comp_streams)]
            # 通信流：所有通信操作共享一个流，避免并发访问问题
            self.comm_stream = torch.cuda.Stream(device=self.device)
        else:
            self.comp_streams = None
            self.comm_stream = None

        # ==================== Orion 调度器初始化 ====================
        self.orion_scheduler = None
        if USE_ORION_SCHEDULER and ORION_AVAILABLE:
            try:
                # 【修复】在启动调度器之前，先初始化 CUDA 上下文
                # 这样可以确保调度器线程和 PyTorch 主线程使用相同的 CUDA 上下文
                if self.device.type == 'cuda':
                    # 【关键修复】先设置当前设备为本 rank 对应的 GPU
                    # 这确保 CUDA 上下文在正确的设备上初始化
                    # cuBLAS handle 会绑定到当前设备，必须在正确的 device 上创建
                    torch.cuda.set_device(self.device)
                    # 显式初始化 CUDA 上下文（现在会在正确的设备上）
                    torch.cuda.init()
                    # 执行一个简单的 CUDA 操作来确保上下文已创建
                    torch.cuda.synchronize()
                    print(f"[Orion] Stage {ID}: CUDA context initialized on device={self.device} before scheduler start")
                
                # 获取 Orion 调度器单例
                self.orion_scheduler = get_scheduler()
                # 设置为多优先级调度模式
                self.orion_scheduler.set_scheduling_mode(SchedulingMode.MULTI_PRIORITY)
                # 启动调度器（如果尚未启动）
                if not self.orion_scheduler.is_running():
                    # 获取当前 GPU ID
                    device_id = self.device.index if self.device.type == 'cuda' else 0
                    self.orion_scheduler.start(num_clients=num_microbatches, device_id=device_id)
                print(f"[Orion] Stage {ID}: Orion scheduler initialized with {num_microbatches} clients (multi-priority mode, device={device_id})")
            except Exception as e:
                print(f"[Orion] Stage {ID}: Failed to initialize Orion scheduler: {e}")
                self.orion_scheduler = None

    def to(self, device):
        """
        将模型和缓冲区移动到指定设备
        
        参数:
            device: str 或 torch.device, 目标设备
        """
        device = torch.device(device) 
        if self.stage_ID == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        self.sub_model.to(device)
        
        # 移动缓冲区
        self.out_x_buffers = [b.to(device) for b in self.out_x_buffers]
        self.grad_y_buffers = [b.to(device) for b in self.grad_y_buffers]
    
    def eval(self):
        """设置为评估模式"""
        self.sub_model.eval()
        
    def train(self):
        """设置为训练模式"""
        self.sub_model.train()
        
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
        
    def forward(self, x, mb_idx):
        """
        前向传播
        
        参数:
            x: Tensor, 输入张量
            mb_idx: int, 微批次索引（用于线程安全地写入缓存）
            
        返回:
            y: Tensor, 输出张量
            
        【修改 2/3】forward 接收 mb_idx 以便线程安全地写入 cache
        """
        if self.stage_ID == 1:
            # 第一阶段：处理原始 token ID
            B, T = x.shape  # B: batch size, T: sequence length
            
            # 生成位置索引 [0, 1, 2, ..., T-1]
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            
            # 计算 token 嵌入和位置嵌入
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            
            # 将两种嵌入相加得到输入表示
            x = tok_emb + pos_emb
            
            # 【修复 B】: 保存 Stage 1 的输入
            # （它不需要 requires_grad_() 因为它是计算图的根）
            self.input_cache[mb_idx] = x 
            
            # 通过 Transformer 块
            y = self.sub_model(x)
        else:
            # 其他阶段：接收上一阶段的激活值
            # 【修复 C】: 
            # 1. 启用输入的 grad，以便 autograd 跟踪它
            # 2. 保存输入，以便在 backward 时检索 .grad
            x.requires_grad_()
            self.input_cache[mb_idx] = x
            y = self.sub_model(x)
        
        # 保留输出的梯度（用于反向传播时获取 dL/dY）
        y.retain_grad()
        # 保存输出到缓存
        self.fwd_cache[mb_idx] = y
        return y
        
    def forward_isend(self, out, mb_idx, iter_num):
        """
        异步发送前向传播的输出到下一阶段
        
        参数:
            out: Tensor, 要发送的输出张量
            mb_idx: int, 微批次索引
            iter_num: int, 当前迭代编号（用于生成唯一的通信标签）
            
        返回:
            handle: 异步通信句柄，可用于等待发送完成
            
        通信标签计算说明:
            tag = iter_num * (总阶段数 * 2 * 微批次数) +
                  (当前阶段-1) * (2 * 微批次数) +
                  方向 * 微批次数 +
                  微批次索引
            
            其中方向: 0 表示前向，1 表示反向
            这种标签设计确保每个通信操作都有唯一的标识符
        """
        dst = global_rank + 1  # 目标是下一个 rank
        direction = 0  # 前向传播方向
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (self.stage_ID - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              mb_idx
        return dist.isend(tensor=out, dst=dst, tag=tag, group=self.model_parallel_group)
    
    def forward_irecv(self, mb_idx, iter_num):
        """
        异步接收上一阶段的前向传播输出
        
        参数:
            mb_idx: int, 微批次索引
            iter_num: int, 当前迭代编号
            
        返回:
            handle: 异步通信句柄
            
        说明:
            接收的数据会存储在预分配的 out_x_buffers[mb_idx] 中
        """
        src = global_rank - 1  # 源是上一个 rank
        sender_stage_id = self.stage_ID - 1  # 发送方的阶段 ID
        direction = 0  # 前向传播方向
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (sender_stage_id - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              mb_idx
        tensor_to_recv = self.out_x_buffers[mb_idx]
        return dist.irecv(tensor=tensor_to_recv, src=src, tag=tag, group=self.model_parallel_group)
    
    def backward(self, microbatch_idx, grad_tensor):
        """
        执行反向传播
        
        参数:
            microbatch_idx: int, 微批次索引
            grad_tensor: Tensor, 从下一阶段接收的梯度 (dL/dY)
            
        说明:
            【不变】backward 和 backward_isend 已经是按索引读，是线程安全的
            从 fwd_cache 中获取缓存的输出，然后调用 backward 计算梯度
        """
        cached_output = self.fwd_cache[microbatch_idx]
        cached_output.backward(grad_tensor)
    
    def backward_isend(self, microbatch_idx, iter_num):
        """
        异步发送反向传播的梯度到上一阶段
        
        参数:
            microbatch_idx: int, 微批次索引
            iter_num: int, 当前迭代编号
            
        返回:
            handle: 异步通信句柄
            
        【修复 D】: 发送 *输入* 的梯度 (dL/dX)
        这是流水线并行中反向传播的关键：
        - 我们需要发送的是 dL/dX（对输入的梯度）
        - 而不是 dL/dY（对输出的梯度）
        """
        cached_input = self.input_cache[microbatch_idx]
        grad = cached_input.grad  # 这才是我们需要的 dL/dX
        dst = global_rank - 1  # 目标是上一个 rank
        direction = 1  # 反向传播方向
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (self.stage_ID - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              microbatch_idx
        return dist.isend(grad, dst=dst, tag=tag, group=self.model_parallel_group)
    
    def backward_irecv(self, microbatch_idx, iter_num):
        """
        异步接收下一阶段的反向传播梯度
        
        参数:
            microbatch_idx: int, 微批次索引
            iter_num: int, 当前迭代编号
            
        返回:
            handle: 异步通信句柄
            
        说明:
            接收的梯度会存储在预分配的 grad_y_buffers[microbatch_idx] 中
        """
        src = global_rank + 1  # 源是下一个 rank
        sender_stage_id = self.stage_ID + 1  # 发送方的阶段 ID
        direction = 1  # 反向传播方向
        tag = iter_num * (model_parallel_size * 2 * num_microbatches) + \
              (sender_stage_id - 1) * (2 * num_microbatches) + \
              direction * num_microbatches + \
              microbatch_idx
        tensor_to_recv = self.grad_y_buffers[microbatch_idx]
        return dist.irecv(tensor_to_recv, src=src, tag=tag, group=self.model_parallel_group)
    
    def all_reduce_gradients(self):
        """
        在数据并行组内对梯度进行 AllReduce 操作
        
        说明:
            当使用数据并行时，不同的数据并行副本处理不同的数据，
            但它们的模型参数应该保持同步。
            AllReduce 操作会：
            1. 对所有副本的梯度求和
            2. 将结果除以副本数（取平均）
            3. 将平均梯度广播回所有副本
        """
        if self.stage_ID == 1:
            params_to_reduce = self.sub_model_opt.parameters()
        else:
            params_to_reduce = self.sub_model.parameters()
        
        for param in params_to_reduce:
            if param.grad is not None:
                # 对梯度进行求和
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                # 取平均
                param.grad /= data_parallel_size


# ==================== 数据加载和预处理 ====================

# 使用 barrier 确保只有 rank 0 先处理数据，避免重复处理和缓存竞争
if global_rank == 0:
    # rank 0 先初始化数据集和分词器
    datasets, tok = init_datasets()
    # 将数据集分割为训练集和测试集
    tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
    # 对数据集应用 tokenization 处理
    tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)

# rank 0 完成后，其他 rank 等待
dist.barrier()

if global_rank != 0:
    # 其他 rank 在 rank 0 完成后初始化
    datasets, tok = init_datasets()
    tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
    tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)

# 所有 rank 同步
dist.barrier()

# 设置数据格式为 PyTorch 张量，并移动到指定设备
tokenized.set_format(type='torch', device=DEVICE)

if global_rank == 0:
    # 获取数据集大小信息
    train_size = len(tokenized['train'])
    print(f"Train dataset size: {train_size} samples")

# 创建分布式采样器（用于数据并行）
# 每个数据并行组内的进程会处理不同的数据子集
train_sampler = DistributedSampler(
    dataset=tokenized['train'],
    num_replicas=data_parallel_size,  # 数据并行的副本数
    rank=global_rank // model_parallel_size,  # 当前进程在数据并行组内的排名
    shuffle=True
) if data_parallel_size > 1 else None

# 创建训练数据加载器
train_loader = DataLoader(
    dataset=tokenized['train'],
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0  # 不使用多进程加载，避免与分布式训练冲突
)

# 创建测试集采样器和加载器
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

# 获取词表大小
vs = len(tok.char2ind)

# ==================== 模型配置 ====================

# 定义完整模型的配置
# 包含：token嵌入、位置嵌入、n_layer个Transformer块、LayerNorm、语言模型头
model_config = {
    "em_tokn": [vs, emb_size],           # Token 嵌入层
    "em_pos": [sequence_len, emb_size],  # 位置嵌入层
    # 动态生成 n_layer 个 decoder 块的配置
    **{f"decoder{i}": [emb_size, head_size] for i in range(1, n_layer+1)},
    "ln": [emb_size],                    # 最终的 LayerNorm
    "lm_head": [emb_size, vs]            # 语言模型输出头
}

# 初始化完整模型
gpt = init_model(model_config=model_config)

# ==================== 流水线阶段划分 ====================

# 定义每个阶段包含的模型层索引
# 这里将 16 层 Transformer 分成 4 个阶段：
# Stage 1: 嵌入层 + 4个Transformer块 (索引 0-5)
# Stage 2: 4个Transformer块 (索引 6-9)
# Stage 3: 4个Transformer块 (索引 10-13)
# Stage 4: 4个Transformer块 + LayerNorm + LM Head (索引 14-19)
model_idx1 = [0, 1, 2, 3, 4, 5]
model_idx2 = [6, 7, 8, 9]
model_idx3 = [10, 11, 12, 13]
model_idx4 = [14, 15, 16, 17, 18, 19]

# 定义 rank 到 stage 的映射
# 支持 8 个 GPU 的 2D 并行配置：
# - Rank 0-3: 第一个数据并行副本的 4 个流水线阶段
# - Rank 4-7: 第二个数据并行副本的 4 个流水线阶段
rank_to_stage = {
    0: 1, 1: 2, 2: 3, 3: 4,
    4: 1, 5: 2, 6: 3, 7: 4,
}

# 获取当前进程对应的阶段
current_stage = rank_to_stage[global_rank]
print("current rank is {}, current stage is {}".format(global_rank, current_stage))

# 根据当前阶段创建对应的 Stage 对象
if current_stage == 1:
    my_stage = Stage(1, gpt, model_idx1, learning_rate, DEVICE, batch_size)
elif current_stage == 2:
    my_stage = Stage(2, gpt, model_idx2, learning_rate, DEVICE, batch_size)
elif current_stage == 3:
    my_stage = Stage(3, gpt, model_idx3, learning_rate, DEVICE, batch_size)
elif current_stage == 4:
    my_stage = Stage(4, gpt, model_idx4, learning_rate, DEVICE, batch_size)

# 将模型移动到对应设备
my_stage.to(DEVICE)

# 用于记录损失的列表
loss_list = []

# 同步所有进程，确保初始化完成
dist.barrier()


# ==================== 训练循环 ====================

# 检查点保存间隔
ck_interval = 1000
loss = 0

# 开始训练循环
for epoch in range(epochs):
    # 设置分布式采样器的 epoch，确保每个 epoch 的数据顺序不同
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    iter_start = time.time()
    
    # 遍历训练数据
    for i, data in tqdm(enumerate(train_loader, 0)):
        # 在第 1000 次迭代时开始 NVTX 标记（用于性能分析）
        if i == 1000:
            nvtx.range_push(f"Epoch {epoch}")
        
        # ==================== Profiler 设置 ====================
        # 在第 20 次迭代时启用 PyTorch Profiler 进行性能分析
        # 其他迭代使用空上下文管理器
        # 根据是否启用 Orion 调度器来区分输出目录
        profiler_dir = f'./profiler_orion/rank{global_rank}' if USE_ORION_SCHEDULER else f'./profiler_baseline/rank{global_rank}'
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,      # 记录张量形状
            with_stack=True,         # 记录调用栈
            profile_memory=True,     # 记录内存使用
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir)
        ) if i == 20 else contextlib.nullcontext()

        train_start = time.time()
        
        # ==================== 数据准备 ====================
        # 将数据移动到设备
        inputs, labels = data['inputs'].to(DEVICE), data['labels'].to(DEVICE)
        
        # 将 batch 分割成多个 micro-batch
        # 这是 GPipe 的核心：通过 micro-batch 实现流水线并行
        micro_inputs = torch.chunk(inputs, num_microbatches)
        micro_labels = torch.chunk(labels, num_microbatches)
        
        # ==================== 通信句柄字典 ====================
        # 用于存储异步通信操作的句柄
        fwd_recv_handles = {}  # 前向传播接收句柄
        fwd_send_handles = {}  # 前向传播发送句柄
        bwd_recv_handles = {}  # 反向传播接收句柄
        bwd_send_handles = {}  # 反向传播发送句柄
        
        # 【修改 3/3】重置 cache 为固定长度列表
        my_stage.fwd_cache = [None] * num_microbatches
        
        # 打印迭代信息（仅 rank 0）
        if global_rank == 0:
            print("\n################## iteration {} ##################".format(i))
        
        # 同步所有进程
        dist.barrier()
        
        # 进入 profiler 上下文
        with profiler_context:
                 
            # ==================== 前向传播阶段 ====================
            # 使用无死锁的通信模型

            # --- 步骤 1：所有接收方 (Rank > 0) 立即提交所有 irecv ---
            # 这确保了接收操作在发送之前就已经准备好，避免死锁
            if current_stage > 1:
                for mb_idx in range(num_microbatches):
                    with record_function(f"Fwd Post Irecv mb_{mb_idx}"):
                        handle = my_stage.forward_irecv(mb_idx, i)
                        fwd_recv_handles[mb_idx] = handle

            # --- 步骤 2：使用多线程并行提交 Compute 到不同 streams ---
            # 定义前向计算函数，每个 micro-batch 在独立的线程中执行
            def fwd_compute_mb(mb_idx):
                """
                单个 micro-batch 的前向计算
                在独立的 CUDA 流中执行，实现计算并行

                【Orion 集成】：
                - 设置 client_idx = mb_idx，使得 micro-batch 0 具有最高优先级
                - Orion 调度器会确保低优先级的 kernel 等待高优先级队列清空

                【修复】在函数开头显式绑定 CUDA context 并初始化 CUBLAS handle
                """
                # 【修复】在函数开头显式绑定 CUDA context 并初始化 CUBLAS
                if my_stage.device.type == 'cuda':
                    torch.cuda.set_device(my_stage.device)
                    # 【关键修复】触发一个 CUBLAS 操作来初始化线程本地的 CUBLAS handle
                    # 这会在线程的 CUDA 上下文中创建必要的 CUBLAS 状态
                    _ = torch.nn.functional.linear(
                        torch.zeros(1, 10, device=my_stage.device),
                        torch.zeros(10, 10, device=my_stage.device)
                    )
                    torch.cuda.synchronize()
                
                # 【Orion】设置当前线程的 client index（用于多优先级调度）
                if my_stage.orion_scheduler is not None:
                    my_stage.orion_scheduler.set_client_idx(mb_idx)
                
                current_comp_stream = my_stage.comp_streams[mb_idx]
                with torch.cuda.stream(current_comp_stream):
                    with record_function(f"Fwd Compute mb_{mb_idx}"):
                        if current_stage > 1:
                            # 非第一阶段：等待接收完成，然后使用接收到的数据
                            fwd_recv_handles[mb_idx].wait()
                            input_tensor = my_stage.out_x_buffers[mb_idx].to(DEVICE)
                        else:
                            # 第一阶段：直接使用输入数据
                            input_tensor = micro_inputs[mb_idx]
                        
                        # 调用修改后的 forward，传入 mb_idx 以便线程安全地写入缓存
                        output_tensor = my_stage.forward(input_tensor, mb_idx)

            # 创建并启动所有前向计算线程
            fwd_threads = []
            for mb_idx in range(num_microbatches):
                t = threading.Thread(target=fwd_compute_mb, args=(mb_idx,))
                t.start()
                fwd_threads.append(t)
            
            # 等待所有 forward compute 线程完成（确保所有 kernel 已 enqueue）
            for t in fwd_threads:
                t.join()

            # --- 步骤 3：串行提交 Send（使用共享 comm_stream，避免并发访问）---
            for mb_idx in range(num_microbatches):
                if current_stage < model_parallel_size:
                    current_comp_stream = my_stage.comp_streams[mb_idx]
                    with record_function(f"Fwd Send mb_{mb_idx}"):
                        with torch.cuda.stream(my_stage.comm_stream):
                            # 等待计算流完成
                            my_stage.comm_stream.wait_stream(current_comp_stream)
                            # 发送前向传播的输出
                            handle = my_stage.forward_isend(my_stage.fwd_cache[mb_idx], mb_idx, i)
                            fwd_send_handles[mb_idx] = handle

            # ==================== 反向传播阶段 ====================
            # 使用无死锁的通信模型

            # --- 步骤 1：所有非最后阶段的 Rank 提交 Bwd irecv ---
            if current_stage < model_parallel_size:
                for mb_idx in reversed(range(num_microbatches)):
                    with record_function(f"Bwd Post Irecv mb_{mb_idx}"):
                        handle = my_stage.backward_irecv(mb_idx, i)
                        bwd_recv_handles[mb_idx] = handle

            # --- 步骤 2：使用多线程并行提交 Bwd Compute 到不同 streams ---
            def bwd_compute_mb(mb_idx):
                """
                单个 micro-batch 的反向计算
                在独立的 CUDA 流中执行，实现计算并行

                【Orion 集成】：
                - 设置 client_idx = mb_idx，保持与前向传播相同的优先级
                - 反向传播时，micro-batch 0 仍然具有最高优先级

                【修复】在函数开头显式绑定 CUDA context 并初始化 CUBLAS handle
                """
                # 【修复】在函数开头显式绑定 CUDA context 并初始化 CUBLAS
                if my_stage.device.type == 'cuda':
                    torch.cuda.set_device(my_stage.device)
                    # 【关键修复】触发一个 CUBLAS 操作来初始化线程本地的 CUBLAS handle
                    _ = torch.nn.functional.linear(
                        torch.zeros(1, 10, device=my_stage.device),
                        torch.zeros(10, 10, device=my_stage.device)
                    )
                    torch.cuda.synchronize()
                
                # 【Orion】设置当前线程的 client index（用于多优先级调度）
                if my_stage.orion_scheduler is not None:
                    my_stage.orion_scheduler.set_client_idx(mb_idx)
                
                current_comp_stream = my_stage.comp_streams[mb_idx]
                with torch.cuda.stream(current_comp_stream):
                    with record_function(f"Bwd Compute mb_{mb_idx}"):
                        grad_tensor = None
                        if current_stage < model_parallel_size:
                            # 非最后阶段：等待接收梯度
                            bwd_recv_handles[mb_idx].wait()
                            grad_tensor = my_stage.grad_y_buffers[mb_idx].to(DEVICE)
                        
                        if current_stage == model_parallel_size:
                            # 最后阶段：计算损失并反向传播
                            logits = my_stage.fwd_cache[mb_idx].transpose(-2, -1)
                            loss = F.cross_entropy(logits, micro_labels[mb_idx])
                            loss.backward()
                        else:
                            # 其他阶段：使用接收到的梯度进行反向传播
                            my_stage.backward(mb_idx, grad_tensor)

            # 创建并启动所有反向计算线程（逆序处理 micro-batch）
            bwd_threads = []
            for mb_idx in reversed(range(num_microbatches)):
                t = threading.Thread(target=bwd_compute_mb, args=(mb_idx,))
                t.start()
                bwd_threads.append(t)
            
            # 等待所有 backward compute 线程完成（确保所有 kernel 已 enqueue）
            for t in bwd_threads:
                t.join()

            # --- 步骤 3：串行提交 Bwd Send（使用共享 comm_stream）---
            for mb_idx in reversed(range(num_microbatches)):
                if current_stage > 1:
                    current_comp_stream = my_stage.comp_streams[mb_idx]
                    with record_function(f"Bwd Send mb_{mb_idx}"):
                        with torch.cuda.stream(my_stage.comm_stream):
                            # 等待计算流完成
                            my_stage.comm_stream.wait_stream(current_comp_stream)
                            # 发送反向传播的梯度
                            handle = my_stage.backward_isend(mb_idx, i)
                            bwd_send_handles[mb_idx] = handle

            # ==================== 同步和参数更新 ====================
            
            # 1. 同步所有 CUDA 流
            # 确保所有计算流的操作都已完成
            if my_stage.comp_streams:
                for s in my_stage.comp_streams:
                    torch.cuda.current_stream().wait_stream(s)
            # 确保通信流的操作都已完成
            if my_stage.comm_stream:
                torch.cuda.current_stream().wait_stream(my_stage.comm_stream)

            # 2. 确保所有 P2P 通信已完成（CPU 句柄等待）
            for handle in fwd_send_handles.values():
                handle.wait()
            for handle in bwd_send_handles.values():
                handle.wait()
            
            # ==================== 参数更新 ====================
            if my_stage.is_training:
                # 如果使用数据并行，先进行梯度 AllReduce
                if data_parallel_size > 1:
                    my_stage.all_reduce_gradients()
                # 执行优化器步骤
                my_stage.optimizer.step()
                # 清零梯度，为下一次迭代做准备
                my_stage.optimizer.zero_grad()

            train_end = time.time()
            
            # 每 50 次迭代打印一次时间信息
            if i % 50 == 0 and i != 0:
                iter_end = time.time()
            
            # 同步所有进程
            dist.barrier()
        
        # 在第 25 次迭代后退出（用于性能分析）
        if i == 25:
            nvtx.range_pop()
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
            
            dist.barrier()  # 确保所有进程同步
            dist.destroy_process_group()  # 清理进程组，避免 NCCL 超时
            sys.exit(0)