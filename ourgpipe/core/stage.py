"""
GPipe 流水线框架抽象 Stage 基类

提供流水线阶段的基础实现，包括：
- 通信缓冲区管理
- CUDA 流管理
- 分布式通信
- Orion 调度器集成
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import threading
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .config import PipelineConfig
from .interfaces import StageInterface


class BaseStage(StageInterface, ABC):
    """流水线阶段基类
    
    封装了所有阶段共有的功能，子类只需要实现模型特定的逻辑。
    
    主要功能:
    - 通信缓冲区管理: 预分配用于 P2P 通信的张量
    - CUDA 流管理: 每个 micro-batch 使用独立的计算流
    - 分布式通信: 异步发送/接收激活值和梯度
    - Orion 调度器集成: 支持多优先级 GPU 调度
    
    子类需要实现:
    - create_sub_model(): 创建该阶段的子模型
    - prepare_input(): 输入预处理（如嵌入层处理）
    - compute_loss(): 损失计算（最后阶段使用）
    
    示例:
        class GPTStage(BaseStage):
            def create_sub_model(self, model_layers, layer_indices):
                if self.stage_id == 1:
                    self.token_embedding = model_layers[0]
                    return nn.Sequential(*[model_layers[i] for i in layer_indices[2:]])
                return nn.Sequential(*[model_layers[i] for i in layer_indices])
    """
    
    def __init__(
        self,
        stage_id: int,
        model_adapter,
        model_config: Dict[str, Any],
        layer_indices: List[int],
        config: PipelineConfig,
        device: torch.device,
        model_parallel_group,
        data_parallel_group,
        global_rank: int,
        model_parallel_size: int,
        data_parallel_size: int
    ):
        """初始化流水线阶段（延迟初始化）
        
        Args:
            stage_id: 阶段 ID（从 1 开始）
            model_adapter: 模型适配器实例，用于创建层
            model_config: 模型配置字典（由 init_model() 返回）
            layer_indices: 该阶段包含的层索引
            config: 流水线配置
            device: 计算设备
            model_parallel_group: 模型并行通信组
            data_parallel_group: 数据并行通信组
            global_rank: 全局进程排名
            model_parallel_size: 模型并行大小（流水线阶段数）
            data_parallel_size: 数据并行大小
        """
        self.stage_id = stage_id
        self.device = device
        self.layer_indices = layer_indices
        self.config = config
        self.is_training = True
        
        # 保存模型适配器和配置（用于延迟初始化）
        self.model_adapter = model_adapter
        self.model_config = model_config
        
        # 分布式相关
        self.model_parallel_group = model_parallel_group
        self.data_parallel_group = data_parallel_group
        self.global_rank = global_rank
        self.model_parallel_size = model_parallel_size
        self.data_parallel_size = data_parallel_size
        self.local_rank = global_rank % model_parallel_size
        
        # 计算 micro-batch 大小
        self.num_microbatches = config.training.num_microbatches
        self.micro_batch_size = config.dataset.batch_size // self.num_microbatches
        
        # 创建子模型（由子类实现，按需创建层）
        self.sub_model = self.create_sub_model(model_adapter, model_config, layer_indices)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.sub_model.parameters(),
            lr=config.training.learning_rate
        )
        
        # 初始化缓冲区和流
        self._init_buffers()
        self._init_streams()
        
        # 缓存
        self.fwd_cache = [None] * self.num_microbatches
        self.input_cache = [None] * self.num_microbatches
        self.grad_buffers = [None] * self.num_microbatches
        
        # 线程锁
        self.comm_stream_lock = threading.Lock()
        
        # 损失记录
        self.lossi = []
        
        # Orion 调度器
        self.orion_scheduler = None
        if config.parallel.use_orion_scheduler:
            self._init_orion_scheduler()
    
    @abstractmethod
    def create_sub_model(
        self,
        model_adapter,
        model_config: Dict[str, Any],
        layer_indices: List[int]
    ) -> nn.Module:
        """创建该阶段的子模型（延迟初始化）
        
        子类需要实现此方法来处理特定模型的层组织方式。
        例如，GPT 的第一阶段需要特殊处理嵌入层。
        
        注意：此方法接收模型配置而非层实例，需要按需创建层。
        这样可以避免每个进程都创建完整模型，节省内存。
        
        Args:
            model_adapter: 模型适配器实例，用于创建层
            model_config: 模型配置字典（由 init_model() 返回）
            layer_indices: 该阶段包含的层索引
            
        Returns:
            该阶段的子模型
            
        示例:
            # 按需创建层
            layer_names = list(model_config.keys())
            layers = []
            for idx in layer_indices:
                layer_name = layer_names[idx]
                layer = model_adapter.create_layer(layer_name, model_config[layer_name])
                layers.append(layer)
            return nn.Sequential(*layers)
        """
        pass
    
    @abstractmethod
    def prepare_input(self, x: torch.Tensor, mb_idx: int) -> torch.Tensor:
        """准备输入
        
        第一阶段可能需要进行嵌入处理，其他阶段通常直接返回输入。
        
        Args:
            x: 输入张量
            mb_idx: micro-batch 索引
            
        Returns:
            处理后的输入张量
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算损失（最后阶段使用）
        
        Args:
            output: 模型输出
            labels: 标签
            
        Returns:
            损失值
        """
        pass
    
    def get_output_for_send(self, mb_idx: int) -> torch.Tensor:
        """获取要发送给下一阶段的输出
        
        Args:
            mb_idx: micro-batch 索引
            
        Returns:
            要发送的张量
        """
        return self.fwd_cache[mb_idx]
    
    def _init_buffers(self):
        """初始化通信缓冲区"""
        hidden_size = self.config.model.hidden_size
        seq_len = self.config.model.sequence_length
        
        # 前向传播输出缓冲区：用于接收上一阶段的激活值
        self.out_x_buffers = [
            torch.zeros(
                self.micro_batch_size, seq_len, hidden_size,
                device=self.device
            )
            for _ in range(self.num_microbatches)
        ]
        
        # 反向传播梯度缓冲区：用于接收下一阶段的梯度
        self.grad_y_buffers = [
            torch.zeros(
                self.micro_batch_size, seq_len, hidden_size,
                device=self.device
            )
            for _ in range(self.num_microbatches)
        ]
    
    def _init_streams(self):
        """初始化 CUDA 流"""
        if self.device.type == 'cuda':
            # 计算流：每个 micro-batch 使用独立的流
            self.comp_streams = [
                torch.cuda.Stream(device=self.device)
                for _ in range(self.num_microbatches)
            ]
            # 通信流：所有通信操作共享一个流
            self.comm_stream = torch.cuda.Stream(device=self.device)
        else:
            self.comp_streams = None
            self.comm_stream = None
    
    def _init_orion_scheduler(self):
        """初始化 Orion 调度器"""
        try:
            # 添加 kernel_intercept/python 到路径
            kernel_intercept_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'kernel_intercept', 'python'
            )
            if kernel_intercept_path not in sys.path:
                sys.path.insert(0, kernel_intercept_path)
            
            from orion_binding import get_scheduler, SchedulingMode
            
            # 在启动调度器之前，先初始化 CUDA 上下文
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
                torch.cuda.init()
                torch.cuda.synchronize()
                print(f"[Orion] Stage {self.stage_id}: CUDA context initialized on device={self.device}")
            
            # 获取 Orion 调度器单例
            self.orion_scheduler = get_scheduler()
            # 设置为多优先级调度模式
            self.orion_scheduler.set_scheduling_mode(SchedulingMode.MULTI_PRIORITY)
            
            # 启动调度器（如果尚未启动）
            if not self.orion_scheduler.is_running():
                device_id = self.device.index if self.device.type == 'cuda' else 0
                self.orion_scheduler.start(
                    num_clients=self.num_microbatches,
                    device_id=device_id
                )
            print(f"[Orion] Stage {self.stage_id}: Orion scheduler initialized with {self.num_microbatches} clients")
            
        except Exception as e:
            print(f"[Orion] Stage {self.stage_id}: Failed to initialize Orion scheduler: {e}")
            self.orion_scheduler = None
    
    def to(self, device):
        """将模型和缓冲区移动到指定设备
        
        Args:
            device: 目标设备
        """
        device = torch.device(device)
        self.sub_model.to(device)
        self.out_x_buffers = [b.to(device) for b in self.out_x_buffers]
        self.grad_y_buffers = [b.to(device) for b in self.grad_y_buffers]
    
    def eval(self):
        """设置为评估模式"""
        self.sub_model.eval()
        self.is_training = False
    
    def train(self):
        """设置为训练模式"""
        self.sub_model.train()
        self.is_training = True
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def forward(self, x: torch.Tensor, mb_idx: int) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            mb_idx: micro-batch 索引
            
        Returns:
            输出张量
        """
        # 准备输入（子类实现）
        x = self.prepare_input(x, mb_idx)
        
        # 非第一阶段需要启用输入的梯度
        if self.stage_id > 1:
            x.requires_grad_()
        
        # 保存输入用于反向传播
        self.input_cache[mb_idx] = x
        
        # 前向计算
        y = self.sub_model(x)
        
        # 保留输出的梯度
        y.retain_grad()
        self.fwd_cache[mb_idx] = y
        
        return y
    
    def backward(self, mb_idx: int, grad_tensor: torch.Tensor):
        """反向传播
        
        Args:
            mb_idx: micro-batch 索引
            grad_tensor: 从下一阶段接收的梯度
        """
        cached_output = self.fwd_cache[mb_idx]
        cached_output.backward(grad_tensor)
    
    def compute_grad_only(self, mb_idx: int, grad_tensor: torch.Tensor):
        """只计算梯度，不累加到参数（用于流水线并行）
        
        Args:
            mb_idx: micro-batch 索引
            grad_tensor: 从下一阶段接收的梯度
        """
        cached_output = self.fwd_cache[mb_idx]
        cached_input = self.input_cache[mb_idx]
        
        params = list(self.sub_model.parameters())
        
        # 确定是否需要计算输入的梯度
        need_input_grad = (self.stage_id > 1) and (cached_input.requires_grad)
        inputs_to_diff = params + [cached_input] if need_input_grad else params
        
        # 计算梯度
        grads = torch.autograd.grad(
            outputs=cached_output,
            inputs=inputs_to_diff,
            grad_outputs=grad_tensor,
            retain_graph=False
        )
        
        # 存储权重梯度
        weight_grads = grads[:len(params)]
        self.grad_buffers[mb_idx] = weight_grads
        
        # 存储输入梯度（用于发送给上一阶段）
        if need_input_grad:
            cached_input.grad = grads[-1]
    
    # ==================== 通信方法 ====================
    
    def forward_isend(self, out: torch.Tensor, mb_idx: int, iter_num: int):
        """异步发送前向传播的输出到下一阶段
        
        Args:
            out: 要发送的输出张量
            mb_idx: micro-batch 索引
            iter_num: 当前迭代编号
            
        Returns:
            异步通信句柄
        """
        dst = self.global_rank + 1
        direction = 0  # 前向传播方向
        tag = self._compute_tag(iter_num, self.stage_id, direction, mb_idx)
        return dist.isend(tensor=out, dst=dst, tag=tag, group=self.model_parallel_group)
    
    def forward_irecv(self, mb_idx: int, iter_num: int):
        """异步接收上一阶段的前向传播输出
        
        Args:
            mb_idx: micro-batch 索引
            iter_num: 当前迭代编号
            
        Returns:
            异步通信句柄
        """
        src = self.global_rank - 1
        sender_stage_id = self.stage_id - 1
        direction = 0  # 前向传播方向
        tag = self._compute_tag(iter_num, sender_stage_id, direction, mb_idx)
        tensor_to_recv = self.out_x_buffers[mb_idx]
        return dist.irecv(tensor=tensor_to_recv, src=src, tag=tag, group=self.model_parallel_group)
    
    def backward_isend(self, mb_idx: int, iter_num: int):
        """异步发送反向传播的梯度到上一阶段
        
        Args:
            mb_idx: micro-batch 索引
            iter_num: 当前迭代编号
            
        Returns:
            异步通信句柄
        """
        cached_input = self.input_cache[mb_idx]
        grad = cached_input.grad  # dL/dX
        dst = self.global_rank - 1
        direction = 1  # 反向传播方向
        tag = self._compute_tag(iter_num, self.stage_id, direction, mb_idx)
        return dist.isend(grad, dst=dst, tag=tag, group=self.model_parallel_group)
    
    def backward_irecv(self, mb_idx: int, iter_num: int):
        """异步接收下一阶段的反向传播梯度
        
        Args:
            mb_idx: micro-batch 索引
            iter_num: 当前迭代编号
            
        Returns:
            异步通信句柄
        """
        src = self.global_rank + 1
        sender_stage_id = self.stage_id + 1
        direction = 1  # 反向传播方向
        tag = self._compute_tag(iter_num, sender_stage_id, direction, mb_idx)
        tensor_to_recv = self.grad_y_buffers[mb_idx]
        return dist.irecv(tensor_to_recv, src=src, tag=tag, group=self.model_parallel_group)
    
    def _compute_tag(
        self,
        iter_num: int,
        stage_id: int,
        direction: int,
        mb_idx: int
    ) -> int:
        """计算通信标签
        
        确保每个通信操作都有唯一的标识符。
        
        Args:
            iter_num: 迭代编号
            stage_id: 阶段 ID
            direction: 方向（0=前向，1=反向）
            mb_idx: micro-batch 索引
            
        Returns:
            唯一的通信标签
        """
        return (
            iter_num * (self.model_parallel_size * 2 * self.num_microbatches) +
            (stage_id - 1) * (2 * self.num_microbatches) +
            direction * self.num_microbatches +
            mb_idx
        )
    
    # ==================== 同步通信方法（用于 Naive 调度器）====================
    
    def forward_send(self, out: torch.Tensor) -> None:
        """同步发送前向传播的输出到下一阶段
        
        阻塞式发送，直到数据被接收方接收。
        
        Args:
            out: 要发送的输出张量
        """
        dst = self.global_rank + 1
        dist.send(tensor=out, dst=dst, tag=self.stage_id, group=self.model_parallel_group)
    
    def forward_recv(self, mb_idx: int) -> None:
        """同步接收上一阶段的前向传播输出
        
        阻塞式接收，直到数据到达。
        接收的数据存储在 self.out_x_buffers[mb_idx] 中。
        
        Args:
            mb_idx: micro-batch 索引
        """
        src = self.global_rank - 1
        tensor_to_recv = self.out_x_buffers[mb_idx]
        dist.recv(tensor=tensor_to_recv, src=src, tag=self.stage_id - 1, group=self.model_parallel_group)
        self.out_x_buffers[mb_idx] = tensor_to_recv.to(self.device)
    
    def backward_send(self, mb_idx: int) -> None:
        """同步发送反向传播的梯度到上一阶段
        
        阻塞式发送，直到数据被接收方接收。
        
        Args:
            mb_idx: micro-batch 索引
        """
        cached_input = self.input_cache[mb_idx]
        grad = cached_input.grad  # dL/dX
        dst = self.global_rank - 1
        dist.send(grad, dst=dst, group=self.model_parallel_group)
    
    def backward_recv(self, mb_idx: int) -> None:
        """同步接收下一阶段的反向传播梯度
        
        阻塞式接收，直到数据到达。
        接收的数据存储在 self.grad_y_buffers[mb_idx] 中。
        
        Args:
            mb_idx: micro-batch 索引
        """
        src = self.global_rank + 1
        tensor_to_recv = self.grad_y_buffers[mb_idx]
        dist.recv(tensor_to_recv, src=src, group=self.model_parallel_group)
        self.grad_y_buffers[mb_idx] = tensor_to_recv.to(self.device)
    
    def all_reduce_gradients(self):
        """在数据并行组内对梯度进行 AllReduce 操作"""
        for param in self.sub_model.parameters():
            if param.grad is not None:
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.data_parallel_group
                )
                param.grad /= self.data_parallel_size
    
    def accumulate_gradients(self):
        """累加 grad_buffers 中的梯度到参数
        
        优化版本：避免 clone 操作，使用 in-place 累加减少内存峰值。
        
        原问题：clone() 操作会在内存中创建梯度副本，当有 16 个 micro-batch 时，
        所有梯度同时存在于 grad_buffers 中，加上 clone 产生的副本，导致内存峰值
        翻倍，最终 OOM。
        
        修复方案：
        1. 第一个梯度直接赋值给 param.grad，不 clone
        2. 后续梯度使用 in-place 加法 (add_)
        """
        params = list(self.sub_model.parameters())
        for param_idx, param in enumerate(params):
            first_grad_assigned = False
            for mb_idx in range(self.num_microbatches):
                grads = self.grad_buffers[mb_idx]
                if grads is not None and param_idx < len(grads):
                    mb_grad = grads[param_idx]
                    if mb_grad is None:
                        continue
                    if not first_grad_assigned:
                        # 第一个梯度：直接赋值，不 clone
                        # 注意：这会使 grad_buffers 中的引用失效，但我们之后会清空它
                        param.grad = mb_grad
                        first_grad_assigned = True
                    else:
                        # 后续梯度：使用 in-place 加法
                        param.grad.add_(mb_grad)
        
        # 清空 buffer，释放内存
        self.grad_buffers = [None] * self.num_microbatches
    
    def step(self):
        """执行优化器步骤"""
        self.optimizer.step()
    
    def reset_cache(self):
        """重置缓存"""
        self.fwd_cache = [None] * self.num_microbatches
        self.input_cache = [None] * self.num_microbatches
        self.grad_buffers = [None] * self.num_microbatches
    
    def stop_orion_scheduler(self):
        """停止 Orion 调度器并打印统计信息"""
        if self.orion_scheduler is not None:
            try:
                for client_idx in range(self.num_microbatches):
                    scheduled, waiting = self.orion_scheduler.get_stats(client_idx)
                    print(f"[Orion] Stage {self.stage_id} client {client_idx}: scheduled={scheduled}, waiting={waiting}")
                self.orion_scheduler.stop()
                print(f"[Orion] Stage {self.stage_id}: Orion scheduler stopped")
            except Exception as e:
                print(f"[Orion] Stage {self.stage_id}: Error stopping scheduler: {e}")