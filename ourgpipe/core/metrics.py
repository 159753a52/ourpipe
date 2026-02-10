"""
GPipe 流水线框架训练指标追踪模块

提供训练过程中的性能指标追踪和计算：
- 吞吐量 (tokens/s)
- MFU (Model FLOPs Utilization)
- 模型参数量统计
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


# GPU 理论峰值 FLOPs (单位: FLOPs/s)
GPU_PEAK_FLOPS = {
    # NVIDIA A100
    'A100-SXM4-80GB': {'fp32': 19.5e12, 'tf32': 156e12, 'fp16': 312e12, 'bf16': 312e12},
    'A100-SXM4-40GB': {'fp32': 19.5e12, 'tf32': 156e12, 'fp16': 312e12, 'bf16': 312e12},
    'A100-PCIE-80GB': {'fp32': 19.5e12, 'tf32': 156e12, 'fp16': 312e12, 'bf16': 312e12},
    'A100-PCIE-40GB': {'fp32': 19.5e12, 'tf32': 156e12, 'fp16': 312e12, 'bf16': 312e12},
    # NVIDIA H100
    'H100-SXM5-80GB': {'fp32': 67e12, 'tf32': 989e12, 'fp16': 1979e12, 'bf16': 1979e12},
    'H100-PCIE-80GB': {'fp32': 51e12, 'tf32': 756e12, 'fp16': 1513e12, 'bf16': 1513e12},
    # NVIDIA V100
    'V100-SXM2-32GB': {'fp32': 15.7e12, 'fp16': 125e12},
    'V100-SXM2-16GB': {'fp32': 15.7e12, 'fp16': 125e12},
    'V100-PCIE-32GB': {'fp32': 14e12, 'fp16': 112e12},
    'V100-PCIE-16GB': {'fp32': 14e12, 'fp16': 112e12},
    # NVIDIA RTX 4090
    'RTX 4090': {'fp32': 82.6e12, 'fp16': 165.2e12, 'bf16': 165.2e12},
    # NVIDIA RTX 3090
    'RTX 3090': {'fp32': 35.6e12, 'fp16': 71e12},
    # 默认值
    'default': {'fp32': 10e12, 'tf32': 50e12, 'fp16': 100e12, 'bf16': 100e12},
}


def get_gpu_peak_flops(device_name: str, dtype: str = 'tf32') -> float:
    """获取 GPU 理论峰值 FLOPs
    
    Args:
        device_name: GPU 设备名称（从 torch.cuda.get_device_name() 获取）
        dtype: 数据类型 ('fp32', 'tf32', 'fp16', 'bf16')
        
    Returns:
        理论峰值 FLOPs/s
    """
    # 尝试匹配 GPU 名称
    for gpu_key, specs in GPU_PEAK_FLOPS.items():
        if gpu_key in device_name:
            if dtype in specs:
                return specs[dtype]
            # 如果没有指定的 dtype，返回 fp32
            return specs.get('fp32', GPU_PEAK_FLOPS['default']['fp32'])
    
    # 未找到匹配，返回默认值
    return GPU_PEAK_FLOPS['default'].get(dtype, GPU_PEAK_FLOPS['default']['fp32'])


@dataclass
class MetricsTracker:
    """训练指标追踪器
    
    追踪训练过程中的性能指标，包括：
    - 时间统计
    - 吞吐量计算
    - MFU 计算
    
    示例:
        tracker = MetricsTracker(
            batch_size=64,
            sequence_length=128,
            num_gpus=4,
            device_name="NVIDIA A100-SXM4-80GB"
        )
        tracker.start()
        
        for iteration in range(num_iterations):
            # ... 训练代码 ...
            tracker.step()
        
        tracker.stop()
        summary = tracker.get_summary(model_params, flops_per_token)
    """
    
    batch_size: int
    sequence_length: int
    num_gpus: int
    device_name: str
    warmup_iterations: int = 10
    dtype: str = 'tf32'
    
    # 内部状态
    _start_time: Optional[float] = field(default=None, repr=False)
    _end_time: Optional[float] = field(default=None, repr=False)
    _warmup_end_time: Optional[float] = field(default=None, repr=False)
    _iteration_count: int = field(default=0, repr=False)
    _tokens_processed: int = field(default=0, repr=False)
    _is_running: bool = field(default=False, repr=False)
    
    def start(self):
        """开始追踪"""
        self._start_time = time.time()
        self._iteration_count = 0
        self._tokens_processed = 0
        self._warmup_end_time = None
        self._is_running = True
    
    def step(self):
        """记录一次迭代完成
        
        每次迭代处理 batch_size × sequence_length 个 token
        """
        self._iteration_count += 1
        tokens_this_iter = self.batch_size * self.sequence_length
        self._tokens_processed += tokens_this_iter
        
        # 记录预热结束时间
        if self._iteration_count == self.warmup_iterations:
            self._warmup_end_time = time.time()
    
    def stop(self):
        """停止追踪"""
        self._end_time = time.time()
        self._is_running = False
    
    @property
    def total_time(self) -> float:
        """总训练时间（秒）"""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time
    
    @property
    def effective_time(self) -> float:
        """有效训练时间（排除预热，秒）"""
        if self._warmup_end_time is None:
            # 预热未完成，返回总时间
            return self.total_time
        end = self._end_time if self._end_time else time.time()
        return end - self._warmup_end_time
    
    @property
    def effective_iterations(self) -> int:
        """有效迭代次数（排除预热）"""
        return max(0, self._iteration_count - self.warmup_iterations)
    
    @property
    def effective_tokens(self) -> int:
        """有效处理的 token 数（排除预热）"""
        return self.effective_iterations * self.batch_size * self.sequence_length
    
    def get_throughput(self) -> float:
        """计算吞吐量 (tokens/s)
        
        Returns:
            每秒处理的 token 数
        """
        if self.effective_time <= 0:
            return 0.0
        return self.effective_tokens / self.effective_time
    
    def get_samples_per_second(self) -> float:
        """计算每秒处理的样本数
        
        Returns:
            每秒处理的样本数
        """
        if self.effective_time <= 0:
            return 0.0
        return self.effective_iterations * self.batch_size / self.effective_time
    
    def get_time_per_iteration(self) -> float:
        """计算每次迭代的平均时间（秒）
        
        Returns:
            平均迭代时间
        """
        if self.effective_iterations <= 0:
            return 0.0
        return self.effective_time / self.effective_iterations
    
    def get_mfu(self, model_params: int, flops_per_token: Optional[int] = None) -> float:
        """计算 MFU (Model FLOPs Utilization)
        
        MFU = 实际 FLOPs/s / 理论峰值 FLOPs/s
        
        Args:
            model_params: 模型总参数量
            flops_per_token: 每个 token 的 FLOPs，如果为 None 则使用 6 × params 估算
            
        Returns:
            MFU 比例 (0.0 ~ 1.0)
        """
        throughput = self.get_throughput()
        if throughput <= 0:
            return 0.0
        
        # 每个 token 的 FLOPs（前向 + 反向 ≈ 6 × params）
        if flops_per_token is None:
            flops_per_token = 6 * model_params
        
        # 实际 FLOPs/s
        actual_flops = throughput * flops_per_token
        
        # 理论峰值 FLOPs/s（所有 GPU 总和）
        peak_flops = get_gpu_peak_flops(self.device_name, self.dtype) * self.num_gpus
        
        if peak_flops <= 0:
            return 0.0
        
        return actual_flops / peak_flops
    
    def get_summary(
        self,
        model_params: int,
        flops_per_token: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取指标汇总
        
        Args:
            model_params: 模型总参数量
            flops_per_token: 每个 token 的 FLOPs
            
        Returns:
            包含所有指标的字典
        """
        throughput = self.get_throughput()
        mfu = self.get_mfu(model_params, flops_per_token)
        
        return {
            # 时间统计
            'total_time_seconds': self.total_time,
            'effective_time_seconds': self.effective_time,
            'time_per_iteration_seconds': self.get_time_per_iteration(),
            
            # 迭代统计
            'total_iterations': self._iteration_count,
            'effective_iterations': self.effective_iterations,
            'warmup_iterations': self.warmup_iterations,
            
            # Token 统计
            'total_tokens': self._tokens_processed,
            'effective_tokens': self.effective_tokens,
            
            # 性能指标
            'throughput_tokens_per_second': throughput,
            'samples_per_second': self.get_samples_per_second(),
            'mfu': mfu,
            
            # 模型信息
            'model_params': model_params,
            'flops_per_token': flops_per_token if flops_per_token else 6 * model_params,
            
            # 硬件信息
            'num_gpus': self.num_gpus,
            'device_name': self.device_name,
            'dtype': self.dtype,
        }
    
    def print_summary(
        self,
        model_params: int,
        flops_per_token: Optional[int] = None,
        model_size_bytes: Optional[int] = None
    ):
        """打印指标汇总
        
        Args:
            model_params: 模型总参数量
            flops_per_token: 每个 token 的 FLOPs
            model_size_bytes: 模型大小（字节）
        """
        summary = self.get_summary(model_params, flops_per_token)
        
        # 格式化模型大小
        if model_size_bytes is None:
            model_size_bytes = model_params * 4  # 假设 FP32
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # 格式化参数量
        if model_params >= 1e9:
            params_str = f"{model_params / 1e9:.2f}B"
        elif model_params >= 1e6:
            params_str = f"{model_params / 1e6:.2f}M"
        else:
            params_str = f"{model_params / 1e3:.2f}K"
        
        print("", flush=True)
        print("=" * 60, flush=True)
        print("TRAINING SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"Hardware:", flush=True)
        print(f"  GPUs: {self.num_gpus} × {self.device_name}", flush=True)
        print(f"  Dtype: {self.dtype}", flush=True)
        print("", flush=True)
        print(f"Training:", flush=True)
        print(f"  Total Iterations: {summary['total_iterations']}", flush=True)
        print(f"  Effective Iterations: {summary['effective_iterations']} (excluding {self.warmup_iterations} warmup)", flush=True)
        print(f"  Total Time: {summary['total_time_seconds']:.2f}s", flush=True)
        print(f"  Effective Time: {summary['effective_time_seconds']:.2f}s", flush=True)
        print(f"  Time per Iteration: {summary['time_per_iteration_seconds'] * 1000:.2f}ms", flush=True)
        print("", flush=True)
        print(f"Model:", flush=True)
        print(f"  Parameters: {model_params:,} ({params_str})", flush=True)
        print(f"  Size: {model_size_mb:.2f} MB", flush=True)
        print("", flush=True)
        print(f"Performance:", flush=True)
        print(f"  Throughput: {summary['throughput_tokens_per_second']:,.0f} tokens/s", flush=True)
        print(f"  Samples/s: {summary['samples_per_second']:.2f}", flush=True)
        print(f"  MFU: {summary['mfu'] * 100:.2f}%", flush=True)
        print("=" * 60, flush=True)
        print("", flush=True)


def collect_model_params(
    stage,
    model_parallel_group,
    device: torch.device
) -> int:
    """收集所有 stage 的模型参数量
    
    通过 all_reduce 汇总所有 stage 的参数量。
    
    Args:
        stage: 当前 stage 对象
        model_parallel_group: 模型并行通信组
        device: 计算设备
        
    Returns:
        模型总参数量
    """
    # 统计当前 stage 的参数量
    local_params = sum(p.numel() for p in stage.sub_model.parameters())
    
    # 如果是第一阶段，加上嵌入层参数
    if stage.stage_id == 1:
        if hasattr(stage, 'token_embedding'):
            local_params += sum(p.numel() for p in stage.token_embedding.parameters())
        if hasattr(stage, 'position_embedding'):
            local_params += sum(p.numel() for p in stage.position_embedding.parameters())
    
    # 汇总所有 stage 的参数量
    total_params_tensor = torch.tensor([local_params], dtype=torch.long, device=device)
    dist.all_reduce(total_params_tensor, op=dist.ReduceOp.SUM, group=model_parallel_group)
    
    return total_params_tensor.item()
