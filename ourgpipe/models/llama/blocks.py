"""
LLaMA 模型的基础组件

包含 RMSNorm, RoPE, LlamaAttention, LlamaMLP, LlamaBlock。
从零实现，不依赖 HuggingFace transformers。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.profiler import record_function


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    LLaMA 使用 RMSNorm 替代 LayerNorm，计算更高效。
    公式: x * rsqrt(mean(x²) + eps) * weight
    
    Args:
        hidden_size: 隐藏层维度
        eps: 数值稳定性的 epsilon
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None
) -> torch.Tensor:
    """预计算 RoPE 的频率张量
    
    Args:
        dim: 每个注意力头的维度（必须为偶数）
        seq_len: 最大序列长度
        theta: RoPE 的 base frequency
        device: 目标设备
        
    Returns:
        freqs_cis: 复数形式的频率张量 (seq_len, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
    # 转为复数: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple:
    """对 Q 和 K 应用旋转位置编码
    
    Args:
        xq: Query 张量 (B, T, n_heads, head_dim)
        xk: Key 张量 (B, T, n_kv_heads, head_dim)
        freqs_cis: 预计算的频率 (T, head_dim // 2)
        
    Returns:
        (xq_out, xk_out): 应用 RoPE 后的 Q 和 K
    """
    # 将实数张量视为复数: (B, T, n_heads, head_dim) -> (B, T, n_heads, head_dim//2) complex
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整 freqs_cis 形状以便广播: (T, head_dim//2) -> (1, T, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # 应用旋转
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaAttention(nn.Module):
    """LLaMA 注意力层
    
    支持标准 MHA 和 GQA (Grouped Query Attention)。
    所有线性层无 bias，Q/K 上应用 RoPE。
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: Query 注意力头数
        num_kv_heads: KV 注意力头数 (GQA)，等于 num_heads 时退化为 MHA
        sequence_len: 最大序列长度
        rope_theta: RoPE base frequency
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        sequence_len: int = 128,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Q/K/V/O 投影，无 bias
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # 预计算 RoPE 频率并注册为 buffer
        freqs_cis = precompute_freqs_cis(self.head_dim, sequence_len, rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # 因果掩码
        mask = torch.tril(torch.ones(sequence_len, sequence_len))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        with record_function("llama_qkv_proj"):
            q = self.q_proj(x)  # (B, T, num_heads * head_dim)
            k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
            v = self.v_proj(x)  # (B, T, num_kv_heads * head_dim)
        
        # Reshape: (B, T, n_heads, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)
        
        # 应用 RoPE
        with record_function("llama_rope"):
            freqs = self.freqs_cis[:T]
            q, k = apply_rotary_emb(q, k, freqs)
        
        # GQA: 扩展 KV 头以匹配 Q 头数
        if self.num_kv_groups > 1:
            # (B, T, num_kv_heads, head_dim) -> (B, T, num_heads, head_dim)
            k = k.unsqueeze(3).expand(B, T, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            k = k.reshape(B, T, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            v = v.reshape(B, T, self.num_heads, self.head_dim)
        
        # 转置为 (B, n_heads, T, head_dim) 以计算注意力
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        with record_function("llama_attention_score"):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 应用因果掩码
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        
        # 合并多头: (B, T, hidden_size)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        
        with record_function("llama_o_proj"):
            output = self.o_proj(output)
        
        return output


class LlamaMLP(nn.Module):
    """LLaMA 前馈网络 (SwiGLU)
    
    使用 gate 机制: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    三个线性层均无 bias。
    
    Args:
        hidden_size: 隐藏层维度
        intermediate_size: FFN 中间维度
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("llama_mlp"):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    """LLaMA Transformer 解码器块
    
    Pre-RMSNorm 架构:
    x = x + Attention(RMSNorm(x))
    x = x + MLP(RMSNorm(x))
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: Query 注意力头数
        num_kv_heads: KV 注意力头数
        intermediate_size: FFN 中间维度
        sequence_len: 最大序列长度
        rms_norm_eps: RMSNorm 的 epsilon
        rope_theta: RoPE base frequency
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        sequence_len: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sequence_len=sequence_len,
            rope_theta=rope_theta
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + Attention + 残差
        x = x + self.self_attn(self.input_layernorm(x))
        # Pre-Norm + MLP + 残差
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
