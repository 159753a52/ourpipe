"""
GPT 模型的基础组件

包含 Transformer Block、Attention、FeedForward 等组件。
这些组件从原始 GPT.py 中提取并重构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.profiler import record_function


class LowRankLinear(nn.Module):
    """低秩线性层
    
    用低秩分解 A[in, r], B[r, out] 近似一个 [in, out] 的 Linear。
    forward: y = x @ A @ B (+ bias)
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: 秩（分解维度）
        bias: 是否使用偏置
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        assert rank > 0 and rank <= min(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # A: [in, r], B: [r, out]
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x @ self.A      # (..., rank)
        x = x @ self.B      # (..., out_features)
        if self.bias is not None:
            x = x + self.bias
        return x


def attention(query, key, value, dropout, mask=None):
    """计算注意力
    
    Args:
        query: 查询张量 (B, T, C)
        key: 键张量 (B, T, C)
        value: 值张量 (B, T, C)
        dropout: Dropout 层
        mask: 注意力掩码 (T, T)
        
    Returns:
        out: 注意力输出 (B, T, C)
        w_att: 注意力权重 (B, T, T)
    """
    B, T, C = query.shape
    # (B, T, C) @ (B, C, T) --> (B, T, T)
    scores = query @ key.transpose(-2, -1) / (C ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    w_att = dropout(F.softmax(scores, dim=-1))
    out = w_att @ value
    return out, w_att


class MaskedAttention(nn.Module):
    """带掩码的单头注意力
    
    Args:
        emb_size: 嵌入维度
        head_size: 注意力头维度
        sequence_len: 序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, emb_size: int, head_size: int, sequence_len: int = 128, dropout: float = 0.4):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        # 上三角掩码（不参与训练）
        self.register_buffer(
            'tril', torch.tril(torch.ones(sequence_len, sequence_len))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        mask = self.tril[:T, :T]
        out, _ = attention(q, k, v, self.dropout, mask)
        return out


class MaskedMultiHeadAttention(nn.Module):
    """带掩码的多头注意力
    
    Args:
        emb_size: 嵌入维度
        head_size: 每个注意力头的维度
        sequence_len: 序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, emb_size: int, head_size: int, sequence_len: int = 128, dropout: float = 0.4):
        super().__init__()
        assert emb_size % head_size == 0
        n_head = emb_size // head_size
        self.heads = nn.ModuleList([
            MaskedAttention(emb_size, head_size, sequence_len, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈网络
    
    Args:
        emb_size: 嵌入维度
        low_rank: 低秩分解的秩，None 表示使用标准 Linear
        dropout: Dropout 概率
    """
    
    def __init__(self, emb_size: int, low_rank: int = None, dropout: float = 0.4):
        super().__init__()
        hidden = 4 * emb_size

        if low_rank is None:
            self.l1 = nn.Linear(emb_size, hidden)
        else:
            self.l1 = LowRankLinear(emb_size, hidden, rank=low_rank)

        self.l2 = nn.Linear(hidden, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.low_rank = low_rank

    def forward(self, x):
        name = "ffn_l1_lowrank" if self.low_rank is not None else "ffn_l1_linear"
        with record_function(name):
            x = self.l1(x)

        x = F.gelu(x)

        with record_function("ffn_l2_linear"):
            out = self.dropout(self.l2(x))
        return out


class Block(nn.Module):
    """Transformer 解码器块
    
    包含多头自注意力和前馈网络，使用残差连接和层归一化。
    
    Args:
        emb_size: 嵌入维度
        head_size: 注意力头维度
        sequence_len: 序列长度
        ff_low_rank: FFN 低秩分解的秩
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        emb_size: int,
        head_size: int,
        sequence_len: int = 128,
        ff_low_rank: int = None,
        dropout: float = 0.4
    ):
        super().__init__()
        self.mha = MaskedMultiHeadAttention(emb_size, head_size, sequence_len, dropout)
        self.ff = FeedForward(emb_size, low_rank=ff_low_rank, dropout=dropout)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # 残差连接
        x = x + self.mha(self.ln1(x))
        out = x + self.ff(self.ln2(x))
        return out


class CharGPT(nn.Module):
    """完整的字符级 GPT 模型
    
    用于独立测试，不用于流水线并行。
    
    Args:
        vocab_size: 词表大小
        emb_size: 嵌入维度
        head_size: 注意力头维度
        n_layer: Transformer 层数
        sequence_len: 序列长度
    """
    
    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 128,
        head_size: int = 8,
        n_layer: int = 12,
        sequence_len: int = 128
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(sequence_len, emb_size)
        self.blocks = nn.Sequential(*[
            Block(emb_size, head_size, sequence_len)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size)
        self.sequence_len = sequence_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits