"""
GPT 模型适配器

实现 ModelInterface 接口，提供 GPT 模型在 GPipe 流水线中的配置和创建逻辑。
"""

import torch.nn as nn
from typing import Dict, List, Any, Optional

from core.interfaces import ModelInterface
from core.registry import MODEL_REGISTRY
from core.config import ModelConfig
from .blocks import Block


@MODEL_REGISTRY.register("gpt")
class GPTModel(ModelInterface):
    """GPT 模型适配器
    
    实现 ModelInterface 接口，提供：
    - 模型配置生成
    - 阶段划分
    - 层创建
    
    示例:
        config = ModelConfig(
            name="gpt",
            hidden_size=512,
            num_layers=16,
            sequence_length=128,
            vocab_size=65,
            extra={"head_size": 32, "ff_low_rank": 256}
        )
        model = GPTModel(config)
        layers = model.init_model()
    """
    
    def __init__(self, config: ModelConfig):
        """初始化 GPT 模型适配器
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.sequence_length = config.sequence_length
        self.vocab_size = config.vocab_size
        
        # GPT 特定参数
        self.head_size = config.extra.get('head_size', self.hidden_size // self.num_heads)
        self.ff_low_rank = config.extra.get('ff_low_rank', None)
        self.dropout = config.extra.get('dropout', 0.4)
    
    def get_model_config(self) -> Dict[str, Any]:
        """返回 GPT 模型配置字典
        
        配置包含：
        - em_tokn: Token 嵌入层
        - em_pos: 位置嵌入层
        - decoder1..N: Transformer 解码器块
        - ln: 最终层归一化
        - lm_head: 语言模型输出头
        
        Returns:
            模型配置字典
        """
        config = {
            "em_tokn": [self.vocab_size, self.hidden_size],
            "em_pos": [self.sequence_length, self.hidden_size],
        }
        
        # 添加 decoder 层
        for i in range(1, self.num_layers + 1):
            config[f"decoder{i}"] = [self.hidden_size, self.head_size]
        
        # 添加最终层
        config["ln"] = [self.hidden_size]
        config["lm_head"] = [self.hidden_size, self.vocab_size]
        
        return config
    
    def get_stage_partition(self, num_stages: int) -> List[List[int]]:
        """返回 GPT 的阶段划分
        
        GPT 结构: [em_tokn, em_pos, decoder1, ..., decoderN, ln, lm_head]
        总层数: 2 + num_layers + 2 = num_layers + 4
        
        Args:
            num_stages: 流水线阶段数
            
        Returns:
            每个阶段包含的层索引列表
        """
        total_layers = self.num_layers + 4  # 2 embedding + N decoder + ln + lm_head
        
        # 简单的均匀划分
        layers_per_stage = total_layers // num_stages
        remainder = total_layers % num_stages
        
        partitions = []
        start = 0
        for i in range(num_stages):
            # 前 remainder 个阶段多分配一层
            extra = 1 if i < remainder else 0
            end = start + layers_per_stage + extra
            partitions.append(list(range(start, end)))
            start = end
        
        return partitions
    
    def create_layer(self, layer_type: str, params: List[Any]) -> nn.Module:
        """根据类型和参数创建 GPT 层
        
        Args:
            layer_type: 层类型名称
            params: 层参数列表
            
        Returns:
            创建的模型层
            
        Raises:
            ValueError: 未知的层类型
        """
        if layer_type == "em_tokn":
            return nn.Embedding(params[0], params[1])
        elif layer_type == "em_pos":
            return nn.Embedding(params[0], params[1])
        elif layer_type.startswith("decoder"):
            return Block(
                emb_size=params[0],
                head_size=params[1],
                sequence_len=self.sequence_length,
                ff_low_rank=self.ff_low_rank,
                dropout=self.dropout
            )
        elif layer_type == "ln":
            return nn.LayerNorm(params[0])
        elif layer_type == "lm_head":
            return nn.Linear(params[0], params[1])
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def get_hidden_size(self) -> int:
        """返回隐藏层维度"""
        return self.hidden_size
    
    def get_sequence_length(self) -> int:
        """返回序列长度"""
        return self.sequence_length
    
    def get_vocab_size(self) -> int:
        """返回词表大小"""
        return self.vocab_size
    
    def get_flops_per_token(self, num_params: Optional[int] = None) -> int:
        """返回每个 token 的 FLOPs（前向 + 反向）
        
        GPT 模型的 FLOPs 计算公式（参考 PaLM 论文）：
        
        每个 token 的前向 FLOPs ≈ 2 × num_params + 2 × num_layers × seq_len × hidden_size
        
        其中：
        - 2 × num_params: 矩阵乘法的 FLOPs
        - 2 × num_layers × seq_len × hidden_size: 注意力的 FLOPs
        
        反向传播 ≈ 2 × 前向，所以总共 ≈ 6 × num_params（近似）
        
        Args:
            num_params: 模型参数量，如果为 None 则使用估算值
            
        Returns:
            每个 token 的 FLOPs
        """
        if num_params is None:
            # 估算参数量
            # GPT: params ≈ 12 × L × H² + V × H
            num_params = (
                12 * self.num_layers * self.hidden_size ** 2 +
                self.vocab_size * self.hidden_size
            )

        # 简化版本：使用 6 × params 的近似
        # 这是业界常用的估算方法
        return 6 * num_params
    
    def estimate_params(self) -> int:
        """估算模型参数量
        
        GPT 参数量公式：
        - Token Embedding: vocab_size × hidden_size
        - Position Embedding: seq_len × hidden_size
        - 每个 Transformer 层: 12 × hidden_size²
          - Attention (Q, K, V, O): 4 × hidden_size²
          - FFN: 8 × hidden_size² (4H → H 和 H → 4H)
          - LayerNorm: 2 × hidden_size (可忽略)
        - Final LayerNorm: hidden_size
        - LM Head: hidden_size × vocab_size
        
        Returns:
            估算的参数量
        """
        params = 0
        
        # Token Embedding
        params += self.vocab_size * self.hidden_size
        
        # Position Embedding
        params += self.sequence_length * self.hidden_size
        
        # Transformer 层
        params += self.num_layers * 12 * self.hidden_size ** 2
        
        # Final LayerNorm
        params += self.hidden_size
        
        # LM Head (通常与 Token Embedding 共享权重，但这里假设不共享)
        params += self.hidden_size * self.vocab_size
        
        return params
    
    def __repr__(self) -> str:
        return (
            f"GPTModel(hidden={self.hidden_size}, layers={self.num_layers}, "
            f"heads={self.num_heads}, seq_len={self.sequence_length}, "
            f"vocab={self.vocab_size})"
        )