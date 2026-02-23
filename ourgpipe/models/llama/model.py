"""
LLaMA 模型适配器

实现 ModelInterface 接口，提供 LLaMA 模型在 GPipe 流水线中的配置和创建逻辑。
"""

import torch.nn as nn
from typing import Dict, List, Any, Optional

from core.interfaces import ModelInterface
from core.registry import MODEL_REGISTRY
from core.config import ModelConfig
from .blocks import LlamaBlock, RMSNorm


@MODEL_REGISTRY.register("llama")
class LlamaModel(ModelInterface):
    """LLaMA 模型适配器
    
    实现 ModelInterface 接口，提供：
    - 模型配置生成
    - 阶段划分
    - 层创建
    
    与 GPT 的关键区别：
    - 没有可学习的位置嵌入（使用 RoPE）
    - 使用 RMSNorm 替代 LayerNorm
    - FFN 使用 SwiGLU (gate + up + down)
    - 支持 GQA
    
    示例:
        config = ModelConfig(
            name="llama",
            hidden_size=512,
            num_layers=16,
            num_heads=8,
            sequence_length=128,
            vocab_size=65,
            extra={
                "num_kv_heads": 8,
                "intermediate_size": 1376,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
        )
        model = LlamaModel(config)
        layers = model.init_model()
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.sequence_length = config.sequence_length
        self.vocab_size = config.vocab_size
        
        # LLaMA 特有参数
        self.num_kv_heads = config.extra.get('num_kv_heads', self.num_heads)
        # 默认 intermediate_size: 取 2/3 * 4 * hidden_size 最近的 32 的倍数
        default_intermediate = int(2 * 4 * self.hidden_size / 3)
        default_intermediate = ((default_intermediate + 31) // 32) * 32
        self.intermediate_size = config.extra.get('intermediate_size', default_intermediate)
        self.rms_norm_eps = config.extra.get('rms_norm_eps', 1e-5)
        self.rope_theta = config.extra.get('rope_theta', 10000.0)
    
    def get_model_config(self) -> Dict[str, Any]:
        """返回 LLaMA 模型配置字典
        
        配置包含：
        - em_tokn: Token 嵌入层（无位置嵌入，RoPE 在 attention 内部处理）
        - decoder1..N: LLaMA Transformer 解码器块
        - ln: 最终 RMSNorm
        - lm_head: 语言模型输出头
        
        Returns:
            模型配置字典
        """
        config = {
            "em_tokn": [self.vocab_size, self.hidden_size],
        }
        
        # 添加 decoder 层
        for i in range(1, self.num_layers + 1):
            config[f"decoder{i}"] = [
                self.hidden_size,
                self.num_heads,
                self.num_kv_heads,
                self.intermediate_size,
            ]
        
        # 最终层
        config["ln"] = [self.hidden_size]
        config["lm_head"] = [self.hidden_size, self.vocab_size]
        
        return config
    
    def get_stage_partition(self, num_stages: int) -> List[List[int]]:
        """返回 LLaMA 的阶段划分
        
        LLaMA 结构: [em_tokn, decoder1, ..., decoderN, ln, lm_head]
        总层数: 1 + num_layers + 2 = num_layers + 3
        
        Args:
            num_stages: 流水线阶段数
            
        Returns:
            每个阶段包含的层索引列表
        """
        total_layers = self.num_layers + 3  # 1 embedding + N decoder + ln + lm_head
        
        layers_per_stage = total_layers // num_stages
        remainder = total_layers % num_stages
        
        partitions = []
        start = 0
        for i in range(num_stages):
            extra = 1 if i < remainder else 0
            end = start + layers_per_stage + extra
            partitions.append(list(range(start, end)))
            start = end
        
        return partitions
    
    def create_layer(self, layer_type: str, params: List[Any]) -> nn.Module:
        """根据类型和参数创建 LLaMA 层
        
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
        elif layer_type.startswith("decoder"):
            return LlamaBlock(
                hidden_size=params[0],
                num_heads=params[1],
                num_kv_heads=params[2],
                intermediate_size=params[3],
                sequence_len=self.sequence_length,
                rms_norm_eps=self.rms_norm_eps,
                rope_theta=self.rope_theta,
            )
        elif layer_type == "ln":
            return RMSNorm(params[0], eps=self.rms_norm_eps)
        elif layer_type == "lm_head":
            return nn.Linear(params[0], params[1], bias=False)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def get_hidden_size(self) -> int:
        return self.hidden_size
    
    def get_sequence_length(self) -> int:
        return self.sequence_length
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_flops_per_token(self, num_params: Optional[int] = None) -> int:
        """返回每个 token 的 FLOPs（前向 + 反向）
        
        LLaMA 的 FLOPs 计算：
        - QKV 投影: 2 * hidden² * (1 + 2 * kv_ratio) per layer
        - O 投影: 2 * hidden² per layer
        - Attention: 2 * seq_len * hidden per layer
        - FFN (SwiGLU): 2 * 3 * hidden * intermediate per layer
        简化为 6 * num_params
        """
        if num_params is None:
            num_params = self.estimate_params()
        return 6 * num_params
    
    def estimate_params(self) -> int:
        """估算模型参数量
        
        LLaMA 参数量：
        - Token Embedding: vocab_size × hidden_size
        - 每个 Transformer 层:
          - Q: hidden² 
          - K: hidden * (hidden / num_heads * num_kv_heads)
          - V: 同 K
          - O: hidden²
          - gate_proj: hidden * intermediate
          - up_proj: hidden * intermediate
          - down_proj: intermediate * hidden
          - 2 × RMSNorm: 2 * hidden
        - Final RMSNorm: hidden
        - LM Head: hidden × vocab_size
        """
        params = 0
        
        # Token Embedding
        params += self.vocab_size * self.hidden_size
        
        kv_dim = self.hidden_size // self.num_heads * self.num_kv_heads
        
        # Transformer 层
        per_layer = (
            self.hidden_size * self.hidden_size +  # Q
            self.hidden_size * kv_dim +             # K
            self.hidden_size * kv_dim +             # V
            self.hidden_size * self.hidden_size +   # O
            self.hidden_size * self.intermediate_size +  # gate
            self.hidden_size * self.intermediate_size +  # up
            self.intermediate_size * self.hidden_size +  # down
            2 * self.hidden_size                         # 2x RMSNorm
        )
        params += self.num_layers * per_layer
        
        # Final RMSNorm
        params += self.hidden_size
        
        # LM Head
        params += self.hidden_size * self.vocab_size
        
        return params
    
    def __repr__(self) -> str:
        return (
            f"LlamaModel(hidden={self.hidden_size}, layers={self.num_layers}, "
            f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
            f"intermediate={self.intermediate_size}, seq_len={self.sequence_length}, "
            f"vocab={self.vocab_size})"
        )
