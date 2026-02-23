"""
LLaMA Stage 实现

提供 LLaMA 模型在流水线阶段中的特定逻辑。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

from core.stage import BaseStage
from core.registry import STAGE_REGISTRY
from core.config import PipelineConfig


@STAGE_REGISTRY.register("llama")
class LlamaStage(BaseStage):
    """LLaMA 特定的 Stage 实现
    
    与 GPT Stage 的关键区别：
    - 第一阶段只有一个 token embedding（没有 position embedding，RoPE 在 attention 内部）
    - layer_indices 偏移从 [1:] 开始而非 GPT 的 [2:]
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
        super().__init__(
            stage_id=stage_id,
            model_adapter=model_adapter,
            model_config=model_config,
            layer_indices=layer_indices,
            config=config,
            device=device,
            model_parallel_group=model_parallel_group,
            data_parallel_group=data_parallel_group,
            global_rank=global_rank,
            model_parallel_size=model_parallel_size,
            data_parallel_size=data_parallel_size
        )
    
    def create_sub_model(
        self,
        model_adapter,
        model_config: Dict[str, Any],
        layer_indices: List[int]
    ) -> nn.Module:
        """创建 LLaMA 子模型（延迟初始化）
        
        第一阶段特殊处理：
        - 创建并保存 token embedding 引用
        - 子模型只包含 decoder 块（如果有）
        - 注意：只有一个嵌入层（无 position embedding）
        
        其他阶段：按需创建层
        """
        layer_names = list(model_config.keys())
        
        if self.stage_id == 1:
            # 第一阶段: layer_indices[0] = em_tokn
            tokn_name = layer_names[layer_indices[0]]
            self.token_embedding = model_adapter.create_layer(
                tokn_name,
                model_config[tokn_name]
            )
            
            # 创建后续层（decoder 块等）
            if len(layer_indices) > 1:
                layers = []
                for idx in layer_indices[1:]:
                    layer_name = layer_names[idx]
                    layer = model_adapter.create_layer(
                        layer_name,
                        model_config[layer_name]
                    )
                    layers.append(layer)
                return nn.Sequential(*layers)
            else:
                return nn.Identity()
        else:
            # 其他阶段：按需创建层
            layers = []
            for idx in layer_indices:
                layer_name = layer_names[idx]
                layer = model_adapter.create_layer(
                    layer_name,
                    model_config[layer_name]
                )
                layers.append(layer)
            return nn.Sequential(*layers)
    
    def prepare_input(self, x: torch.Tensor, mb_idx: int) -> torch.Tensor:
        """准备 LLaMA 输入
        
        第一阶段：只做 token embedding（RoPE 在 attention 内部处理，无需位置编码）
        其他阶段：直接返回输入
        
        Args:
            x: 输入张量
               - 第一阶段: token IDs (B, T)
               - 其他阶段: 激活值 (B, T, C)
            mb_idx: micro-batch 索引
            
        Returns:
            处理后的输入张量 (B, T, C)
        """
        if self.stage_id == 1:
            return self.token_embedding(x)
        else:
            return x
    
    def compute_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算语言模型损失
        
        与 GPT 相同，使用 cross entropy loss。
        
        Args:
            output: 模型输出 (B, T, vocab_size)
            labels: 标签 (B, T)
            
        Returns:
            损失值
        """
        logits = output.transpose(-2, -1)  # (B, vocab_size, T)
        return F.cross_entropy(logits, labels)
    
    def to(self, device):
        """将模型和缓冲区移动到指定设备"""
        device = torch.device(device)
        if self.stage_id == 1:
            self.token_embedding.to(device)
        super().to(device)
