"""
GPT Stage 实现

提供 GPT 模型在流水线阶段中的特定逻辑。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

from core.stage import BaseStage
from core.registry import STAGE_REGISTRY
from core.config import PipelineConfig


@STAGE_REGISTRY.register("gpt")
class GPTStage(BaseStage):
    """GPT 特定的 Stage 实现
    
    处理 GPT 模型在流水线阶段中的特殊逻辑：
    - 第一阶段：处理嵌入层（token + position embedding）
    - 最后阶段：计算语言模型损失
    
    示例:
        stage = GPTStage(
            stage_id=1,
            model_layers=gpt_layers,
            layer_indices=[0, 1, 2, 3, 4, 5],
            config=pipeline_config,
            device=torch.device("cuda:0"),
            ...
        )
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
        """初始化 GPT Stage（延迟初始化）
        
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
            model_parallel_size: 模型并行大小
            data_parallel_size: 数据并行大小
        """
        # 调用父类初始化（会调用 create_sub_model）
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
        """创建 GPT 子模型（延迟初始化，按需创建层）
        
        第一阶段需要特殊处理：
        - 创建并保存嵌入层引用（用于 prepare_input）
        - 子模型只包含 Transformer 块
        
        其他阶段：
        - 按需创建 Transformer 块
        
        Args:
            model_adapter: 模型适配器实例，用于创建层
            model_config: 模型配置字典
            layer_indices: 该阶段包含的层索引
            
        Returns:
            该阶段的子模型
        """
        # 获取层名称列表
        layer_names = list(model_config.keys())
        
        if self.stage_id == 1:
            # 第一阶段：创建嵌入层
            # layer_indices[0] = em_tokn, layer_indices[1] = em_pos
            tokn_name = layer_names[layer_indices[0]]
            pos_name = layer_names[layer_indices[1]]
            
            # 按需创建嵌入层
            self.token_embedding = model_adapter.create_layer(
                tokn_name,
                model_config[tokn_name]
            )
            self.position_embedding = model_adapter.create_layer(
                pos_name,
                model_config[pos_name]
            )
            
            # 创建 Transformer 块（如果有）
            if len(layer_indices) > 2:
                transformer_layers = []
                for idx in layer_indices[2:]:
                    layer_name = layer_names[idx]
                    layer = model_adapter.create_layer(
                        layer_name,
                        model_config[layer_name]
                    )
                    transformer_layers.append(layer)
                return nn.Sequential(*transformer_layers)
            else:
                # 如果只有嵌入层，返回一个恒等映射
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
        """准备 GPT 输入
        
        第一阶段需要进行嵌入处理：
        - Token embedding
        - Position embedding
        - 两者相加
        
        Args:
            x: 输入张量
               - 第一阶段: token IDs (B, T)
               - 其他阶段: 激活值 (B, T, C)
            mb_idx: micro-batch 索引
            
        Returns:
            处理后的输入张量 (B, T, C)
        """
        if self.stage_id == 1:
            B, T = x.shape
            # 生成位置索引
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            # 计算嵌入
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            # 返回嵌入之和
            return tok_emb + pos_emb
        else:
            # 其他阶段直接返回输入
            return x
    
    def compute_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算语言模型损失
        
        使用 cross entropy loss 计算语言模型的损失。
        
        Args:
            output: 模型输出 (B, T, vocab_size)
            labels: 标签 (B, T)
            
        Returns:
            损失值
        """
        # GPT 使用 cross entropy loss
        # 需要将 output 转置为 (B, vocab_size, T) 以匹配 cross_entropy 的输入格式
        logits = output.transpose(-2, -1)
        return F.cross_entropy(logits, labels)
    
    def to(self, device):
        """将模型和缓冲区移动到指定设备
        
        重写父类方法以处理嵌入层。
        
        Args:
            device: 目标设备
        """
        device = torch.device(device)
        
        # 移动嵌入层（如果是第一阶段）
        if self.stage_id == 1:
            self.token_embedding.to(device)
            self.position_embedding.to(device)
        
        # 调用父类方法移动子模型和缓冲区
        super().to(device)