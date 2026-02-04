"""
GPT Stage 实现

提供 GPT 模型在流水线阶段中的特定逻辑。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
        model_layers: nn.ModuleList,
        layer_indices: List[int],
        config: PipelineConfig,
        device: torch.device,
        model_parallel_group,
        data_parallel_group,
        global_rank: int,
        model_parallel_size: int,
        data_parallel_size: int
    ):
        """初始化 GPT Stage
        
        Args:
            stage_id: 阶段 ID（从 1 开始）
            model_layers: 完整模型的所有层
            layer_indices: 该阶段包含的层索引
            config: 流水线配置
            device: 计算设备
            model_parallel_group: 模型并行通信组
            data_parallel_group: 数据并行通信组
            global_rank: 全局进程排名
            model_parallel_size: 模型并行大小
            data_parallel_size: 数据并行大小
        """
        # 保存嵌入层引用（在调用父类初始化之前）
        self._model_layers = model_layers
        self._layer_indices = layer_indices
        
        # 调用父类初始化
        super().__init__(
            stage_id=stage_id,
            model_layers=model_layers,
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
        model_layers: nn.ModuleList,
        layer_indices: List[int]
    ) -> nn.Module:
        """创建 GPT 子模型
        
        第一阶段需要特殊处理：
        - 保存嵌入层引用（用于 prepare_input）
        - 子模型只包含 Transformer 块
        
        Args:
            model_layers: 完整模型的所有层
            layer_indices: 该阶段包含的层索引
            
        Returns:
            该阶段的子模型
        """
        if self.stage_id == 1:
            # 第一阶段：保存嵌入层引用
            # layer_indices[0] = em_tokn, layer_indices[1] = em_pos
            self.token_embedding = model_layers[layer_indices[0]]
            self.position_embedding = model_layers[layer_indices[1]]
            
            # 子模型只包含 Transformer 块（跳过嵌入层）
            if len(layer_indices) > 2:
                return nn.Sequential(*[model_layers[i] for i in layer_indices[2:]])
            else:
                # 如果只有嵌入层，返回一个恒等映射
                return nn.Identity()
        else:
            # 其他阶段：直接使用指定的层
            return nn.Sequential(*[model_layers[i] for i in layer_indices])
    
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