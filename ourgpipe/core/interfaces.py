"""
GPipe 流水线框架抽象接口定义

本模块定义了模型、Stage 和数据集的抽象接口，
所有具体实现都需要继承这些接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Callable
import torch
import torch.nn as nn


class ModelInterface(ABC):
    """模型抽象接口
    
    所有要在 GPipe 流水线中运行的模型都需要实现此接口。
    该接口定义了模型如何被分割成多个阶段，以及如何创建各层。
    
    示例:
        @MODEL_REGISTRY.register("gpt")
        class GPTModel(ModelInterface):
            def get_model_config(self):
                return {"em_tokn": [vocab_size, hidden_size], ...}
    """
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """返回模型配置字典
        
        配置字典的键是层名称，值是该层的参数列表。
        这个字典将被用于 init_model() 函数创建模型层。
        
        Returns:
            Dict[str, Any]: 模型配置字典
            
        示例:
            {
                "em_tokn": [vocab_size, emb_size],
                "em_pos": [seq_len, emb_size],
                "decoder1": [emb_size, head_size],
                ...
            }
        """
        pass
    
    @abstractmethod
    def get_stage_partition(self, num_stages: int) -> List[List[int]]:
        """返回模型层的阶段划分
        
        将模型的所有层划分到指定数量的流水线阶段中。
        
        Args:
            num_stages: 流水线阶段数
            
        Returns:
            List[List[int]]: 每个阶段包含的层索引列表
            
        示例:
            对于 4 阶段划分:
            [[0,1,2,3,4,5], [6,7,8,9], [10,11,12,13], [14,15,16,17,18,19]]
        """
        pass
    
    @abstractmethod
    def create_layer(self, layer_type: str, params: List[Any]) -> nn.Module:
        """根据类型和参数创建模型层
        
        Args:
            layer_type: 层类型名称（如 "decoder", "embedding" 等）
            params: 层参数列表
            
        Returns:
            nn.Module: 创建的模型层实例
        """
        pass
    
    @abstractmethod
    def get_hidden_size(self) -> int:
        """返回隐藏层维度
        
        用于创建流水线阶段之间的通信缓冲区。
        
        Returns:
            int: 隐藏层维度大小
        """
        pass
    
    @abstractmethod
    def get_sequence_length(self) -> int:
        """返回序列长度
        
        用于创建流水线阶段之间的通信缓冲区。
        
        Returns:
            int: 序列长度
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """返回词表大小
        
        对于语言模型，返回词表大小。
        对于其他模型（如图像分类），返回 -1。
        
        Returns:
            int: 词表大小，不适用时返回 -1
        """
        pass
    
    def get_flops_per_token(self, num_params: Optional[int] = None) -> int:
        """返回每个 token 的 FLOPs（前向 + 反向）
        
        用于计算 MFU (Model FLOPs Utilization)。
        默认实现使用 6 × num_params 的近似公式。
        子类可以重写此方法提供更精确的计算。
        
        Args:
            num_params: 模型参数量，如果为 None 则需要子类自行计算
            
        Returns:
            int: 每个 token 的 FLOPs
        """
        if num_params is None:
            raise ValueError("num_params is required for default FLOPs estimation")
        # 默认使用 6 × params 的近似公式
        # 前向传播 ≈ 2 × params，反向传播 ≈ 4 × params
        return 6 * num_params
    
    def init_model(self) -> nn.ModuleList:
        """根据配置初始化完整模型
        
        这是一个便捷方法，使用 get_model_config() 和 create_layer() 
        创建完整的模型层列表。
        
        Returns:
            nn.ModuleList: 包含所有模型层的列表
        """
        model = nn.ModuleList()
        config = self.get_model_config()
        for layer_type, params in config.items():
            layer = self.create_layer(layer_type, params)
            model.append(layer)
        return model


class StageInterface(ABC):
    """Stage 抽象接口
    
    定义流水线阶段的标准行为。每个具体的模型需要实现
    自己的 Stage 类来处理模型特定的逻辑。
    
    主要职责:
    - 输入预处理（如嵌入层处理）
    - 损失计算
    - 输出格式化
    """
    
    @abstractmethod
    def prepare_input(self, x: torch.Tensor, mb_idx: int) -> torch.Tensor:
        """准备阶段输入
        
        第一阶段可能需要进行嵌入处理，其他阶段通常直接返回输入。
        
        Args:
            x: 输入张量
            mb_idx: micro-batch 索引
            
        Returns:
            torch.Tensor: 处理后的输入张量
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self, 
        output: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算损失（仅最后阶段使用）
        
        不同的任务有不同的损失函数:
        - 语言模型: cross_entropy
        - 图像分类: cross_entropy
        - 序列标注: cross_entropy (per token)
        - 回归任务: mse_loss
        
        Args:
            output: 模型输出
            labels: 标签
            
        Returns:
            torch.Tensor: 损失值
        """
        pass
    
    @abstractmethod
    def get_output_for_send(self, mb_idx: int) -> torch.Tensor:
        """获取要发送给下一阶段的输出
        
        Args:
            mb_idx: micro-batch 索引
            
        Returns:
            torch.Tensor: 要发送的张量
        """
        pass
    
    @abstractmethod
    def create_sub_model(
        self, 
        model_layers: nn.ModuleList, 
        layer_indices: List[int]
    ) -> nn.Module:
        """创建该阶段的子模型
        
        子类需要实现此方法来处理特定模型的层组织方式。
        例如，GPT 的第一阶段需要特殊处理嵌入层。
        
        Args:
            model_layers: 完整模型的所有层
            layer_indices: 该阶段包含的层索引
            
        Returns:
            nn.Module: 该阶段的子模型
        """
        pass


class DatasetInterface(ABC):
    """数据集抽象接口
    
    定义数据集加载和处理的标准行为。
    不同的任务需要不同的数据处理方式。
    
    示例:
        @DATASET_REGISTRY.register("code_search_net")
        class CodeSearchNetDataset(DatasetInterface):
            def load_dataset(self):
                return load_dataset('code_search_net', 'python')
    """
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """加载原始数据集
        
        Returns:
            Any: 原始数据集对象（如 HuggingFace Dataset）
        """
        pass
    
    @abstractmethod
    def get_tokenizer(self) -> Any:
        """获取分词器/预处理器
        
        Returns:
            Any: 分词器或预处理器对象
        """
        pass
    
    @abstractmethod
    def process_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """处理一个批次的数据
        
        将原始数据转换为模型可以使用的格式。
        
        Args:
            batch: 原始批次数据
            
        Returns:
            Dict[str, torch.Tensor]: 包含 'inputs' 和 'labels' 的字典
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """获取词表大小
        
        Returns:
            int: 词表大小，如果不适用返回 -1
        """
        pass
    
    @abstractmethod
    def get_collate_fn(self) -> Optional[Callable]:
        """获取 DataLoader 的 collate 函数
        
        Returns:
            Optional[Callable]: collate 函数，如果使用默认则返回 None
        """
        pass
    
    @abstractmethod
    def get_train_test_split(
        self, 
        test_size: float = 0.1, 
        seed: int = 1024
    ) -> Tuple[Any, Any]:
        """获取训练集和测试集
        
        Args:
            test_size: 测试集比例
            seed: 随机种子
            
        Returns:
            Tuple[Any, Any]: (训练集, 测试集)
        """
        pass