"""
GPT 数据集适配器

提供 CodeSearchNet 数据集的加载和处理逻辑。
"""

import torch
from typing import Dict, Any, Optional, Callable, Tuple
from datasets import load_dataset

from core.interfaces import DatasetInterface
from core.registry import DATASET_REGISTRY
from core.config import DatasetConfig


class CharTokenizer:
    """字符级分词器
    
    将文本转换为字符级别的 token ID 序列。
    
    Args:
        data: 用于构建词表的文本列表
    """
    
    def __init__(self, data):
        """初始化分词器
        
        Args:
            data: 文本列表，用于构建字符词表
        """
        # 从数据中提取所有唯一字符
        chars = sorted(list(set(''.join(data))))
        # 构建字符到索引的映射（预留 0 给结束符）
        self.char2ind = {s: i + 1 for i, s in enumerate(chars)}
        self.char2ind['<|e|>'] = 0  # 结束符
        # 构建索引到字符的映射
        self.ind2char = {i: s for s, i in self.char2ind.items()}
    
    def encode(self, text: str) -> list:
        """将文本编码为 token ID 列表
        
        Args:
            text: 输入文本
            
        Returns:
            token ID 列表
        """
        return [self.char2ind[c] for c in text]
    
    def decode(self, enc) -> str:
        """将 token ID 解码为文本
        
        Args:
            enc: token ID 或 token ID 列表
            
        Returns:
            解码后的文本
        """
        if isinstance(enc, int):
            return self.ind2char[enc]
        return ''.join([self.ind2char[i] for i in enc])
    
    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.char2ind)


@DATASET_REGISTRY.register("code_search_net")
class CodeSearchNetDataset(DatasetInterface):
    """CodeSearchNet 数据集适配器
    
    加载和处理 CodeSearchNet Python 代码数据集，
    用于字符级语言模型训练。
    
    示例:
        config = DatasetConfig(name="code_search_net", batch_size=64)
        dataset = CodeSearchNetDataset(config, sequence_length=128)
        train_data, test_data = dataset.get_train_test_split()
    """
    
    def __init__(self, config: DatasetConfig, sequence_length: int):
        """初始化数据集适配器
        
        Args:
            config: 数据集配置
            sequence_length: 序列长度
        """
        self.config = config
        self.sequence_length = sequence_length
        self._dataset = None
        self._tokenizer = None
        self._processed_dataset = None
    
    def load_dataset(self):
        """加载原始数据集
        
        Returns:
            过滤后的数据集
        """
        if self._dataset is None:
            # 加载 CodeSearchNet Python 数据集
            raw_datasets = load_dataset('code_search_net', 'python')
            # 过滤只保留 Apache Spark 相关的代码
            self._dataset = raw_datasets['train'].filter(
                lambda x: 'apache/spark' in x['repository_name']
            )
        return self._dataset
    
    def get_tokenizer(self) -> CharTokenizer:
        """获取字符级分词器
        
        Returns:
            CharTokenizer 实例
        """
        if self._tokenizer is None:
            dataset = self.load_dataset()
            self._tokenizer = CharTokenizer(dataset['whole_func_string'])
        return self._tokenizer
    
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理一个批次的数据
        
        将原始文本转换为模型输入格式。
        
        Args:
            batch: 原始批次数据
            
        Returns:
            包含 'inputs' 和 'labels' 的字典
        """
        tok = self.get_tokenizer()
        text = batch['whole_func_string']
        inputs, labels = [], []
        
        for t in text:
            enc = tok.encode(t)
            enc += [0]  # 添加结束符
            
            # 将文本转换为多个训练样本
            for i in range(len(enc) - self.sequence_length):
                inputs.append(enc[i: i + self.sequence_length])
                labels.append(enc[i + 1: i + 1 + self.sequence_length])
        
        return {'inputs': inputs, 'labels': labels}
    
    def get_vocab_size(self) -> int:
        """获取词表大小
        
        Returns:
            词表大小
        """
        tok = self.get_tokenizer()
        return len(tok)
    
    def get_collate_fn(self) -> Optional[Callable]:
        """获取 DataLoader 的 collate 函数
        
        Returns:
            None（使用默认 collate）
        """
        return None
    
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
            (训练集, 测试集) 元组
        """
        dataset = self.load_dataset()
        
        # 分割数据集
        split = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        
        # 处理数据集
        processed = split.map(
            self.process_batch,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed['train'], processed['test']
    
    def get_processed_dataset(self, device: str = 'cpu'):
        """获取处理后的完整数据集
        
        Args:
            device: 目标设备
            
        Returns:
            处理后的数据集字典，包含 'train' 和 'test'
        """
        if self._processed_dataset is None:
            train_data, test_data = self.get_train_test_split()
            
            # 设置格式为 PyTorch 张量
            train_data.set_format(type='torch', device=device)
            test_data.set_format(type='torch', device=device)
            
            self._processed_dataset = {
                'train': train_data,
                'test': test_data
            }
        
        return self._processed_dataset