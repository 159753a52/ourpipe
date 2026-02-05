"""
GPipe 流水线框架配置系统

提供配置类和 YAML 配置文件加载功能。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class ModelConfig:
    """模型配置
    
    Attributes:
        name: 模型名称，用于从注册表中查找模型类
        hidden_size: 隐藏层维度
        num_layers: 模型层数
        num_heads: 注意力头数（如适用）
        sequence_length: 序列长度
        vocab_size: 词表大小，-1 表示从数据集获取
        extra: 模型特定的额外参数
    """
    name: str
    hidden_size: int = 512
    num_layers: int = 16
    num_heads: int = 16
    sequence_length: int = 128
    vocab_size: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class DatasetConfig:
    """数据集配置
    
    Attributes:
        name: 数据集名称，用于从注册表中查找数据集类
        batch_size: 批次大小
        num_workers: DataLoader 的 worker 数量
        extra: 数据集特定的额外参数
    """
    name: str
    batch_size: int = 64
    num_workers: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class TrainingConfig:
    """训练配置
    
    Attributes:
        learning_rate: 学习率
        epochs: 训练轮数
        num_microbatches: GPipe micro-batch 数量
        gradient_accumulation: 梯度累积步数
        max_iterations: 最大迭代次数，-1 表示不限制
        profile_iteration: 在哪个迭代进行性能分析，-1 表示不分析
        exit_after_profile: 性能分析后是否退出
    """
    learning_rate: float = 1e-4
    epochs: int = 2
    num_microbatches: int = 4
    gradient_accumulation: int = 1
    max_iterations: int = -1
    profile_iteration: int = -1
    exit_after_profile: bool = False


@dataclass
class ParallelConfig:
    """并行配置
    
    Attributes:
        model_parallel_size: 流水线阶段数（模型并行大小）
        data_parallel_size: 数据并行副本数
        scheduler: 调度器类型，可选 'naive' 或 'async_threaded'
        use_orion_scheduler: 是否使用 Orion 多优先级调度器
    """
    model_parallel_size: int = 4
    data_parallel_size: int = 1
    scheduler: str = "async_threaded"  # 'naive' 或 'async_threaded'
    use_orion_scheduler: bool = False


@dataclass
class PipelineConfig:
    """完整的流水线配置
    
    包含模型、数据集、训练和并行配置。
    支持从 YAML 文件加载和保存。
    
    示例:
        # 从 YAML 加载
        config = PipelineConfig.from_yaml("configs/gpt_small.yaml")
        
        # 编程方式创建
        config = PipelineConfig(
            model=ModelConfig(name="gpt", hidden_size=512),
            dataset=DatasetConfig(name="code_search_net"),
            training=TrainingConfig(learning_rate=1e-4),
            parallel=ParallelConfig(model_parallel_size=4)
        )
    """
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    parallel: ParallelConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """从 YAML 文件加载配置
        
        Args:
            path: YAML 配置文件路径
            
        Returns:
            PipelineConfig: 加载的配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML 解析错误
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for loading YAML configs. Install with: pip install pyyaml")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """从字典创建配置
        
        Args:
            data: 配置字典
            
        Returns:
            PipelineConfig: 配置对象
        """
        model_data = data.get('model', {})
        dataset_data = data.get('dataset', {})
        training_data = data.get('training', {})
        parallel_data = data.get('parallel', {})
        
        return cls(
            model=ModelConfig(**model_data),
            dataset=DatasetConfig(**dataset_data),
            training=TrainingConfig(**training_data),
            parallel=ParallelConfig(**parallel_data)
        )
    
    def to_yaml(self, path: str):
        """保存配置到 YAML 文件
        
        Args:
            path: 目标文件路径
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for saving YAML configs. Install with: pip install pyyaml")
        
        data = self.to_dict()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        import dataclasses
        return {
            'model': dataclasses.asdict(self.model),
            'dataset': dataclasses.asdict(self.dataset),
            'training': dataclasses.asdict(self.training),
            'parallel': dataclasses.asdict(self.parallel)
        }
    
    def validate(self) -> bool:
        """验证配置的有效性
        
        Returns:
            bool: 配置是否有效
            
        Raises:
            ValueError: 配置无效时抛出
        """
        # 验证模型配置
        if not self.model.name:
            raise ValueError("Model name is required")
        if self.model.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.model.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.model.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        
        # 验证数据集配置
        if not self.dataset.name:
            raise ValueError("Dataset name is required")
        if self.dataset.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # 验证训练配置
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training.num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")
        if self.dataset.batch_size % self.training.num_microbatches != 0:
            raise ValueError("batch_size must be divisible by num_microbatches")
        
        # 验证并行配置
        if self.parallel.model_parallel_size <= 0:
            raise ValueError("model_parallel_size must be positive")
        if self.parallel.data_parallel_size <= 0:
            raise ValueError("data_parallel_size must be positive")
        
        # 验证调度器类型
        valid_schedulers = ['naive', 'async_threaded']
        if self.parallel.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got '{self.parallel.scheduler}'")
        
        return True
    
    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  model={self.model.name}(hidden={self.model.hidden_size}, layers={self.model.num_layers}),\n"
            f"  dataset={self.dataset.name}(batch={self.dataset.batch_size}),\n"
            f"  training(lr={self.training.learning_rate}, epochs={self.training.epochs}, microbatches={self.training.num_microbatches}),\n"
            f"  parallel(mp={self.parallel.model_parallel_size}, dp={self.parallel.data_parallel_size}, scheduler={self.parallel.scheduler}, orion={self.parallel.use_orion_scheduler})\n"
            f")"
        )