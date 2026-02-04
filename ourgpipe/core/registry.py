"""
GPipe 流水线框架注册机制

提供模型、数据集和 Stage 的注册表，支持通过名称动态创建实例。
"""

from typing import Dict, Type, Any, Optional, List


class Registry:
    """通用注册表
    
    用于注册和管理类，支持通过名称查找和创建实例。
    
    示例:
        # 创建注册表
        MODEL_REGISTRY = Registry("models")
        
        # 使用装饰器注册
        @MODEL_REGISTRY.register("gpt")
        class GPTModel:
            pass
        
        # 创建实例
        model = MODEL_REGISTRY.create("gpt", config)
    """
    
    def __init__(self, name: str):
        """初始化注册表
        
        Args:
            name: 注册表名称，用于错误信息
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str, force: bool = False):
        """装饰器：注册一个类
        
        Args:
            name: 注册名称
            force: 是否强制覆盖已存在的注册
            
        Returns:
            装饰器函数
            
        Raises:
            ValueError: 名称已存在且 force=False
            
        示例:
            @MODEL_REGISTRY.register("gpt")
            class GPTModel(ModelInterface):
                pass
        """
        def decorator(cls):
            if name in self._registry and not force:
                raise ValueError(
                    f"'{name}' is already registered in {self.name}. "
                    f"Use force=True to override."
                )
            self._registry[name] = cls
            # 在类上添加注册名称属性
            cls._registry_name = name
            return cls
        return decorator
    
    def register_class(self, name: str, cls: Type, force: bool = False):
        """直接注册一个类（非装饰器方式）
        
        Args:
            name: 注册名称
            cls: 要注册的类
            force: 是否强制覆盖
            
        Raises:
            ValueError: 名称已存在且 force=False
        """
        if name in self._registry and not force:
            raise ValueError(
                f"'{name}' is already registered in {self.name}. "
                f"Use force=True to override."
            )
        self._registry[name] = cls
        cls._registry_name = name
    
    def get(self, name: str) -> Optional[Type]:
        """获取注册的类
        
        Args:
            name: 注册名称
            
        Returns:
            注册的类，如果不存在返回 None
        """
        return self._registry.get(name)
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """创建注册类的实例
        
        Args:
            name: 注册名称
            *args: 传递给构造函数的位置参数
            **kwargs: 传递给构造函数的关键字参数
            
        Returns:
            创建的实例
            
        Raises:
            ValueError: 名称未注册
        """
        cls = self.get(name)
        if cls is None:
            available = self.list_registered()
            raise ValueError(
                f"'{name}' not found in {self.name}. "
                f"Available: {available}"
            )
        return cls(*args, **kwargs)
    
    def list_registered(self) -> List[str]:
        """列出所有注册的名称
        
        Returns:
            注册名称列表
        """
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        """检查名称是否已注册
        
        Args:
            name: 要检查的名称
            
        Returns:
            是否已注册
        """
        return name in self._registry
    
    def __len__(self) -> int:
        """返回注册的类数量"""
        return len(self._registry)
    
    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', registered={self.list_registered()})"


# ============================================================================
# 全局注册表实例
# ============================================================================

# 模型注册表
MODEL_REGISTRY = Registry("models")

# 数据集注册表
DATASET_REGISTRY = Registry("datasets")

# Stage 注册表
STAGE_REGISTRY = Registry("stages")


def get_model(name: str, *args, **kwargs):
    """便捷函数：从模型注册表创建模型实例
    
    Args:
        name: 模型名称
        *args, **kwargs: 传递给模型构造函数的参数
        
    Returns:
        模型实例
    """
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def get_dataset(name: str, *args, **kwargs):
    """便捷函数：从数据集注册表创建数据集实例
    
    Args:
        name: 数据集名称
        *args, **kwargs: 传递给数据集构造函数的参数
        
    Returns:
        数据集实例
    """
    return DATASET_REGISTRY.create(name, *args, **kwargs)


def get_stage(name: str, *args, **kwargs):
    """便捷函数：从 Stage 注册表创建 Stage 实例
    
    Args:
        name: Stage 名称
        *args, **kwargs: 传递给 Stage 构造函数的参数
        
    Returns:
        Stage 实例
    """
    return STAGE_REGISTRY.create(name, *args, **kwargs)


def list_models() -> List[str]:
    """列出所有注册的模型"""
    return MODEL_REGISTRY.list_registered()


def list_datasets() -> List[str]:
    """列出所有注册的数据集"""
    return DATASET_REGISTRY.list_registered()


def list_stages() -> List[str]:
    """列出所有注册的 Stage"""
    return STAGE_REGISTRY.list_registered()