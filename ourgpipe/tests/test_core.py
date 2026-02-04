"""
核心框架单元测试

测试配置系统、注册表和接口。
"""

import pytest
import os
import sys
import tempfile

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    PipelineConfig, ModelConfig, DatasetConfig, 
    TrainingConfig, ParallelConfig
)
from core.registry import Registry, MODEL_REGISTRY, DATASET_REGISTRY, STAGE_REGISTRY


class TestRegistry:
    """测试注册表功能"""
    
    def test_register_and_get(self):
        """测试注册和获取"""
        registry = Registry("test")
        
        @registry.register("test_class")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        assert "test_class" in registry
        assert registry.get("test_class") == TestClass
    
    def test_create_instance(self):
        """测试创建实例"""
        registry = Registry("test")
        
        @registry.register("my_class")
        class MyClass:
            def __init__(self, x, y=10):
                self.x = x
                self.y = y
        
        instance = registry.create("my_class", 5, y=20)
        assert instance.x == 5
        assert instance.y == 20
    
    def test_duplicate_registration(self):
        """测试重复注册"""
        registry = Registry("test")
        
        @registry.register("dup")
        class First:
            pass
        
        with pytest.raises(ValueError):
            @registry.register("dup")
            class Second:
                pass
    
    def test_force_override(self):
        """测试强制覆盖"""
        registry = Registry("test")
        
        @registry.register("override")
        class First:
            pass
        
        @registry.register("override", force=True)
        class Second:
            pass
        
        assert registry.get("override") == Second
    
    def test_list_registered(self):
        """测试列出注册项"""
        registry = Registry("test")
        
        @registry.register("a")
        class A:
            pass
        
        @registry.register("b")
        class B:
            pass
        
        registered = registry.list_registered()
        assert "a" in registered
        assert "b" in registered
    
    def test_not_found(self):
        """测试未找到的情况"""
        registry = Registry("test")
        
        assert registry.get("nonexistent") is None
        
        with pytest.raises(ValueError):
            registry.create("nonexistent")


class TestConfig:
    """测试配置系统"""
    
    def test_model_config(self):
        """测试模型配置"""
        config = ModelConfig(
            name="gpt",
            hidden_size=512,
            num_layers=16
        )
        assert config.name == "gpt"
        assert config.hidden_size == 512
        assert config.num_layers == 16
        assert config.extra == {}
    
    def test_pipeline_config(self):
        """测试完整配置"""
        config = PipelineConfig(
            model=ModelConfig(name="gpt", hidden_size=256),
            dataset=DatasetConfig(name="test", batch_size=32),
            training=TrainingConfig(learning_rate=1e-3),
            parallel=ParallelConfig(model_parallel_size=2)
        )
        
        assert config.model.name == "gpt"
        assert config.dataset.batch_size == 32
        assert config.training.learning_rate == 1e-3
        assert config.parallel.model_parallel_size == 2
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            'model': {'name': 'gpt', 'hidden_size': 128},
            'dataset': {'name': 'test', 'batch_size': 16},
            'training': {'learning_rate': 0.001},
            'parallel': {'model_parallel_size': 4}
        }
        
        config = PipelineConfig.from_dict(data)
        assert config.model.name == 'gpt'
        assert config.model.hidden_size == 128
        assert config.dataset.batch_size == 16
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = PipelineConfig(
            model=ModelConfig(name="gpt"),
            dataset=DatasetConfig(name="test"),
            training=TrainingConfig(),
            parallel=ParallelConfig()
        )
        
        data = config.to_dict()
        assert data['model']['name'] == 'gpt'
        assert 'dataset' in data
        assert 'training' in data
        assert 'parallel' in data
    
    def test_config_yaml_roundtrip(self):
        """测试 YAML 保存和加载"""
        config = PipelineConfig(
            model=ModelConfig(name="gpt", hidden_size=256),
            dataset=DatasetConfig(name="test", batch_size=32),
            training=TrainingConfig(learning_rate=1e-4),
            parallel=ParallelConfig(model_parallel_size=4)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded = PipelineConfig.from_yaml(f.name)
        
        assert loaded.model.name == config.model.name
        assert loaded.model.hidden_size == config.model.hidden_size
        assert loaded.dataset.batch_size == config.dataset.batch_size
        
        os.unlink(f.name)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = PipelineConfig(
            model=ModelConfig(name="gpt", hidden_size=512),
            dataset=DatasetConfig(name="test", batch_size=64),
            training=TrainingConfig(num_microbatches=4),
            parallel=ParallelConfig()
        )
        assert valid_config.validate() == True
        
        # 无效配置：batch_size 不能被 num_microbatches 整除
        invalid_config = PipelineConfig(
            model=ModelConfig(name="gpt"),
            dataset=DatasetConfig(name="test", batch_size=65),
            training=TrainingConfig(num_microbatches=4),
            parallel=ParallelConfig()
        )
        with pytest.raises(ValueError):
            invalid_config.validate()


class TestGlobalRegistries:
    """测试全局注册表"""
    
    def test_model_registry_exists(self):
        """测试模型注册表存在"""
        assert MODEL_REGISTRY is not None
        assert MODEL_REGISTRY.name == "models"
    
    def test_dataset_registry_exists(self):
        """测试数据集注册表存在"""
        assert DATASET_REGISTRY is not None
        assert DATASET_REGISTRY.name == "datasets"
    
    def test_stage_registry_exists(self):
        """测试 Stage 注册表存在"""
        assert STAGE_REGISTRY is not None
        assert STAGE_REGISTRY.name == "stages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])