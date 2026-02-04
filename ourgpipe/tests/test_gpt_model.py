"""
GPT 模型适配器测试

测试 GPT 模型的配置、层创建和阶段划分。
"""

import pytest
import torch
import torch.nn as nn
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig
from core.registry import MODEL_REGISTRY, STAGE_REGISTRY

# 导入模型以触发注册
import models.gpt


class TestGPTModel:
    """测试 GPT 模型适配器"""
    
    @pytest.fixture
    def model_config(self):
        """创建测试用的模型配置"""
        return ModelConfig(
            name="gpt",
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            sequence_length=32,
            vocab_size=100,
            extra={'head_size': 32, 'ff_low_rank': None}
        )
    
    @pytest.fixture
    def gpt_model(self, model_config):
        """创建 GPT 模型适配器"""
        return MODEL_REGISTRY.create("gpt", model_config)
    
    def test_model_registered(self):
        """测试模型已注册"""
        assert "gpt" in MODEL_REGISTRY
    
    def test_get_model_config(self, gpt_model):
        """测试获取模型配置"""
        config = gpt_model.get_model_config()
        
        assert "em_tokn" in config
        assert "em_pos" in config
        assert "decoder1" in config
        assert "decoder4" in config
        assert "ln" in config
        assert "lm_head" in config
        
        # 检查参数
        assert config["em_tokn"] == [100, 128]  # [vocab_size, hidden_size]
        assert config["em_pos"] == [32, 128]    # [seq_len, hidden_size]
    
    def test_get_stage_partition(self, gpt_model):
        """测试阶段划分"""
        # 4 层模型，总共 8 层（2 embedding + 4 decoder + ln + lm_head）
        partitions = gpt_model.get_stage_partition(4)
        
        assert len(partitions) == 4
        
        # 检查所有层都被分配
        all_indices = []
        for p in partitions:
            all_indices.extend(p)
        assert sorted(all_indices) == list(range(8))
    
    def test_create_embedding_layer(self, gpt_model):
        """测试创建嵌入层"""
        layer = gpt_model.create_layer("em_tokn", [100, 128])
        assert isinstance(layer, nn.Embedding)
        assert layer.num_embeddings == 100
        assert layer.embedding_dim == 128
    
    def test_create_decoder_layer(self, gpt_model):
        """测试创建解码器层"""
        from models.gpt.blocks import Block
        layer = gpt_model.create_layer("decoder1", [128, 32])
        assert isinstance(layer, Block)
    
    def test_create_layernorm(self, gpt_model):
        """测试创建 LayerNorm"""
        layer = gpt_model.create_layer("ln", [128])
        assert isinstance(layer, nn.LayerNorm)
        assert layer.normalized_shape == (128,)
    
    def test_create_lm_head(self, gpt_model):
        """测试创建语言模型头"""
        layer = gpt_model.create_layer("lm_head", [128, 100])
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 128
        assert layer.out_features == 100
    
    def test_init_model(self, gpt_model):
        """测试初始化完整模型"""
        layers = gpt_model.init_model()
        
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 8  # 2 + 4 + 1 + 1
        
        # 检查层类型
        assert isinstance(layers[0], nn.Embedding)  # em_tokn
        assert isinstance(layers[1], nn.Embedding)  # em_pos
        assert isinstance(layers[-2], nn.LayerNorm)  # ln
        assert isinstance(layers[-1], nn.Linear)     # lm_head
    
    def test_get_hidden_size(self, gpt_model):
        """测试获取隐藏层维度"""
        assert gpt_model.get_hidden_size() == 128
    
    def test_get_sequence_length(self, gpt_model):
        """测试获取序列长度"""
        assert gpt_model.get_sequence_length() == 32
    
    def test_get_vocab_size(self, gpt_model):
        """测试获取词表大小"""
        assert gpt_model.get_vocab_size() == 100


class TestGPTBlocks:
    """测试 GPT 基础组件"""
    
    def test_block_forward(self):
        """测试 Block 前向传播"""
        from models.gpt.blocks import Block
        
        block = Block(emb_size=64, head_size=16, sequence_len=16)
        x = torch.randn(2, 16, 64)  # (batch, seq, hidden)
        
        y = block(x)
        
        assert y.shape == x.shape
    
    def test_feedforward(self):
        """测试 FeedForward"""
        from models.gpt.blocks import FeedForward
        
        ff = FeedForward(emb_size=64)
        x = torch.randn(2, 16, 64)
        
        y = ff(x)
        
        assert y.shape == x.shape
    
    def test_feedforward_lowrank(self):
        """测试低秩 FeedForward"""
        from models.gpt.blocks import FeedForward
        
        ff = FeedForward(emb_size=64, low_rank=16)
        x = torch.randn(2, 16, 64)
        
        y = ff(x)
        
        assert y.shape == x.shape
    
    def test_attention(self):
        """测试注意力机制"""
        from models.gpt.blocks import MaskedMultiHeadAttention
        
        mha = MaskedMultiHeadAttention(emb_size=64, head_size=16, sequence_len=16)
        x = torch.randn(2, 16, 64)
        
        y = mha(x)
        
        assert y.shape == x.shape
    
    def test_char_gpt(self):
        """测试完整 CharGPT 模型"""
        from models.gpt.blocks import CharGPT
        
        model = CharGPT(
            vocab_size=50,
            emb_size=64,
            head_size=16,
            n_layer=2,
            sequence_len=16
        )
        
        x = torch.randint(0, 50, (2, 16))  # (batch, seq)
        
        logits = model(x)
        
        assert logits.shape == (2, 16, 50)  # (batch, seq, vocab)


class TestGPTStage:
    """测试 GPT Stage"""
    
    def test_stage_registered(self):
        """测试 Stage 已注册"""
        assert "gpt" in STAGE_REGISTRY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])