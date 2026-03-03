"""
GPT Hanayo Stage 实现

Hanayo 双向流水线的 Stage：
- Wave A: stage1(emb_A+dec) -> stage2(dec) -> ... -> stageN(dec+ln_A+head_A) -> loss
- Wave B: stageN(emb_B+dec) -> ... -> stage1(dec+ln_B+head_B) -> loss
- decoder 层在两个 wave 间共享
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from core.stage import BaseStage
from core.registry import STAGE_REGISTRY
from core.config import PipelineConfig


@STAGE_REGISTRY.register("gpt_hanayo")
class GPTHanayoStage(BaseStage):
    """GPT Hanayo 双向 Stage

    第一阶段 (entry for Wave A, exit for Wave B):
        - token_embedding_A, position_embedding_A  (Wave A 入口)
        - decoder layers (共享)
        - ln_B, lm_head_B  (Wave B 出口)

    最后阶段 (exit for Wave A, entry for Wave B):
        - decoder layers (共享)
        - ln_A, lm_head_A  (Wave A 出口)
        - token_embedding_B, position_embedding_B  (Wave B 入口)

    中间阶段:
        - decoder layers (共享，双向都经过)
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
        # Wave 缓存（dict of dict）
        self.fwd_cache_wave = {'A': {}, 'B': {}}
        self.input_cache_wave = {'A': {}, 'B': {}}

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

        # 重新创建优化器，包含所有参数（BaseStage 只包含了 sub_model 的参数）
        import torch.optim as optim
        all_params = list(self.sub_model.parameters())
        if self.stage_id == 1:
            all_params += list(self.token_embedding_A.parameters())
            all_params += list(self.position_embedding_A.parameters())
            all_params += list(self.ln_B.parameters())
            all_params += list(self.lm_head_B.parameters())
        elif self.stage_id == self.model_parallel_size:
            all_params += list(self.ln_A.parameters())
            all_params += list(self.lm_head_A.parameters())
            all_params += list(self.token_embedding_B.parameters())
            all_params += list(self.position_embedding_B.parameters())
        self.optimizer = optim.Adam(all_params, lr=config.training.learning_rate)

    def create_sub_model(
        self,
        model_adapter,
        model_config: Dict[str, Any],
        layer_indices: List[int]
    ) -> nn.Module:
        """创建 Hanayo 双向子模型"""
        layer_names = list(model_config.keys())
        print(f"[CREATE_MODEL] rank={self.global_rank} stage={self.stage_id} layer_indices={layer_indices}", flush=True)

        if self.stage_id == 1:
            print(f"[CREATE_MODEL] Stage 1: Creating Wave A entry + Wave B exit", flush=True)
            # Wave A 入口：embedding
            tokn_name = layer_names[layer_indices[0]]
            pos_name = layer_names[layer_indices[1]]
            print(f"[CREATE_MODEL] Creating token_embedding_A: {tokn_name}", flush=True)
            self.token_embedding_A = model_adapter.create_layer(
                tokn_name, model_config[tokn_name])
            print(f"[CREATE_MODEL] Creating position_embedding_A: {pos_name}", flush=True)
            self.position_embedding_A = model_adapter.create_layer(
                pos_name, model_config[pos_name])

            # 共享 decoder layers
            decoder_layers = []
            for idx in layer_indices[2:]:
                layer_name = layer_names[idx]
                print(f"[CREATE_MODEL] Creating decoder layer: {layer_name}", flush=True)
                layer = model_adapter.create_layer(
                    layer_name, model_config[layer_name])
                decoder_layers.append(layer)
            decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
            print(f"[CREATE_MODEL] Created {len(decoder_layers)} decoder layers", flush=True)

            # Wave B 出口：ln + lm_head
            hidden_size = model_config[layer_names[0]][1]  # em_tokn 的第二个参数
            vocab_size = model_config[layer_names[0]][0]
            print(f"[CREATE_MODEL] Creating ln_B and lm_head_B: hidden={hidden_size}, vocab={vocab_size}", flush=True)
            self.ln_B = nn.LayerNorm(hidden_size)
            self.lm_head_B = nn.Linear(hidden_size, vocab_size)

            return decoder

        elif self.stage_id == self.model_parallel_size:
            print(f"[CREATE_MODEL] Stage {self.stage_id}: Creating Wave A exit + Wave B entry", flush=True)
            # 共享 decoder layers (不含最后两层 ln + lm_head)
            decoder_layers = []
            for idx in layer_indices[:-2]:
                layer_name = layer_names[idx]
                print(f"[CREATE_MODEL] Creating decoder layer: {layer_name}", flush=True)
                layer = model_adapter.create_layer(
                    layer_name, model_config[layer_name])
                decoder_layers.append(layer)
            decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
            print(f"[CREATE_MODEL] Created {len(decoder_layers)} decoder layers", flush=True)

            # Wave A 出口：ln + lm_head
            ln_name = layer_names[layer_indices[-2]]
            head_name = layer_names[layer_indices[-1]]
            print(f"[CREATE_MODEL] Creating ln_A: {ln_name}", flush=True)
            self.ln_A = model_adapter.create_layer(
                ln_name, model_config[ln_name])
            print(f"[CREATE_MODEL] Creating lm_head_A: {head_name}", flush=True)
            self.lm_head_A = model_adapter.create_layer(
                head_name, model_config[head_name])

            # Wave B 入口：embedding
            hidden_size = model_config[layer_names[0]][1]
            vocab_size = model_config[layer_names[0]][0]
            seq_len = model_config[layer_names[1]][0]
            print(f"[CREATE_MODEL] Creating token_embedding_B and position_embedding_B: vocab={vocab_size}, hidden={hidden_size}, seq_len={seq_len}", flush=True)
            self.token_embedding_B = nn.Embedding(vocab_size, hidden_size)
            self.position_embedding_B = nn.Embedding(seq_len, hidden_size)

            return decoder

        else:
            print(f"[CREATE_MODEL] Stage {self.stage_id}: Creating middle stage (decoder only)", flush=True)
            # 中间阶段：只有 decoder layers
            layers = []
            for idx in layer_indices:
                layer_name = layer_names[idx]
                print(f"[CREATE_MODEL] Creating decoder layer: {layer_name}", flush=True)
                layer = model_adapter.create_layer(
                    layer_name, model_config[layer_name])
                layers.append(layer)
            print(f"[CREATE_MODEL] Created {len(layers)} decoder layers", flush=True)
            return nn.Sequential(*layers)

    def prepare_input(self, x: torch.Tensor, mb_idx: int) -> torch.Tensor:
        """标准 prepare_input（Wave A 第一阶段用）"""
        if self.stage_id == 1:
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            return self.token_embedding_A(x) + self.position_embedding_A(pos)
        return x

    def compute_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算语言模型损失"""
        logits = output.transpose(-2, -1)
        return F.cross_entropy(logits, labels)

    def to(self, device):
        """移动所有模块到设备"""
        device = torch.device(device)
        self.sub_model.to(device)
        self.out_x_buffers = [b.to(device) for b in self.out_x_buffers]
        self.grad_y_buffers = [b.to(device) for b in self.grad_y_buffers]

        if self.stage_id == 1:
            self.token_embedding_A.to(device)
            self.position_embedding_A.to(device)
            self.ln_B.to(device)
            self.lm_head_B.to(device)
        elif self.stage_id == self.model_parallel_size:
            self.ln_A.to(device)
            self.lm_head_A.to(device)
            self.token_embedding_B.to(device)
            self.position_embedding_B.to(device)

    def train(self):
        """设置训练模式"""
        self.sub_model.train()
        self.is_training = True
        if self.stage_id == 1:
            self.token_embedding_A.train()
            self.position_embedding_A.train()
            self.ln_B.train()
            self.lm_head_B.train()
        elif self.stage_id == self.model_parallel_size:
            self.ln_A.train()
            self.lm_head_A.train()
            self.token_embedding_B.train()
            self.position_embedding_B.train()

    def zero_grad(self):
        """清零梯度和缓存"""
        self.optimizer.zero_grad()
        for w in ('A', 'B'):
            self.fwd_cache_wave[w].clear()
            self.input_cache_wave[w].clear()

    # ==================== Hanayo 特有方法 ====================

    def is_entry(self, wave: str) -> bool:
        """判断当前 stage 是否为指定 wave 的入口"""
        return ((wave == 'A' and self.stage_id == 1) or
                (wave == 'B' and self.stage_id == self.model_parallel_size))

    def is_exit(self, wave: str) -> bool:
        """判断当前 stage 是否为指定 wave 的出口"""
        return ((wave == 'A' and self.stage_id == self.model_parallel_size) or
                (wave == 'B' and self.stage_id == 1))

    def _embed(self, x: torch.Tensor, wave: str) -> torch.Tensor:
        """对输入做 embedding"""
        print(f"[EMBED_DEBUG] rank={self.global_rank} stage={self.stage_id} wave={wave} x.shape={x.shape}", flush=True)
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        if wave == 'A':
            print(f"[EMBED_DEBUG] Using token_embedding_A and position_embedding_A", flush=True)
            tok_emb = self.token_embedding_A(x)
            pos_emb = self.position_embedding_A(pos)
            result = tok_emb + pos_emb
            print(f"[EMBED_DEBUG] result.shape={result.shape}", flush=True)
            return result
        else:
            print(f"[EMBED_DEBUG] Using token_embedding_B and position_embedding_B", flush=True)
            tok_emb = self.token_embedding_B(x)
            pos_emb = self.position_embedding_B(pos)
            result = tok_emb + pos_emb
            print(f"[EMBED_DEBUG] result.shape={result.shape}", flush=True)
            return result

    def _head(self, y: torch.Tensor, wave: str) -> torch.Tensor:
        """对输出做 ln + lm_head"""
        print(f"[HEAD_DEBUG] rank={self.global_rank} stage={self.stage_id} wave={wave} y.shape={y.shape}", flush=True)
        if wave == 'A':
            print(f"[HEAD_DEBUG] Using ln_A and lm_head_A", flush=True)
            ln_out = self.ln_A(y)
            print(f"[HEAD_DEBUG] After ln_A: ln_out.shape={ln_out.shape}", flush=True)
            result = self.lm_head_A(ln_out)
            print(f"[HEAD_DEBUG] After lm_head_A: result.shape={result.shape}", flush=True)
            return result
        else:
            print(f"[HEAD_DEBUG] Using ln_B and lm_head_B", flush=True)
            ln_out = self.ln_B(y)
            print(f"[HEAD_DEBUG] After ln_B: ln_out.shape={ln_out.shape}", flush=True)
            result = self.lm_head_B(ln_out)
            print(f"[HEAD_DEBUG] After lm_head_B: result.shape={result.shape}", flush=True)
            return result

    def forward_wave(
        self,
        wave: str,
        mb_idx: int,
        micro_input: Optional[torch.Tensor] = None,
        recv_buf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """指定 wave 的前向计算

        Args:
            wave: 'A' 或 'B'
            mb_idx: micro-batch 索引
            micro_input: 入口阶段的原始输入
            recv_buf: 非入口阶段从上游接收的 activation

        Returns:
            输出张量
        """
        print(f"[FW_DEBUG] rank={self.global_rank} stage={self.stage_id} wave={wave} mb={mb_idx} "
              f"entry={self.is_entry(wave)} exit={self.is_exit(wave)}", flush=True)
        
        if self.is_entry(wave):
            print(f"[FW_DEBUG] Entry path: micro_input.shape={micro_input.shape}", flush=True)
            x_emb = self._embed(micro_input, wave)
            print(f"[FW_DEBUG] After embed: x_emb.shape={x_emb.shape}", flush=True)
            self.input_cache_wave[wave][mb_idx] = x_emb
            print(f"[FW_DEBUG] Before sub_model", flush=True)
            y = self.sub_model(x_emb)
            print(f"[FW_DEBUG] After sub_model: y.shape={y.shape}", flush=True)
        elif self.is_exit(wave):
            print(f"[FW_DEBUG] Exit path: recv_buf.shape={recv_buf.shape}", flush=True)
            x = recv_buf.requires_grad_(True)
            self.input_cache_wave[wave][mb_idx] = x
            print(f"[FW_DEBUG] Before sub_model", flush=True)
            y_hidden = self.sub_model(x)
            print(f"[FW_DEBUG] After sub_model: y_hidden.shape={y_hidden.shape}", flush=True)
            print(f"[FW_DEBUG] Before head", flush=True)
            y = self._head(y_hidden, wave)
            print(f"[FW_DEBUG] After head: y.shape={y.shape}", flush=True)
        else:
            print(f"[FW_DEBUG] Middle path: recv_buf.shape={recv_buf.shape}", flush=True)
            x = recv_buf.requires_grad_(True)
            self.input_cache_wave[wave][mb_idx] = x
            print(f"[FW_DEBUG] Before sub_model, x.device={x.device}, x.requires_grad={x.requires_grad}", flush=True)
            print(f"[FW_DEBUG] sub_model type: {type(self.sub_model)}", flush=True)
            print(f"[FW_DEBUG] sub_model device: {next(self.sub_model.parameters()).device if len(list(self.sub_model.parameters())) > 0 else 'no params'}", flush=True)
            print(f"[FW_DEBUG] Calling sub_model...", flush=True)
            y = self.sub_model(x)
            print(f"[FW_DEBUG] After sub_model: y.shape={y.shape}, y.device={y.device}", flush=True)

        print(f"[FW_DEBUG] Before retain_grad", flush=True)
        y.retain_grad()
        print(f"[FW_DEBUG] After retain_grad", flush=True)
        self.fwd_cache_wave[wave][mb_idx] = y
        print(f"[FW_DEBUG] Done: returning y.shape={y.shape}", flush=True)
        return y

    def backward_wave(
        self,
        wave: str,
        mb_idx: int,
        grad_or_label
    ) -> Tuple[Optional[float], Optional[torch.Tensor]]:
        """指定 wave 的反向计算

        Args:
            wave: 'A' 或 'B'
            mb_idx: micro-batch 索引
            grad_or_label: 出口阶段传入 label，其他阶段传入梯度

        Returns:
            (loss_value, input_grad) 元组
        """
        cached_output = self.fwd_cache_wave[wave].pop(mb_idx)

        if self.is_exit(wave):
            logits = cached_output.transpose(-2, -1)
            loss = F.cross_entropy(logits, grad_or_label)
            loss.backward()
            cached_input = self.input_cache_wave[wave].pop(mb_idx)
            return loss.item(), cached_input.grad
        elif self.is_entry(wave):
            cached_output.backward(grad_or_label)
            self.input_cache_wave[wave].pop(mb_idx, None)
            return None, None
        else:
            cached_output.backward(grad_or_label)
            cached_input = self.input_cache_wave[wave].pop(mb_idx)
            return None, cached_input.grad
