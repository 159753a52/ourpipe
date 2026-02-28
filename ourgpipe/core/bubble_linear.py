"""
BubbleLinear — ZeroBubble 专用的自定义 Linear 层

将 nn.Linear 的反向传播拆分为：
- B (Input Gradient): 立即计算，因为上一层的反向传播依赖它
- W (Weight Gradient): 通过 WeightGradStore 延迟执行，填充 pipeline bubble

用法：
    from core.bubble_linear import convert_to_bubble_model
    convert_to_bubble_model(sub_model)  # 将模型中所有 nn.Linear 替换为 BubbleLinear
"""

import math
import torch
import torch.nn as nn
from torch.profiler import record_function

from .weight_grad_store import WeightGradStore


class BubbleLinearFunction(torch.autograd.Function):
    """自定义 autograd Function，实现 B/W 分离"""

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None

        # 1. 计算 B (Input Gradient) — 立即执行
        if ctx.needs_input_grad[0]:
            with record_function("Bubble_Backward_B"):
                grad_input = grad_output.matmul(weight)

        # 2. 定义 W (Weight Gradient) 的计算闭包
        def compute_w_grad():
            with torch.no_grad():
                input_reshaped = input.reshape(-1, input.shape[-1])
                grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

                if ctx.needs_input_grad[1]:
                    gw = grad_output_reshaped.t().matmul(input_reshaped)
                    if weight.grad is None:
                        weight.grad = gw
                    else:
                        weight.grad += gw

                if bias is not None and ctx.needs_input_grad[2]:
                    gb = grad_output_reshaped.sum(dim=0)
                    if bias.grad is None:
                        bias.grad = gb
                    else:
                        bias.grad += gb

        # 3. 根据 split_bw 开关决定立即执行还是延迟
        WeightGradStore.put(compute_w_grad)

        # weight/bias 的梯度通过 side-effect 更新，这里返回 None
        return grad_input, None, None


class BubbleLinear(nn.Module):
    """替换 nn.Linear，使用 BubbleLinearFunction 实现 B/W 分离"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return BubbleLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')


def convert_to_bubble_model(module: nn.Module) -> None:
    """将模型中所有 nn.Linear 替换为 BubbleLinear（原地修改）

    Args:
        module: 要转换的模型模块
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = BubbleLinear(
                child.in_features, child.out_features,
                child.bias is not None
            )
            new_layer.weight.data = child.weight.data
            if child.bias is not None:
                new_layer.bias.data = child.bias.data
            setattr(module, name, new_layer)
        else:
            convert_to_bubble_model(child)
