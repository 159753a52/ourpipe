"""
P2P 批量通信工具

提供基于 batch_isend_irecv 的异步 P2P 通信原语，
供 1F1B / Hanayo / ZeroBubble 等调度器使用。
"""

import torch
import torch.distributed as dist
from typing import List, Tuple


def make_p2p_send(
    tensor: torch.Tensor,
    peer: int,
    tag: int,
    group=None
) -> Tuple[dist.P2POp, torch.Tensor]:
    """构造一个异步 send P2P 操作

    Args:
        tensor: 要发送的张量
        peer: 目标 rank
        tag: 通信标签
        group: 通信组

    Returns:
        (P2POp, send_buffer) 元组，send_buffer 需要保持引用防止 GC
    """
    buf = tensor.detach().clone().contiguous()
    op = dist.P2POp(dist.isend, buf, peer, group=group, tag=tag)
    return op, buf


def make_p2p_recv(
    shape: tuple,
    peer: int,
    tag: int,
    device: torch.device,
    group=None,
    dtype: torch.dtype = torch.float32
) -> Tuple[dist.P2POp, torch.Tensor]:
    """构造一个异步 recv P2P 操作

    Args:
        shape: 接收缓冲区形状
        peer: 源 rank
        tag: 通信标签
        device: 缓冲区所在设备
        group: 通信组
        dtype: 数据类型

    Returns:
        (P2POp, recv_buffer) 元组
    """
    buf = torch.zeros(*shape, device=device, dtype=dtype)
    op = dist.P2POp(dist.irecv, buf, peer, group=group, tag=tag)
    return op, buf


def execute_p2p_ops(ops: List[dist.P2POp]) -> None:
    """批量提交 P2P 操作并等待完成

    使用 batch_isend_irecv 一次性提交所有操作，
    这是避免 NCCL 死锁的关键：send 和 recv 必须成对出现在同一个 batch 中。

    Args:
        ops: P2P 操作列表
    """
    if not ops:
        return
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
