#!/usr/bin/env python3
"""
GPipe 流水线并行训练 - 兼容入口

此文件为向后兼容保留，实际逻辑已迁移到 pipeline_runner.py。
pipeline_runner.py 支持所有调度策略：naive, async_threaded, 1f1b, hanayo, zerobubble。

使用方法:
    # 以下两种方式等价
    torchrun --nproc_per_node=4 --master_port=29500 gpipe_runner.py --config configs/gpt_small_naive.yaml
    torchrun --nproc_per_node=4 --master_port=29500 pipeline_runner.py --config configs/gpt_small_naive.yaml
"""

from pipeline_runner import main

if __name__ == "__main__":
    main()
