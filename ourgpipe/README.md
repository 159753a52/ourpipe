# OurGPipe

一个模型无关的 GPipe 流水线并行训练框架，支持多种调度策略和 Orion GPU 调度器集成。

## 项目结构

```
ourgpipe/
├── core/                       # 核心框架模块
│   ├── config.py               # 配置系统 (PipelineConfig, ModelConfig 等)
│   ├── interfaces.py           # 抽象接口定义 (ModelInterface, StageInterface, DatasetInterface)
│   ├── stage.py                # Stage 基类 (通信、缓冲区、CUDA 流管理)
│   ├── registry.py             # 注册表机制 (MODEL_REGISTRY, DATASET_REGISTRY, STAGE_REGISTRY)
│   ├── metrics.py              # 性能指标追踪 (吞吐量、MFU)
│   └── schedulers/             # 调度器模块
│       ├── base.py             # 调度器基类
│       ├── naive.py            # 同步阻塞调度器 (简单，适合调试)
│       └── async_threaded.py   # 异步多线程调度器 (高性能)
├── models/                     # 模型实现
│   └── gpt/                    # GPT 模型
│       ├── model.py            # GPT 模型适配器 (实现 ModelInterface)
│       ├── stage.py            # GPT Stage 实现 (实现 StageInterface)
│       ├── blocks.py           # Transformer 组件 (Block, Attention, FFN)
│       └── dataset.py          # CodeSearchNet 数据集适配器
├── configs/                    # YAML 配置文件
│   ├── gpt_small.yaml          # 小模型配置 (4 阶段)
│   ├── gpt_8stage.yaml         # 8 阶段配置
│   └── gpt_16stage.yaml        # 16 阶段配置
├── gpipe_runner.py             # 主入口程序
├── submit_gpipe_1node.sh       # 单节点 SLURM 提交脚本
├── submit_gpipe_2nodes.sh      # 双节点 SLURM 提交脚本
└── submit_gpipe_4nodes.sh      # 四节点 SLURM 提交脚本
```

## 核心模块说明

### 1. 配置系统 (`core/config.py`)
- `ModelConfig`: 模型参数 (hidden_size, num_layers, num_heads 等)
- `DatasetConfig`: 数据集参数 (batch_size, num_workers)
- `TrainingConfig`: 训练参数 (learning_rate, num_microbatches, epochs)
- `ParallelConfig`: 并行参数 (model_parallel_size, scheduler 类型)
- `PipelineConfig`: 完整配置，支持 YAML 加载/保存

### 2. 抽象接口 (`core/interfaces.py`)
- `ModelInterface`: 模型需实现 `get_model_config()`, `get_stage_partition()`, `create_layer()`
- `StageInterface`: Stage 需实现 `prepare_input()`, `compute_loss()`, `create_sub_model()`
- `DatasetInterface`: 数据集需实现 `load_dataset()`, `get_tokenizer()`, `process_batch()`

### 3. Stage 基类 (`core/stage.py`)
- 通信缓冲区管理 (out_x_buffers, grad_y_buffers)
- CUDA 流管理 (每个 micro-batch 独立计算流)
- 分布式通信 (forward_isend/irecv, backward_isend/irecv)
- Orion 调度器集成

### 4. 调度器 (`core/schedulers/`)
- **NaiveScheduler**: 同步阻塞，串行执行所有 micro-batch，适合调试
- **AsyncThreadedScheduler**: 异步多线程，并行执行 micro-batch，高性能

### 5. 注册表 (`core/registry.py`)
使用装饰器注册模型、数据集、Stage：
```python
@MODEL_REGISTRY.register("gpt")
class GPTModel(ModelInterface):
    ...
```

## 快速开始

### 环境准备
```bash
source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron
cd /home/bingxing2/home/scx9kvs/orion-docker/ourgpipe
```

### 本地运行 (4 GPU)
```bash
# 使用异步调度器 (默认，高性能)
torchrun --nproc_per_node=4 --master_port=29500 \
    gpipe_runner.py --config configs/gpt_small.yaml

# 使用 Naive 调度器 (调试用)
torchrun --nproc_per_node=4 --master_port=29500 \
    gpipe_runner.py --config configs/gpt_small.yaml --scheduler naive
```

### 集群提交 (SLURM)
```bash
mkdir -p logs

# 单节点 4 GPU
sbatch submit_gpipe_1node.sh

# 双节点 8 GPU
sbatch submit_gpipe_2nodes.sh

# 四节点 16 GPU
sbatch submit_gpipe_4nodes.sh
```

### 启用 Orion 调度器
```bash
export USE_ORION_SCHEDULER=1
LD_PRELOAD=/home/bingxing2/home/scx9kvs/orion-docker/kernel_intercept/build/libgpu_scheduler.so \
    torchrun --nproc_per_node=4 --master_port=29500 \
    gpipe_runner.py --config configs/gpt_small_orion.yaml
```

## 配置文件示例

```yaml
model:
  name: gpt
  hidden_size: 512
  num_layers: 16
  num_heads: 16
  sequence_length: 128
  vocab_size: -1  # 从数据集获取

dataset:
  name: code_search_net
  batch_size: 4

training:
  learning_rate: 0.0001
  epochs: 2
  num_microbatches: 4
  profile_iteration: 100
  exit_after_profile: true

parallel:
  model_parallel_size: 4
  scheduler: async_threaded  # 或 naive
  use_orion_scheduler: false
```

## 扩展新模型

1. 在 `models/` 下创建新模型目录
2. 实现 `ModelInterface` 并注册到 `MODEL_REGISTRY`
3. 实现 `StageInterface` 并注册到 `STAGE_REGISTRY`
4. 实现 `DatasetInterface` 并注册到 `DATASET_REGISTRY`
5. 在 `models/__init__.py` 中导入新模型
6. 创建对应的 YAML 配置文件

## 性能分析

训练结束后会输出性能汇总：
- 吞吐量 (tokens/s)
- MFU (Model FLOPs Utilization)
- 每次迭代时间

Profiler 输出保存在 `./profiler_<scheduler>/rank<N>/` 目录，可用 TensorBoard 查看：
```bash
tensorboard --logdir=./profiler_async_threaded
```
