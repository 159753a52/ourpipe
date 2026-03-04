# OurPipe Benchmark 自动化方案设计

## 一、问题分析

### 1.1 当前现状

目前 ourgpipe 框架已经具备：
- **5 种调度器**：`naive`(GPipe)、`async_threaded`、`1f1b`、`hanayo`、`zerobubble`
- **2 种模型**：`gpt`、`llama`
- **指标追踪器** [`MetricsTracker`](ourgpipe/core/metrics.py:65)：已能计算 throughput、MFU、time_per_iteration 等
- **配置系统** [`PipelineConfig`](ourgpipe/core/config.py:97)：YAML 驱动，支持命令行覆盖
- **运行入口** [`pipeline_runner.py`](ourgpipe/pipeline_runner.py:109)：统一的 `main()` 入口

**但存在以下痛点：**
1. 每次实验需要手动创建一个 YAML 配置文件
2. 实验结果只通过 [`print_summary()`](ourgpipe/core/metrics.py:266) 输出到终端/日志，无结构化存储
3. 无法方便地对比不同模型大小 × 不同调度器的结果
4. 要导入 Excel 需要手动从日志中提取数据

### 1.2 目标

设计一套 benchmark 方案，实现：
1. **一键运行**多组实验（不同模型大小 × 不同调度器）
2. **自动收集**所有 metrics 到**结构化文件（CSV）**
3. CSV 可以直接**拖入 Excel / Google Sheets**，无需手动整理

---

## 二、方案概览

整体方案分为 **3 个改动层**，从底层到顶层依次是：

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: benchmark_runner.py (批量实验编排)          │
│     - 定义实验矩阵（模型大小 × 调度器）                  │
│     - 自动生成临时 YAML                                │
│     - 调用 torchrun 执行每个实验                        │
│     - 解析日志 → 汇总到 CSV                            │
├─────────────────────────────────────────────────────┤
│  Layer 2: pipeline_runner.py (小幅修改)               │
│     - 添加 --output-json 参数                         │
│     - rank 0 训练结束后输出 JSON 格式的 metrics         │
├─────────────────────────────────────────────────────┤
│  Layer 1: MetricsTracker (小幅扩展)                   │
│     - 添加 to_json() / save_json() 方法               │
│     - 在 summary 中增加配置信息字段                     │
└─────────────────────────────────────────────────────┘
```

---

## 三、逐层设计详解

### 3.1 Layer 1：扩展 `MetricsTracker`

**文件**：[`ourgpipe/core/metrics.py`](ourgpipe/core/metrics.py)

**当前问题**：[`get_summary()`](ourgpipe/core/metrics.py:219) 返回的字典只有性能数据，缺少"这个实验是什么配置"的信息。而且没有持久化方法。

**修改内容**：

#### 3.1.1 新增 `get_full_summary()` 方法

在 [`MetricsTracker`](ourgpipe/core/metrics.py:65) 类中新增一个方法，将**配置信息 + 性能指标**合并为一个扁平字典：

```python
def get_full_summary(
    self,
    model_params: int,
    config: 'PipelineConfig',
    flops_per_token: Optional[int] = None
) -> Dict[str, Any]:
    """获取包含配置信息的完整指标汇总（适合导出 CSV 的一行数据）"""
    perf = self.get_summary(model_params, flops_per_token)
    
    return {
        # ===== 实验配置 =====
        'model_name': config.model.name,
        'hidden_size': config.model.hidden_size,
        'num_layers': config.model.num_layers,
        'num_heads': config.model.num_heads,
        'sequence_length': config.model.sequence_length,
        'vocab_size': config.model.vocab_size,
        'batch_size': config.dataset.batch_size,
        'num_microbatches': config.training.num_microbatches,
        'scheduler': config.parallel.scheduler,
        'use_orion': config.parallel.use_orion_scheduler,
        'num_stages': config.parallel.model_parallel_size,
        'data_parallel_size': config.parallel.data_parallel_size,
        
        # ===== 性能指标 =====
        **perf,
    }
```

**为什么这样做**：把"这条数据是什么实验跑出来的"和"跑出了什么结果"放在同一行，导出 CSV 后 Excel 里一目了然。

#### 3.1.2 新增 `save_json()` 方法

```python
def save_json(
    self,
    path: str,
    model_params: int,
    config: 'PipelineConfig',
    flops_per_token: Optional[int] = None
):
    """将完整指标保存为 JSON 文件"""
    import json
    summary = self.get_full_summary(model_params, config, flops_per_token)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
```

**为什么用 JSON 而不是直接 CSV**：每次实验只产出一行数据。如果直接写 CSV，多进程/多任务并发写同一个 CSV 容易出问题。用 JSON 文件（每次实验一个文件），最后由上层脚本合并成 CSV 更安全。

---

### 3.2 Layer 2：修改 `pipeline_runner.py`

**文件**：[`ourgpipe/pipeline_runner.py`](ourgpipe/pipeline_runner.py)

**修改内容**：

#### 3.2.1 新增命令行参数 `--output-json`

在 [`main()`](ourgpipe/pipeline_runner.py:109) 的 `argparse` 部分添加：

```python
parser.add_argument('--output-json', type=str, default=None,
                    help='Path to save metrics JSON (rank 0 only)')
```

#### 3.2.2 训练结束后保存 JSON

在现有 [`print_summary()`](ourgpipe/pipeline_runner.py:393) 调用之后添加 JSON 保存逻辑：

```python
# 输出训练汇总（只在 rank 0）
if global_rank == 0:
    metrics_tracker.print_summary(
        model_params=total_model_params,
        flops_per_token=flops_per_token
    )
    
    # ===== 新增：保存 JSON =====
    if args.output_json:
        metrics_tracker.save_json(
            path=args.output_json,
            model_params=total_model_params,
            config=config,
            flops_per_token=flops_per_token
        )
        print(f"Metrics saved to: {args.output_json}", flush=True)
```

**同样需要在 [`exit_after_profile`](ourgpipe/pipeline_runner.py:374) 的提前退出分支中添加相同的 JSON 保存逻辑。**

**为什么这样做**：
- 改动量极小（只加 ~10 行代码）
- 完全向后兼容（不加 `--output-json` 则行为不变）
- 保证只有 rank 0 写文件，避免多进程冲突

---

### 3.3 Layer 3：新建 `benchmark_runner.py`

**文件**：新建 `ourgpipe/benchmark_runner.py`

这是核心编排脚本，职责是：
1. 定义实验矩阵
2. 为每个实验动态生成临时 YAML 配置
3. 调用 `torchrun` 运行实验
4. 收集所有 JSON → 合并为一个 CSV

#### 3.3.1 实验矩阵定义

```python
# ===== 实验矩阵 =====
MODEL_CONFIGS = {
    # 名称 → (hidden_size, num_layers, num_heads, batch_size, sequence_length)
    "gpt-small":  {"name": "gpt",   "hidden_size": 512,  "num_layers": 16, "num_heads": 16, "batch_size": 64,  "seq_len": 128},
    "gpt-medium": {"name": "gpt",   "hidden_size": 1024, "num_layers": 24, "num_heads": 16, "batch_size": 32,  "seq_len": 256},
    "gpt-large":  {"name": "gpt",   "hidden_size": 1536, "num_layers": 24, "num_heads": 24, "batch_size": 16,  "seq_len": 512},
    "gpt-xl":     {"name": "gpt",   "hidden_size": 2048, "num_layers": 32, "num_heads": 32, "batch_size": 8,   "seq_len": 512},
    "llama-small":{"name": "llama", "hidden_size": 512,  "num_layers": 16, "num_heads": 8,  "batch_size": 64,  "seq_len": 128},
    "llama-medium":{"name": "llama","hidden_size": 1024, "num_layers": 24, "num_heads": 16, "batch_size": 32,  "seq_len": 256},
}

SCHEDULERS = ["naive", "1f1b", "zerobubble", "hanayo"]

NUM_STAGES = 4           # 流水线阶段数
NUM_MICROBATCHES = 8     # micro-batch 数量
MAX_ITERATIONS = 50      # 每个实验跑多少迭代（benchmark 不需要跑完整 epoch）
WARMUP_ITERATIONS = 10   # 前 10 迭代预热
```

**为什么这样定义**：
- 字典形式方便增删模型配置
- `SCHEDULERS` 列表可以轻松添加新调度器
- 所有调度器共用相同的模型/数据配置，保证对比公平

#### 3.3.2 动态生成 YAML

```python
def generate_config(model_key: str, scheduler: str, output_dir: str) -> str:
    """动态生成 YAML 配置文件，返回文件路径"""
    mc = MODEL_CONFIGS[model_key]
    
    # Hanayo 需要 batch_size 能被 2*num_microbatches 整除
    batch_size = mc["batch_size"]
    num_mb = NUM_MICROBATCHES
    if scheduler == "hanayo":
        while batch_size % (num_mb * 2) != 0:
            batch_size += 1  # 微调 batch_size

    config = {
        "model": {
            "name": mc["name"],
            "hidden_size": mc["hidden_size"],
            "num_layers": mc["num_layers"],
            "num_heads": mc["num_heads"],
            "sequence_length": mc["seq_len"],
            "vocab_size": -1,
            "extra": {
                "head_size": mc["hidden_size"] // mc["num_heads"],
                "ff_low_rank": None,
                "dropout": 0.1,
            } if mc["name"] == "gpt" else {
                "num_kv_heads": mc["num_heads"],
                "intermediate_size": ((int(2 * 4 * mc["hidden_size"] / 3) + 31) // 32) * 32,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }
        },
        "dataset": {
            "name": "code_search_net",
            "batch_size": batch_size,
            "num_workers": 0,
            "extra": {}
        },
        "training": {
            "learning_rate": 0.0001,
            "epochs": 1,
            "num_microbatches": num_mb,
            "gradient_accumulation": 1,
            "max_iterations": MAX_ITERATIONS,
            "profile_iteration": -1,
            "exit_after_profile": False
        },
        "parallel": {
            "model_parallel_size": NUM_STAGES,
            "data_parallel_size": 1,
            "scheduler": scheduler,
            "use_orion_scheduler": False
        }
    }
    
    filename = f"{model_key}_{scheduler}.yaml"
    filepath = os.path.join(output_dir, filename)
    
    import yaml
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return filepath
```

#### 3.3.3 运行单个实验

```python
def run_experiment(config_path: str, json_output_path: str, num_gpus: int = 4) -> bool:
    """运行单个实验并等待完成"""
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        "pipeline_runner.py",
        "--config", config_path,
        "--output-json", json_output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0
```

#### 3.3.4 合并 JSON → CSV

```python
def collect_results(json_dir: str, csv_output: str):
    """将所有实验的 JSON 结果合并为 CSV"""
    import csv
    import json
    
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print("No results found!")
        return
    
    # 读取所有 JSON
    all_results = []
    for jf in json_files:
        with open(jf) as f:
            all_results.append(json.load(f))
    
    # 写 CSV
    fieldnames = all_results[0].keys()
    with open(csv_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Results saved to: {csv_output}")
    print(f"Total experiments: {len(all_results)}")
```

#### 3.3.5 主函数

```python
def main():
    parser = argparse.ArgumentParser(description='Pipeline Benchmark Runner')
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='benchmark_results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of models to run (e.g., gpt-small gpt-medium)')
    parser.add_argument('--schedulers', nargs='+', default=None,
                        help='Subset of schedulers (e.g., naive 1f1b)')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    config_dir = os.path.join(output_dir, "configs")
    json_dir = os.path.join(output_dir, "json_results")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    models = args.models or list(MODEL_CONFIGS.keys())
    schedulers = args.schedulers or SCHEDULERS
    
    total = len(models) * len(schedulers)
    print(f"Running {total} experiments: {len(models)} models × {len(schedulers)} schedulers")
    
    for model_key in models:
        for sched in schedulers:
            exp_name = f"{model_key}_{sched}"
            print(f"\n{'='*60}")
            print(f"Running: {exp_name}")
            print(f"{'='*60}")
            
            config_path = generate_config(model_key, sched, config_dir)
            json_path = os.path.join(json_dir, f"{exp_name}.json")
            
            success = run_experiment(config_path, json_path, args.num_gpus)
            
            if success:
                print(f"✓ {exp_name} completed")
            else:
                print(f"✗ {exp_name} FAILED")
    
    # 合并结果
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    collect_results(json_dir, csv_path)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete! CSV saved to: {csv_path}")
    print(f"Open in Excel: just drag and drop the CSV file")
    print(f"{'='*60}")
```

---

## 四、产出的 CSV 格式示例

最终产出的 `benchmark_results.csv` 长这样：

| model_name | hidden_size | num_layers | num_heads | sequence_length | batch_size | scheduler | num_stages | model_params | throughput_tokens_per_second | time_per_iteration_seconds | mfu | samples_per_second | num_gpus | device_name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt | 512 | 16 | 16 | 128 | 64 | naive | 4 | 50,331,776 | 125,000 | 0.065 | 0.12 | 976.56 | 4 | A100-SXM4-80GB |
| gpt | 512 | 16 | 16 | 128 | 64 | 1f1b | 4 | 50,331,776 | 180,000 | 0.045 | 0.17 | 1,406.25 | 4 | A100-SXM4-80GB |
| gpt | 512 | 16 | 16 | 128 | 64 | zerobubble | 4 | 50,331,776 | 200,000 | 0.041 | 0.19 | 1,562.50 | 4 | A100-SXM4-80GB |
| gpt | 1024 | 24 | 16 | 256 | 32 | naive | 4 | 302,235,648 | 45,000 | 0.182 | 0.08 | 175.78 | 4 | A100-SXM4-80GB |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Excel 友好**：直接拖入 Excel 后可以做数据透视表、画图。

---

## 五、修改清单汇总

### 需要修改的文件

| 文件 | 修改类型 | 改动量 | 说明 |
|------|---------|-------|------|
| [`ourgpipe/core/metrics.py`](ourgpipe/core/metrics.py) | **修改** | ~30 行 | 新增 `get_full_summary()` 和 `save_json()` 方法 |
| [`ourgpipe/pipeline_runner.py`](ourgpipe/pipeline_runner.py) | **修改** | ~15 行 | 添加 `--output-json` 参数和保存逻辑 |
| `ourgpipe/benchmark_runner.py` | **新建** | ~200 行 | 批量实验编排 + 结果汇总 |

### 不需要修改的文件

- [`config.py`](ourgpipe/core/config.py)：配置系统已足够灵活，不需要改
- 各 scheduler 实现：不需要改
- 各 model 实现：不需要改
- 已有的 YAML 配置文件：不受影响

---

## 六、使用方式

### 6.1 交互式运行（单节点）

```bash
cd ourgpipe

# 运行所有实验
python benchmark_runner.py --num-gpus 4

# 只跑指定模型
python benchmark_runner.py --num-gpus 4 --models gpt-small gpt-medium

# 只跑指定调度器
python benchmark_runner.py --num-gpus 4 --schedulers naive 1f1b

# 自定义输出目录
python benchmark_runner.py --num-gpus 4 --output-dir my_results
```

### 6.2 通过 Slurm 提交

可以创建一个 `submit_benchmark.sh`：

```bash
#!/bin/bash
#SBATCH --job-name=pipeline_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_%j.out

source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron
cd /home/bingxing2/home/scx9kvs/orion-docker/ourgpipe

python benchmark_runner.py --num-gpus 4 --output-dir benchmark_results
```

### 6.3 查看结果

```bash
# 结果目录结构
benchmark_results/
└── 20260303_172200/
    ├── configs/          # 每个实验的 YAML 配置
    ├── json_results/     # 每个实验的 JSON 结果
    └── benchmark_results.csv   # ← 这个拖入 Excel
```

---

## 七、关键设计决策说明

### 7.1 为什么不直接在 `pipeline_runner.py` 里写循环？

`pipeline_runner.py` 是被 `torchrun` 启动的**分布式进程**，每个 GPU 都会运行一份。如果在里面写实验循环，会导致：
- 4 个 GPU 各自独立跑循环 → 混乱
- `dist.init_process_group()` 只能调用一次

所以**实验编排必须在外部脚本**，每次实验启动一个新的 `torchrun` 进程。

### 7.2 为什么用 JSON 中间格式而不是直接追加 CSV？

- `torchrun` 启动多个进程，只有 rank 0 写文件
- 如果实验失败了，中间结果不会污染 CSV
- JSON 文件易于调试（可以单独查看某个实验的结果）
- 最后一次性合并为 CSV，保证数据完整性

### 7.3 为什么选 CSV 作为最终输出？

- Excel、Google Sheets、WPS 都能直接打开
- Python `pandas.read_csv()` 可以直接读
- 纯文本格式，git 友好
- 比 `.xlsx` 更轻量，不依赖 openpyxl 等库

### 7.4 Hanayo 调度器的特殊处理

[`hanayo`](ourgpipe/core/config.py:234) 要求 `batch_size % (2 * num_microbatches) == 0`。在动态生成配置时需要自动调整 `batch_size`，确保满足约束。

---

## 八、后续可扩展方向

1. **多节点支持**：`benchmark_runner.py` 可以扩展为生成 sbatch 脚本并行提交
2. **GPU 内存监控**：在 `MetricsTracker` 中增加 `torch.cuda.max_memory_allocated()` 记录
3. **可视化报告**：自动生成 matplotlib 图表（throughput 对比柱状图、MFU 对比图等）
4. **回归测试**：保存历史 CSV，自动对比新旧结果是否有性能退化
