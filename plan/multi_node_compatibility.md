# 多节点兼容性分析与修改方案

## 一、分析目标

分析 ourgpipe 框架中 5 种调度器（naive、async_threaded、1f1b、zerobubble、hanayo）在从**单节点 4 GPU** 扩展到**多节点（2 节点 × 4 GPU = 8 GPU）**时的兼容性问题，并给出修改建议。

---

## 二、通信机制分析

### 2.1 两类通信方式

框架中存在两套通信 API，不同调度器使用不同的 API：

| 通信方式 | 使用者 | 实现位置 |
|---------|--------|---------|
| **同步阻塞** `dist.send`/`dist.recv` | [`naive`](ourgpipe/core/schedulers/naive.py:27), [`async_threaded`](ourgpipe/core/schedulers/async_threaded.py:28) | [`BaseStage.forward_send()`](ourgpipe/core/stage.py:480) 等 |
| **P2P 批量** `batch_isend_irecv` / 逐个 `isend`/`irecv` | [`1f1b`](ourgpipe/core/schedulers/one_f_one_b.py:28), [`zerobubble`](ourgpipe/core/schedulers/zerobubble.py:34), [`hanayo`](ourgpipe/core/schedulers/hanayo.py:122) | [`BaseStage.p2p_forward_send()`](ourgpipe/core/stage.py:597), [`comm.py`](ourgpipe/core/comm.py:78) 等 |

### 2.2 通信对端的计算方式

**这是多节点兼容性的核心关注点**。每种通信方法通过不同方式确定对端 rank：

#### 同步通信（naive / async_threaded）

```python
# forward_send: 发给 rank+1
dst = self.global_rank + 1                    # stage.py:488

# forward_recv: 从 rank-1 接收
src = self.global_rank - 1                    # stage.py:500

# backward_send: 发给 rank-1
dst = self.global_rank - 1                    # stage.py:515

# backward_recv: 从 rank+1 接收
src = self.global_rank + 1                    # stage.py:527
```

#### P2P 通信（1f1b / zerobubble / hanayo）

```python
# p2p_forward_send: 发给 rank+1
dst = self.global_rank + 1                    # stage.py:618

# p2p_forward_recv: 从 rank-1 接收
src = self.global_rank - 1                    # stage.py:632

# p2p_backward_send: 发给 rank-1
dst = self.global_rank - 1                    # stage.py:658

# p2p_backward_recv: 从 rank+1 接收
src = self.global_rank + 1                    # stage.py:672
```

---

## 三、逐调度器兼容性分析

### 3.1 `naive` — ✅ 多节点理论兼容

**通信方式**：`dist.send` / `dist.recv`（阻塞式），通过 `model_parallel_group` 限定通信组。

**分析**：
- 所有通信都带 `group=self.model_parallel_group` 参数
- `model_parallel_group` 在 [`pipeline_runner.py:96`](ourgpipe/pipeline_runner.py:96) 中由 `dist.new_group(list(range(i * model_parallel_size, (i + 1) * model_parallel_size)))` 创建
- `global_rank + 1` / `global_rank - 1` 作为 `dst`/`src`，只要 rank 编号在 model_parallel_group 中连续，就没问题
- `torchrun` 跨节点启动时，rank 编号是全局连续的（Node 0: rank 0-3, Node 1: rank 4-7）

**结论**：naive 调度器**不需要修改**即可多节点运行。现有的 [`gpt_8gpu_naive.yaml`](ourgpipe/configs/gpt_8gpu_naive.yaml) 配置已验证了这一点。

---

### 3.2 `async_threaded` — ✅ 多节点理论兼容

**通信方式**：`dist.isend` / `dist.irecv`（非阻塞异步），通过 `model_parallel_group` 限定通信组。

**分析**：
- [`forward_isend()`](ourgpipe/core/stage.py:384) / [`forward_irecv()`](ourgpipe/core/stage.py:400) 等方法也都带 `group=self.model_parallel_group`
- 使用 `self._compute_tag()` 生成唯一 tag，tag 计算与节点无关
- 多线程 + 异步通信在跨节点 NCCL 下依然有效（NCCL 本身是线程安全的）

**潜在风险**：
- 跨节点网络延迟更大，多线程竞争可能导致**性能下降**，但不会导致**正确性问题**
- 现有的 [`gpt_8stage.yaml`](ourgpipe/configs/gpt_8stage.yaml) 已经用 `async_threaded` + 8 stage 运行过

**结论**：async_threaded 调度器**不需要修改**即可多节点运行。

---

### 3.3 `1f1b` — ⚠️ 需要关注但理论兼容

**通信方式**：通过 [`execute_p2p_ops()`](ourgpipe/core/comm.py:78) 逐个提交 `isend`/`irecv`，使用 `model_parallel_group`。

**分析**：

1. **P2P 对端计算**：`p2p_forward_send()` 使用 `self.global_rank + 1`，`p2p_forward_recv()` 使用 `self.global_rank - 1`。同样只要 rank 连续就正确。

2. **group 参数**：所有 P2P 操作都通过 [`make_p2p_send()`](ourgpipe/core/comm.py:13) / [`make_p2p_recv()`](ourgpipe/core/comm.py:35) 传入 `group=self.model_parallel_group`，确保通信在正确的组内。

3. **`execute_p2p_ops()` 实现**（[`comm.py:78`](ourgpipe/core/comm.py:78)）：
   ```python
   def execute_p2p_ops(ops):
       for op in ops:
           req = op.op(op.tensor, op.peer, group=op.group, tag=op.tag)
           reqs.append(req)
       for req in reqs:
           req.wait()
   ```
   这里逐个提交 P2P 操作而非用 `batch_isend_irecv`，正是为了避免 NCCL group 语义问题。**这是正确的做法**。

4. **死锁风险**：1F1B steady state 阶段同时提交 `send_fwd + recv_bwd`（[`one_f_one_b.py:108-118`](ourgpipe/core/schedulers/one_f_one_b.py:108)）。在跨节点场景下，如果 `send_fwd` 的对端在另一个节点，需要确保对端同时也在等待接收。由于 1F1B 的调度逻辑是确定性的（每个 stage 的 F/B 顺序由 `pipeline_rank` 和 `num_warmup` 严格决定），**不会产生死锁**。

**潜在风险**：
- Tag 值使用简单的 `mb_idx`（前向）和 `1000 + mb_idx`（反向）。在同一个 `model_parallel_group` 内这是唯一的，**跨节点不影响**。

**结论**：1f1b 调度器**不需要修改**即可多节点运行。

---

### 3.4 `zerobubble` — ⚠️ 需要关注但理论兼容

**通信方式**：与 1f1b 完全相同的 P2P 通信（`execute_p2p_ops`）。

**分析**：

1. **通信对端**：同 1f1b，使用 `global_rank ± 1`。
2. **B/W 分离逻辑**（[`WeightGradStore`](ourgpipe/core/weight_grad_store.py:19)）：这是**纯本地逻辑**，不涉及跨进程通信。`WeightGradStore` 是一个全局变量，但因为每个进程都是独立的 Python 进程，不会有跨进程的状态污染。
3. **`BubbleLinear`**：也是纯本地替换，不影响通信。

**潜在风险**：
- 跨节点网络延迟增大后，ZeroBubble 的 "用 W 填充 bubble" 策略的效果可能打折——跨节点通信延迟变大导致 bubble 更大，但 W 计算量不变。**这是性能问题，不是正确性问题**。

**结论**：zerobubble 调度器**不需要修改**即可多节点运行。

---

### 3.5 `hanayo` — ⚠️⚠️ 需要仔细验证

**通信方式**：与 1f1b 类似的 P2P 通信，但**方向更复杂**。

**分析**：

Hanayo 有 4 种通信方向：

| 操作 | 方向 | 对端 | Tag 范围 |
|------|------|------|---------|
| `p2p_forward_send` (Wave A) | rank → rank+1 | `global_rank + 1` | `mb_idx` |
| `p2p_forward_recv` (Wave A) | rank-1 → rank | `global_rank - 1` | `mb_idx` |
| `p2p_backward_send` (Wave A) | rank → rank-1 | `global_rank - 1` | `1000 + mb_idx` |
| `p2p_backward_recv` (Wave A) | rank+1 → rank | `global_rank + 1` | `1000 + mb_idx` |
| `p2p_forward_send_B` (Wave B) | rank → rank-1 | `global_rank - 1` | `2000 + mb_idx` |
| `p2p_forward_recv_B` (Wave B) | rank+1 → rank | `global_rank + 1` | `2000 + mb_idx` |
| `p2p_backward_send_B` (Wave B) | rank → rank+1 | `global_rank + 1` | `3000 + mb_idx` |
| `p2p_backward_recv_B` (Wave B) | rank-1 → rank | `global_rank - 1` | `3000 + mb_idx` |

**关键观察**：

1. **对端计算**：全部是 `global_rank ± 1`，与 1f1b 一致。只要 rank 连续就正确。

2. **Tag 唯一性**：4 组操作使用 4 个 tag 空间（`[0, N)`, `[1000, 1000+N)`, `[2000, 2000+N)`, `[3000, 3000+N)`），互不冲突。

3. **entry/exit 判断**（[`hanayo_stage.py:216-224`](ourgpipe/models/gpt/hanayo_stage.py:216)）：
   ```python
   def is_entry(self, wave):
       return ((wave == 'A' and self.stage_id == 1) or
               (wave == 'B' and self.stage_id == self.model_parallel_size))
   
   def is_exit(self, wave):
       return ((wave == 'A' and self.stage_id == self.model_parallel_size) or
               (wave == 'B' and self.stage_id == 1))
   ```
   这里使用 `self.stage_id` 和 `self.model_parallel_size`，与全局 rank 无关，**是正确的**。

4. **调度表生成**（[`build_hanayo_schedule()`](ourgpipe/core/schedulers/hanayo.py:33)）：
   ```python
   def build_hanayo_schedule(pipeline_rank, num_stages, num_mb):
   ```
   使用 `pipeline_rank`（0-indexed stage ID），与全局 rank 无关，**是正确的**。

5. **死锁风险**：Hanayo 的调度是确定性的，每个 stage 的操作顺序由 `build_hanayo_schedule()` 完全确定。Wave A 和 Wave B 的通信方向相反但 tag 不同，不会混淆。在多节点场景下，只要 NCCL 能正确路由跨节点的 P2P 消息（`isend`/`irecv` 通过 `model_parallel_group`），就不会死锁。

**潜在风险**：

1. **跨节点 P2P 性能**：Hanayo 的通信模式比 1f1b 更密集（双向流水线意味着更多 P2P 操作），跨节点延迟可能导致性能显著下降。

2. **NCCL P2P 兼容性**：某些 NCCL 版本在跨节点 P2P 通信中可能有 bug。当前 [`execute_p2p_ops()`](ourgpipe/core/comm.py:78) 已经采用逐个提交的方式（而非 `batch_isend_irecv`），这是比较安全的做法。

**结论**：hanayo 调度器**理论上不需要修改**即可多节点运行，但建议作为最后一个测试。

---

## 四、总体评估

### 4.1 结论汇总

| 调度器 | 多节点兼容 | 需要修改 | 风险等级 | 说明 |
|--------|-----------|---------|---------|------|
| `naive` | ✅ | 无 | 低 | 已有 8GPU 配置证明可用 |
| `async_threaded` | ✅ | 无 | 低 | 已有 8stage 配置 |
| `1f1b` | ✅ | 无 | 中 | 需跨节点 P2P 验证 |
| `zerobubble` | ✅ | 无 | 中 | 同 1f1b，B/W 分离是本地逻辑 |
| `hanayo` | ✅ | 无 | 中-高 | 双向通信更复杂，需仔细测试 |

### 4.2 为什么所有调度器都兼容

核心原因是框架的**良好抽象**：

1. **通信组隔离**：所有通信操作都通过 `model_parallel_group` 限定范围，不会与其他组混淆
2. **rank 编号连续**：`torchrun` 跨节点时 rank 是全局连续编号，`global_rank ± 1` 的逻辑在单节点和多节点下一致
3. **stage_id 与 global_rank 分离**：[`pipeline_runner.py:168-172`](ourgpipe/pipeline_runner.py:168) 通过 `rank_to_stage` 映射将全局 rank 转换为 stage_id（1-indexed），调度器使用 stage_id 做逻辑判断
4. **确定性调度**：所有调度器的 F/B 操作顺序由 `pipeline_rank` 和 `num_microbatches` 确定，不依赖节点拓扑

---

## 五、建议的测试顺序

由于所有调度器理论上都不需要代码修改，建议按以下顺序在 2 节点 8 GPU 上逐步测试：

### Step 1: naive（最安全）
```bash
torchrun --nnodes=2 --nproc_per_node=4 ... pipeline_runner.py --config configs/gpt_8gpu_1f1b.yaml --scheduler naive
```

### Step 2: 1f1b
```bash
torchrun --nnodes=2 --nproc_per_node=4 ... pipeline_runner.py --config configs/gpt_8gpu_1f1b.yaml
```

### Step 3: zerobubble
```bash
torchrun --nnodes=2 --nproc_per_node=4 ... pipeline_runner.py --config configs/gpt_8gpu_zerobubble.yaml
```

### Step 4: hanayo
```bash
torchrun --nnodes=2 --nproc_per_node=4 ... pipeline_runner.py --config configs/gpt_8gpu_hanayo.yaml
```

### Step 5: async_threaded
```bash
torchrun --nnodes=2 --nproc_per_node=4 ... pipeline_runner.py --config configs/gpt_8gpu_async.yaml
```

---

## 六、可能遇到的问题及排查

### 6.1 NCCL 超时

**症状**：训练卡住，最终报 `NCCL timeout` 错误。

**原因**：跨节点 P2P 操作中某个 send/recv 对不匹配。

**排查**：
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```
查看每个 rank 的通信日志，确认 send/recv 的对端和 tag 是否匹配。

### 6.2 InfiniBand 配置

**症状**：跨节点通信极慢或失败。

**解决**：
```bash
export NCCL_IB_DISABLE=0          # 启用 InfiniBand
export NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3        # 根据集群配置调整
```

### 6.3 Hanayo batch_size 约束

**症状**：Hanayo 报 `batch_size must be divisible by 2*num_microbatches` 错误。

**解决**：确保 `batch_size % (2 * num_microbatches) == 0`。8 GPU 下如果不开数据并行，batch_size 不变；如果开了数据并行（比如 4 stage + 2 DP），需要重新计算。

### 6.4 WeightGradStore 全局状态

**注意**：[`WeightGradStore`](ourgpipe/core/weight_grad_store.py:19) 使用 class-level 变量。在多节点场景下，每个进程是独立的 Python 进程，**不会共享状态**，所以没有问题。但如果将来改为多线程架构，需要加锁。

---

## 七、需要创建的配置文件

以下配置文件用于 2 节点 8 GPU 测试各调度器。已创建在 `ourgpipe/configs/` 目录下：

| 配置文件 | 调度器 | 模型 | 说明 |
|---------|--------|------|------|
| `gpt_8gpu_1f1b.yaml` | 1f1b | GPT | 2 节点 8 stage |
| `gpt_8gpu_zerobubble.yaml` | zerobubble | GPT | 2 节点 8 stage |
| `gpt_8gpu_hanayo.yaml` | hanayo | GPT | 2 节点 8 stage, batch_size 需整除 2*num_mb |
| `gpt_8gpu_async.yaml` | async_threaded | GPT | 2 节点 8 stage |

---

## 八、提交脚本模板

创建一个通用的 2 节点提交脚本 `submit_2nodes_all.sh`，支持通过参数选择调度器：

```bash
#!/bin/bash
#SBATCH --job-name=pipeline_8gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/pipeline_8gpu_%j.out
#SBATCH --error=logs/pipeline_8gpu_%j.err

# 用法: sbatch submit_2nodes_all.sh [config_file]
# 例如: sbatch submit_2nodes_all.sh configs/gpt_8gpu_1f1b.yaml

CONFIG=${1:-configs/gpt_8gpu_1f1b.yaml}

SCRIPT_DIR="/home/bingxing2/home/scx9kvs/orion-docker/ourgpipe"
cd "$SCRIPT_DIR"
mkdir -p logs

source /home/bingxing2/home/scx9kvs/mxy/env.sh
conda activate pai-megatron

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
GPUS_PER_NODE=4
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=4

echo "Config: $CONFIG"
echo "Nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES)"
echo "Total GPUs: $WORLD_SIZE"

srun --kill-on-bad-exit=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    pipeline_runner.py --config $CONFIG
```
