# Hanayo 集成启动卡死（NCCL watchdog timeout）修复计划

> 目标：让 `torchrun --nproc_per_node=4 ... --config configs/gpt_small_hanayo.yaml` 能在 **iteration 0** 正常完成一次完整的 Hanayo 迭代，不再出现 `ProcessGroupNCCL watchdog timeout`；并在跑通后补齐训练正确性（optimizer 覆盖 embedding/head 等参数）。

## 0. 背景与复现信息

### 0.1 复现命令

```bash
torchrun --nproc_per_node=4 --master_port=29500 pipeline_runner.py --config configs/gpt_small_hanayo.yaml
```

入口脚本：[`pipeline_runner.py`](../ourgpipe/pipeline_runner.py)

### 0.2 典型报错现象（摘要）

- 训练进度停在 iteration 0（无进一步输出）
- 约 600s 后某个 rank 报：`Watchdog caught collective operation timeout`
- 日志里出现 `PG 1 Rank 2`，说明卡在 **model_parallel_group（非 world group）** 上

你提供的关键信息是：
- `[PG 1 Rank 2] Watchdog caught collective operation timeout ... OpType=COALESCED ...`

## 1. 结论（高概率根因）

**根因：Hanayo 会把一个 batch 拆成 `2 * num_microbatches` 份，但 Stage 的通信 shape 仍按 `batch_size / num_microbatches` 去算，导致 P2P send/recv 的 message size 不一致，NCCL 会一直等待，从而卡住并触发 watchdog 超时。**

证据链（从“拆分”到“通信 shape”）：

1) Hanayo 拆分 micro-batch 的地方：
- 在 [`pipeline_runner.py`](../ourgpipe/pipeline_runner.py:341) 里，当 `scheduler == 'hanayo'` 时：
  - `micro_inputs = torch.chunk(inputs, num_microbatches * 2)`
  - 所以真实 micro-batch 大小是 `batch_size / (2*num_microbatches)`

2) Stage 当前 micro-batch 大小的计算：
- 在 [`BaseStage.__init__()`](../ourgpipe/core/stage.py:50) 里：
  - `self.micro_batch_size = config.dataset.batch_size // self.num_microbatches`（见 [`core/stage.py`](../ourgpipe/core/stage.py:97)）
  - 这对 Hanayo 来说偏大一倍

3) P2P recv buffer 的 shape 来源：
- [`BaseStage.get_activation_shape()`](../ourgpipe/core/stage.py:586) 返回 `(micro_batch_size, seq_len, hidden_size)`
- Hanayo 的 recv buffer 由 [`make_p2p_recv()`](../ourgpipe/core/comm.py:35) 按这个 shape 分配

4) 结果：
- 发送端实际发的是 `(batch_size/(2*num_mb), T, H)`
- 接收端按 `(batch_size/num_mb, T, H)` 去收
- **message size 不一致** ⇒ NCCL 等不到匹配的 recv/send ⇒ 卡死

对照可跑通脚本的正确做法：
- 可跑通脚本把 micro-batch 明确设为 `batch_size // (num_microbatches * 2)`（见 [`HanayoStage.__init__()`](../ourgpipe/hanayo_naive.py:152)）
- 通信 shape 也同样按 `num_microbatches * 2` 计算（见 [`act_shape()`](../ourgpipe/hanayo_naive.py:107)）

## 2. 给初学者的“为什么会卡死”的解释

NCCL 的 `send/recv` 并不会像普通 Python 函数那样在“参数不匹配”时立刻抛异常。

- 发送方发出 N 个元素
- 接收方按自己的 buffer 期望收 M 个元素
- 如果 N != M，NCCL 可能进入“等待/阻塞”状态（对方永远给不出它期望的大小），最终 watchdog 超时。

所以这种问题最典型的现象就是：
- **程序不报 Python Exception**
- **训练停住**
- **10 分钟左右**（你的 timeout=600s）出现 NCCL watchdog 报错

## 3. 修改计划（按优先级，从“先跑通”到“再正确”）

> 原则：每一步改动尽量小；每改一步就跑一个最小验证（只跑 1-2 step）确认现象变化。

### Phase A：把“卡死”变成“秒报错”（强烈建议先做）

**目的**：即使后续仍有问题，也能第一时间看见“哪个 rank 期望的 shape vs 实际 shape”。

A1) 在发送前加断言
- 修改位置：[`BaseStage.p2p_forward_send()`](../ourgpipe/core/stage.py:594)、[`BaseStage.p2p_backward_send()`](../ourgpipe/core/stage.py:624) 及 Wave B 对应函数
- 内容：断言 `tensor.shape == stage.get_activation_shape()`（或至少 `tensor.numel()` 相等）

A2) 在 Hanayo 调度器里对 recv buffer 做断言
- 修改位置：[`HanayoScheduler.run_iteration()`](../ourgpipe/core/schedulers/hanayo.py:103)
- 在 `execute_p2p_ops([op])` 之后，检查 `buf.shape` 是否等于 `stage.get_activation_shape()`

> 预期效果：如果形状不一致，不再“等 10 分钟超时”，而是 iteration 0 的第一个 mb 直接抛出明确错误。

### Phase B：修复 Hanayo micro-batch 形状口径（解决卡死的最小修复）

B1) 统一 `micro_batch_size` 的计算口径
- 修改位置：[`BaseStage.__init__()`](../ourgpipe/core/stage.py:50)
- 修改点：把
  - `micro_batch_size = batch_size // num_microbatches`
  改为
  - `effective_microbatches = num_microbatches * (2 if scheduler == 'hanayo' else 1)`
  - `micro_batch_size = batch_size // effective_microbatches`

说明：
- 这里的 `scheduler` 直接来自 `config.parallel.scheduler`
- Hanayo 的 schedule 仍然以 `num_microbatches` 为单位（mb_idx 仍是 0..num_mb-1），**不用**把 `self.num_microbatches` 改成 2 倍；只需要把“每个 mb 的 batch 维度”改成正确的 1/2。

B2) 加配置校验（避免 chunk 出不等长 micro-batch）
- 修改位置：[`core/config.py`](../ourgpipe/core/config.py)
- 规则：当 `scheduler == 'hanayo'` 时，必须满足：
  - `batch_size % (2*num_microbatches) == 0`
- 若不满足直接抛错，并提示用户调整 `batch_size` 或 `num_microbatches`

B3) 最小验证
- 使用现有配置 [`gpt_small_hanayo.yaml`](../ourgpipe/configs/gpt_small_hanayo.yaml)
- 预期：iteration 0 不再卡死在 P2P；至少能走过第一次完整 `run_iteration`，并进入 iteration 1

### Phase C：设备绑定健壮性（降低 NCCL 诡异卡顿概率）

C1) 在初始化 NCCL 之前绑定 `LOCAL_RANK` 对应的 GPU
- 参考可跑通脚本里 [`get_device()`](../ourgpipe/hanayo_naive.py:68) 的做法：`torch.cuda.set_device(local_rank)`
- 建议修改点：增强 [`get_device()`](../ourgpipe/pipeline_runner.py:64) 或在 [`setup_distributed()`](../ourgpipe/pipeline_runner.py:72) 里 `dist.init_process_group` 之前设置

> 注意：这一步通常不会解决“形状不一致”导致的死锁，但会减少“默认 device 不一致”造成的玄学问题。

### Phase D：训练正确性修复（跑通后必须补，不然参数不更新）

#### D1) 现状问题：optimizer 只覆盖 `sub_model`
- BaseStage 创建 optimizer：只用 `self.sub_model.parameters()`（见 [`BaseStage.__init__()`](../ourgpipe/core/stage.py:105)）

但 GPT/Hanayo 的 embedding/head/ln 在 sub_model 之外：
- Hanayo stage1 的 `token_embedding_A / position_embedding_A / ln_B / lm_head_B`（见 [`GPTHanayoStage.create_sub_model()`](../ourgpipe/models/gpt/hanayo_stage.py:79)）
- Hanayo stageN 的 `ln_A / lm_head_A / token_embedding_B / position_embedding_B`（见 [`GPTHanayoStage.create_sub_model()`](../ourgpipe/models/gpt/hanayo_stage.py:105)）
- 普通 GPT stage1 的 embedding 也在 sub_model 之外（见 [`GPTStage.create_sub_model()`](../ourgpipe/models/gpt/stage.py:106)）

结论：即使通信跑通，embedding/head 等参数也不会被优化器更新，训练不正确。

#### D2) 推荐修法（两种方案，推荐方案 1）

方案 1（推荐，侵入较小、对所有模型通用）：
- 在 BaseStage 增加一个可覆盖的接口，比如：
  - `BaseStage.get_optimizer_params()`（默认返回 `self.sub_model.parameters()`）
- GPTStage / GPTHanayoStage 覆盖它，把 embedding/head/ln 的参数也加进去
- 同理把 `all_reduce_gradients()` 的遍历对象改成“optimizer param groups 中的 params”（或提供 `get_all_params()` 用于 allreduce）

方案 2（结构化，长期更干净）：
- 让 Stage 自身继承 `nn.Module` 并注册所有子模块（embedding/head 作为 stage 的子模块）
- optimizer/allreduce 直接用 `stage.parameters()`

> 建议先走方案 1，把正确性补齐，再考虑方案 2 的重构。

### Phase E：回归验证/辅助调试

E1) 启用更详细的分布式调试（用于定位仍可能存在的 P2P 顺序问题）
- 环境变量建议（仅调试时开启）：
  - `NCCL_DEBUG=INFO`
  - `TORCH_DISTRIBUTED_DEBUG=DETAIL`

E2) 快速 smoke test（只跑 1-2 step）
- 将配置里的 `training.max_iterations` 临时设为 1 或 2（见 [`gpt_small_hanayo.yaml`](../ourgpipe/configs/gpt_small_hanayo.yaml)）
- 观察是否能稳定结束而不报 watchdog timeout

## 4. 风险与备选排查项

如果按 Phase B 修了 micro_batch_size 仍卡住（概率较低），再按以下顺序排查：

1) Hanayo schedule 顺序是否与 wave A/B 的 send/recv 对应
- 重点看：[`build_hanayo_schedule()`](../ourgpipe/core/schedulers/hanayo.py:31) 是否在某些 stage 上会出现“双向同时 recv”导致双方都等

2) tag 是否发生冲突
- wave A: `tag=mb_idx / 1000+mb_idx`
- wave B: `tag=2000+mb_idx / 3000+mb_idx`
- 对照可跑通脚本的 tag 设计（见 [`COMM` 映射](../ourgpipe/hanayo_naive.py:327)）

3) device 未正确 set 导致 NCCL 在错误 GPU 上初始化
- 优先完成 Phase C

---

## 5. 预计实施顺序（建议你按这个顺序让我逐步改）

- 第 1 次改动：Phase A（断言/日志）
- 第 2 次改动：Phase B（micro_batch_size 修复 + config 校验）
- 第 3 次改动：Phase C（set_device）
- 第 4 次改动：Phase D（optimizer 参数覆盖，先方案 1）

每次改完你跑一次最小 step，把日志/报错贴回来，我们再进入下一步。
