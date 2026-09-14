# 阶段三：实现公平调度策略

以仓库根目录的 [实验计划](../../实验计划.md) 为准。本阶段主要改动是 `wait_time` 等待阈值策略；之前的 `alternating` 只作为额外对照。原始三策略实验保存在 [REPORT.md](REPORT.md) 和 `evidence/`；本次补充见 [WAIT_TIME_REPORT.md](WAIT_TIME_REPORT.md) 和 `evidence/wait_time/`。

## 策略与等待定义

| 策略 | 规则 |
|---|---|
| `prefill_first` | 原始默认策略，优先 prefill |
| `decode_first` | 优先 decode，作为另一端的对照 |
| `alternating` | 按成功选择的批次交替，额外对照 |
| `wait_time` | decode 达到阈值则优先 decode；prefill 达到阈值则优先 prefill；同时触发时选择超出各自阈值更久的一类，相同时优先 decode；都未触发时优先 prefill |

优先阶段不能组批时尝试另一阶段，所有策略保留 prefill token 预算。

每个请求分别记录当前阶段的等待起点：prefill 从引擎收到有效请求并加入 pending 队列开始；decode 从进入 running 集合开始。实际选中的请求在批次选择时重置起点；未选中的请求保持原时间。分块请求仍处于 prefill；完成 prefill 后进入 decode，重新计时。完成、取消、空闲时同步清理状态，新请求不会重置旧请求的等待。

这是**距进入可调度集合或上次获得调度的 CPU 时间间隔**，包含上次批次执行及结果处理时间；不是纯 GPU 空闲时间，也不是从计划到达计算的端到端排队延迟。取每一类请求的最大间隔参与决策。pending 存在不代表资源足够，所以日志显式记录回退。该策略提供两类可执行请求的调度机会，不承诺单请求严格公平或延迟上限，也不改变 prefill manager 的队列内顺序。

首版参数 decode 50 ms、prefill 200 ms 是实验起点，未调优。阈值不抢占已经执行的 GPU 工作。`wait_time` 暂只支持 TP=1，避免各 rank 使用本地时钟产生不同 batch 决策；其他策略保留原有 TP 行为。overlap 和 Radix 组合留待后续阶段验证。

## 配置与日志

服务参数示例（接在原有服务启动命令后）：

```text
--scheduling-policy wait_time --decode-wait-ms 50 --prefill-wait-ms 200 --log-scheduling-decisions
```

对应 Python 配置字段为 `scheduling_policy`、`decode_wait_ms`、`prefill_wait_ms`、`log_scheduling_decisions`。阈值必须有限且大于 0。服务决策日志默认关闭，开启后包含两类等待时间、阈值、最老请求 UID、跟踪请求数、触发原因、优先阶段、实际阶段、回退和 token 预算。回放脚本把这些字段作为 `decision` 事件保存在 JSONL，批次事件另含选中 UID 与实际 token 数。

## 复现

在仓库根目录激活已有环境后执行：

```bash
export PYTHONPATH="$PWD/python"
python -m unittest discover -s tests/stage03 -v
CUDA_VISIBLE_DEVICES=0 python study/stage03_scheduling/run.py \
  --model /path/to/model --output study/stage03_scheduling/runs/wait_time/main \
  --policies prefill_first decode_first wait_time --decode-wait-ms 50 --prefill-wait-ms 200
python study/stage03_scheduling/summarize.py study/stage03_scheduling/runs/wait_time/main
python study/stage03_scheduling/audit_decisions.py study/stage03_scheduling/runs/wait_time/main
CUDA_VISIBLE_DEVICES=0 python study/stage03_scheduling/check_lifecycle.py \
  --model /path/to/model --output study/stage03_scheduling/runs/wait_time/lifecycle.json
```

默认三种原计划策略、三个负载、每配置五次测量；每策略预热一次，正式配置轮换顺序。预算固定 1024，单卡、Graph 开启、overlap 关闭、naive cache。复用短请求、长请求突发及有限持续到达负载，与原实验输入相同。复现此前交替对照需显式指定 `--policies prefill_first decode_first alternating`。

测试覆盖精确阈值、双重触发与平局、持续双类需求、分块计时、保留未服务请求年龄、资源回退、取消/完成/UID 复用和非法参数。GPU 生命周期验证使用真实前向与可控采样触发 EOS、长度结束及等待/分块/decode 取消，检查资源和等待状态清理；不等于语义质量验证。
