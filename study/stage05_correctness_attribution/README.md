# 阶段五：正确性与性能归因

本阶段按 [实验计划](../../实验计划.md) 推进。首先调查阶段四的真实提示词输出差异，再覆盖 EOS、取消、分块、前缀缓存、overlap 和长时间资源回收，最后单独运行 Nsight Systems 做性能归因。

## 实验 5.1：固定阶段四批次轨迹，检查首次输出分歧

`inventory.py` 比较固定预算矩阵中 wait_time/1024 与 prefill_first/1024，汇总首次分歧位置类型；按文件名固定顺序选择第一个有差异的 mixed_steady/near 运行对。选择不是随机总体抽样，单例不用于推断全部差异原因。

`replay_trace.py` 按记录重放每个批次前的请求准入、prefill/decode 阶段及 token 预算，逐批断言请求顺序和处理 token 数一致。自然生成，不强制回填已存 token。在原有首次分歧位置记录前缀哈希、logits、候选分数和批次形状。Graph/eager 各两次，用于检查重复性与模式差异。诊断绕过实时策略选择，不用于测试策略触发，也不用于性能统计；CPU logits 拷贝会同步 GPU。

```bash
export PYTHONPATH="$PWD/python"
python study/stage05_correctness_attribution/inventory.py \
  --source study/stage04_system_evaluation/runs/main/private \
  --output study/stage05_correctness_attribution/runs
python study/stage05_correctness_attribution/replay_trace.py \
  --model /path/to/model \
  --source study/stage04_system_evaluation/runs/main/private \
  --pools study/stage04_system_evaluation/prepared/pools.json \
  --selection study/stage05_correctness_attribution/runs/selected.json \
  --output study/stage05_correctness_attribution/runs/graph --graph 8
# 在新进程中改为 --graph 0、--output .../runs/eager，执行 eager 对照。
python -m unittest discover -s tests/stage05 -v
```

`runs/` 含原始请求标识、输出 token、候选 token 与 logits，仅保留本地。公开结果仅为计数与匿名数值摘要。阶段四发布分支和测量代码保持不变。

已完成实验 5.1，结果和未解决问题见 [中文报告](REPORT.md)。

```bash
python study/stage05_correctness_attribution/summarize.py --runs study/stage05_correctness_attribution/runs --output study/stage05_correctness_attribution/evidence
```
