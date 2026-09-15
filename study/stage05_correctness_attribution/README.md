# 阶段五：正确性与性能归因

总体验收与结论见 [阶段五中文报告](REPORT.md)。本阶段按 [实验计划](../../实验计划.md) 推进。首先调查阶段四的真实提示词输出差异，再覆盖 EOS、取消、分块、前缀缓存、overlap 和长时间资源回收，最后单独运行 Nsight Systems 做性能归因。

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

已完成实验 5.1，结果和未解决问题见 [中文报告](DIVERGENCE_REPORT.md)。

```bash
python study/stage05_correctness_attribution/summarize.py --runs study/stage05_correctness_attribution/runs --output study/stage05_correctness_attribution/evidence
```

## 实验 5.2：隐藏状态与 KV 定位

已完成，见 [中文报告](HIDDEN_REPORT.md)。定点采集自然 eager 生成的隐藏状态及 KV，先检查探针不改变原输出，再按逻辑位置对齐。参考注意力使用 CPU float64；中间张量仍属本地敏感数据。

```bash
python study/stage05_correctness_attribution/probe_hidden.py \
  --model /path/to/model \
  --source study/stage04_system_evaluation/runs/main/private \
  --pools study/stage04_system_evaluation/prepared/pools.json \
  --previous study/stage05_correctness_attribution/runs \
  --output study/stage05_correctness_attribution/runs/hidden_backend
python study/stage05_correctness_attribution/summarize_hidden.py \
  --runs study/stage05_correctness_attribution/runs/hidden_backend \
  --output study/stage05_correctness_attribution/evidence/hidden_summary.json
# 使用包含 matplotlib 的独立绘图环境：
python study/stage05_correctness_attribution/plot_hidden.py study/stage05_correctness_attribution/evidence
```

## 实验 5.3：生命周期、缓存与持续资源回收

先激活包含 torch、FlashInfer、ninja 的推理环境，从仓库根目录执行。四种配置按顺序运行，避免分布式初始化端口冲突。基础矩阵约数分钟，最后的 radix＋overlap 额外持续运行至少十分钟。

```bash
export PYTHONPATH="$PWD/python"
export CUDA_VISIBLE_DEVICES=0
python study/stage05_correctness_attribution/run_lifecycle.py \
  --model /path/to/model \
  --output study/stage05_correctness_attribution/runs/lifecycle \
  --waves 100 --soak-seconds 600
for cache in naive radix; do
  for overlap in 0 1; do
    python study/stage05_correctness_attribution/extra_lifecycle.py \
      --model /path/to/model --cache "$cache" --overlap "$overlap" \
      --output "study/stage05_correctness_attribution/runs/extra/${cache}_overlap${overlap}.json"
  done
done
python -m unittest discover -s tests/stage05 -v
# 在安装 pytest 和 pytest-cov 的环境中运行原项目缓存测试：
python -m pytest tests/core/test_cache_allocate.py -q \
  -o 'addopts=--cov=minisgl --cov-report=term-missing'
```

`export_validation.py` 验证本次完整交付的已有记录，包括四次修复后 Graph 回放、阶段一至五 CPU 测试记录及缓存 pytest 日志，再导出匿名摘要。只执行上述 GPU 命令不足以生成完整交付证据。

```bash
python study/stage05_correctness_attribution/export_validation.py \
  --runs study/stage05_correctness_attribution/runs \
  --output study/stage05_correctness_attribution/evidence
```

## 实验 5.4：Nsight GPU 时间线

需要安装 Nsight Systems；绘图步骤另需 matplotlib。采集脚本读取阶段四本地准备的 pools 与运行记录，使用相同输入/到达计划重新执行实际策略。每个策略单独启动进程，排除模型加载与预热。

```bash
python study/stage05_correctness_attribution/run_profiles.py \
  --model /path/to/model \
  --output study/stage05_correctness_attribution/runs/profiles
python study/stage05_correctness_attribution/summarize_profiles.py \
  --runs study/stage05_correctness_attribution/runs/profiles \
  --output study/stage05_correctness_attribution/evidence
```

原始 `.nsys-rep`、SQLite、CSV 和生成输出只留在 `runs/`。公开的时间线来自 Nsight GPU 投影范围；graph span 与显式 kernel 分开统计。两次采集用于解释调度顺序，不替代阶段四多次重复的系统评估。
