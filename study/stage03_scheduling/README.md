# 阶段三：分阶段调度策略

首次修改核心Scheduler的批次选择。默认`prefill_first`保持原始优先级；`decode_first`优先decode；`alternating`优先选择上次成功执行阶段的另一阶段，无可执行batch时回退，空闲后从prefill开始。交替按批次而非执行时间，也不保证资源不足请求的等待上限。

服务参数：`--scheduling-policy prefill_first|decode_first|alternating`。
Python配置字段：`scheduling_policy`。

预算固定1024，单卡、Graph开启、重叠关闭、naive cache。复用阶段二短请求/长请求突发负载，额外增加有限持续到达组：四个短请求输出512 token，八个长请求从0.3秒起每0.15秒到达（输入4096、输出128）。各策略预热一次，九种组合各测五次，旋转顺序。

```bash
export PYTHONPATH="$PWD/python"
python -m unittest discover -s tests/stage03 -v
CUDA_VISIBLE_DEVICES=0 python study/stage03_scheduling/run.py --model /path/to/model --output study/stage03_scheduling/runs/main
```

策略测试覆盖默认顺序、预算传递、交替、资源阻塞回退、无decode/空闲与非法配置。真实GPU回放验证输出数量、分块预算、重复完成防护和资源回收。`check_lifecycle.py`额外通过真实GPU前向和可控采样验证EOS、长度结束及等待/分块/decode取消。重叠调度、多卡和Radix缓存组合未在本阶段验证，不将有限工作负载无遗漏等同于完整正确性。

```bash
python study/stage03_scheduling/summarize.py study/stage03_scheduling/runs/main
CUDA_VISIBLE_DEVICES=0 python study/stage03_scheduling/check_lifecycle.py --model /path/to/model --output study/stage03_scheduling/runs/lifecycle.json
```

已完成 45 次正式性能回放、9 个 GPU 生命周期用例及 14 个阶段一至三单元测试。结果、指标定义与局限见 [中文实验报告](REPORT.md)，完整原始数据见 [evidence](evidence)。交替调度改善最大间隔，但 P95 间隔有所增大。
