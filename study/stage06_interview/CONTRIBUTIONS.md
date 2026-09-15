# 贡献与证据

上游基点为 `9a91cfafe754aa85daee49998176275667eb58f2`，保留 MIT 版权与 [原始说明](../../docs/upstream-readme.md)。本项目不是从零实现的推理引擎，也未声称向上游合入补丁。

| 能力 | 来源或新增范围 | 阅读入口 |
|---|---|---|
| 模型加载、attention backend、采样、CUDA Graph、TP | 继承上游；没有据此声称自研 kernel | [引擎](../../python/minisgl/engine/engine.py) |
| paged KV、Radix prefix cache、chunked prefill、overlap 框架 | 继承上游；后续仅修复明确缺陷 | [缓存管理](../../python/minisgl/scheduler/cache.py) |
| 可控到达回放、TTFT/ITL、测量开销诊断 | 本项目实验工具 | [阶段一](../stage01_measurement/README.md) |
| 三策略比较、等待状态、固定双阈值冲突规则、回退与日志 | 本项目主要调度改动 | [policy.py](../../python/minisgl/scheduler/policy.py)、[阶段三](../stage03_scheduling/README.md) |
| ShareGPT 预处理、分组划分、六类负载、预算选择与独立确认 | 本项目评估设计与实现 | [阶段四](../stage04_system_evaluation/REPORT.md) |
| overlap 在途引用与终止状态、Radix 活跃页表重映射 | 本项目阶段五修复，提交 `29850fa` | [调度器](../../python/minisgl/scheduler/scheduler.py)、[测试](../../tests/stage05/test_overlap_lifecycle.py)、[Radix 回归](../../tests/stage05/test_radix_live_remap.py) |
| 固定轨迹/隐藏状态探针、生命周期矩阵、Nsight 汇总 | 本项目验证与归因工具 | [阶段五](../stage05_correctness_attribution/REPORT.md)，交付提交 `3a37966` |

## 面试数字的证据入口

| 可使用的结论 | 证据与统计口径 |
|---|---|
| 稳定混合短请求 P99 改善 53.31%，吞吐变化 −0.16% | [comparison.json](../stage04_system_evaluation/evidence/comparison.json)，mixed_steady / confirm，80 请求×5 轮，中位数之比 |
| 突发调优后仅改善 5.53%，默认配置退化 | 同上 long_burst / confirm；预算不同，不能称作固定预算纯策略收益 |
| 61 项测试，持续测试完成 9744 请求 | [lifecycle_summary.json](../stage05_correctness_attribution/evidence/lifecycle_summary.json)，CPU/GPU 分开计数，强制采样检验生命周期 |
| GPU 连续 prefill 阻塞链缩短 | [profile_summary.json](../stage05_correctness_attribution/evidence/profile_summary.json)，单次每策略 Nsight GPU 投影跨度 |
| 一个案例最早观测差异在 attention backend | [隐藏状态报告](../stage05_correctness_attribution/HIDDEN_REPORT.md)，不能推广到所有输出差异 |

## 建议现场打开的代码

按 `BatchPolicy.select` → `Scheduler._schedule_next_batch` → `_process_last_data` → `CacheManager.cache_req` 阅读。先讲策略与资源准入的边界，再讲等待计时和资源安全。用测试中的具体状态解释改动，不需要逐行背完整引擎。

简历采用“基于 Mini-SGLang 实现与验证……”的表述。不要写“从零开发推理引擎”“自研 FlashAttention”“生产环境降低延迟 53%”“严格保证 50 ms”“已实现多卡公平调度”。如果被问到 AI 工具使用，应如实说明辅助范围，并以能够解释、修改、复现的代码证明掌握程度。
