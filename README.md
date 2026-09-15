# LLM Serving Lab

面向 LLM 推理系统学习、调度优化与可复现性能实验的独立项目。

本项目基于 [Mini-SGLang](https://github.com/sgl-project/mini-sglang)，保留完整引擎源码与原始 MIT 许可证。它不是从零实现的推理引擎。上游使用说明见 [原始 README](docs/upstream-readme.md)。

## 项目结果与阅读入口

核心工作是带等待阈值的批次调度及其验证。单卡 A800、Llama-3.1-8B-Instruct 的稳定混合受控负载中，每轮 80 请求、五轮确认的短请求 P99 ITL 中位数由 215.56 降至 100.64 ms（降低 53.31%），输出吞吐变化 −0.16%。突发默认配置出现退化，收益有场景限制；详见 [正式评估](study/stage04_system_evaluation/REPORT.md)。

- [项目总报告](study/stage06_interview/PROJECT_REPORT.md)：问题、方案、结果、代价与局限。
- [贡献与证据](study/stage06_interview/CONTRIBUTIONS.md)：明确区分上游基础设施与新增工作。
- [复现入口](study/stage06_interview/README.md)：公开证据核验、测试和独立目录重新测量。
- [面试讲解](study/stage06_interview/INTERVIEW.md) 与 [简历描述](study/stage06_interview/RESUME.md)。
- [六阶段索引](study/README.md) 与 [原定实验计划](实验计划.md)。

阶段五另修复 overlap 请求终止/资源释放和 Radix 活跃页表问题，并完成生命周期验证与 Nsight 归因，见 [报告](study/stage05_correctness_attribution/REPORT.md)。这些修复不被计作上述阶段四性能收益的来源。

```bash
# 无需 GPU 或私有数据，从公开汇总重新计算核心结果。
python study/stage06_interview/reproduce.py evidence
```

## 环境与实验

需要 Linux、兼容的 NVIDIA GPU/CUDA 环境。按原始 README 安装依赖：

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
export MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python study/stage00_baselines/bench_cuda_graph.py --graph 8 --requests 8
CUDA_VISIBLE_DEVICES=0 MINISGL_DISABLE_OVERLAP_SCHEDULING=0 python study/stage00_baselines/bench_overlap.py --graph 8 --requests 8
CUDA_VISIBLE_DEVICES=0,1 python study/stage00_baselines/bench_tp.py --tp 2
```

TP 自定义通信扩展需要链接 NCCL；确保编译器可以找到 `libnccl.so`，运行时可以找到 `libnccl.so.2`。

阶段零基线来自 A800、Llama-3.1-8B、固定合成 token 负载，不代表通用性能。TP=1 与 TP=2 输出哈希不同，差异原因尚未定位；两卡内部输出一致。初始化和预热不计入正式测量。

## 来源与许可证

基于上游提交 `9a91cfafe754aa85daee49998176275667eb58f2`。保留 [MIT LICENSE](LICENSE) 中的原始版权声明。新增实验脚本与后续优化通过本仓库提交历史记录。
