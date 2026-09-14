# LLM Serving Lab

面向 LLM 推理系统学习、调度优化与可复现性能实验的独立项目。

本项目基于 [Mini-SGLang](https://github.com/sgl-project/mini-sglang)，保留完整引擎源码与原始 MIT 许可证。它不是从零实现的推理引擎。上游使用说明见 [原始 README](docs/upstream-readme.md)。

## 当前内容

- `python/minisgl/`：继承的推理引擎，模块名称保持 `minisgl`。
- `study/stage00_baselines/`：CUDA Graph、重叠调度和 TP 基线。
- `study/stage01_measurement/`：测量工具和输出差异调查。
- `study/stage02_prefill_interference/`：长 prefill 干扰 decode 的预算对照。
- [实验阶段索引](study/README.md)：各阶段入口。
- `tests/`：上游测试。
- `docs/experiment-plan.md`：调度优化计划。

阶段零到二测量原始策略；阶段三新增可切换的批次策略，默认仍为原始 prefill 优先。详见 [阶段三说明](study/stage03_scheduling/README.md)。

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

已有结果来自 A800、Llama-3.1-8B、固定合成 token 负载，不代表通用性能。TP=1 与 TP=2 输出哈希不同，差异原因尚未定位；两卡内部输出一致。初始化和预热不计入正式测量。

## 来源与许可证

基于上游提交 `9a91cfafe754aa85daee49998176275667eb58f2`。保留 [MIT LICENSE](LICENSE) 中的原始版权声明。新增实验脚本与后续优化通过本仓库提交历史记录。
