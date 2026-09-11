# 阶段二：长 prefill 对 decode 的干扰

固定四个短请求（输入128、输出256 token）在0秒进入；混合组额外在0.3秒注入四个长请求（输入4096、输出128 token）。短请求对照组没有长请求。两组各扫描 prefill 预算256/1024/4096，预热后各测五次，轮换执行顺序。

保持原始 prefill 优先策略、单卡、Graph开启、重叠关闭、naive缓存。32768个单token KV页、最大8请求、8192上下文，确保容量不会成为本场景的主要约束。只改变运行时 prefill 预算，Engine 初始化容量保持一致。

```bash
export PYTHONPATH="$PWD/python"
CUDA_VISIBLE_DEVICES=0 python study/stage02_prefill_interference/run.py --model /path/to/model --output study/stage02_prefill_interference/runs/main
python study/stage02_prefill_interference/summarize.py study/stage02_prefill_interference/runs/main
python -m unittest discover -s tests/stage02 -v
```

复用阶段一的CPU测量口径。新增 execution span 从提交前到原有结果处理完成，包含GPU执行、等待和CPU处理，不是纯GPU耗时，也没有逐轮额外CUDA同步。对最差短请求输出间隔，统计其中完整落入的prefill批次数、处理token数、CPU观测执行跨度和decode次数。

只读指标不能单独证明因果：通过无长请求对照、固定突发和预算干预共同分析。P99仅作小样本描述，不做生产尾延迟承诺。原始输出均保存；调度变化会影响BF16结果，数量正确不等于语义正确。
