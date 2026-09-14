# 阶段四：系统评估

按根目录 [实验计划](../../实验计划.md) 推进，先固定预算比较策略，再对有价值的策略扫预算。已完成 357 组测量与独立校验，结果、负面案例和验证边界见 [中文实验报告](REPORT.md)。

## 数据准备

保留合成负载作为受控对照，并引入 ShareGPT 对话作为真实提示词与长度分布来源。数据集不包含本项目的请求到达时间；到达序列需要单独生成，调参与最终评估使用不同种子，并按会话划分不重叠样本。

- 数据集：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- 文件：`ShareGPT_V3_unfiltered_cleaned_split.json`，文件页面标注约 673 MB。
- 文件页面：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json
- 下载：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json?download=true
- 2026-09-14 文件页面标注 SHA256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

下载到本目录下 `datasets/ShareGPT_V3_unfiltered_cleaned_split.json`。不需要下载整个数据仓库、训练脚本、模型或其他 JSON 变体。

下载后先校验文件哈希和数据结构，再用本地 Llama tokenizer 统计实际输入长度。输出长度采用明确的固定上限；若后续使用参考回答长度作为生成预算，需要单独标记，不将数据集参考文本当成模型生成正确性的标准。

原始对话、逐请求文本及可逆 token 数据只保留本地，不推送。`datasets/`、`runs/`、`prepared/` 均被 Git 忽略。只发布经过检查的脚本、配置、来源哈希与聚合统计。

## 评估范围

1. 先导实验确定纯短、纯长、稳定混合、长请求突发、持续 decode 加新请求、KV 紧张六类场景的低/中/近饱和到达率。
2. 固定预算 1024，对比 prefill_first、decode_first、wait_time；阈值沿用阶段三 50/200 ms，不在评估期间修改规则。
3. 每配置预热并至少测量五次，记录 TTFT、ITL、端到端延迟、吞吐、错误率、等待时间与资源使用。
4. 根据固定预算结果再安排预算扫描；P99 样本不足时扩充请求数，明确说明相关性与统计局限。
5. 输出中文报告、延迟—吞吐对照图和适用边界。后续 overlap/Radix 与深入正确性、Nsight 归因按原计划继续验证。

## 当前实验实现与复现

`prepare.py` 校验固定文件 SHA256，保留完整起始 user 提问（可带首条 system 消息），过滤空或助手开头的分片，按渲染后的 prompt 精确去重。抽样类别按固定顺序遍历，并通过跨进程哈希种子测试，避免集合遍历顺序影响随机抽样。使用本地模型 chat template 和 tokenizer，保留 32–4096 token；short 为 32–512，long 为 2048–4096，中间长度保留在池中但不参加这轮二类对照。按去掉数字分片后缀的原始会话 ID 哈希划分 tune/eval，验证会话与相同 prompt 均不跨组。

六类负载使用真实提示词，但混合场景刻意按长短 1:1 采样，不等于 ShareGPT 总体分布；到达时间是人工生成的。稳定组使用指数分布间隔，突发组长请求每四个一批，decode_new 使用四个输出 512 token 的背景短请求并注入长请求，背景请求不是无限持续流。其他短请求输出 256、长请求输出 128，全部 ignore_eos。输入不会为了凑长度被截断或填充。

正常场景 65536 个单 token KV 页，最大运行请求数 8，避免正常长请求准入被 KV 总容量限制；kv_tight 为 4608 页。单卡 A800，Graph 开启，overlap 关闭，naive cache。预算 1024、阈值 50/200 ms。每配置用同类型的前八条请求、输出八个 token 预热，预热时间不计入正式指标。

```bash
export PYTHONPATH="$PWD/python"
python study/stage04_system_evaluation/prepare.py \
  --dataset study/stage04_system_evaluation/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /path/to/model --output study/stage04_system_evaluation/prepared
python -m unittest discover -s tests/stage04 -v
python study/stage04_system_evaluation/run.py --model /path/to/model --stage pilot
python study/stage04_system_evaluation/run.py --model /path/to/model --stage matrix
python study/stage04_system_evaluation/run.py --model /path/to/model --stage sweep
python study/stage04_system_evaluation/run.py --model /path/to/model --stage confirm
```

先导用 tune 数据进行同时到达容量测量，再探测到达率；若队列或运行请求数尚未形成压力，最多增加两次探测。低/中/near 取压力探测锚点的 0.35/0.65/0.95；这些是有限负载的压力档位，不是已证明的生产系统饱和点。突发场景的 rate 控制短请求间隔和长请求批间隔，不等于合并后所有请求的平均到达率。

固定矩阵为 6 场景 × 3 档位 × 3 策略 × 5 轮，每轮 24 个请求。然后只在 tune 集的稳定混合与突发近容量组比较 prefill_first/1024 与 wait_time 的 256、1024、4096 预算。选择满足基线吞吐 95% 门槛的候选中短请求 P99 最小的预算；无候选时保留 1024，明确标记门槛未满足。最后用不同 seed 的 eval 样本，每轮 80 个请求，对比三策略及选中预算；每轮有 10200 个短请求 ITL 样本。相同 GPU batch 的间隔相关，不能将这些样本当作独立同分布数据。

每轮检查输出长度、完成数、重复完成、token 预算、cache 完整性、槽位与等待状态清理。worker 有超时保护和完成检查点；恢复时检查作业配置及输入/到达序列哈希一致。失败轮保留记录并中止，不跳过失败后继续发布成功结果。固定预算、预算扫描与确认分别记录，禁止混合不同 seed 或样本数报告提升。

```bash
python study/stage04_system_evaluation/summarize.py \
  --runs study/stage04_system_evaluation/runs/main \
  --pools study/stage04_system_evaluation/prepared/pools.json \
  --output study/stage04_system_evaluation/evidence
# 绘图可使用独立安装 matplotlib 的环境，不要求修改推理环境：
python study/stage04_system_evaluation/plot.py study/stage04_system_evaluation/evidence
```

聚合导出会从本地事件重新计算指标，核对输出长度与公平比较的工作负载哈希，只导出允许的聚合字段。原始输入与输出 token、会话 ID 和逐请求轨迹不导出。正式性能结果与图表已收录于 [REPORT.md](REPORT.md)，原始敏感数据仅留在本地。
