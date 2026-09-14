# 阶段四：系统评估

按根目录 [实验计划](../../实验计划.md) 推进，先固定预算比较策略，再对有价值的策略扫预算。当前为数据准备阶段，尚未运行阶段四先导实验或正式评估。

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

## 待完成评估

1. 先导实验确定纯短、纯长、稳定混合、长请求突发、持续 decode 加新请求、KV 紧张六类场景的低/中/近饱和到达率。
2. 固定预算 1024，对比 prefill_first、decode_first、wait_time；阈值沿用阶段三 50/200 ms，不在评估期间修改规则。
3. 每配置预热并至少测量五次，记录 TTFT、ITL、端到端延迟、吞吐、错误率、等待时间与资源使用。
4. 根据固定预算结果再安排预算扫描；P99 样本不足时扩充请求数，明确说明相关性与统计局限。
5. 输出中文报告、延迟—吞吐对照图和适用边界。后续 overlap/Radix 与深入正确性、Nsight 归因按原计划继续验证。
