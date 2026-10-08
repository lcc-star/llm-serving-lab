# 实验复现

## 无 GPU 核验公开证据

从仓库根目录执行，只需 Python 标准库：

```bash
python study/reproduction/reproduce.py evidence
```

该命令从公开 JSON 重算两组扩大样本确认的改善比例，检查样本数、轮数及阶段五验收状态。它核验已发布汇总，不等于从原始数据独立重做实验。原始对话、token 和 Nsight 文件因隐私约束未发布。

## 重新执行

Linux、兼容 NVIDIA GPU/CUDA、本地 Llama-3.1-8B-Instruct 权重。沿用 [上游安装说明](../../docs/upstream-readme.md)，激活包含 torch、FlashInfer、ninja 的环境；完整评估绘图还需 matplotlib，原项目缓存 pytest 测试另需 pytest、pytest-cov。历史硬件为单卡 A800，入口固定使用 GPU 0。模型权重与数据集自行准备，不自动下载或安装依赖。

```bash
# 55 项阶段一至五测试；无需模型或 GPU，但需安装推理包依赖。
python study/reproduction/reproduce.py cpu

# 先查看将执行的命令，不加载模型、不创建输出目录。
python study/reproduction/reproduce.py stage3 --model /path/to/model --dry-run

# 三策略合成负载对照、汇总及决策审计。
python study/reproduction/reproduce.py stage3 --model /path/to/model

# 完整系统评估：数据准备、先导、固定预算、预算扫描、扩大样本确认、聚合与绘图。
python study/reproduction/reproduce.py evaluation --model /path/to/model \
  --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json

# 四组缓存/overlap 生命周期矩阵、十分钟持续测试及补充边界测试。
python study/reproduction/reproduce.py lifecycle --model /path/to/model
```

所有新输出在 `runs/<模式>_<UTC时间>/`，包含版本、完整命令及分步骤日志，默认被 Git 忽略。失败立即停止，不把部分成功当作完整验收；重新调用创建新目录，不覆盖旧记录。完整系统评估需要持续占用 GPU，十分钟仅是 lifecycle 最后的持续测试时间，不是整个矩阵耗时。

数据来源、固定文件 SHA256 和划分方法见 [阶段四数据说明](../stage04_system_evaluation/README.md)。注意：入口在**当前提交**上重新测量；阶段四发布数字来自冻结版本 `bfb319a20860047afedc090339e9b347f27c05e0`，不能把新运行写回旧报告或要求逐位复现时间。先导到达率也会随硬件和运行环境变化。

输出分歧探针和 Nsight 涉及本地阶段四轨迹，复现方法单独保留在 [阶段五入口](../stage05_correctness_attribution/README.md)。统一入口不声称自动复现所有历史诊断。
