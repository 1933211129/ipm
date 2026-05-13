# MLISE 2026 诊断版实验方案：从规模效应转向因果失败瓶颈定位

## 1. 为什么调整方案

上一轮 `Qwen3-0.6B / 4B / 8B × CLadder` 正式实验说明，原来的“模型规模越大，因果一致性越强”主线支撑不足：

- `Accuracy` 只有有限提升，且 `8B` 并未稳定优于 `4B`。
- `PCC` 基本停留在 `0.50` 附近，不能形成清晰规模趋势。
- `Story All-Correct Rate` 全部接近地板，区分度很弱。
- 原有 patching 虽有 logit recovery，但与“因果一致性失败”绑定不够紧。

因此，会议稿不再强行写成 scaling paper，而改成诊断型问题：

> Qwen 模型在 CLadder 上的失败，到底主要来自自然语言因果结构抽取失败，还是来自形式化因果运算失败？

这个问题比单纯比较模型规模更有解释力，也更适合当前结果。

## 2. 新论文主线

建议英文题目：

**Scaling Alone Is Not Enough: Diagnosing Natural-Language and Formal-Causal Bottlenecks in Qwen Models on CLadder**

中文理解：

> 规模提升不足以带来因果一致性：Qwen 模型在 CLadder 上的自然语言理解瓶颈与形式化因果瓶颈诊断。

核心观点：

1. 自然语言条件下，模型规模提升只能带来有限 accuracy 改善。
2. 粗粒度 PCC 不足以解释失败，需要更严格的 contrast-level consistency。
3. 如果 formal scaffold 明显提升表现，说明瓶颈主要在自然语言到因果结构的抽取。
4. 如果 formal scaffold 仍不能提升表现，说明模型对形式化因果运算本身也不稳定。
5. formal-to-natural activation patching 可作为探索性证据，检验 formal scaffold 的隐藏状态是否能恢复自然语言失败样本。

## 3. 行为实验设计

### 3.1 模型

仍然只使用一个模型家族，避免横向模型比较引入额外变量：

- `Qwen3-0.6B`
- `Qwen3-4B`
- `Qwen3-8B`

### 3.2 主数据集

主实验继续使用：

```text
datasets/cladder/data/full_v1.5_default.csv
```

正式版主样本仍使用 `640` 条分层平衡样本，保持与上一轮结果可比。

### 3.3 输入条件

每个样本测试三种输入条件：

| 条件 | 含义 | 诊断目的 |
|---|---|---|
| `nl` | 原始自然语言题干 | 基线表现 |
| `nl_formal` | 原始题干 + 变量映射、因果图、formal query | 检验显式结构提示是否能提升 |
| `formula_only` | 事实条件 + 变量映射、因果图、formal query 和问题句 | 弱化原始故事包装，检验形式化因果运算能力 |

注意：formal scaffold 不给 gold label，也不提供完整推导步骤。`formula_only` 仍保留题干中的事实条件和概率信息，否则许多 CLadder 问题在信息上不可解；它去掉的是原始叙事包装和完整推导，而不是必要的因果机制信息。

### 3.4 Stress splits

利用 CLadder 已有子集，不引入外部新数据集：

- `test-commonsense-v1.5.csv`
- `test-anticommonsense-v1.5.csv`
- `test-noncommonsense-v1.5.csv`
- `test-easy-v1.5.csv`
- `test-hard-v1.5.csv`

每个 split 抽取小规模平衡样本，测试 `nl` 和 `nl_formal` 两种输入条件，观察：

- commonsense 与 anticommonsense 是否有明显差距；
- formal scaffold 是否能缩小常识偏置；
- easy/hard 差距是否比模型规模差距更大。

## 4. 指标设计

### 4.1 Accuracy 与 Parse Rate

保留常规准确率和解析率，但不再把它们当作唯一结论。

### 4.2 Strict Contrast Causal Consistency, CCC

替代原先过粗的 PCC。

只统计严格 contrast pair：

- 同一 `story_id`
- 同一 `query_type`
- 同一 `formal_form`
- gold label 相反

定义：

```text
CCC = 1, if gold labels are opposite and model predictions are also opposite
CCC = 0, otherwise
```

这个指标更直接检验模型是否能跟随最小因果对照发生预测翻转。

### 4.3 Scaffold Gain

定义：

```text
Scaffold Gain = Accuracy(scaffold condition) - Accuracy(nl)
```

分别计算：

- `nl_formal_gain`
- `formula_only_gain`

它回答的是：显式因果结构是否真的帮了模型。

### 4.4 Rescue / Harm 分析

对同一样本比较 `nl` 与 `nl_formal`：

- `rescue`：`nl` 错，`nl_formal` 对。
- `harm`：`nl` 对，`nl_formal` 错。

如果 rescue 明显高于 harm，说明 formal scaffold 有实质帮助。

## 5. 白盒实验设计

上一轮 patching 是 clean/corrupted pair，和主问题绑定不够紧。诊断版改成：

> natural prompt 错、formal scaffold prompt 对；把 formal scaffold 的 residual stream hidden state patch 到 natural prompt，看自然语言失败样本的 gold-label logit margin 是否恢复。

默认模型：

- `Qwen3-4B`

默认设置：

- `16` 个 natural-fail / scaffold-success 样本；
- 只做 residual stream；
- 全层扫描；
- 用 HuggingFace forward hook 离线实现，避免服务器无法访问 HuggingFace 时卡在 TransformerLens 元数据请求。

解释边界：

- 不声称发现完整 causal circuit。
- 只说 formal scaffold 条件下的隐藏状态在部分层中包含可转移的正确答案方向信号。

## 6. 图表要求

实验图表统一使用英文标题和坐标，便于后续直接放入英文会议论文。报告和实验记录仍全部使用中文。

建议图表：

1. `Accuracy by Input Condition`
2. `Scaffold Gain by Model`
3. `Strict Contrast Consistency`
4. `Stress Split Accuracy`
5. `Rescue vs Harm Rate`
6. `Formal-to-Natural Patching Recovery`

不再把标签分布图作为核心图，因为它对论文论证帮助很小。

## 7. 服务器执行方案

本轮使用独立脚本和独立输出目录，避免覆盖上一轮 scaling 结果：

```text
scripts/mlise2026_qwen_diagnostic.py
outputs/mlise2026_qwen_diagnostic/
```

推荐顺序：

1. `sample`：生成主样本和 stress split 样本。
2. `behavior`：三模型主实验，输入条件为 `nl`、`nl_formal`、`formula_only`。
3. `stress`：三模型 stress split 实验，输入条件为 `nl`、`nl_formal`。
4. `patch`：默认在 `Qwen3-4B` 上做 formal-to-natural residual patching。
5. `aggregate`：生成英文图表、中文聚合报告和全部 CSV 表。

核心输出：

- `tables/aggregate/aggregate_diagnostic_results.csv`
- `tables/aggregate/metrics_summary.csv`
- `tables/aggregate/metrics_ccc.csv`
- `tables/aggregate/metrics_scaffold_gain.csv`
- `tables/aggregate/metrics_rescue_harm.csv`
- `patching/qwen3_4b/formal_to_natural_patching_results.csv`
- `figures/*.png`
- `reports/MLISE2026_qwen_diagnostic_report.md`

## 8. 成功标准

这轮实验最理想的结果是：

- `nl_formal` 相比 `nl` 有明显提升；
- `formula_only` 能揭示模型是否具备形式化运算能力；
- CCC 比原 PCC 更有区分度；
- commonsense / anticommonsense 或 easy / hard split 展现出可解释差异；
- formal-to-natural patching 在若干层出现正向 recovery。

如果 scaffold 没有提升，也可以形成更强的 negative 结论：

> Qwen 模型不仅无法从自然语言中稳定抽取因果结构，即使给出形式化因果 scaffold，也没有形成稳定的 contrast-level causal consistency。

这比上一轮单纯说 PCC 接近 0.5 更有诊断价值。
