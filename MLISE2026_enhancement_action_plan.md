# MLISE 2026 轻量增强实验包行动计划

## 1. 目标与边界

本行动计划用于增强当前 `Qwen3 × CLadder` 诊断实验的证据链。核心目标不是更换模型或数据集，而是在现有实验基础上补充细粒度误差分析、对照 pair 正误分解、输入条件转移分析、stress robustness 统计、形式脚手架成分消融和 patching 对照分析。

当前已完成的基础实验为：

```text
outputs/mlise2026_qwen_diagnostic/runs/diagnostic_20260513_151831
```

基础实验已经包含：

- 主数据集：`CLadder full_v1.5_default`
- 主样本规模：`640`
- stress split：`commonsense`、`anticommonsense`、`noncommonsense`、`easy`、`hard`
- 模型：`Qwen3-0.6B`、`Qwen3-4B`、`Qwen3-8B`
- 输入条件：`nl`、`nl_formal`、`formula_only`
- 已有指标：`Accuracy`、`Parse Rate`、`CCC`、`Scaffold Gain`、`Rescue/Harm`
- 已有白盒实验：`Qwen3-4B` formal-to-natural residual stream patching

本计划的执行边界：

1. 不更换模型家族。
2. 不新增外部数据集。
3. 不覆盖已有输出。
4. 不删除已有结果。
5. Markdown 报告、实验记录、行动记录使用中文。
6. 实验图表标题、坐标轴和图例使用英文。
7. 在用户确认前，只修改行动文档，不启动新增实验。

建议新增输出根目录：

```text
outputs/mlise2026_qwen_enhanced/
```

每次正式执行写入新的 run 目录，例如：

```text
outputs/mlise2026_qwen_enhanced/runs/enhanced_YYYYMMDD_HHMMSS/
```

该目录通过 `source_run_id` 明确引用基础实验：

```text
source_run_id = diagnostic_20260513_151831
```

## 2. 需要补强的证据链

当前基础结果已经说明：

- 形式脚手架没有稳定提升主实验 accuracy。
- Qwen3-4B 和 Qwen3-8B 的 CCC 明显高于 Qwen3-0.6B。
- patching 在部分层和部分样本中存在正向 recovery。

但论文证据链仍需回答三个更细的问题：

1. 模型错误集中在哪些 query type 和 rung？
2. CCC 较高时，模型到底是在正确翻转，还是在错误翻转？
3. 形式脚手架为什么会同时产生 rescue 和 harm？

因此，增强实验包按如下逻辑组织：

```text
已有预测结果重分析
  -> query type / rung 细粒度表现
  -> CCC 正误翻转分解
  -> 输入条件正确性转移矩阵
  -> stress split worst-case 与方差
  -> bootstrap 置信区间与配对检验

小规模新增推理
  -> formal component ablation

小规模新增白盒对照
  -> patching layer profile
  -> random patch control
```

## 3. 新增实验与分析清单

### 3.1 Experiment 1: Fine-Grained Causal Task Analysis

目的：定位模型错误是否集中在特定因果问题类型或 rung 上，而不是只报告整体 accuracy。

是否需要重新跑模型：不需要。

输入文件：

```text
outputs/mlise2026_qwen_diagnostic/runs/diagnostic_20260513_151831/tables/aggregate/all_main_eval.csv
```

新增统计：

```text
Model × Input × Query Type Accuracy
Model × Input × Query Type Scaffold Gain
Model × Input × Rung Accuracy
Model × Input × Rung Scaffold Gain
```

新增输出表：

```text
tables/enhanced/metrics_query_type_accuracy.csv
tables/enhanced/metrics_query_type_scaffold_gain.csv
tables/enhanced/metrics_rung_accuracy.csv
tables/enhanced/metrics_rung_scaffold_gain.csv
```

新增英文图：

```text
figures/01_query_type_accuracy_heatmap.png
figures/02_query_type_scaffold_gain_heatmap.png
figures/03_rung_accuracy_by_model.png
```

论文中可支撑的问题：

- 哪些因果任务类型最容易失败？
- 形式脚手架是否只对部分 query type 有帮助？
- rung 越高是否越难？

### 3.2 Experiment 2: Contrast Pair Correctness Decomposition

目的：解释 CCC 与 accuracy 之间的张力。CCC 只说明模型预测是否随严格对照发生翻转，但不区分翻转是否正确。

是否需要重新跑模型：不需要。

输入文件：

```text
tables/aggregate/metrics_ccc_pairs.csv
tables/aggregate/all_main_eval.csv
```

严格对照 pair 满足：

```text
same story_id
same query_type
same formal_form
gold labels are opposite
```

新增 pair-level 分类：

```text
Correct Flip = 1, if pred_i = gold_i and pred_j = gold_j
Wrong Flip   = 1, if pred_i != gold_i and pred_j != gold_j
Invariant-Yes = 1, if pred_i = pred_j = yes
Invariant-No  = 1, if pred_i = pred_j = no
```

新增指标：

```text
SCCA = mean(Correct Flip)
Signed CCC = Correct Flip Rate - Wrong Flip Rate
```

需要注意：

```text
CCC = Correct Flip Rate + Wrong Flip Rate
```

因此，较高 CCC 不必然表示正确因果判断。它可能包含一部分方向翻转但整体翻错的 pair。

新增输出表：

```text
tables/enhanced/metrics_contrast_decomposition.csv
tables/enhanced/metrics_contrast_decomposition_by_query.csv
tables/enhanced/contrast_pair_level_labels.csv
```

新增英文图：

```text
figures/04_correct_vs_wrong_flip.png
figures/05_signed_ccc_by_model.png
```

论文中可支撑的问题：

- 较高 CCC 是否来自正确翻转？
- 4B/8B 的局部对照敏感性是否等价于正确判断？
- 哪些 query type 中 wrong flip 更常见？

### 3.3 Experiment 3: Scaffold Transition Analysis

目的：将 Rescue/Harm 从简单计数扩展成输入条件正确性轨迹，观察形式输入如何改变样本层面的答案稳定性。

是否需要重新跑模型：不需要。

输入文件：

```text
tables/aggregate/all_main_eval.csv
```

对每个模型和每个样本，统计三种输入条件的正确性轨迹：

```text
nl -> nl_formal -> formula_only
```

其中 `C` 表示 correct，`W` 表示 wrong。重点轨迹包括：

```text
C-C-C: 三种输入都正确
W-W-W: 三种输入都错误
W-C-C: 形式信息稳定救回
C-W-W: 形式信息稳定破坏
W-C-W: nl_formal 短暂救回，但 formula_only 又失败
C-W-C: nl_formal 破坏，但 formula_only 恢复
```

新增输出表：

```text
tables/enhanced/metrics_transition_patterns.csv
tables/enhanced/metrics_transition_patterns_by_query.csv
tables/enhanced/sample_transition_labels.csv
```

新增英文图：

```text
figures/06_transition_pattern_distribution.png
figures/07_transition_patterns_by_query.png
```

论文中可支撑的问题：

- 形式脚手架是稳定修复，还是诱发样本级答案迁移？
- 被修复样本和被破坏样本是否集中在不同 query type？
- `formula_only` 是否延续 `nl_formal` 的修复效果？

### 3.4 Experiment 4: Stress Robustness Analysis

目的：将 stress split 从均值描述扩展为鲁棒性统计，量化 best/worst split 差异和跨 split 方差。

是否需要重新跑模型：不需要。

输入文件：

```text
tables/aggregate/all_stress_eval.csv
```

新增统计：

```text
Accuracy per split
Mean Accuracy
Worst-split Accuracy
Best-split Accuracy
Std across splits
Scaffold Gain per split
Worst-split Scaffold Gain
```

新增输出表：

```text
tables/enhanced/metrics_stress_robustness.csv
tables/enhanced/metrics_stress_scaffold_gain.csv
```

新增英文图：

```text
figures/08_stress_split_robustness.png
figures/09_stress_worst_case_accuracy.png
```

论文中可支撑的问题：

- 形式脚手架的平均改善是否被个别 split 拉动？
- 模型是否在 worst-case split 中退化到随机水平？
- commonsense、anticommonsense、hard 等 split 的差异是否足以支撑鲁棒性结论？

### 3.5 Experiment 5: Formal Component Ablation

目的：定位 `nl_formal` 中的具体形式成分对模型的影响，区分变量映射、causal graph 和 formal query 的作用。

是否需要重新跑模型：需要，新增小规模主样本推理。

建议新增两个输入条件：

```text
nl_var_query:
原始自然语言题干 + 变量映射 + formal query
不提供 causal graph

nl_var_graph:
原始自然语言题干 + 变量映射 + causal graph
不提供 formal query
```

形成如下消融链条：

```text
nl
nl_var_query
nl_var_graph
nl_formal
formula_only
```

优先执行范围：

```text
Qwen3-4B
Qwen3-8B
640 主样本
```

可选扩展：

```text
Qwen3-0.6B
```

推荐原因：Qwen3-0.6B 已接近随机水平，正文可以重点讨论 4B/8B；若需要模型集合完全一致，再补跑 0.6B。

新增输出表：

```text
tables/ablation/<model>/ablation_eval.csv
tables/enhanced/metrics_component_ablation.csv
tables/enhanced/metrics_component_ablation_contrast.csv
tables/enhanced/metrics_component_ablation_gain.csv
```

新增英文图：

```text
figures/10_component_ablation_accuracy.png
figures/11_component_ablation_ccc.png
figures/12_component_ablation_gain.png
```

论文中可支撑的问题：

- formal query 是否比 causal graph 更有帮助？
- causal graph 是否引入额外负担？
- 完整 `nl_formal` 的下降是否来自信息过载或结构成分干扰？

### 3.6 Experiment 6: Bootstrap CI and Paired Tests

目的：为主表和新增表补充不确定性估计，避免只报告点估计。

是否需要重新跑模型：不需要。

建议统计：

```text
Accuracy 95% bootstrap CI
CCC 95% bootstrap CI
SCCA 95% bootstrap CI
Scaffold Gain 95% bootstrap CI
nl vs nl_formal paired bootstrap
nl vs nl_formal McNemar test
```

bootstrap 设置：

```text
n_bootstrap = 1000
seed = 42
confidence = 95%
```

新增输出表：

```text
tables/enhanced/bootstrap_accuracy_ci.csv
tables/enhanced/bootstrap_ccc_ci.csv
tables/enhanced/bootstrap_scca_ci.csv
tables/enhanced/bootstrap_scaffold_gain_ci.csv
tables/enhanced/paired_tests_nl_vs_nl_formal.csv
```

论文中可支撑的问题：

- Qwen3-4B 的 `nl` 与 `nl_formal` accuracy 差异是否稳定？
- Qwen3-8B 的 CCC 差异是否有足够置信度？
- scaffold gain 的负值是否可能只是抽样波动？

### 3.7 Experiment 7: Exploratory Patching Enhancement

目的：保留当前 patching 的探索性定位，同时补充层级分布和随机对照，避免只报告均值与最大值。

是否需要重新跑模型：部分需要。

不需要重新跑模型的部分：

```text
从已有 formal_to_natural_patching_results.csv 计算 layer-wise profile
列出 top-5 recovery layers
按样本统计 max recovery layer
```

需要新增 forward hook 的部分：

```text
random patch control
```

random patch control 定义：

```text
matched patch:
将同一样本的 nl_formal residual patch 到该样本 nl prompt

random patch:
将另一个随机样本的 nl_formal residual patch 到当前样本 nl prompt
```

建议执行范围：

```text
Qwen3-4B
当前 16 个 patching 样本
residual stream
36 层
random seed = 42
```

新增输出表：

```text
tables/enhanced/patching_layer_profile.csv
tables/enhanced/patching_top_layers.csv
tables/enhanced/patching_random_control.csv
tables/enhanced/patching_matched_vs_random_summary.csv
```

新增英文图：

```text
figures/13_patching_layer_profile.png
figures/14_matched_vs_random_patching.png
```

论文中可支撑的问题：

- recovery 是否集中在少数层？
- matched formal-to-natural patch 是否优于随机 formal residual patch？
- patching 信号是否只是任意形式输入 residual 带来的偶然扰动？

## 4. 推荐执行顺序

### 4.1 第一批：完全不需要重新跑模型

第一批只读取已有预测结果和 patching 结果，适合先快速完成：

1. Fine-Grained Causal Task Analysis
2. Contrast Pair Correctness Decomposition
3. Scaffold Transition Analysis
4. Stress Robustness Analysis
5. Bootstrap CI and Paired Tests
6. Patching layer-wise profile 和 top-layer summary

建议新增脚本：

```text
scripts/mlise2026_qwen_enhanced_analysis.py
```

建议命令形式：

```bash
python scripts/mlise2026_qwen_enhanced_analysis.py \
  --source-run outputs/mlise2026_qwen_diagnostic/runs/diagnostic_20260513_151831 \
  --output-root outputs/mlise2026_qwen_enhanced \
  --run-id enhanced_YYYYMMDD_HHMMSS \
  --stage analysis
```

### 4.2 第二批：小规模新增推理

第二批需要服务器 GPU，但只新增两个输入条件，主样本规模仍为 640：

1. Formal Component Ablation
2. Random Patch Control

建议命令形式：

```bash
python scripts/mlise2026_qwen_enhanced_analysis.py \
  --source-run outputs/mlise2026_qwen_diagnostic/runs/diagnostic_20260513_151831 \
  --output-root outputs/mlise2026_qwen_enhanced \
  --run-id enhanced_YYYYMMDD_HHMMSS \
  --stage ablation \
  --model qwen3_4b
```

```bash
python scripts/mlise2026_qwen_enhanced_analysis.py \
  --source-run outputs/mlise2026_qwen_diagnostic/runs/diagnostic_20260513_151831 \
  --output-root outputs/mlise2026_qwen_enhanced \
  --run-id enhanced_YYYYMMDD_HHMMSS \
  --stage patch_control \
  --patch-model qwen3_4b
```

### 4.3 第三批：聚合、图表和论文段落更新

第三批只在第一批和第二批完成后执行：

```text
aggregate
generate figures
write Chinese experiment notes
update paper experiment/conclusion draft
```

注意：正式论文正文更新只根据新增结果写，不写执行过程，不写“上一轮”或“方案调整”之类表述。

## 5. 最小增强版与完整增强版

### 5.1 最小增强版

如果时间紧，优先完成以下内容：

1. Query type / rung 细粒度 accuracy 与 scaffold gain。
2. Correct Flip / Wrong Flip / SCCA / Signed CCC。
3. `nl -> nl_formal -> formula_only` 正确性转移矩阵。
4. Stress split worst-case accuracy 与 std。
5. Bootstrap 95% CI 与 paired test。
6. Patching layer-wise profile。

优点：

- 不需要重新跑模型。
- 能显著增强论文解释力。
- 风险低，最快形成新表和新图。

### 5.2 完整增强版

在最小增强版基础上增加：

1. `nl_var_query` 与 `nl_var_graph` formal component ablation。
2. random patch control。

优点：

- 能解释形式脚手架到底哪个成分造成帮助或干扰。
- 能增强 patching 结果的可信度。
- 论文更像完整实验研究，而不是只做结果描述。

## 6. 验收标准

### 6.1 表格验收

至少生成以下表格：

```text
metrics_query_type_accuracy.csv
metrics_query_type_scaffold_gain.csv
metrics_rung_accuracy.csv
metrics_contrast_decomposition.csv
metrics_transition_patterns.csv
metrics_stress_robustness.csv
bootstrap_accuracy_ci.csv
bootstrap_ccc_ci.csv
patching_layer_profile.csv
```

完整增强版还应生成：

```text
metrics_component_ablation.csv
metrics_component_ablation_contrast.csv
patching_random_control.csv
patching_matched_vs_random_summary.csv
```

### 6.2 图表验收

图表标题、坐标轴和图例必须使用英文。至少生成：

```text
Query Type Accuracy Heatmap
Query Type Scaffold Gain
Rung Accuracy by Model
Correct vs Wrong Flip
Transition Pattern Distribution
Stress Split Robustness
Patching Layer Profile
```

完整增强版还应生成：

```text
Component Ablation Accuracy
Component Ablation CCC
Matched vs Random Patching
```

### 6.3 论文叙事验收

增强后的论文结论应能支撑以下表述：

> 模型在 CLadder 上的失败不是单一的准确率不足，而表现为多层结构性不稳定：不同 query type 和 rung 的错误分布不均衡；严格对照中的预测翻转同时包含 correct flip 和 wrong flip；形式脚手架在样本层面同时产生 rescue 与 harm；形式输入在部分隐藏层中包含可转移答案信号，但该信号没有稳定转化为行为收益。

正式论文中不写：

- 实验方案修改过程。
- “上一轮结果如何”。
- “与单纯比较模型规模不同”。
- 任何报告式流水账。

## 7. 当前待确认事项

需要用户确认后再执行：

1. 是否先做最小增强版。
2. formal component ablation 是否只跑 `Qwen3-4B` 和 `Qwen3-8B`，还是三模型都跑。
3. random patch control 是否保留 16 个样本，还是扩到更多 natural-fail / scaffold-success 样本。
4. bootstrap 是否采用 1000 次，还是为了速度先用 500 次预览。
