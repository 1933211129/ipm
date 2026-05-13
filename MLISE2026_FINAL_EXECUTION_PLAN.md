# MLISE 2026 最终完整执行方案

## 1. 当前状态

本地仓库已清理旧实验产物。已删除内容包括：

- `outputs/` 下所有已提交的实验结果、表格、日志、报告和图表；
- 旧版 MLISE scaling / diagnostic / enhancement 过程文档；
- 旧版 pilot 报告；
- 本地缓存文件。

保留内容包括：

- `datasets/`：CLadder 数据集；
- `scripts/`：已有实验脚本，作为最终脚本实现参考；
- `requirements-mlise-a100.txt`：服务器实验环境依赖；
- `AGENTS.md`：服务器、目录和协作约束；
- `GPT55_HANDOFF.md`：接手说明，已更新为指向本最终执行方案；
- `IP&M.md`、`IPM_rewrite.md`：长期项目参考，不作为 MLISE 2026 当前执行主线。

后续所有正式结果必须重新生成，不复用已删除的旧结果。

## 2. 研究目标

本文围绕 CLadder 因果推理任务，评估 Qwen3 同一模型家族在自然语言因果问题、形式结构提示、严格对照一致性和隐藏层信号上的表现。实验不更换模型家族，不新增外部数据集，而是在同一数据框架内增强误差分析、消融分析和机制探索。

核心研究问题：

1. 模型在不同 query type 和 rung 上的错误是否均匀分布？
2. 严格对照样本中的预测翻转是正确翻转，还是错误翻转？
3. 形式脚手架在样本层面是稳定修复，还是同时产生 rescue 与 harm？
4. 形式脚手架中的变量映射、因果图和 formal query 分别如何影响模型表现？
5. stress split 下模型的 worst-case 表现和跨 split 方差如何？
6. 形式输入条件下的 hidden state 是否包含可转移到自然语言条件的答案方向信号？

## 3. 数据与模型

### 3.1 数据集

主数据集：

```text
datasets/cladder/data/full_v1.5_default.csv
```

主实验样本：

```text
640 samples
```

抽样规则：

| Query Type | Samples |
|---|---:|
| marginal | 100 |
| correlation | 100 |
| ate | 100 |
| backadj | 100 |
| det-counterfactual | 60 |
| ett | 60 |
| nie | 60 |
| nde | 60 |

每个 query type 内保持 yes/no 标签平衡。抽样结果保存为：

```text
tables/selected_main_subset.csv
```

stress split：

```text
datasets/cladder/data/test-commonsense-v1.5.csv
datasets/cladder/data/test-anticommonsense-v1.5.csv
datasets/cladder/data/test-noncommonsense-v1.5.csv
datasets/cladder/data/test-easy-v1.5.csv
datasets/cladder/data/test-hard-v1.5.csv
```

每个 split 抽取：

```text
100 samples
```

每个 split 内保持 yes/no 标签平衡。stress 样本保存为：

```text
tables/selected_stress_subset.csv
```

### 3.2 模型

正式模型：

| Model Key | Model | Server Path |
|---|---|---|
| `qwen3_0_6b` | Qwen3-0.6B | `/data/LLM/Qwen/Qwen3-0___6B` |
| `qwen3_4b` | Qwen3-4B | `/data/LLM/Qwen/Qwen3-4B` |
| `qwen3_8b` | Qwen3-8B | `/data/LLM/Qwen/Qwen3-8B` |

行为实验三模型全部运行。白盒 patching 默认只在 Qwen3-4B 上运行。

## 4. 输入条件

### 4.1 主输入条件

主实验使用三种输入条件：

| Condition | 内容 | 目的 |
|---|---|---|
| `nl` | 原始自然语言题干 | 基线 |
| `nl_formal` | 原始题干 + 变量映射 + causal graph + formal query | 完整形式脚手架 |
| `formula_only` | 事实条件 + 变量映射 + causal graph + formal query + 问题句 | 形式化输入 |

注意：

- 不提供 oracle label；
- 不提供 CLadder 完整推导步骤；
- `formula_only` 保留事实条件和概率信息，因为这些是任务求解所必需的信息。

### 4.2 形式成分消融条件

新增两个消融输入条件：

| Condition | 内容 | 目的 |
|---|---|---|
| `nl_var_query` | 原始题干 + 变量映射 + formal query | 检验 formal query 的作用 |
| `nl_var_graph` | 原始题干 + 变量映射 + causal graph | 检验 causal graph 的作用 |

完整输入链条：

```text
nl
nl_var_query
nl_var_graph
nl_formal
formula_only
```

消融实验默认三模型全部运行，以保持表格一致性。正文分析可以重点讨论 Qwen3-4B 和 Qwen3-8B。

## 5. 指标体系

### 5.1 基础行为指标

```text
Accuracy
Parse Rate
Invalid Rate
Latency
```

解析规则：

1. 优先解析 `Final answer: yes/no`；
2. 其次解析输出文本中的独立 `yes` 或 `no`；
3. 解析失败记为 `invalid`；
4. 不使用外部模型或人工修补主实验答案。

### 5.2 细粒度任务指标

按以下维度统计：

```text
Model × Input × Query Type Accuracy
Model × Input × Query Type Scaffold Gain
Model × Input × Rung Accuracy
Model × Input × Rung Scaffold Gain
```

用于回答：

- 哪些 query type 错误集中？
- rung 越高是否越难？
- 形式脚手架是否只对部分因果任务有效？

### 5.3 严格对照指标

严格对照 pair 定义：

```text
same story_id
same query_type
same formal_form
gold labels are opposite
```

基础 CCC：

```text
CCC = 1, if pred_i != pred_j
CCC = 0, otherwise
```

新增正误分解：

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

解释原则：

```text
CCC = Correct Flip Rate + Wrong Flip Rate
```

因此 CCC 高不必然表示正确因果判断，需要同时报告 Correct Flip、Wrong Flip、SCCA 和 Signed CCC。

### 5.4 输入条件转移指标

对每个模型和每个样本统计：

```text
nl -> nl_formal -> formula_only
```

其中：

```text
C = correct
W = wrong
```

重点轨迹：

```text
C-C-C: 三种输入都正确
W-W-W: 三种输入都错误
W-C-C: 形式信息稳定救回
C-W-W: 形式信息稳定破坏
W-C-W: nl_formal 短暂救回，但 formula_only 又失败
C-W-C: nl_formal 破坏，但 formula_only 恢复
```

同时按 query type 统计轨迹分布。

### 5.5 Stress robustness 指标

按五个 stress split 分别统计：

```text
Accuracy per split
Scaffold Gain per split
Mean Accuracy
Worst-split Accuracy
Best-split Accuracy
Std across splits
Worst-split Scaffold Gain
```

用于量化跨 split 鲁棒性，而不是只报告均值。

### 5.6 不确定性估计与显著性检验

bootstrap 设置：

```text
n_bootstrap = 1000
seed = 42
confidence = 95%
```

需要报告：

```text
Accuracy 95% CI
CCC 95% CI
SCCA 95% CI
Scaffold Gain 95% CI
```

配对比较：

```text
nl vs nl_formal paired bootstrap
nl vs nl_formal McNemar test
```

## 6. 白盒分析

### 6.1 Matched formal-to-natural patching

默认模型：

```text
Qwen3-4B
```

样本选择：

```text
nl condition is wrong
nl_formal condition is correct
parsed labels are valid
```

样本数：

```text
16 samples
```

方法：

```text
HuggingFace forward hook
residual stream
last-token hidden state
all layers
```

恢复指标：

```text
absolute recovery = patched gold margin - nl gold margin
normalized recovery = (patched gold margin - nl gold margin) / (nl_formal gold margin - nl gold margin)
```

### 6.2 Layer-wise profile

从 matched patching 结果中计算：

```text
mean recovery by layer
median recovery by layer
positive recovery ratio by layer
top-5 layers by mean recovery
top-5 layers by max recovery
max recovery layer per sample
```

### 6.3 Random patch control

随机对照定义：

```text
matched patch:
同一样本 nl_formal residual -> 该样本 nl prompt

random patch:
另一个随机样本 nl_formal residual -> 当前样本 nl prompt
```

默认设置：

```text
Qwen3-4B
16 matched samples
36 layers
random seed = 42
```

比较指标：

```text
matched mean recovery
random mean recovery
matched positive recovery ratio
random positive recovery ratio
matched vs random paired difference
```

解释边界：

- 不声称发现完整 causal circuit；
- 不声称模型具备人类式因果推理；
- patching 只作为探索性 hidden-state evidence。

## 7. 最终脚本规划

建议新增统一最终脚本：

```text
scripts/mlise2026_qwen_final.py
```

该脚本可以复用现有：

```text
scripts/mlise2026_qwen_diagnostic.py
scripts/mlise2026_qwen_scaling.py
scripts/qwen3_cladder_feasibility.py
```

最终脚本需要支持：

```text
--stage sample|behavior|ablation|stress|analysis|patch|patch-control|aggregate|all
--model qwen3_0_6b|qwen3_4b|qwen3_8b|all
--run-id <name>
--resume
--output-root outputs/mlise2026_qwen_final
--source-run <path>
--patch-model qwen3_4b
--batch-size <int>
--max-new-tokens 48
--bootstrap 1000
```

最终输出根目录：

```text
outputs/mlise2026_qwen_final/
```

正式 run 示例：

```text
outputs/mlise2026_qwen_final/runs/final_YYYYMMDD_HHMMSS/
```

## 8. 服务器执行顺序

### 8.1 本地准备

1. 基于本方案实现或整理 `scripts/mlise2026_qwen_final.py`。
2. 本地只运行语法检查：

```bash
python -m py_compile scripts/mlise2026_qwen_final.py
```

3. 提交并推送代码。
4. 服务器 `/data/kongyb/ipm` 执行：

```bash
git pull origin main
```

### 8.2 服务器环境检查

使用 `ipm-mlise` 环境：

```bash
conda activate ipm-mlise
```

检查内容：

```text
Python
PyTorch
CUDA
Transformers
GPU 列表
模型路径存在性
数据集路径存在性
```

环境检查写入：

```text
reports/server_environment_report.md
```

### 8.3 抽样

```bash
python scripts/mlise2026_qwen_final.py \
  --stage sample \
  --run-id final_YYYYMMDD_HHMMSS \
  --output-root outputs/mlise2026_qwen_final
```

输出：

```text
tables/selected_main_subset.csv
tables/selected_stress_subset.csv
reports/sample_report.md
```

### 8.4 主行为实验

三模型并行运行，输入条件为：

```text
nl
nl_formal
formula_only
```

示例：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/mlise2026_qwen_final.py \
  --stage behavior \
  --model qwen3_0_6b \
  --run-id final_YYYYMMDD_HHMMSS \
  --resume \
  > outputs/mlise2026_qwen_final/runs/final_YYYYMMDD_HHMMSS/logs/qwen3_0_6b_behavior.log 2>&1 &
```

4B 和 8B 使用独立 GPU 与独立日志。

### 8.5 形式成分消融

三模型并行运行，新增输入条件为：

```text
nl_var_query
nl_var_graph
```

示例：

```bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/mlise2026_qwen_final.py \
  --stage ablation \
  --model qwen3_4b \
  --run-id final_YYYYMMDD_HHMMSS \
  --resume \
  > outputs/mlise2026_qwen_final/runs/final_YYYYMMDD_HHMMSS/logs/qwen3_4b_ablation.log 2>&1 &
```

### 8.6 Stress split

三模型并行运行，输入条件为：

```text
nl
nl_formal
```

输出用于 robustness 统计。

### 8.7 纯分析阶段

不加载模型，只读取已有 `main_eval.csv`、`ablation_eval.csv` 和 `stress_eval.csv`：

```bash
python scripts/mlise2026_qwen_final.py \
  --stage analysis \
  --run-id final_YYYYMMDD_HHMMSS \
  --bootstrap 1000
```

生成：

```text
query type / rung metrics
contrast decomposition
transition matrix
stress robustness
bootstrap CI
paired tests
```

### 8.8 Patching

先运行 matched patching：

```bash
CUDA_VISIBLE_DEVICES=7 nohup python scripts/mlise2026_qwen_final.py \
  --stage patch \
  --patch-model qwen3_4b \
  --run-id final_YYYYMMDD_HHMMSS \
  --resume \
  > outputs/mlise2026_qwen_final/runs/final_YYYYMMDD_HHMMSS/logs/qwen3_4b_patch.log 2>&1 &
```

再运行 random patch control：

```bash
CUDA_VISIBLE_DEVICES=7 nohup python scripts/mlise2026_qwen_final.py \
  --stage patch-control \
  --patch-model qwen3_4b \
  --run-id final_YYYYMMDD_HHMMSS \
  --resume \
  > outputs/mlise2026_qwen_final/runs/final_YYYYMMDD_HHMMSS/logs/qwen3_4b_patch_control.log 2>&1 &
```

### 8.9 聚合、作图与报告

```bash
python scripts/mlise2026_qwen_final.py \
  --stage aggregate \
  --run-id final_YYYYMMDD_HHMMSS
```

要求：

- 图表标题、坐标轴、图例使用英文；
- 自动生成的实验记录、阶段总结、Markdown 报告使用中文；
- 论文正文草稿使用正式学术写法，不写执行过程，不写旧方案对比。

## 9. 输出结构

最终 run 目录：

```text
outputs/mlise2026_qwen_final/runs/final_YYYYMMDD_HHMMSS/
├── config.json
├── logs/
├── tables/
│   ├── selected_main_subset.csv
│   ├── selected_stress_subset.csv
│   ├── qwen3_0_6b/
│   ├── qwen3_4b/
│   ├── qwen3_8b/
│   ├── aggregate/
│   └── enhanced/
├── figures/
├── patching/
│   └── qwen3_4b/
├── reports/
└── paper/
```

核心表格：

```text
tables/aggregate/main_results.csv
tables/enhanced/metrics_query_type_accuracy.csv
tables/enhanced/metrics_query_type_scaffold_gain.csv
tables/enhanced/metrics_rung_accuracy.csv
tables/enhanced/metrics_contrast_decomposition.csv
tables/enhanced/metrics_contrast_decomposition_by_query.csv
tables/enhanced/metrics_transition_patterns.csv
tables/enhanced/metrics_transition_patterns_by_query.csv
tables/enhanced/metrics_stress_robustness.csv
tables/enhanced/bootstrap_accuracy_ci.csv
tables/enhanced/bootstrap_ccc_ci.csv
tables/enhanced/bootstrap_scca_ci.csv
tables/enhanced/bootstrap_scaffold_gain_ci.csv
tables/enhanced/paired_tests_nl_vs_nl_formal.csv
tables/enhanced/metrics_component_ablation.csv
tables/enhanced/metrics_component_ablation_contrast.csv
tables/enhanced/patching_layer_profile.csv
tables/enhanced/patching_top_layers.csv
tables/enhanced/patching_random_control.csv
tables/enhanced/patching_matched_vs_random_summary.csv
```

核心图表：

```text
figures/01_overall_accuracy_and_ccc.png
figures/02_query_type_accuracy_heatmap.png
figures/03_query_type_scaffold_gain_heatmap.png
figures/04_rung_accuracy_by_model.png
figures/05_correct_vs_wrong_flip.png
figures/06_signed_ccc_by_model.png
figures/07_transition_pattern_distribution.png
figures/08_transition_patterns_by_query.png
figures/09_stress_split_robustness.png
figures/10_stress_worst_case_accuracy.png
figures/11_component_ablation_accuracy.png
figures/12_component_ablation_ccc.png
figures/13_component_ablation_gain.png
figures/14_patching_layer_profile.png
figures/15_matched_vs_random_patching.png
```

报告与论文草稿：

```text
reports/MLISE2026_final_experiment_report.md
paper/MLISE2026_experiments_and_conclusion_zh.md
```

## 10. 论文正文结构

最终论文实验部分建议组织为：

```text
4. Experiments
4.1 Experimental Setup
4.2 Overall Performance
4.3 Fine-Grained Causal Task Analysis
4.4 Contrast Pair Correctness Analysis
4.5 Scaffold Transition Analysis
4.6 Formal Component Ablation
4.7 Stress Robustness
4.8 Exploratory Hidden-State Patching

5. Conclusion
```

正文叙事要求：

- 只写正式论文内容；
- 不写“上一轮”；
- 不写“方案调整”；
- 不写“与单纯比较模型规模不同”；
- 不写报告式流水账；
- 结论围绕结构性不稳定、对照敏感性、形式脚手架成分影响和局部 hidden-state 信号展开。

## 11. 验收标准

### 11.1 本地验收

```bash
python -m py_compile scripts/mlise2026_qwen_final.py
```

本地不加载模型，不生成正式实验结果。

### 11.2 服务器行为实验验收

每个模型必须生成：

```text
main_eval.csv
ablation_eval.csv
stress_eval.csv
```

行数预期：

```text
main_eval.csv: 640 × 3 = 1920
ablation_eval.csv: 640 × 2 = 1280
stress_eval.csv: 5 × 100 × 2 = 1000
```

三模型总行数预期：

```text
behavior main: 5760
ablation: 3840
stress: 3000
```

parse rate 应接近或等于 1.0；若明显下降，单独报告 invalid 分布。

### 11.3 分析验收

必须生成：

```text
query type / rung metrics
contrast decomposition
transition patterns
stress robustness
bootstrap CI
paired tests
component ablation metrics
patching layer profile
random patch control
```

### 11.4 图表验收

所有图表：

- 英文标题；
- 英文坐标轴；
- 英文图例；
- 不使用中文图注；
- 不使用旧结果；
- 能直接放入英文会议论文初稿。

### 11.5 报告验收

Markdown 报告必须使用中文，并包括：

- 实验时间；
- 服务器环境；
- 模型路径；
- 数据集路径；
- 样本规模；
- 输入条件；
- 解析规则；
- 主结果；
- 细粒度分析；
- contrast decomposition；
- transition analysis；
- stress robustness；
- ablation；
- patching；
- bootstrap CI；
- 图表索引；
- 异常与下一步建议。

## 12. 安全边界

1. 不写入服务器密码、token 或密钥。
2. 不提交 `.env`。
3. 不覆盖已有正式输出；每次 run 使用新目录。
4. 不删除数据集或模型权重。
5. 不修改服务器系统级配置。
6. 长任务使用 `nohup`，日志写入当前 run 的 `logs/`。
7. 所有正式实验在服务器 `/data/kongyb/ipm` 完成。

## 13. 下一步

等待用户确认后再执行：

1. 实现 `scripts/mlise2026_qwen_final.py`；
2. 本地语法检查；
3. 提交推送；
4. 服务器拉取；
5. 按本方案从零重新运行所有实验。
