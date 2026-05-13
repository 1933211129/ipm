# MLISE 2026 最终实验结果分析

- 生成时间：2026-05-13 09:22:00
- run_id：`final_20260513_090357`
- 输出目录：`outputs/mlise2026_qwen_final/runs/final_20260513_090357`
- 图表语言：英文
- 本文档语言：中文

## 1. 验收状态

本轮正式实验已完整完成。三种行为阶段的行数均符合方案：每个模型 `main_eval.csv = 640 × 3 = 1920` 行，`ablation_eval.csv = 640 × 2 = 1280` 行，`stress_eval.csv = 5 × 100 × 2 = 1000` 行。Qwen3-4B 的 matched patching 与 random patch control 均为 `16 × 36 = 576` 行。

## 2. 主结果

| model_display_name   | prompt_mode   |   n |   accuracy |   parse_rate |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   scca |   signed_ccc |
|:---------------------|:--------------|----:|-----------:|-------------:|-------------:|--------------------:|------------------:|-------:|-------------:|
| Qwen3-0.6B           | formula_only  | 640 |     0.5062 |       1      |       0.0112 |              0.0112 |            0      | 0.0112 |       0.0112 |
| Qwen3-0.6B           | nl            | 640 |     0.5    |       1      |       0      |              0      |            0      | 0      |       0      |
| Qwen3-0.6B           | nl_formal     | 640 |     0.5    |       1      |       0      |              0      |            0      | 0      |       0      |
| Qwen3-0.6B           | nl_var_graph  | 640 |     0.5    |       1      |       0      |              0      |            0      | 0      |       0      |
| Qwen3-0.6B           | nl_var_query  | 640 |     0.5    |       0.9922 |       0      |              0      |            0      | 0      |       0      |
| Qwen3-4B             | formula_only  | 640 |     0.5641 |       1      |       0.3771 |              0.257  |            0.1201 | 0.257  |       0.1369 |
| Qwen3-4B             | nl            | 640 |     0.5953 |       1      |       0.4078 |              0.2905 |            0.1173 | 0.2905 |       0.1732 |
| Qwen3-4B             | nl_formal     | 640 |     0.5688 |       1      |       0.4078 |              0.2821 |            0.1257 | 0.2821 |       0.1564 |
| Qwen3-4B             | nl_var_graph  | 640 |     0.5953 |       1      |       0.4525 |              0.3464 |            0.1061 | 0.3464 |       0.2402 |
| Qwen3-4B             | nl_var_query  | 640 |     0.5688 |       1      |       0.3966 |              0.2709 |            0.1257 | 0.2709 |       0.1453 |
| Qwen3-8B             | formula_only  | 640 |     0.5531 |       1      |       0.4078 |              0.2765 |            0.1313 | 0.2765 |       0.1453 |
| Qwen3-8B             | nl            | 640 |     0.5609 |       1      |       0.4441 |              0.3017 |            0.1425 | 0.3017 |       0.1592 |
| Qwen3-8B             | nl_formal     | 640 |     0.5453 |       1      |       0.4721 |              0.3128 |            0.1592 | 0.3128 |       0.1536 |
| Qwen3-8B             | nl_var_graph  | 640 |     0.5609 |       1      |       0.4609 |              0.3073 |            0.1536 | 0.3073 |       0.1536 |
| Qwen3-8B             | nl_var_query  | 640 |     0.5375 |       1      |       0.4162 |              0.2626 |            0.1536 | 0.2626 |       0.1089 |

核心观察：Qwen3-0.6B 基本退化为标签平衡下的 0.5 accuracy；Qwen3-4B 在 `nl` 条件下达到最高主结果 accuracy 0.5953，`nl_var_graph` 也达到 0.5953；Qwen3-8B 的 `nl` 和 `nl_var_graph` 均为 0.5609。完整形式脚手架 `nl_formal` 没有带来稳定提升，4B 和 8B 分别下降 0.0266 与 0.0156。

## 3. 形式成分消融

| model_display_name   | scaffold_mode   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------------------|:----------------|--------------:|--------------------:|----------------:|
| Qwen3-0.6B           | nl_var_query    |        0.5    |              0.5    |          0      |
| Qwen3-0.6B           | nl_var_graph    |        0.5    |              0.5    |          0      |
| Qwen3-0.6B           | nl_formal       |        0.5    |              0.5    |          0      |
| Qwen3-0.6B           | formula_only    |        0.5    |              0.5062 |          0.0062 |
| Qwen3-4B             | nl_var_query    |        0.5953 |              0.5688 |         -0.0266 |
| Qwen3-4B             | nl_var_graph    |        0.5953 |              0.5953 |          0      |
| Qwen3-4B             | nl_formal       |        0.5953 |              0.5688 |         -0.0266 |
| Qwen3-4B             | formula_only    |        0.5953 |              0.5641 |         -0.0312 |
| Qwen3-8B             | nl_var_query    |        0.5609 |              0.5375 |         -0.0234 |
| Qwen3-8B             | nl_var_graph    |        0.5609 |              0.5609 |          0      |
| Qwen3-8B             | nl_formal       |        0.5609 |              0.5453 |         -0.0156 |
| Qwen3-8B             | formula_only    |        0.5609 |              0.5531 |         -0.0078 |

消融结果最有解释力的一点是：单独加入 causal graph 并不伤害 4B/8B，`nl_var_graph` 与 `nl` 持平；而加入 formal query 的 `nl_var_query` 与完整 `nl_formal` 都会降低 4B/8B 的 accuracy。这说明本轮结果不应简单写成“形式信息无效”，更准确的说法是：形式成分的作用不均匀，formal query 或完整形式包装可能引入额外解析负担，而 graph 成分本身未表现出同等负面影响。

## 4. Query Type 与 Rung 差异

### Query Type Accuracy

| model_display_name   | prompt_mode   | query_type         |   n |   accuracy |   parse_rate |
|:---------------------|:--------------|:-------------------|----:|-----------:|-------------:|
| Qwen3-0.6B           | formula_only  | ate                | 100 |     0.5    |            1 |
| Qwen3-0.6B           | formula_only  | backadj            | 100 |     0.5    |            1 |
| Qwen3-0.6B           | formula_only  | correlation        | 100 |     0.53   |            1 |
| Qwen3-0.6B           | formula_only  | det-counterfactual |  60 |     0.5167 |            1 |
| Qwen3-0.6B           | formula_only  | ett                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | formula_only  | marginal           | 100 |     0.5    |            1 |
| Qwen3-0.6B           | formula_only  | nde                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | formula_only  | nie                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | ate                | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | backadj            | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | correlation        | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | det-counterfactual |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | ett                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | marginal           | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | nde                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl            | nie                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | ate                | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | backadj            | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | correlation        | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | det-counterfactual |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | ett                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | marginal           | 100 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | nde                |  60 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     | nie                |  60 |     0.5    |            1 |
| Qwen3-4B             | formula_only  | ate                | 100 |     0.71   |            1 |
| Qwen3-4B             | formula_only  | backadj            | 100 |     0.5    |            1 |
| Qwen3-4B             | formula_only  | correlation        | 100 |     0.59   |            1 |
| Qwen3-4B             | formula_only  | det-counterfactual |  60 |     0.5667 |            1 |
| Qwen3-4B             | formula_only  | ett                |  60 |     0.4    |            1 |
| Qwen3-4B             | formula_only  | marginal           | 100 |     0.56   |            1 |
| Qwen3-4B             | formula_only  | nde                |  60 |     0.5333 |            1 |
| Qwen3-4B             | formula_only  | nie                |  60 |     0.5833 |            1 |
| Qwen3-4B             | nl            | ate                | 100 |     0.74   |            1 |
| Qwen3-4B             | nl            | backadj            | 100 |     0.52   |            1 |
| Qwen3-4B             | nl            | correlation        | 100 |     0.65   |            1 |
| Qwen3-4B             | nl            | det-counterfactual |  60 |     0.5833 |            1 |
| Qwen3-4B             | nl            | ett                |  60 |     0.4667 |            1 |
| Qwen3-4B             | nl            | marginal           | 100 |     0.53   |            1 |
| Qwen3-4B             | nl            | nde                |  60 |     0.6167 |            1 |
| Qwen3-4B             | nl            | nie                |  60 |     0.6167 |            1 |
| Qwen3-4B             | nl_formal     | ate                | 100 |     0.73   |            1 |
| Qwen3-4B             | nl_formal     | backadj            | 100 |     0.52   |            1 |
| Qwen3-4B             | nl_formal     | correlation        | 100 |     0.56   |            1 |
| Qwen3-4B             | nl_formal     | det-counterfactual |  60 |     0.5667 |            1 |
| Qwen3-4B             | nl_formal     | ett                |  60 |     0.4    |            1 |
| Qwen3-4B             | nl_formal     | marginal           | 100 |     0.57   |            1 |
| Qwen3-4B             | nl_formal     | nde                |  60 |     0.6167 |            1 |
| Qwen3-4B             | nl_formal     | nie                |  60 |     0.5167 |            1 |
| Qwen3-8B             | formula_only  | ate                | 100 |     0.7    |            1 |
| Qwen3-8B             | formula_only  | backadj            | 100 |     0.5    |            1 |
| Qwen3-8B             | formula_only  | correlation        | 100 |     0.55   |            1 |
| Qwen3-8B             | formula_only  | det-counterfactual |  60 |     0.6    |            1 |
| Qwen3-8B             | formula_only  | ett                |  60 |     0.4167 |            1 |
| Qwen3-8B             | formula_only  | marginal           | 100 |     0.51   |            1 |
| Qwen3-8B             | formula_only  | nde                |  60 |     0.6    |            1 |
| Qwen3-8B             | formula_only  | nie                |  60 |     0.5167 |            1 |
| Qwen3-8B             | nl            | ate                | 100 |     0.76   |            1 |
| Qwen3-8B             | nl            | backadj            | 100 |     0.44   |            1 |
| Qwen3-8B             | nl            | correlation        | 100 |     0.55   |            1 |
| Qwen3-8B             | nl            | det-counterfactual |  60 |     0.5333 |            1 |
| Qwen3-8B             | nl            | ett                |  60 |     0.45   |            1 |
| Qwen3-8B             | nl            | marginal           | 100 |     0.53   |            1 |
| Qwen3-8B             | nl            | nde                |  60 |     0.6333 |            1 |
| Qwen3-8B             | nl            | nie                |  60 |     0.5667 |            1 |
| Qwen3-8B             | nl_formal     | ate                | 100 |     0.71   |            1 |
| Qwen3-8B             | nl_formal     | backadj            | 100 |     0.45   |            1 |
| Qwen3-8B             | nl_formal     | correlation        | 100 |     0.55   |            1 |
| Qwen3-8B             | nl_formal     | det-counterfactual |  60 |     0.5    |            1 |
| Qwen3-8B             | nl_formal     | ett                |  60 |     0.4333 |            1 |
| Qwen3-8B             | nl_formal     | marginal           | 100 |     0.56   |            1 |
| Qwen3-8B             | nl_formal     | nde                |  60 |     0.55   |            1 |
| Qwen3-8B             | nl_formal     | nie                |  60 |     0.55   |            1 |

4B/8B 的错误并非均匀分布。ATE 是最强项，4B `nl` 为 0.7400，8B `nl` 为 0.7600；backadj、ett 和部分 mediation/counterfactual 类任务更弱。形式脚手架对 marginal 有小幅正效应，但对 correlation、nie、nde、ett 等类型经常带来负效应。

### Rung Accuracy

| model_display_name   | prompt_mode   |   rung |   n |   accuracy |   parse_rate |
|:---------------------|:--------------|-------:|----:|-----------:|-------------:|
| Qwen3-0.6B           | formula_only  |      1 | 200 |     0.515  |            1 |
| Qwen3-0.6B           | formula_only  |      2 | 200 |     0.5    |            1 |
| Qwen3-0.6B           | formula_only  |      3 | 240 |     0.5042 |            1 |
| Qwen3-0.6B           | nl            |      1 | 200 |     0.5    |            1 |
| Qwen3-0.6B           | nl            |      2 | 200 |     0.5    |            1 |
| Qwen3-0.6B           | nl            |      3 | 240 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     |      1 | 200 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     |      2 | 200 |     0.5    |            1 |
| Qwen3-0.6B           | nl_formal     |      3 | 240 |     0.5    |            1 |
| Qwen3-4B             | formula_only  |      1 | 200 |     0.575  |            1 |
| Qwen3-4B             | formula_only  |      2 | 200 |     0.605  |            1 |
| Qwen3-4B             | formula_only  |      3 | 240 |     0.5208 |            1 |
| Qwen3-4B             | nl            |      1 | 200 |     0.59   |            1 |
| Qwen3-4B             | nl            |      2 | 200 |     0.63   |            1 |
| Qwen3-4B             | nl            |      3 | 240 |     0.5708 |            1 |
| Qwen3-4B             | nl_formal     |      1 | 200 |     0.565  |            1 |
| Qwen3-4B             | nl_formal     |      2 | 200 |     0.625  |            1 |
| Qwen3-4B             | nl_formal     |      3 | 240 |     0.525  |            1 |
| Qwen3-8B             | formula_only  |      1 | 200 |     0.53   |            1 |
| Qwen3-8B             | formula_only  |      2 | 200 |     0.6    |            1 |
| Qwen3-8B             | formula_only  |      3 | 240 |     0.5333 |            1 |
| Qwen3-8B             | nl            |      1 | 200 |     0.54   |            1 |
| Qwen3-8B             | nl            |      2 | 200 |     0.6    |            1 |
| Qwen3-8B             | nl            |      3 | 240 |     0.5458 |            1 |
| Qwen3-8B             | nl_formal     |      1 | 200 |     0.555  |            1 |
| Qwen3-8B             | nl_formal     |      2 | 200 |     0.58   |            1 |
| Qwen3-8B             | nl_formal     |      3 | 240 |     0.5083 |            1 |

rung 维度也显示结构性差异。4B 在 rung 2 的自然语言条件最好，rung 3 明显下降；8B 在 rung 2 相对较好，但形式脚手架在 rung 3 上的表现更弱。这支持“高阶因果问题更容易触发不稳定判断”的叙事。

## 5. CCC 正误分解

主表显示，CCC 高并不等同于判断正确。8B 的 `nl_formal` CCC 为 0.4721，高于其 `nl` 的 0.4441，但 Correct Flip 仅 0.3128，Wrong Flip 达到 0.1592；4B 的 `nl_var_graph` CCC 为 0.4525，同时 Correct Flip 0.3464、Wrong Flip 0.1061，是本轮更稳的对照敏感性结果。

按 query type 看，8B 在 `det-counterfactual` 的 `nl_formal` 条件下 CCC 为 1.0000，但 Correct Flip 为 0，Wrong Flip 为 1.0000。这是非常关键的负例：模型确实随对照变化翻转了预测，但翻转方向完全错误。因此论文中应强调“局部对照敏感性不等价于正确因果判断”。

## 6. Rescue / Harm 与转移轨迹

| model_display_name   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| Qwen3-0.6B           |             640 |              0 |            0 |                         0      |                        0      |            0      |
| Qwen3-4B             |             640 |             26 |           43 |                         0.1004 |                        0.1129 |           -0.0266 |
| Qwen3-8B             |             640 |             23 |           33 |                         0.0819 |                        0.0919 |           -0.0156 |

4B 在 `nl_formal` 条件下 rescue 26 个样本，但 harm 43 个样本，净效应为 -0.0266；8B rescue 23 个、harm 33 个，净效应为 -0.0156。形式脚手架不是单向修复机制，而是在样本层面同时造成修复与破坏。

| model      | model_display_name   | diagnostic_source   | dataset_variant   | transition_pattern   |   n |   total |   rate |
|:-----------|:---------------------|:--------------------|:------------------|:---------------------|----:|--------:|-------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | C-C-C                | 320 |     640 | 0.5    |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | W-W-W                | 316 |     640 | 0.4938 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | W-W-C                |   4 |     640 | 0.0062 |
| qwen3_4b   | Qwen3-4B             | main                | main              | C-C-C                | 304 |     640 | 0.475  |
| qwen3_4b   | Qwen3-4B             | main                | main              | W-W-W                | 210 |     640 | 0.3281 |
| qwen3_4b   | Qwen3-4B             | main                | main              | C-C-W                |  34 |     640 | 0.0531 |
| qwen3_4b   | Qwen3-4B             | main                | main              | W-W-C                |  23 |     640 | 0.0359 |
| qwen3_4b   | Qwen3-4B             | main                | main              | C-W-W                |  22 |     640 | 0.0344 |
| qwen3_4b   | Qwen3-4B             | main                | main              | C-W-C                |  21 |     640 | 0.0328 |
| qwen3_4b   | Qwen3-4B             | main                | main              | W-C-C                |  13 |     640 | 0.0203 |
| qwen3_4b   | Qwen3-4B             | main                | main              | W-C-W                |  13 |     640 | 0.0203 |
| qwen3_8b   | Qwen3-8B             | main                | main              | C-C-C                | 278 |     640 | 0.4344 |
| qwen3_8b   | Qwen3-8B             | main                | main              | W-W-W                | 212 |     640 | 0.3312 |
| qwen3_8b   | Qwen3-8B             | main                | main              | C-C-W                |  48 |     640 | 0.075  |
| qwen3_8b   | Qwen3-8B             | main                | main              | W-W-C                |  46 |     640 | 0.0719 |
| qwen3_8b   | Qwen3-8B             | main                | main              | C-W-W                |  18 |     640 | 0.0281 |
| qwen3_8b   | Qwen3-8B             | main                | main              | C-W-C                |  15 |     640 | 0.0234 |
| qwen3_8b   | Qwen3-8B             | main                | main              | W-C-C                |  15 |     640 | 0.0234 |
| qwen3_8b   | Qwen3-8B             | main                | main              | W-C-W                |   8 |     640 | 0.0125 |

转移矩阵显示，4B 的 C-C-C 为 47.5%，W-W-W 为 32.8%；8B 的 C-C-C 为 43.4%，W-W-W 为 33.1%。这意味着相当一部分样本在三种输入下保持稳定，但仍有不可忽略的 C-W-W、C-W-C、W-W-C 等迁移轨迹，可以用来支撑“形式输入诱发答案迁移，但迁移方向不稳定”的分析。

## 7. Stress Robustness

| model_display_name   | prompt_mode   |   commonsense_accuracy |   anticommonsense_accuracy |   noncommonsense_accuracy |   easy_accuracy |   hard_accuracy |   mean_accuracy |   worst_split_accuracy |   std_across_splits |   worst_split_scaffold_gain |
|:---------------------|:--------------|-----------------------:|---------------------------:|--------------------------:|----------------:|----------------:|----------------:|-----------------------:|--------------------:|----------------------------:|
| Qwen3-0.6B           | nl            |                   0.5  |                       0.5  |                      0.5  |            0.5  |            0.5  |           0.5   |                   0.5  |              0      |                        0    |
| Qwen3-0.6B           | nl_formal     |                   0.5  |                       0.5  |                      0.5  |            0.5  |            0.5  |           0.5   |                   0.5  |              0      |                        0    |
| Qwen3-4B             | nl            |                   0.51 |                       0.47 |                      0.48 |            0.49 |            0.49 |           0.488 |                   0.47 |              0.0133 |                       -0.01 |
| Qwen3-4B             | nl_formal     |                   0.55 |                       0.52 |                      0.49 |            0.48 |            0.5  |           0.508 |                   0.48 |              0.0248 |                       -0.01 |
| Qwen3-8B             | nl            |                   0.52 |                       0.49 |                      0.49 |            0.51 |            0.46 |           0.494 |                   0.46 |              0.0206 |                       -0.04 |
| Qwen3-8B             | nl_formal     |                   0.57 |                       0.45 |                      0.47 |            0.53 |            0.46 |           0.496 |                   0.45 |              0.0463 |                       -0.04 |

stress split 上，4B 的 `nl_formal` 平均 accuracy 从 0.488 提高到 0.508，但 worst split 仍为 0.48，跨 split 标准差从 0.0133 增至 0.0248。8B 的 `nl_formal` 平均值几乎不变，但 worst split 从 0.46 降至 0.45，标准差从 0.0206 增至 0.0463。换言之，形式脚手架偶尔改善 commonsense，但会放大跨 split 波动。

## 8. Patching 与 Random Control

| model    | model_display_name   | method                          |   n_rows |   n_samples |   mean_absolute_recovery |   max_absolute_recovery |   mean_normalized_recovery |
|:---------|:---------------------|:--------------------------------|---------:|------------:|-------------------------:|------------------------:|---------------------------:|
| qwen3_4b | Qwen3-4B             | hf_last_token_formal_to_natural |      576 |          16 |                  -0.0225 |                  3.7188 |                     0.5157 |

| model_display_name   | patch_condition      |   n_rows |   n_samples |   mean_absolute_recovery |   median_absolute_recovery |   max_absolute_recovery |   positive_recovery_rate |   mean_normalized_recovery |
|:---------------------|:---------------------|---------:|------------:|-------------------------:|---------------------------:|------------------------:|-------------------------:|---------------------------:|
| Qwen3-4B             | matched              |      576 |          16 |                  -0.0225 |                     0      |                  3.7188 |                   0.4618 |                     0.5157 |
| Qwen3-4B             | random               |      576 |          16 |                  -0.061  |                    -0.0312 |                  2.9375 |                   0.467  |                     0.3573 |
| Qwen3-4B             | matched_minus_random |      576 |          16 |                   0.0385 |                     0.0312 |                  2.125  |                   0.5069 |                   nan      |

Matched patching 的 mean absolute recovery 为 -0.0225，max recovery 为 3.7188，mean normalized recovery 为 0.5157。Random control 的 mean absolute recovery 为 -0.0610，matched-minus-random 的平均差为 0.0385，正差比例约 0.5069。这个结果不支持强机制宣称，但可以作为弱的探索性证据：匹配形式输入 residual 在部分层和部分样本上比随机形式 residual 更有利，不过这种信号没有稳定转化为总体行为收益。

## 9. 论文叙事建议

本轮结果可以写，而且比只报告总体 accuracy 更完整。建议主线写成“形式结构提示下的因果判断不稳定性诊断”：

1. 模型错误集中在特定因果类型和高阶 rung，而非均匀随机。
2. CCC 需要拆解；高 CCC 可由错误翻转贡献，不能直接解释为正确因果推理。
3. 形式脚手架同时产生 rescue 和 harm，formal query 成分比 graph 成分更可能引入负效应。
4. stress split 中平均值变化很小，但 worst-case 和方差揭示鲁棒性不足。
5. Patching 只提供探索性 hidden-state evidence，应谨慎表述为局部可转移信号，而非完整因果回路。

## 10. 当前限制

- 本文只使用 CLadder，结论不外推到全部现实因果问答场景。
- 三个模型属于同一家族，不能解释所有模型架构差异。
- 消融只覆盖两个形式成分，不等于完整 prompt 搜索。
- Patching 只在 Qwen3-4B 上做 16 个样本，适合作为探索性机制分析。
