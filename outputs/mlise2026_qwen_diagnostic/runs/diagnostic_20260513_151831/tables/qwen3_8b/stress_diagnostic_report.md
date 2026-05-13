# Qwen3-8B 诊断评测记录

- 生成时间：2026-05-13 07:21:13
- 模型路径：`/data/LLM/Qwen/Qwen3-8B`
- 评测来源：`stress`
- 样本数：`495`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl            | Natural Language     | 100 |       0.49 |            1 |              0 |     0.045145  |
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl_formal     | NL + Formal Scaffold | 100 |       0.46 |            1 |              0 |     0.0426129 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl            | Natural Language     | 100 |       0.52 |            1 |              0 |     0.0348544 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl_formal     | NL + Formal Scaffold | 100 |       0.59 |            1 |              0 |     0.0434225 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl            | Natural Language     | 100 |       0.51 |            1 |              0 |     0.0344218 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl_formal     | NL + Formal Scaffold | 100 |       0.56 |            1 |              0 |     0.043615  |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl            | Natural Language     | 100 |       0.45 |            1 |              0 |     0.0346972 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl_formal     | NL + Formal Scaffold | 100 |       0.45 |            1 |              0 |     0.0435336 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl            | Natural Language     | 100 |       0.49 |            1 |              0 |     0.0345759 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl_formal     | NL + Formal Scaffold | 100 |       0.46 |            1 |              0 |     0.0473085 |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   strict_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|-------------:|--------------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl            |        22 |     0.454545 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl_formal     |        22 |     0.227273 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl            |        25 |     0.16     |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl_formal     |        25 |     0.24     |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl            |        12 |     0.333333 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl_formal     |        12 |     0.166667 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl            |        17 |     0.470588 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl_formal     |        17 |     0.294118 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl            |        29 |     0.448276 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl_formal     |        29 |     0.310345 |                   0 |

## Scaffold Gain

| model    | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl_formal       | NL + Formal Scaffold |          0.49 |                0.46 |           -0.03 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl_formal       | NL + Formal Scaffold |          0.52 |                0.59 |            0.07 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl_formal       | NL + Formal Scaffold |          0.51 |                0.56 |            0.05 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl_formal       | NL + Formal Scaffold |          0.45 |                0.45 |            0    |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl_formal       | NL + Formal Scaffold |          0.49 |                0.46 |           -0.03 |

## Rescue / Harm

| model    | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   |             100 |             15 |           18 |                       0.294118 |                      0.367347 |             -0.03 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       |             100 |             17 |           10 |                       0.354167 |                      0.192308 |              0.07 |
| qwen3_8b | Qwen3-8B             | stress              | easy              |             100 |             13 |            8 |                       0.265306 |                      0.156863 |              0.05 |
| qwen3_8b | Qwen3-8B             | stress              | hard              |             100 |             16 |           16 |                       0.290909 |                      0.355556 |              0    |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    |             100 |             10 |           13 |                       0.196078 |                      0.265306 |             -0.03 |
