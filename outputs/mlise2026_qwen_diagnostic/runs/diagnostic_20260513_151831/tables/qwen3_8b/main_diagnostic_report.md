# Qwen3-8B 诊断评测记录

- 生成时间：2026-05-13 07:21:42
- 模型路径：`/data/LLM/Qwen/Qwen3-8B`
- 评测来源：`main`
- 样本数：`640`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_8b | Qwen3-8B             | main                | main              | formula_only  | Formalized Input     | 640 |   0.557813 |            1 |              0 |     0.0458544 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl            | Natural Language     | 640 |   0.559375 |            1 |              0 |     0.0390638 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl_formal     | NL + Formal Scaffold | 640 |   0.540625 |            1 |              0 |     0.0454929 |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   strict_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|-------------:|--------------------:|
| qwen3_8b | Qwen3-8B             | main                | main              | formula_only  |       358 |     0.424581 |                   0 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl            |       358 |     0.455307 |                   0 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl_formal     |       358 |     0.458101 |                   0 |

## Scaffold Gain

| model    | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_8b | Qwen3-8B             | main                | main              | nl_formal       | NL + Formal Scaffold |      0.559375 |            0.540625 |      -0.01875   |
| qwen3_8b | Qwen3-8B             | main                | main              | formula_only    | Formalized Input     |      0.559375 |            0.557813 |      -0.0015625 |

## Rescue / Harm

| model    | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_8b | Qwen3-8B             | main                | main              |             640 |             23 |           35 |                      0.0815603 |                     0.0977654 |          -0.01875 |
