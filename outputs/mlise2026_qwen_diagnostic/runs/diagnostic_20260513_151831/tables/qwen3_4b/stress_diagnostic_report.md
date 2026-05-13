# Qwen3-4B 诊断评测记录

- 生成时间：2026-05-13 07:20:42
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 评测来源：`stress`
- 样本数：`495`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_4b | Qwen3-4B             | stress              | anticommonsense   | nl            | Natural Language     | 100 |       0.47 |            1 |              0 |     0.0384943 |
| qwen3_4b | Qwen3-4B             | stress              | anticommonsense   | nl_formal     | NL + Formal Scaffold | 100 |       0.52 |            1 |              0 |     0.0339048 |
| qwen3_4b | Qwen3-4B             | stress              | commonsense       | nl            | Natural Language     | 100 |       0.51 |            1 |              0 |     0.0317815 |
| qwen3_4b | Qwen3-4B             | stress              | commonsense       | nl_formal     | NL + Formal Scaffold | 100 |       0.55 |            1 |              0 |     0.0342597 |
| qwen3_4b | Qwen3-4B             | stress              | easy              | nl            | Natural Language     | 100 |       0.49 |            1 |              0 |     0.0292231 |
| qwen3_4b | Qwen3-4B             | stress              | easy              | nl_formal     | NL + Formal Scaffold | 100 |       0.48 |            1 |              0 |     0.0352283 |
| qwen3_4b | Qwen3-4B             | stress              | hard              | nl            | Natural Language     | 100 |       0.49 |            1 |              0 |     0.0292048 |
| qwen3_4b | Qwen3-4B             | stress              | hard              | nl_formal     | NL + Formal Scaffold | 100 |       0.5  |            1 |              0 |     0.0348525 |
| qwen3_4b | Qwen3-4B             | stress              | noncommonsense    | nl            | Natural Language     | 100 |       0.48 |            1 |              0 |     0.0302554 |
| qwen3_4b | Qwen3-4B             | stress              | noncommonsense    | nl_formal     | NL + Formal Scaffold | 100 |       0.49 |            1 |              0 |     0.0343946 |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   strict_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|-------------:|--------------------:|
| qwen3_4b | Qwen3-4B             | stress              | anticommonsense   | nl            |        22 |    0         |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | anticommonsense   | nl_formal     |        22 |    0.136364  |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | commonsense       | nl            |        25 |    0.04      |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | commonsense       | nl_formal     |        25 |    0.12      |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | easy              | nl            |        12 |    0.0833333 |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | easy              | nl_formal     |        12 |    0.25      |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | hard              | nl            |        17 |    0.0588235 |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | hard              | nl_formal     |        17 |    0.0588235 |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | noncommonsense    | nl            |        29 |    0.103448  |                   0 |
| qwen3_4b | Qwen3-4B             | stress              | noncommonsense    | nl_formal     |        29 |    0.137931  |                   0 |

## Scaffold Gain

| model    | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_4b | Qwen3-4B             | stress              | anticommonsense   | nl_formal       | NL + Formal Scaffold |          0.47 |                0.52 |            0.05 |
| qwen3_4b | Qwen3-4B             | stress              | commonsense       | nl_formal       | NL + Formal Scaffold |          0.51 |                0.55 |            0.04 |
| qwen3_4b | Qwen3-4B             | stress              | easy              | nl_formal       | NL + Formal Scaffold |          0.49 |                0.48 |           -0.01 |
| qwen3_4b | Qwen3-4B             | stress              | hard              | nl_formal       | NL + Formal Scaffold |          0.49 |                0.5  |            0.01 |
| qwen3_4b | Qwen3-4B             | stress              | noncommonsense    | nl_formal       | NL + Formal Scaffold |          0.48 |                0.49 |            0.01 |

## Rescue / Harm

| model    | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_4b | Qwen3-4B             | stress              | anticommonsense   |             100 |              9 |            4 |                      0.169811  |                     0.0851064 |              0.05 |
| qwen3_4b | Qwen3-4B             | stress              | commonsense       |             100 |              7 |            3 |                      0.142857  |                     0.0588235 |              0.04 |
| qwen3_4b | Qwen3-4B             | stress              | easy              |             100 |              4 |            5 |                      0.0784314 |                     0.102041  |             -0.01 |
| qwen3_4b | Qwen3-4B             | stress              | hard              |             100 |              5 |            4 |                      0.0980392 |                     0.0816327 |              0.01 |
| qwen3_4b | Qwen3-4B             | stress              | noncommonsense    |             100 |              7 |            6 |                      0.134615  |                     0.125     |              0.01 |
