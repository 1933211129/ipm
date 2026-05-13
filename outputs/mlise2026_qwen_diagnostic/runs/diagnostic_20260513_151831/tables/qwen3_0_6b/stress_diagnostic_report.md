# Qwen3-0.6B 诊断评测记录

- 生成时间：2026-05-13 07:20:00
- 模型路径：`/data/LLM/Qwen/Qwen3-0___6B`
- 评测来源：`stress`
- 样本数：`495`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model      | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:-----------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_0_6b | Qwen3-0.6B           | stress              | anticommonsense   | nl            | Natural Language     | 100 |        0.5 |            1 |              0 |     0.0244398 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | anticommonsense   | nl_formal     | NL + Formal Scaffold | 100 |        0.5 |            1 |              0 |     0.0211607 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | commonsense       | nl            | Natural Language     | 100 |        0.5 |            1 |              0 |     0.0208242 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | commonsense       | nl_formal     | NL + Formal Scaffold | 100 |        0.5 |            1 |              0 |     0.0194552 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | easy              | nl            | Natural Language     | 100 |        0.5 |            1 |              0 |     0.0194252 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | easy              | nl_formal     | NL + Formal Scaffold | 100 |        0.5 |            1 |              0 |     0.018322  |
| qwen3_0_6b | Qwen3-0.6B           | stress              | hard              | nl            | Natural Language     | 100 |        0.5 |            1 |              0 |     0.0195036 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | hard              | nl_formal     | NL + Formal Scaffold | 100 |        0.5 |            1 |              0 |     0.0190545 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | noncommonsense    | nl            | Natural Language     | 100 |        0.5 |            1 |              0 |     0.0213403 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | noncommonsense    | nl_formal     | NL + Formal Scaffold | 100 |        0.5 |            1 |              0 |     0.0188684 |

## Strict Contrast Consistency

| model      | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   strict_ccc |   invalid_pair_rate |
|:-----------|:---------------------|:--------------------|:------------------|:--------------|----------:|-------------:|--------------------:|
| qwen3_0_6b | Qwen3-0.6B           | stress              | anticommonsense   | nl            |        22 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | anticommonsense   | nl_formal     |        22 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | commonsense       | nl            |        25 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | commonsense       | nl_formal     |        25 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | easy              | nl            |        12 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | easy              | nl_formal     |        12 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | hard              | nl            |        17 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | hard              | nl_formal     |        17 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | noncommonsense    | nl            |        29 |            0 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | noncommonsense    | nl_formal     |        29 |            0 |                   0 |

## Scaffold Gain

| model      | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:-----------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_0_6b | Qwen3-0.6B           | stress              | anticommonsense   | nl_formal       | NL + Formal Scaffold |           0.5 |                 0.5 |               0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | commonsense       | nl_formal       | NL + Formal Scaffold |           0.5 |                 0.5 |               0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | easy              | nl_formal       | NL + Formal Scaffold |           0.5 |                 0.5 |               0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | hard              | nl_formal       | NL + Formal Scaffold |           0.5 |                 0.5 |               0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | noncommonsense    | nl_formal       | NL + Formal Scaffold |           0.5 |                 0.5 |               0 |

## Rescue / Harm

| model      | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:-----------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_0_6b | Qwen3-0.6B           | stress              | anticommonsense   |             100 |              0 |            0 |                              0 |                             0 |                 0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | commonsense       |             100 |              0 |            0 |                              0 |                             0 |                 0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | easy              |             100 |              0 |            0 |                              0 |                             0 |                 0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | hard              |             100 |              0 |            0 |                              0 |                             0 |                 0 |
| qwen3_0_6b | Qwen3-0.6B           | stress              | noncommonsense    |             100 |              0 |            0 |                              0 |                             0 |                 0 |
