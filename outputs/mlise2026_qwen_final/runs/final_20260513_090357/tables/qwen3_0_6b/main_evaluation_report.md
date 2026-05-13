# Qwen3-0.6B 评测记录

- 生成时间：2026-05-13 09:05:40
- 模型路径：`/data/LLM/Qwen/Qwen3-0___6B`
- 评测来源：`main`
- 样本数：`640`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model      | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:-----------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | formula_only  | Formalized Input     | 640 |    0.50625 |            1 |              0 |     0.0118315 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl            | Natural Language     | 640 |    0.5     |            1 |              0 |     0.0109315 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal     | NL + Formal Scaffold | 640 |    0.5     |            1 |              0 |     0.0112851 |

## Strict Contrast Consistency

| model      | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |      scca |   signed_ccc |   invalid_pair_rate |
|:-----------|:---------------------|:--------------------|:------------------|:--------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|----------:|-------------:|--------------------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | formula_only  |       358 |             358 |    0.0111732 |           0.0111732 |                 0 |             0.988827 |                   0 | 0.0111732 |    0.0111732 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl            |       358 |             358 |    0         |           0         |                 0 |             1        |                   0 | 0         |    0         |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal     |       358 |             358 |    0         |           0         |                 0 |             1        |                   0 | 0         |    0         |                   0 |

## Scaffold Gain

| model      | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:-----------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal       | NL + Formal Scaffold |           0.5 |             0.5     |         0       |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | formula_only    | Formalized Input     |           0.5 |             0.50625 |         0.00625 |

## Rescue / Harm

| model      | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:-----------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              |             640 |              0 |            0 |                              0 |                             0 |                 0 |
