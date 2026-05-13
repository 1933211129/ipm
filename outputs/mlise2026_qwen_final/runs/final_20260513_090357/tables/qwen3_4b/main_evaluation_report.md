# Qwen3-4B 评测记录

- 生成时间：2026-05-13 09:06:37
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 评测来源：`main`
- 样本数：`640`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_4b | Qwen3-4B             | main                | main              | formula_only  | Formalized Input     | 640 |   0.564063 |            1 |              0 |     0.0343917 |
| qwen3_4b | Qwen3-4B             | main                | main              | nl            | Natural Language     | 640 |   0.595313 |            1 |              0 |     0.0291895 |
| qwen3_4b | Qwen3-4B             | main                | main              | nl_formal     | NL + Formal Scaffold | 640 |   0.56875  |            1 |              0 |     0.033753  |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |     scca |   signed_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|---------:|-------------:|--------------------:|
| qwen3_4b | Qwen3-4B             | main                | main              | formula_only  |       358 |             358 |     0.377095 |            0.256983 |          0.120112 |             0.23743  |            0.385475 | 0.256983 |     0.136872 |                   0 |
| qwen3_4b | Qwen3-4B             | main                | main              | nl            |       358 |             358 |     0.407821 |            0.290503 |          0.117318 |             0.209497 |            0.382682 | 0.290503 |     0.173184 |                   0 |
| qwen3_4b | Qwen3-4B             | main                | main              | nl_formal     |       358 |             358 |     0.407821 |            0.282123 |          0.125698 |             0.192737 |            0.399441 | 0.282123 |     0.156425 |                   0 |

## Scaffold Gain

| model    | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_4b | Qwen3-4B             | main                | main              | nl_formal       | NL + Formal Scaffold |      0.595313 |            0.56875  |      -0.0265625 |
| qwen3_4b | Qwen3-4B             | main                | main              | formula_only    | Formalized Input     |      0.595313 |            0.564063 |      -0.03125   |

## Rescue / Harm

| model    | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_4b | Qwen3-4B             | main                | main              |             640 |             26 |           43 |                       0.100386 |                      0.112861 |        -0.0265625 |
