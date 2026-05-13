# Qwen3-8B 评测记录

- 生成时间：2026-05-13 09:07:56
- 模型路径：`/data/LLM/Qwen/Qwen3-8B`
- 评测来源：`main`
- 样本数：`640`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_8b | Qwen3-8B             | main                | main              | formula_only  | Formalized Input     | 640 |   0.553125 |            1 |              0 |     0.0622966 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl            | Natural Language     | 640 |   0.560937 |            1 |              0 |     0.0539684 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl_formal     | NL + Formal Scaffold | 640 |   0.545312 |            1 |              0 |     0.0625321 |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |     scca |   signed_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|---------:|-------------:|--------------------:|
| qwen3_8b | Qwen3-8B             | main                | main              | formula_only  |       358 |             358 |     0.407821 |            0.276536 |          0.131285 |             0.298883 |            0.293296 | 0.276536 |     0.145251 |                   0 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl            |       358 |             358 |     0.444134 |            0.301676 |          0.142458 |             0.337989 |            0.217877 | 0.301676 |     0.159218 |                   0 |
| qwen3_8b | Qwen3-8B             | main                | main              | nl_formal     |       358 |             358 |     0.472067 |            0.312849 |          0.159218 |             0.287709 |            0.240223 | 0.312849 |     0.153631 |                   0 |

## Scaffold Gain

| model    | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_8b | Qwen3-8B             | main                | main              | nl_formal       | NL + Formal Scaffold |      0.560937 |            0.545312 |      -0.015625  |
| qwen3_8b | Qwen3-8B             | main                | main              | formula_only    | Formalized Input     |      0.560937 |            0.553125 |      -0.0078125 |

## Rescue / Harm

| model    | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_8b | Qwen3-8B             | main                | main              |             640 |             23 |           33 |                      0.0818505 |                      0.091922 |         -0.015625 |
