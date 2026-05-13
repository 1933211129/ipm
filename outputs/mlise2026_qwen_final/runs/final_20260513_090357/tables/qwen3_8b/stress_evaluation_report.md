# Qwen3-8B 评测记录

- 生成时间：2026-05-13 09:11:50
- 模型路径：`/data/LLM/Qwen/Qwen3-8B`
- 评测来源：`stress`
- 样本数：`495`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:---------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl            | Natural Language     | 100 |       0.49 |            1 |              0 |     0.0566596 |
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl_formal     | NL + Formal Scaffold | 100 |       0.45 |            1 |              0 |     0.0610723 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl            | Natural Language     | 100 |       0.52 |            1 |              0 |     0.0519194 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl_formal     | NL + Formal Scaffold | 100 |       0.57 |            1 |              0 |     0.0607018 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl            | Natural Language     | 100 |       0.51 |            1 |              0 |     0.0526062 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl_formal     | NL + Formal Scaffold | 100 |       0.53 |            1 |              0 |     0.0624314 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl            | Natural Language     | 100 |       0.46 |            1 |              0 |     0.0528848 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl_formal     | NL + Formal Scaffold | 100 |       0.46 |            1 |              0 |     0.0621478 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl            | Natural Language     | 100 |       0.49 |            1 |              0 |     0.0528735 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl_formal     | NL + Formal Scaffold | 100 |       0.47 |            1 |              0 |     0.062307  |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |      scca |   signed_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|----------:|-------------:|--------------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl            |        22 |              22 |    0.454545  |           0.318182  |         0.136364  |            0.318182  |            0.227273 | 0.318182  |    0.181818  |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl_formal     |        22 |              22 |    0.227273  |           0.0909091 |         0.136364  |            0.272727  |            0.5      | 0.0909091 |   -0.0454545 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl            |        25 |              25 |    0.16      |           0.08      |         0.08      |            0.48      |            0.36     | 0.08      |    0         |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl_formal     |        25 |              25 |    0.32      |           0.12      |         0.2       |            0.4       |            0.28     | 0.12      |   -0.08      |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl            |        12 |              12 |    0.333333  |           0.0833333 |         0.25      |            0.25      |            0.416667 | 0.0833333 |   -0.166667  |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl_formal     |        12 |              12 |    0.0833333 |           0         |         0.0833333 |            0.416667  |            0.5      | 0         |   -0.0833333 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl            |        17 |              17 |    0.470588  |           0.117647  |         0.352941  |            0.294118  |            0.235294 | 0.117647  |   -0.235294  |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl_formal     |        17 |              17 |    0.235294  |           0         |         0.235294  |            0.294118  |            0.470588 | 0         |   -0.235294  |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl            |        29 |              29 |    0.37931   |           0.206897  |         0.172414  |            0.103448  |            0.517241 | 0.206897  |    0.0344828 |                   0 |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl_formal     |        29 |              29 |    0.344828  |           0.137931  |         0.206897  |            0.0344828 |            0.62069  | 0.137931  |   -0.0689655 |                   0 |

## Scaffold Gain

| model    | model_display_name   | diagnostic_source   | dataset_variant   | scaffold_mode   | scaffold_condition   |   nl_accuracy |   scaffold_accuracy |   scaffold_gain |
|:---------|:---------------------|:--------------------|:------------------|:----------------|:---------------------|--------------:|--------------------:|----------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   | nl_formal       | NL + Formal Scaffold |          0.49 |                0.45 |           -0.04 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       | nl_formal       | NL + Formal Scaffold |          0.52 |                0.57 |            0.05 |
| qwen3_8b | Qwen3-8B             | stress              | easy              | nl_formal       | NL + Formal Scaffold |          0.51 |                0.53 |            0.02 |
| qwen3_8b | Qwen3-8B             | stress              | hard              | nl_formal       | NL + Formal Scaffold |          0.46 |                0.46 |            0    |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    | nl_formal       | NL + Formal Scaffold |          0.49 |                0.47 |           -0.02 |

## Rescue / Harm

| model    | model_display_name   | diagnostic_source   | dataset_variant   |   n_valid_pairs |   rescue_count |   harm_count |   rescue_rate_over_nl_failures |   harm_rate_over_nl_successes |   net_rescue_rate |
|:---------|:---------------------|:--------------------|:------------------|----------------:|---------------:|-------------:|-------------------------------:|------------------------------:|------------------:|
| qwen3_8b | Qwen3-8B             | stress              | anticommonsense   |             100 |             14 |           18 |                       0.27451  |                      0.367347 |             -0.04 |
| qwen3_8b | Qwen3-8B             | stress              | commonsense       |             100 |             17 |           12 |                       0.354167 |                      0.230769 |              0.05 |
| qwen3_8b | Qwen3-8B             | stress              | easy              |             100 |             12 |           10 |                       0.244898 |                      0.196078 |              0.02 |
| qwen3_8b | Qwen3-8B             | stress              | hard              |             100 |             19 |           19 |                       0.351852 |                      0.413043 |              0    |
| qwen3_8b | Qwen3-8B             | stress              | noncommonsense    |             100 |             10 |           12 |                       0.196078 |                      0.244898 |             -0.02 |
