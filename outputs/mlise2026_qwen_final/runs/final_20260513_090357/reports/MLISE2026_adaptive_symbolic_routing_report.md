# MLISE 2026 自适应 Symbolic Routing 补实验

- 生成时间：2026-05-13 13:15:00
- 训练信号：2048 条 matched symbolic control 的 NL 与 symbolic binary score。
- 测试设置：主评测 640 样本，以及去除 calibration 重叠样本后的 non-overlap 子集。
- 路由策略：Query Router 按 query type 选择 NL 或 symbolic；Confidence Router 在每个 query type 内按 yes/no margin 差值选择输入源。

## Transfer Accuracy

| eval_set                 | model    | model_display_name   | prompt_mode         | prompt_condition   |   n |   accuracy |   symbolic_selection_rate |
|:-------------------------|:---------|:---------------------|:--------------------|:-------------------|----:|-----------:|--------------------------:|
| main_full_transfer       | qwen3_4b | Qwen3-4B             | always_nl           | Always NL          | 640 |     0.6    |                    0      |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | always_symbolic     | Always Symbolic    | 640 |     0.5844 |                    1      |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | confidence_by_query | Confidence Router  | 640 |     0.6078 |                    0.4609 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | query_router        | Query Router       | 640 |     0.6125 |                    0.4062 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | always_nl           | Always NL          | 640 |     0.5547 |                    0      |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | always_symbolic     | Always Symbolic    | 640 |     0.5828 |                    1      |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | confidence_by_query | Confidence Router  | 640 |     0.5922 |                    0.6516 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | query_router        | Query Router       | 640 |     0.5938 |                    0.75   |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | always_nl           | Always NL          | 398 |     0.6055 |                    0      |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | always_symbolic     | Always Symbolic    | 398 |     0.6055 |                    1      |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | confidence_by_query | Confidence Router  | 398 |     0.6156 |                    0.5729 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | query_router        | Query Router       | 398 |     0.6181 |                    0.5352 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | always_nl           | Always NL          | 398 |     0.5427 |                    0      |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | always_symbolic     | Always Symbolic    | 398 |     0.5754 |                    1      |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | confidence_by_query | Confidence Router  | 398 |     0.5754 |                    0.6357 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | query_router        | Query Router       | 398 |     0.5804 |                    0.6859 |

## Transfer Paired Bootstrap

| eval_set                 | model    | model_display_name   | condition_a         | condition_b     |   n |   acc_a |   acc_b |   paired_accuracy_diff |   ci_low |   ci_high |   p_diff_le_0 |
|:-------------------------|:---------|:---------------------|:--------------------|:----------------|----:|--------:|--------:|-----------------------:|---------:|----------:|--------------:|
| main_full_transfer       | qwen3_4b | Qwen3-4B             | query_router        | always_nl       | 640 |  0.6125 |  0.6    |                 0.0125 |  -0.0094 |    0.0359 |         0.155 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | confidence_by_query | always_nl       | 640 |  0.6078 |  0.6    |                 0.0078 |  -0.0203 |    0.0328 |         0.328 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | always_symbolic     | always_nl       | 640 |  0.5844 |  0.6    |                -0.0156 |  -0.0516 |    0.0203 |         0.836 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | query_router        | always_symbolic | 640 |  0.6125 |  0.5844 |                 0.0281 |   0.0016 |    0.0562 |         0.022 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | confidence_by_query | always_symbolic | 640 |  0.6078 |  0.5844 |                 0.0234 |  -0.0016 |    0.0469 |         0.045 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | query_router        | always_nl       | 640 |  0.5938 |  0.5547 |                 0.0391 |   0.0109 |    0.0688 |         0.002 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | confidence_by_query | always_nl       | 640 |  0.5922 |  0.5547 |                 0.0375 |   0.0109 |    0.0656 |         0.002 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | always_symbolic     | always_nl       | 640 |  0.5828 |  0.5547 |                 0.0281 |  -0.0031 |    0.0625 |         0.051 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | query_router        | always_symbolic | 640 |  0.5938 |  0.5828 |                 0.0109 |  -0.0047 |    0.0281 |         0.102 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | confidence_by_query | always_symbolic | 640 |  0.5922 |  0.5828 |                 0.0094 |  -0.0078 |    0.0266 |         0.172 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | query_router        | always_nl       | 398 |  0.6181 |  0.6055 |                 0.0126 |  -0.0201 |    0.0452 |         0.221 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | confidence_by_query | always_nl       | 398 |  0.6156 |  0.6055 |                 0.0101 |  -0.0226 |    0.0452 |         0.289 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | always_symbolic     | always_nl       | 398 |  0.6055 |  0.6055 |                 0      |  -0.0402 |    0.0452 |         0.477 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | query_router        | always_symbolic | 398 |  0.6181 |  0.6055 |                 0.0126 |  -0.0151 |    0.0402 |         0.236 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | confidence_by_query | always_symbolic | 398 |  0.6156 |  0.6055 |                 0.0101 |  -0.0176 |    0.0377 |         0.266 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | query_router        | always_nl       | 398 |  0.5804 |  0.5427 |                 0.0377 |   0.0075 |    0.0704 |         0.011 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | confidence_by_query | always_nl       | 398 |  0.5754 |  0.5427 |                 0.0327 |   0      |    0.0678 |         0.031 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | always_symbolic     | always_nl       | 398 |  0.5754 |  0.5427 |                 0.0327 |  -0.0075 |    0.0704 |         0.07  |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | query_router        | always_symbolic | 398 |  0.5804 |  0.5754 |                 0.005  |  -0.0176 |    0.0251 |         0.377 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | confidence_by_query | always_symbolic | 398 |  0.5754 |  0.5754 |                 0      |  -0.0226 |    0.0201 |         0.548 |

## Control Cross-Validation

| eval_set   | model    | model_display_name   | prompt_mode         | prompt_condition   |   n_splits |   mean_accuracy |   accuracy_ci_low |   accuracy_ci_high |   mean_gain_vs_nl |   gain_ci_low |   gain_ci_high |   p_gain_le_0 |   mean_symbolic_selection_rate |
|:-----------|:---------|:---------------------|:--------------------|:-------------------|-----------:|----------------:|------------------:|-------------------:|------------------:|--------------:|---------------:|--------------:|-------------------------------:|
| control_cv | qwen3_4b | Qwen3-4B             | always_nl           | Always NL          |        100 |          0.5814 |            0.5528 |             0.6086 |            0      |        0      |         0      |          1    |                         0      |
| control_cv | qwen3_4b | Qwen3-4B             | always_symbolic     | Always Symbolic    |        100 |          0.5871 |            0.5543 |             0.6166 |            0.0057 |       -0.0203 |         0.0313 |          0.38 |                         1      |
| control_cv | qwen3_4b | Qwen3-4B             | confidence_by_query | Confidence Router  |        100 |          0.5932 |            0.5634 |             0.6196 |            0.0117 |       -0.0104 |         0.0288 |          0.14 |                         0.5059 |
| control_cv | qwen3_4b | Qwen3-4B             | query_router        | Query Router       |        100 |          0.5863 |            0.5557 |             0.6136 |            0.0049 |       -0.0173 |         0.0192 |          0.32 |                         0.5268 |
| control_cv | qwen3_8b | Qwen3-8B             | always_nl           | Always NL          |        100 |          0.5618 |            0.5334 |             0.5829 |            0      |        0      |         0      |          1    |                         0      |
| control_cv | qwen3_8b | Qwen3-8B             | always_symbolic     | Always Symbolic    |        100 |          0.5635 |            0.535  |             0.5894 |            0.0017 |       -0.0171 |         0.018  |          0.46 |                         1      |
| control_cv | qwen3_8b | Qwen3-8B             | confidence_by_query | Confidence Router  |        100 |          0.5714 |            0.5419 |             0.5959 |            0.0096 |       -0.0056 |         0.0258 |          0.13 |                         0.5689 |
| control_cv | qwen3_8b | Qwen3-8B             | query_router        | Query Router       |        100 |          0.5709 |            0.548  |             0.5961 |            0.0091 |       -0.0042 |         0.0237 |          0.08 |                         0.6699 |

## Transfer Strict Contrast Metrics

| eval_set                 | model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode         |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |   scca |   signed_ccc |   invalid_pair_rate |
|:-------------------------|:---------|:---------------------|:--------------------|:------------------|:--------------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|-------:|-------------:|--------------------:|
| main_full_transfer       | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | always_nl           |       358 |             358 |       0.405  |              0.3017 |            0.1034 |               0.2291 |              0.3659 | 0.3017 |       0.1983 |                   0 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | always_symbolic     |       358 |             358 |       0.4693 |              0.338  |            0.1313 |               0.162  |              0.3687 | 0.338  |       0.2067 |                   0 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | confidence_by_query |       358 |             358 |       0.4469 |              0.338  |            0.1089 |               0.1983 |              0.3547 | 0.338  |       0.2291 |                   0 |
| main_full_transfer       | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | query_router        |       358 |             358 |       0.4609 |              0.3603 |            0.1006 |               0.1676 |              0.3715 | 0.3603 |       0.2598 |                   0 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | always_nl           |       358 |             358 |       0.4441 |              0.3045 |            0.1397 |               0.3436 |              0.2123 | 0.3045 |       0.1648 |                   0 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | always_symbolic     |       358 |             358 |       0.4218 |              0.2849 |            0.1369 |               0.4413 |              0.1369 | 0.2849 |       0.148  |                   0 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | confidence_by_query |       358 |             358 |       0.4302 |              0.3017 |            0.1285 |               0.4274 |              0.1425 | 0.3017 |       0.1732 |                   0 |
| main_full_transfer       | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | query_router        |       358 |             358 |       0.4832 |              0.3408 |            0.1425 |               0.3687 |              0.148  | 0.3408 |       0.1983 |                   0 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | always_nl           |       163 |             163 |       0.4233 |              0.3006 |            0.1227 |               0.227  |              0.3497 | 0.3006 |       0.1779 |                   0 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | always_symbolic     |       163 |             163 |       0.454  |              0.3558 |            0.0982 |               0.1656 |              0.3804 | 0.3558 |       0.2577 |                   0 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | confidence_by_query |       163 |             163 |       0.4663 |              0.362  |            0.1043 |               0.1902 |              0.3436 | 0.362  |       0.2577 |                   0 |
| main_nonoverlap_transfer | qwen3_4b | Qwen3-4B             | adaptive_routing    | adaptive_routing  | query_router        |       163 |             163 |       0.4969 |              0.3865 |            0.1104 |               0.1534 |              0.3497 | 0.3865 |       0.2761 |                   0 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | always_nl           |       163 |             163 |       0.5092 |              0.319  |            0.1902 |               0.3313 |              0.1595 | 0.319  |       0.1288 |                   0 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | always_symbolic     |       163 |             163 |       0.4847 |              0.3067 |            0.1779 |               0.3497 |              0.1656 | 0.3067 |       0.1288 |                   0 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | confidence_by_query |       163 |             163 |       0.4847 |              0.3006 |            0.184  |               0.362  |              0.1534 | 0.3006 |       0.1166 |                   0 |
| main_nonoverlap_transfer | qwen3_8b | Qwen3-8B             | adaptive_routing    | adaptive_routing  | query_router        |       163 |             163 |       0.5337 |              0.3558 |            0.1779 |               0.2883 |              0.1779 | 0.3558 |       0.1779 |                   0 |
