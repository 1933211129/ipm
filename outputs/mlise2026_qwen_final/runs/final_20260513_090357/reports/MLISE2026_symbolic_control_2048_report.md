# MLISE 2026 Symbolic Control 实验结果

- 生成时间：2026-05-13 12:58:56
- 样本数：`2048`
- 目标：检验 matched symbolic decomposition 是否优于 shuffled symbolic decomposition。

## 总体结果

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode                    | prompt_condition               |    n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:-------------------------------|:-------------------------------|-----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | NL Binary Score                | 2048 |     0.583  |            1 |              0 |        0.0136 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | Matched Symbolic Binary Score  | 2048 |     0.5854 |            1 |              0 |        0.0279 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | Shuffled Symbolic Binary Score | 2048 |     0.543  |            1 |              0 |        0.0282 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | NL Binary Score                | 2048 |     0.5591 |            1 |              0 |        0.0211 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | Matched Symbolic Binary Score  | 2048 |     0.562  |            1 |              0 |        0.0433 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | Shuffled Symbolic Binary Score | 2048 |     0.5332 |            1 |              0 |        0.0439 |

## 配对 Bootstrap

| model    | condition_a                    | condition_b                    |    n |   acc_a |   acc_b |   paired_gain |   ci_low |   ci_high |   p_gain_le_0 |
|:---------|:-------------------------------|:-------------------------------|-----:|--------:|--------:|--------------:|---------:|----------:|--------------:|
| qwen3_4b | symbolic_matched_binary_score  | nl_binary_score                | 2048 |  0.5854 |  0.583  |        0.0024 |  -0.0156 |    0.022  |         0.432 |
| qwen3_4b | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 2048 |  0.5854 |  0.543  |        0.0425 |   0.0239 |    0.0605 |         0     |
| qwen3_4b | symbolic_shuffled_binary_score | nl_binary_score                | 2048 |  0.543  |  0.583  |       -0.04   |  -0.0591 |   -0.021  |         1     |
| qwen3_8b | symbolic_matched_binary_score  | nl_binary_score                | 2048 |  0.562  |  0.5591 |        0.0029 |  -0.0137 |    0.0205 |         0.396 |
| qwen3_8b | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 2048 |  0.562  |  0.5332 |        0.0288 |   0.0049 |    0.0513 |         0.011 |
| qwen3_8b | symbolic_shuffled_binary_score | nl_binary_score                | 2048 |  0.5332 |  0.5591 |       -0.0259 |  -0.0483 |   -0.0044 |         0.991 |

## Query Type 结果

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode                    | query_type         |   n |   accuracy |   parse_rate |
|:---------|:---------------------|:--------------------|:------------------|:-------------------------------|:-------------------|----:|-----------:|-------------:|
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | ate                | 256 |     0.7578 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | backadj            | 256 |     0.5586 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | correlation        | 256 |     0.5508 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | det-counterfactual | 256 |     0.6016 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | ett                | 256 |     0.457  |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | marginal           | 256 |     0.5195 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | nde                | 256 |     0.6172 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                | nie                | 256 |     0.6016 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | ate                | 256 |     0.8203 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | backadj            | 256 |     0.6016 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | correlation        | 256 |     0.543  |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | det-counterfactual | 256 |     0.5977 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | ett                | 256 |     0.4531 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | marginal           | 256 |     0.5039 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | nde                | 256 |     0.5469 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | nie                | 256 |     0.6172 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | ate                | 256 |     0.6992 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | backadj            | 256 |     0.5742 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | correlation        | 256 |     0.5234 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | det-counterfactual | 256 |     0.5664 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | ett                | 256 |     0.4023 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | marginal           | 256 |     0.5    |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | nde                | 256 |     0.5469 |            1 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | nie                | 256 |     0.5312 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | ate                | 256 |     0.7969 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | backadj            | 256 |     0.4492 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | correlation        | 256 |     0.5195 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | det-counterfactual | 256 |     0.5703 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | ett                | 256 |     0.3984 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | marginal           | 256 |     0.5117 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | nde                | 256 |     0.6445 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                | nie                | 256 |     0.582  |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | ate                | 256 |     0.8359 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | backadj            | 256 |     0.457  |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | correlation        | 256 |     0.5    |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | det-counterfactual | 256 |     0.5898 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | ett                | 256 |     0.3047 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | marginal           | 256 |     0.5312 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | nde                | 256 |     0.6875 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  | nie                | 256 |     0.5898 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | ate                | 256 |     0.7109 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | backadj            | 256 |     0.4414 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | correlation        | 256 |     0.4805 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | det-counterfactual | 256 |     0.5664 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | ett                | 256 |     0.4414 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | marginal           | 256 |     0.5    |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | nde                | 256 |     0.5938 |            1 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score | nie                | 256 |     0.5312 |            1 |

## Strict Contrast 指标

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode                    |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |   scca |   signed_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:-------------------------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|-------:|-------------:|--------------------:|
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | nl_binary_score                |      3393 |            3393 |       0.3867 |              0.2765 |            0.1102 |               0.2202 |              0.3932 | 0.2765 |       0.1662 |                   0 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  |      3393 |            3393 |       0.4922 |              0.3327 |            0.1594 |               0.1303 |              0.3775 | 0.3327 |       0.1733 |                   0 |
| qwen3_4b | Qwen3-4B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score |      3393 |            3393 |       0.3705 |              0.2284 |            0.1421 |               0.0905 |              0.5391 | 0.2284 |       0.0864 |                   0 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | nl_binary_score                |      3393 |            3393 |       0.4468 |              0.303  |            0.1438 |               0.277  |              0.2762 | 0.303  |       0.1592 |                   0 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_matched_binary_score  |      3393 |            3393 |       0.4777 |              0.3295 |            0.1482 |               0.331  |              0.1913 | 0.3295 |       0.1813 |                   0 |
| qwen3_8b | Qwen3-8B             | symbolic_control    | symbolic_control  | symbolic_shuffled_binary_score |      3393 |            3393 |       0.402  |              0.2479 |            0.1541 |               0.0893 |              0.5087 | 0.2479 |       0.0937 |                   0 |

## Strict Contrast 配对 Bootstrap

| model    | model_display_name   | condition_a                    | condition_b                    | metric     |   n_pairs |   metric_a |   metric_b |   paired_metric_diff |   ci_low |   ci_high |   p_diff_le_0 |
|:---------|:---------------------|:-------------------------------|:-------------------------------|:-----------|----------:|-----------:|-----------:|---------------------:|---------:|----------:|--------------:|
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | nl_binary_score                | strict_ccc |      3393 |     0.4922 |     0.3867 |               0.1055 |   0.0863 |    0.1235 |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | nl_binary_score                | scca       |      3393 |     0.3327 |     0.2765 |               0.0563 |   0.0415 |    0.0707 |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | nl_binary_score                | wrong_flip |      3393 |     0.1594 |     0.1102 |               0.0492 |   0.0365 |    0.0613 |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | nl_binary_score                | signed_ccc |      3393 |     0.1733 |     0.1662 |               0.0071 |  -0.0136 |    0.0283 |         0.253 |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | strict_ccc |      3393 |     0.4922 |     0.3705 |               0.1217 |   0.102  |    0.1409 |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | scca       |      3393 |     0.3327 |     0.2284 |               0.1043 |   0.0893 |    0.1197 |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | wrong_flip |      3393 |     0.1594 |     0.1421 |               0.0174 |   0.0053 |    0.0298 |         0.001 |
| qwen3_4b | Qwen3-4B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | signed_ccc |      3393 |     0.1733 |     0.0864 |               0.0869 |   0.0669 |    0.107  |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_shuffled_binary_score | nl_binary_score                | strict_ccc |      3393 |     0.3705 |     0.3867 |              -0.0162 |  -0.0345 |    0.0021 |         0.96  |
| qwen3_4b | Qwen3-4B             | symbolic_shuffled_binary_score | nl_binary_score                | scca       |      3393 |     0.2284 |     0.2765 |              -0.048  |  -0.0634 |   -0.0333 |         1     |
| qwen3_4b | Qwen3-4B             | symbolic_shuffled_binary_score | nl_binary_score                | wrong_flip |      3393 |     0.1421 |     0.1102 |               0.0318 |   0.02   |    0.0433 |         0     |
| qwen3_4b | Qwen3-4B             | symbolic_shuffled_binary_score | nl_binary_score                | signed_ccc |      3393 |     0.0864 |     0.1662 |              -0.0799 |  -0.099  |   -0.0601 |         1     |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | nl_binary_score                | strict_ccc |      3393 |     0.4777 |     0.4468 |               0.0309 |   0.0144 |    0.0489 |         0     |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | nl_binary_score                | scca       |      3393 |     0.3295 |     0.303  |               0.0265 |   0.0115 |    0.0413 |         0.001 |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | nl_binary_score                | wrong_flip |      3393 |     0.1482 |     0.1438 |               0.0044 |  -0.005  |    0.0142 |         0.207 |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | nl_binary_score                | signed_ccc |      3393 |     0.1813 |     0.1592 |               0.0221 |   0.0015 |    0.0395 |         0.018 |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | strict_ccc |      3393 |     0.4777 |     0.402  |               0.0757 |   0.0551 |    0.0976 |         0     |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | scca       |      3393 |     0.3295 |     0.2479 |               0.0816 |   0.0645 |    0.0987 |         0     |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | wrong_flip |      3393 |     0.1482 |     0.1541 |              -0.0059 |  -0.0192 |    0.0071 |         0.823 |
| qwen3_8b | Qwen3-8B             | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | signed_ccc |      3393 |     0.1813 |     0.0937 |               0.0875 |   0.0642 |    0.1093 |         0     |
| qwen3_8b | Qwen3-8B             | symbolic_shuffled_binary_score | nl_binary_score                | strict_ccc |      3393 |     0.402  |     0.4468 |              -0.0448 |  -0.0651 |   -0.0256 |         1     |
| qwen3_8b | Qwen3-8B             | symbolic_shuffled_binary_score | nl_binary_score                | scca       |      3393 |     0.2479 |     0.303  |              -0.0551 |  -0.0716 |   -0.0404 |         1     |
| qwen3_8b | Qwen3-8B             | symbolic_shuffled_binary_score | nl_binary_score                | wrong_flip |      3393 |     0.1541 |     0.1438 |               0.0103 |  -0.0027 |    0.0224 |         0.056 |
| qwen3_8b | Qwen3-8B             | symbolic_shuffled_binary_score | nl_binary_score                | signed_ccc |      3393 |     0.0937 |     0.1592 |              -0.0654 |  -0.087  |   -0.0457 |         1     |

## Query Type 配对差异

| model    | model_display_name   | query_type         | condition_a                    | condition_b                    |   n |   acc_a |   acc_b |   paired_accuracy_diff |   ci_low |   ci_high |   p_diff_le_0 |   prediction_change_rate |   a_correct_b_wrong_rate |   a_wrong_b_correct_rate |
|:---------|:---------------------|:-------------------|:-------------------------------|:-------------------------------|----:|--------:|--------:|-----------------------:|---------:|----------:|--------------:|-------------------------:|-------------------------:|-------------------------:|
| qwen3_4b | Qwen3-4B             | ate                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.8203 |  0.7578 |                 0.0625 |   0.0039 |    0.1251 |         0.023 |                   0.2422 |                   0.1523 |                   0.0898 |
| qwen3_4b | Qwen3-4B             | ate                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.8203 |  0.6992 |                 0.1211 |   0.0586 |    0.1875 |         0     |                   0.3008 |                   0.2109 |                   0.0898 |
| qwen3_4b | Qwen3-4B             | ate                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.6992 |  0.7578 |                -0.0586 |  -0.1172 |    0.0078 |         0.967 |                   0.2617 |                   0.1016 |                   0.1602 |
| qwen3_4b | Qwen3-4B             | backadj            | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.6016 |  0.5586 |                 0.043  |   0.0078 |    0.082  |         0.01  |                   0.082  |                   0.0625 |                   0.0195 |
| qwen3_4b | Qwen3-4B             | backadj            | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.6016 |  0.5742 |                 0.0273 |  -0.0078 |    0.0664 |         0.087 |                   0.0898 |                   0.0586 |                   0.0312 |
| qwen3_4b | Qwen3-4B             | backadj            | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5742 |  0.5586 |                 0.0156 |  -0.0273 |    0.0625 |         0.253 |                   0.125  |                   0.0703 |                   0.0547 |
| qwen3_4b | Qwen3-4B             | correlation        | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.543  |  0.5508 |                -0.0078 |  -0.0586 |    0.0391 |         0.645 |                   0.1641 |                   0.0781 |                   0.0859 |
| qwen3_4b | Qwen3-4B             | correlation        | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.543  |  0.5234 |                 0.0195 |  -0.0234 |    0.0625 |         0.228 |                   0.1289 |                   0.0742 |                   0.0547 |
| qwen3_4b | Qwen3-4B             | correlation        | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5234 |  0.5508 |                -0.0273 |  -0.0703 |    0.0157 |         0.909 |                   0.1289 |                   0.0508 |                   0.0781 |
| qwen3_4b | Qwen3-4B             | det-counterfactual | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5977 |  0.6016 |                -0.0039 |  -0.0586 |    0.0508 |         0.592 |                   0.1914 |                   0.0938 |                   0.0977 |
| qwen3_4b | Qwen3-4B             | det-counterfactual | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5977 |  0.5664 |                 0.0312 |  -0.0118 |    0.0781 |         0.104 |                   0.1406 |                   0.0859 |                   0.0547 |
| qwen3_4b | Qwen3-4B             | det-counterfactual | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5664 |  0.6016 |                -0.0352 |  -0.0859 |    0.0157 |         0.922 |                   0.168  |                   0.0664 |                   0.1016 |
| qwen3_4b | Qwen3-4B             | ett                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.4531 |  0.457  |                -0.0039 |  -0.0625 |    0.0587 |         0.602 |                   0.2617 |                   0.1289 |                   0.1328 |
| qwen3_4b | Qwen3-4B             | ett                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.4531 |  0.4023 |                 0.0508 |  -0.0078 |    0.1134 |         0.056 |                   0.2617 |                   0.1562 |                   0.1055 |
| qwen3_4b | Qwen3-4B             | ett                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.4023 |  0.457  |                -0.0547 |  -0.1095 |    0      |         0.978 |                   0.1953 |                   0.0703 |                   0.125  |
| qwen3_4b | Qwen3-4B             | marginal           | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5039 |  0.5195 |                -0.0156 |  -0.0665 |    0.0352 |         0.749 |                   0.1797 |                   0.082  |                   0.0977 |
| qwen3_4b | Qwen3-4B             | marginal           | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5039 |  0.5    |                 0.0039 |  -0.0625 |    0.0703 |         0.48  |                   0.2617 |                   0.1328 |                   0.1289 |
| qwen3_4b | Qwen3-4B             | marginal           | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5    |  0.5195 |                -0.0195 |  -0.0781 |    0.0391 |         0.737 |                   0.2539 |                   0.1172 |                   0.1367 |
| qwen3_4b | Qwen3-4B             | nde                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5469 |  0.6172 |                -0.0703 |  -0.125  |   -0.0156 |         0.994 |                   0.2109 |                   0.0703 |                   0.1406 |
| qwen3_4b | Qwen3-4B             | nde                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5469 |  0.5469 |                 0      |  -0.0547 |    0.0508 |         0.53  |                   0.1719 |                   0.0859 |                   0.0859 |
| qwen3_4b | Qwen3-4B             | nde                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5469 |  0.6172 |                -0.0703 |  -0.125  |   -0.0156 |         0.996 |                   0.1875 |                   0.0586 |                   0.1289 |
| qwen3_4b | Qwen3-4B             | nie                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.6172 |  0.6016 |                 0.0156 |  -0.0508 |    0.0821 |         0.329 |                   0.2578 |                   0.1367 |                   0.1211 |
| qwen3_4b | Qwen3-4B             | nie                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.6172 |  0.5312 |                 0.0859 |   0.0273 |    0.1445 |         0.003 |                   0.2344 |                   0.1602 |                   0.0742 |
| qwen3_4b | Qwen3-4B             | nie                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5312 |  0.6016 |                -0.0703 |  -0.1367 |   -0.0038 |         0.982 |                   0.3516 |                   0.1406 |                   0.2109 |
| qwen3_8b | Qwen3-8B             | ate                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.8359 |  0.7969 |                 0.0391 |  -0.0117 |    0.0938 |         0.076 |                   0.1797 |                   0.1094 |                   0.0703 |
| qwen3_8b | Qwen3-8B             | ate                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.8359 |  0.7109 |                 0.125  |   0.0586 |    0.1914 |         0     |                   0.3125 |                   0.2188 |                   0.0938 |
| qwen3_8b | Qwen3-8B             | ate                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.7109 |  0.7969 |                -0.0859 |  -0.1523 |   -0.0194 |         0.995 |                   0.3203 |                   0.1172 |                   0.2031 |
| qwen3_8b | Qwen3-8B             | backadj            | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.457  |  0.4492 |                 0.0078 |  -0.0195 |    0.0352 |         0.377 |                   0.0469 |                   0.0273 |                   0.0195 |
| qwen3_8b | Qwen3-8B             | backadj            | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.457  |  0.4414 |                 0.0156 |  -0.0117 |    0.0469 |         0.172 |                   0.0547 |                   0.0352 |                   0.0195 |
| qwen3_8b | Qwen3-8B             | backadj            | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.4414 |  0.4492 |                -0.0078 |  -0.0312 |    0.0156 |         0.792 |                   0.0391 |                   0.0156 |                   0.0234 |
| qwen3_8b | Qwen3-8B             | correlation        | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5    |  0.5195 |                -0.0195 |  -0.0703 |    0.0273 |         0.797 |                   0.1758 |                   0.0781 |                   0.0977 |
| qwen3_8b | Qwen3-8B             | correlation        | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5    |  0.4805 |                 0.0195 |  -0.0469 |    0.0859 |         0.305 |                   0.3242 |                   0.1719 |                   0.1523 |
| qwen3_8b | Qwen3-8B             | correlation        | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.4805 |  0.5195 |                -0.0391 |  -0.0977 |    0.0195 |         0.91  |                   0.2344 |                   0.0977 |                   0.1367 |
| qwen3_8b | Qwen3-8B             | det-counterfactual | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5898 |  0.5703 |                 0.0195 |  -0.0273 |    0.0625 |         0.205 |                   0.1211 |                   0.0703 |                   0.0508 |
| qwen3_8b | Qwen3-8B             | det-counterfactual | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5898 |  0.5664 |                 0.0234 |  -0.0273 |    0.0781 |         0.222 |                   0.1875 |                   0.1055 |                   0.082  |
| qwen3_8b | Qwen3-8B             | det-counterfactual | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5664 |  0.5703 |                -0.0039 |  -0.0547 |    0.0508 |         0.569 |                   0.1836 |                   0.0898 |                   0.0938 |
| qwen3_8b | Qwen3-8B             | ett                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.3047 |  0.3984 |                -0.0938 |  -0.1445 |   -0.043  |         1     |                   0.1797 |                   0.043  |                   0.1367 |
| qwen3_8b | Qwen3-8B             | ett                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.3047 |  0.4414 |                -0.1367 |  -0.1914 |   -0.082  |         1     |                   0.2148 |                   0.0391 |                   0.1758 |
| qwen3_8b | Qwen3-8B             | ett                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.4414 |  0.3984 |                 0.043  |  -0.0234 |    0.1094 |         0.11  |                   0.2617 |                   0.1523 |                   0.1094 |
| qwen3_8b | Qwen3-8B             | marginal           | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5312 |  0.5117 |                 0.0195 |  -0.0312 |    0.0703 |         0.264 |                   0.1836 |                   0.1016 |                   0.082  |
| qwen3_8b | Qwen3-8B             | marginal           | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5312 |  0.5    |                 0.0312 |  -0.0547 |    0.1211 |         0.251 |                   0.4922 |                   0.2617 |                   0.2305 |
| qwen3_8b | Qwen3-8B             | marginal           | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5    |  0.5117 |                -0.0117 |  -0.0978 |    0.0703 |         0.636 |                   0.4414 |                   0.2148 |                   0.2266 |
| qwen3_8b | Qwen3-8B             | nde                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.6875 |  0.6445 |                 0.043  |  -0.0234 |    0.1016 |         0.107 |                   0.2617 |                   0.1523 |                   0.1094 |
| qwen3_8b | Qwen3-8B             | nde                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.6875 |  0.5938 |                 0.0938 |   0.0156 |    0.1642 |         0.012 |                   0.3828 |                   0.2383 |                   0.1445 |
| qwen3_8b | Qwen3-8B             | nde                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5938 |  0.6445 |                -0.0508 |  -0.1055 |    0.0078 |         0.962 |                   0.2148 |                   0.082  |                   0.1328 |
| qwen3_8b | Qwen3-8B             | nie                | symbolic_matched_binary_score  | nl_binary_score                | 256 |  0.5898 |  0.582  |                 0.0078 |  -0.0352 |    0.0547 |         0.401 |                   0.1328 |                   0.0703 |                   0.0625 |
| qwen3_8b | Qwen3-8B             | nie                | symbolic_matched_binary_score  | symbolic_shuffled_binary_score | 256 |  0.5898 |  0.5312 |                 0.0586 |  -0.0117 |    0.1328 |         0.06  |                   0.3789 |                   0.2188 |                   0.1602 |
| qwen3_8b | Qwen3-8B             | nie                | symbolic_shuffled_binary_score | nl_binary_score                | 256 |  0.5312 |  0.582  |                -0.0508 |  -0.1289 |    0.0195 |         0.915 |                   0.3867 |                   0.168  |                   0.2188 |
