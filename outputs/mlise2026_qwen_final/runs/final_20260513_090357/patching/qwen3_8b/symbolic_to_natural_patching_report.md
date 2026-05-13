# MLISE 2026 symbolic-to-natural patching 结果

- 生成时间：2026-05-13 12:54:30
- 模型：`Qwen3-8B`
- 候选样本：`24`
- 候选条件：`nl` 错，`symbolic_solver_concise` 对。

## Control Summary

| model    | model_display_name   | patch_condition      |   n_rows |   n_samples |   mean_absolute_recovery |   median_absolute_recovery |   max_absolute_recovery |   positive_recovery_rate |   mean_normalized_recovery |
|:---------|:---------------------|:---------------------|---------:|------------:|-------------------------:|---------------------------:|------------------------:|-------------------------:|---------------------------:|
| qwen3_8b | Qwen3-8B             | matched_symbolic     |      864 |          24 |                  -0.4201 |                    -0.25   |                  7.7656 |                   0.3993 |                   0.407772 |
| qwen3_8b | Qwen3-8B             | random_symbolic      |      864 |          24 |                  -0.11   |                    -0.1953 |                 11.0312 |                   0.4306 |                  -0.114056 |
| qwen3_8b | Qwen3-8B             | matched_minus_random |      864 |          24 |                  -0.3101 |                    -0.0625 |                 11.3906 |                   0.4653 |                            |

## Top Layers

| model    | model_display_name   |   layer |   mean_recovery |   median_recovery |   max_recovery |   positive_recovery_rate |   mean_normalized_recovery |
|:---------|:---------------------|--------:|----------------:|------------------:|---------------:|-------------------------:|---------------------------:|
| qwen3_8b | Qwen3-8B             |      10 |          0.1159 |            0.0938 |         3.375  |                   0.5    |                     0.6392 |
| qwen3_8b | Qwen3-8B             |      11 |          0.071  |           -0.1875 |         4.1875 |                   0.4583 |                     0.635  |
| qwen3_8b | Qwen3-8B             |       1 |          0.0247 |            0.0234 |         0.75   |                   0.5417 |                     0.0829 |
| qwen3_8b | Qwen3-8B             |       2 |          0.0078 |            0.0078 |         0.7188 |                   0.5    |                    -0.0314 |
| qwen3_8b | Qwen3-8B             |       5 |         -0.0182 |           -0.0938 |         1.9688 |                   0.4167 |                     0.1354 |
| qwen3_8b | Qwen3-8B             |       3 |         -0.0182 |           -0.1094 |         1.1875 |                   0.4167 |                    -0.0398 |
| qwen3_8b | Qwen3-8B             |      13 |         -0.0417 |           -0.25   |         2.875  |                   0.4167 |                     0.0165 |
| qwen3_8b | Qwen3-8B             |       0 |         -0.0469 |           -0.125  |         0.625  |                   0.2917 |                     0.0048 |
