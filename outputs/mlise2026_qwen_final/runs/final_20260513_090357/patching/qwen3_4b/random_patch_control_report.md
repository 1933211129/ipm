# Qwen3-4B Random Patch Control 记录

- 生成时间：2026-05-13 09:14:38
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 候选样本数：`16`
- 方法：随机选择另一样本的 `nl_formal` residual，patch 到当前样本的 `nl` prompt。

## 结果概述

本轮 random patch control 使用 `16` 个目标样本，将其他样本的 `nl_formal` residual patch 到当前样本的 `nl` 条件，用于检验 matched patching 的恢复是否只是任意形式输入 residual 的偶然效应。

## Matched vs. Random 汇总

| model    | model_display_name   | patch_condition      |   n_rows |   n_samples |   mean_absolute_recovery |   median_absolute_recovery |   max_absolute_recovery |   positive_recovery_rate |   mean_normalized_recovery |
|:---------|:---------------------|:---------------------|---------:|------------:|-------------------------:|---------------------------:|------------------------:|-------------------------:|---------------------------:|
| qwen3_4b | Qwen3-4B             | matched              |      576 |          16 |               -0.0224609 |                    0       |                 3.71875 |                 0.461806 |                   0.515671 |
| qwen3_4b | Qwen3-4B             | random               |      576 |          16 |               -0.0609809 |                   -0.03125 |                 2.9375  |                 0.467014 |                   0.35728  |
| qwen3_4b | Qwen3-4B             | matched_minus_random |      576 |          16 |                0.03852   |                    0.03125 |                 2.125   |                 0.506944 |                 nan        |
