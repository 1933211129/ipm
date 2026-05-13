# Qwen3-4B Formal-to-Natural Patching 记录

- 生成时间：2026-05-13 07:23:58
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 候选样本数：`16`
- 方法：HuggingFace forward hook；将 `nl_formal` 条件下每层最后 token 的 residual 输出 patch 到 `nl` 条件。

## 结果概述

本轮共分析 `16` 个 natural-fail / scaffold-success 样本，其中最大 absolute recovery 为正的样本数为 `14`。

## 汇总表

| model    | model_display_name   | method                          |   n_rows |   n_samples |   mean_absolute_recovery |   max_absolute_recovery |   mean_normalized_recovery |
|:---------|:---------------------|:--------------------------------|---------:|------------:|-------------------------:|------------------------:|---------------------------:|
| qwen3_4b | Qwen3-4B             | hf_last_token_formal_to_natural |      576 |          16 |               -0.0224609 |                 3.71875 |                   0.515671 |

## 解释边界

该结果只说明 formal scaffold 条件下的隐藏状态可能携带可转移的答案方向信号，不能解释为完整 causal circuit。
