# Qwen3-4B Residual Stream Patching 记录

- 生成时间：2026-05-13 06:54:45
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 候选来源：`selected_subset`
- TransformerLens 状态：`transformer_lens_load_failed: Can't load the configuration of 'Qwen/Qwen3-4B'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'Qwen/Qwen3-4B' is the correct path to a directory containing a config.json file`
- 选中 pair 数：`16`

## 结果概述

本轮共得到 `16` 对 clean/corrupted pair，其中最大 recovery 为正的 pair 数为 `15`。

本实验报告中的 `recovery` 表示把 clean 样本的 residual stream 激活写入 corrupted 样本后，输出 logits 向 clean 样本正确答案方向移动的幅度。同时保留 `corrupted_gold_recovery` 字段，用于记录该 patch 对 corrupted 样本自身 gold label 的影响，避免过度解释。

## 汇总表

| model    | model_display_name   | method                   |   n_rows |   n_pairs |   mean_recovery |   max_recovery |   mean_corrupted_gold_recovery |
|:---------|:---------------------|:-------------------------|---------:|----------:|----------------:|---------------:|-------------------------------:|
| qwen3_4b | Qwen3-4B             | hf_forward_hook_residual |      576 |        16 |         1.18164 |        4.28125 |                       -1.18164 |
