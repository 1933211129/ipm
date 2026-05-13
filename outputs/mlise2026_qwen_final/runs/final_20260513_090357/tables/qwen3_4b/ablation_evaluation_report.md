# Qwen3-4B 评测记录

- 生成时间：2026-05-13 09:07:47
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 评测来源：`ablation`
- 样本数：`640`
- 报告语言：中文。
- 图表语言：英文。
- 解析规则：自动正则解析，不调用外部模型修补答案。

## Summary

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   | prompt_condition              |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------------|:------------------|:--------------|:------------------------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_4b | Qwen3-4B             | main                | main              | nl_var_graph  | NL + Variables + Graph        | 640 |   0.595313 |            1 |              0 |     0.0320955 |
| qwen3_4b | Qwen3-4B             | main                | main              | nl_var_query  | NL + Variables + Formal Query | 640 |   0.56875  |            1 |              0 |     0.0328662 |

## Strict Contrast Consistency

| model    | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode   |   n_pairs |   n_valid_pairs |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   invariant_yes_rate |   invariant_no_rate |     scca |   signed_ccc |   invalid_pair_rate |
|:---------|:---------------------|:--------------------|:------------------|:--------------|----------:|----------------:|-------------:|--------------------:|------------------:|---------------------:|--------------------:|---------:|-------------:|--------------------:|
| qwen3_4b | Qwen3-4B             | main                | main              | nl_var_graph  |       358 |             358 |     0.452514 |            0.346369 |          0.106145 |             0.231844 |            0.315642 | 0.346369 |     0.240223 |                   0 |
| qwen3_4b | Qwen3-4B             | main                | main              | nl_var_query  |       358 |             358 |     0.396648 |            0.27095  |          0.125698 |             0.178771 |            0.424581 | 0.27095  |     0.145251 |                   0 |

## Scaffold Gain

暂无结果。

## Rescue / Harm

暂无结果。
