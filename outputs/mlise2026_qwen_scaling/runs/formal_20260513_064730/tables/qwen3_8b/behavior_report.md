# Qwen3-8B 行为评测记录

- 生成时间：2026-05-13 06:49:22
- 模型路径：`/data/LLM/Qwen/Qwen3-8B`
- 样本数：`640`
- prompt 设置：`direct`、`structured`
- 解析规则：自动正则解析，不调用外部模型修补答案。

## 主结果

| model    | model_display_name   | sample_mode   | prompt_mode   |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------|:--------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_8b | Qwen3-8B             | formal        | direct        | 640 |   0.55     |            1 |              0 |     0.0399283 |
| qwen3_8b | Qwen3-8B             | formal        | structured    | 640 |   0.539062 |            1 |              0 |     0.0389747 |

## PCC

| model    | model_display_name   | sample_mode   | prompt_mode   | transition   |    n |      pcc |   invalid_pair_rate |
|:---------|:---------------------|:--------------|:--------------|:-------------|-----:|---------:|--------------------:|
| qwen3_8b | Qwen3-8B             | formal        | direct        | 1→2          |  997 | 0.528586 |                   0 |
| qwen3_8b | Qwen3-8B             | formal        | direct        | 1→3          | 1238 | 0.498384 |                   0 |
| qwen3_8b | Qwen3-8B             | formal        | direct        | 2→3          | 1138 | 0.502636 |                   0 |
| qwen3_8b | Qwen3-8B             | formal        | structured    | 1→2          |  997 | 0.496489 |                   0 |
| qwen3_8b | Qwen3-8B             | formal        | structured    | 1→3          | 1238 | 0.491115 |                   0 |
| qwen3_8b | Qwen3-8B             | formal        | structured    | 2→3          | 1138 | 0.498243 |                   0 |

## Story All-Correct Rate

| model    | model_display_name   | sample_mode   | prompt_mode   |   n_stories |   story_all_correct_rate |
|:---------|:---------------------|:--------------|:--------------|------------:|-------------------------:|
| qwen3_8b | Qwen3-8B             | formal        | direct        |          47 |                0.0212766 |
| qwen3_8b | Qwen3-8B             | formal        | structured    |          47 |                0.0212766 |

## 输出标签分布

| model    | model_display_name   | prompt_mode   | parsed_label   |   n |
|:---------|:---------------------|:--------------|:---------------|----:|
| qwen3_8b | Qwen3-8B             | direct        | no             | 400 |
| qwen3_8b | Qwen3-8B             | direct        | yes            | 240 |
| qwen3_8b | Qwen3-8B             | structured    | no             | 369 |
| qwen3_8b | Qwen3-8B             | structured    | yes            | 271 |
