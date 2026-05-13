# Qwen3-4B 行为评测记录

- 生成时间：2026-05-13 06:48:55
- 模型路径：`/data/LLM/Qwen/Qwen3-4B`
- 样本数：`640`
- prompt 设置：`direct`、`structured`
- 解析规则：自动正则解析，不调用外部模型修补答案。

## 主结果

| model    | model_display_name   | sample_mode   | prompt_mode   |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:---------|:---------------------|:--------------|:--------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_4b | Qwen3-4B             | formal        | direct        | 640 |   0.578125 |            1 |              0 |     0.0344998 |
| qwen3_4b | Qwen3-4B             | formal        | structured    | 640 |   0.560937 |            1 |              0 |     0.0324599 |

## PCC

| model    | model_display_name   | sample_mode   | prompt_mode   | transition   |    n |      pcc |   invalid_pair_rate |
|:---------|:---------------------|:--------------|:--------------|:-------------|-----:|---------:|--------------------:|
| qwen3_4b | Qwen3-4B             | formal        | direct        | 1→2          |  997 | 0.533601 |                   0 |
| qwen3_4b | Qwen3-4B             | formal        | direct        | 1→3          | 1238 | 0.505654 |                   0 |
| qwen3_4b | Qwen3-4B             | formal        | direct        | 2→3          | 1138 | 0.496485 |                   0 |
| qwen3_4b | Qwen3-4B             | formal        | structured    | 1→2          |  997 | 0.514544 |                   0 |
| qwen3_4b | Qwen3-4B             | formal        | structured    | 1→3          | 1238 | 0.496769 |                   0 |
| qwen3_4b | Qwen3-4B             | formal        | structured    | 2→3          | 1138 | 0.501757 |                   0 |

## Story All-Correct Rate

| model    | model_display_name   | sample_mode   | prompt_mode   |   n_stories |   story_all_correct_rate |
|:---------|:---------------------|:--------------|:--------------|------------:|-------------------------:|
| qwen3_4b | Qwen3-4B             | formal        | direct        |          47 |                0.0212766 |
| qwen3_4b | Qwen3-4B             | formal        | structured    |          47 |                0.0212766 |

## 输出标签分布

| model    | model_display_name   | prompt_mode   | parsed_label   |   n |
|:---------|:---------------------|:--------------|:---------------|----:|
| qwen3_4b | Qwen3-4B             | direct        | no             | 302 |
| qwen3_4b | Qwen3-4B             | direct        | yes            | 338 |
| qwen3_4b | Qwen3-4B             | structured    | no             | 269 |
| qwen3_4b | Qwen3-4B             | structured    | yes            | 371 |
