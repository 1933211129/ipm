# Qwen3-0.6B 行为评测记录

- 生成时间：2026-05-13 06:48:19
- 模型路径：`/data/LLM/Qwen/Qwen3-0___6B`
- 样本数：`640`
- prompt 设置：`direct`、`structured`
- 解析规则：自动正则解析，不调用外部模型修补答案。

## 主结果

| model      | model_display_name   | sample_mode   | prompt_mode   |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |
|:-----------|:---------------------|:--------------|:--------------|----:|-----------:|-------------:|---------------:|--------------:|
| qwen3_0_6b | Qwen3-0.6B           | formal        | direct        | 640 |        0.5 |            1 |              0 |     0.0238246 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | structured    | 640 |        0.5 |            1 |              0 |     0.0195785 |

## PCC

| model      | model_display_name   | sample_mode   | prompt_mode   | transition   |    n |      pcc |   invalid_pair_rate |
|:-----------|:---------------------|:--------------|:--------------|:-------------|-----:|---------:|--------------------:|
| qwen3_0_6b | Qwen3-0.6B           | formal        | direct        | 1→2          |  997 | 0.52658  |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | direct        | 1→3          | 1238 | 0.494346 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | direct        | 2→3          | 1138 | 0.521968 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | structured    | 1→2          |  997 | 0.52658  |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | structured    | 1→3          | 1238 | 0.494346 |                   0 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | structured    | 2→3          | 1138 | 0.521968 |                   0 |

## Story All-Correct Rate

| model      | model_display_name   | sample_mode   | prompt_mode   |   n_stories |   story_all_correct_rate |
|:-----------|:---------------------|:--------------|:--------------|------------:|-------------------------:|
| qwen3_0_6b | Qwen3-0.6B           | formal        | direct        |          47 |                0.0212766 |
| qwen3_0_6b | Qwen3-0.6B           | formal        | structured    |          47 |                0.0212766 |

## 输出标签分布

| model      | model_display_name   | prompt_mode   | parsed_label   |   n |
|:-----------|:---------------------|:--------------|:---------------|----:|
| qwen3_0_6b | Qwen3-0.6B           | direct        | yes            | 640 |
| qwen3_0_6b | Qwen3-0.6B           | structured    | yes            | 640 |
