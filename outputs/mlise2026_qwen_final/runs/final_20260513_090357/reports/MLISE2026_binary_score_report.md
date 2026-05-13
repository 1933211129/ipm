# MLISE 2026 二分类似然打分实验结果

- 生成时间：2026-05-13 12:30:20
- run_id：`final_20260513_090357`
- 方法：对 `yes` 与 `no` 两个候选 continuation 进行条件 log-likelihood 打分，选择分数更高者作为答案。
- 该方法不依赖自由文本解析，所有样本的解析率为 100%。

## 方法定义

设输入提示为 p，候选答案集合为 A={yes,no}。二分类似然打分选择：

$$
\hat{y}=\arg\max_{a\in A}\log P_\theta(a\mid p,\text{``Final answer:''}).
$$

其中 p 可以是自然语言题干、形式脚手架题干，或符号分解辅助题干。该设置把答案空间固定为二元集合，减少自由生成中的格式漂移和冗余推理文本。

## 总体比较

## binary_score_comparison

| model      | model_display_name   |   nl_generation |   nl_formal_generation |   symbolic_generation |   nl_binary_score |   nl_formal_binary_score |   symbolic_binary_score |   best_binary_gain_vs_nl |
|:-----------|:---------------------|----------------:|-----------------------:|----------------------:|------------------:|-------------------------:|------------------------:|-------------------------:|
| qwen3_0_6b | Qwen3-0.6B           |          0.5    |                 0.5    |                0.4953 |            0.5    |                   0.5    |                  0.5    |                   0      |
| qwen3_4b   | Qwen3-4B             |          0.5953 |                 0.5688 |                0.5672 |            0.6    |                   0.5688 |                  0.5844 |                   0.0047 |
| qwen3_8b   | Qwen3-8B             |          0.5609 |                 0.5453 |                0.5719 |            0.5547 |                   0.5359 |                  0.5828 |                   0.0219 |

## binary_score_summary

| model      | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode                          | prompt_condition                     |   n |   accuracy |   parse_rate |   invalid_rate |   latency_sec |   strict_ccc |   correct_flip_rate |   wrong_flip_rate |   scca |   signed_ccc |
|:-----------|:---------------------|:--------------------|:------------------|:-------------------------------------|:-------------------------------------|----:|-----------:|-------------:|---------------:|--------------:|-------------:|--------------------:|------------------:|-------:|-------------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl                                   | Natural Language                     | 640 |     0.5    |       1      |         0      |        0.0109 |       0      |              0      |            0      | 0      |       0      |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal                            | NL + Formal Scaffold                 | 640 |     0.5    |       1      |         0      |        0.0113 |       0      |              0      |            0      | 0      |       0      |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | symbolic_solver_concise              | Concise Symbolic Solver              | 640 |     0.4953 |       0.9875 |         0.0125 |        0.0895 |       0      |              0      |            0      | 0      |       0      |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_binary_score                      | NL Binary Score                      | 640 |     0.5    |       1      |         0      |        0.004  |       0      |              0      |            0      | 0      |       0      |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal_binary_score               | Formal Scaffold Binary Score         | 640 |     0.5    |       1      |         0      |        0.0039 |       0      |              0      |            0      | 0      |       0      |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | symbolic_solver_concise_binary_score | Concise Symbolic Solver Binary Score | 640 |     0.5    |       1      |         0      |        0.0055 |       0      |              0      |            0      | 0      |       0      |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl                                   | Natural Language                     | 640 |     0.5953 |       1      |         0      |        0.0292 |       0.4078 |              0.2905 |            0.1173 | 0.2905 |       0.1732 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl_formal                            | NL + Formal Scaffold                 | 640 |     0.5688 |       1      |         0      |        0.0338 |       0.4078 |              0.2821 |            0.1257 | 0.2821 |       0.1564 |
| qwen3_4b   | Qwen3-4B             | main                | main              | symbolic_solver_concise              | Concise Symbolic Solver              | 640 |     0.5672 |       0.9719 |         0.0281 |        0.4928 |       0.4645 |              0.3521 |            0.1124 | 0.3521 |       0.2396 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl_binary_score                      | NL Binary Score                      | 640 |     0.6    |       1      |         0      |        0.0141 |       0.405  |              0.3017 |            0.1034 | 0.3017 |       0.1983 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl_formal_binary_score               | Formal Scaffold Binary Score         | 640 |     0.5688 |       1      |         0      |        0.019  |       0.3911 |              0.2709 |            0.1201 | 0.2709 |       0.1508 |
| qwen3_4b   | Qwen3-4B             | main                | main              | symbolic_solver_concise_binary_score | Concise Symbolic Solver Binary Score | 640 |     0.5844 |       1      |         0      |        0.0277 |       0.4693 |              0.338  |            0.1313 | 0.338  |       0.2067 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl                                   | Natural Language                     | 640 |     0.5609 |       1      |         0      |        0.054  |       0.4441 |              0.3017 |            0.1425 | 0.3017 |       0.1592 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl_formal                            | NL + Formal Scaffold                 | 640 |     0.5453 |       1      |         0      |        0.0625 |       0.4721 |              0.3128 |            0.1592 | 0.3128 |       0.1536 |
| qwen3_8b   | Qwen3-8B             | main                | main              | symbolic_solver_concise              | Concise Symbolic Solver              | 640 |     0.5719 |       0.9656 |         0.0344 |        0.9164 |       0.5062 |              0.3789 |            0.1273 | 0.3789 |       0.2516 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl_binary_score                      | NL Binary Score                      | 640 |     0.5547 |       1      |         0      |        0.0213 |       0.4441 |              0.3045 |            0.1397 | 0.3045 |       0.1648 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl_formal_binary_score               | Formal Scaffold Binary Score         | 640 |     0.5359 |       1      |         0      |        0.0301 |       0.4609 |              0.2989 |            0.162  | 0.2989 |       0.1369 |
| qwen3_8b   | Qwen3-8B             | main                | main              | symbolic_solver_concise_binary_score | Concise Symbolic Solver Binary Score | 640 |     0.5828 |       1      |         0      |        0.043  |       0.4218 |              0.2849 |            0.1369 | 0.2849 |       0.148  |

## binary_score_by_query

| model      | model_display_name   | query_type         |   nl_generation |   symbolic_generation |   nl_binary_score |   nl_formal_binary_score |   symbolic_binary_score |
|:-----------|:---------------------|:-------------------|----------------:|----------------------:|------------------:|-------------------------:|------------------------:|
| qwen3_0_6b | Qwen3-0.6B           | ate                |          0.5    |                0.5    |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | backadj            |          0.5    |                0.5    |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | correlation        |          0.5    |                0.5    |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | det-counterfactual |          0.5    |                0.4833 |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | ett                |          0.5    |                0.5    |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | marginal           |          0.5    |                0.5    |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | nde                |          0.5    |                0.5    |            0.5    |                   0.5    |                  0.5    |
| qwen3_0_6b | Qwen3-0.6B           | nie                |          0.5    |                0.4667 |            0.5    |                   0.5    |                  0.5    |
| qwen3_4b   | Qwen3-4B             | ate                |          0.74   |                0.76   |            0.75   |                   0.71   |                  0.78   |
| qwen3_4b   | Qwen3-4B             | backadj            |          0.52   |                0.56   |            0.52   |                   0.52   |                  0.54   |
| qwen3_4b   | Qwen3-4B             | correlation        |          0.65   |                0.6    |            0.65   |                   0.56   |                  0.62   |
| qwen3_4b   | Qwen3-4B             | det-counterfactual |          0.5833 |                0.5333 |            0.5833 |                   0.5833 |                  0.5333 |
| qwen3_4b   | Qwen3-4B             | ett                |          0.4667 |                0.3667 |            0.4833 |                   0.4    |                  0.45   |
| qwen3_4b   | Qwen3-4B             | marginal           |          0.53   |                0.53   |            0.54   |                   0.56   |                  0.53   |
| qwen3_4b   | Qwen3-4B             | nde                |          0.6167 |                0.5167 |            0.6333 |                   0.6167 |                  0.4833 |
| qwen3_4b   | Qwen3-4B             | nie                |          0.6167 |                0.55   |            0.6    |                   0.55   |                  0.65   |
| qwen3_8b   | Qwen3-8B             | ate                |          0.76   |                0.82   |            0.76   |                   0.7    |                  0.8    |
| qwen3_8b   | Qwen3-8B             | backadj            |          0.44   |                0.46   |            0.44   |                   0.45   |                  0.48   |
| qwen3_8b   | Qwen3-8B             | correlation        |          0.55   |                0.5    |            0.55   |                   0.54   |                  0.53   |
| qwen3_8b   | Qwen3-8B             | det-counterfactual |          0.5333 |                0.5833 |            0.4833 |                   0.5167 |                  0.5    |
| qwen3_8b   | Qwen3-8B             | ett                |          0.45   |                0.3333 |            0.4333 |                   0.4333 |                  0.35   |
| qwen3_8b   | Qwen3-8B             | marginal           |          0.53   |                0.5    |            0.52   |                   0.53   |                  0.55   |
| qwen3_8b   | Qwen3-8B             | nde                |          0.6333 |                0.6833 |            0.6667 |                   0.5667 |                  0.7667 |
| qwen3_8b   | Qwen3-8B             | nie                |          0.5667 |                0.7    |            0.55   |                   0.5    |                  0.6667 |

## binary_score_accuracy_ci

| model      | model_display_name   | diagnostic_source   | dataset_variant   | prompt_mode                          | metric   |   estimate |   ci_low |   ci_high |   n_bootstrap |
|:-----------|:---------------------|:--------------------|:------------------|:-------------------------------------|:---------|-----------:|---------:|----------:|--------------:|
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl                                   | accuracy |     0.5    |   0.4641 |    0.5422 |          1000 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_binary_score                      | accuracy |     0.5    |   0.4625 |    0.5375 |          1000 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal                            | accuracy |     0.5    |   0.4625 |    0.5406 |          1000 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | nl_formal_binary_score               | accuracy |     0.5    |   0.4625 |    0.5391 |          1000 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | symbolic_solver_concise              | accuracy |     0.4953 |   0.4578 |    0.5328 |          1000 |
| qwen3_0_6b | Qwen3-0.6B           | main                | main              | symbolic_solver_concise_binary_score | accuracy |     0.5    |   0.4593 |    0.5406 |          1000 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl                                   | accuracy |     0.5953 |   0.5578 |    0.6297 |          1000 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl_binary_score                      | accuracy |     0.6    |   0.5609 |    0.6391 |          1000 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl_formal                            | accuracy |     0.5688 |   0.5312 |    0.6078 |          1000 |
| qwen3_4b   | Qwen3-4B             | main                | main              | nl_formal_binary_score               | accuracy |     0.5688 |   0.5312 |    0.6078 |          1000 |
| qwen3_4b   | Qwen3-4B             | main                | main              | symbolic_solver_concise              | accuracy |     0.5672 |   0.525  |    0.6047 |          1000 |
| qwen3_4b   | Qwen3-4B             | main                | main              | symbolic_solver_concise_binary_score | accuracy |     0.5844 |   0.5438 |    0.6219 |          1000 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl                                   | accuracy |     0.5609 |   0.5234 |    0.5985 |          1000 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl_binary_score                      | accuracy |     0.5547 |   0.5172 |    0.5906 |          1000 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl_formal                            | accuracy |     0.5453 |   0.5078 |    0.5828 |          1000 |
| qwen3_8b   | Qwen3-8B             | main                | main              | nl_formal_binary_score               | accuracy |     0.5359 |   0.4953 |    0.575  |          1000 |
| qwen3_8b   | Qwen3-8B             | main                | main              | symbolic_solver_concise              | accuracy |     0.5719 |   0.5344 |    0.6109 |          1000 |
| qwen3_8b   | Qwen3-8B             | main                | main              | symbolic_solver_concise_binary_score | accuracy |     0.5828 |   0.5469 |    0.6188 |          1000 |
