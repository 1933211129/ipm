# MLISE 2026 最终实验抽样记录

- 抽样时间：2026-05-13 09:04:00
- 主数据集路径：`/data/kongyb/ipm/datasets/cladder/data/full_v1.5_default.csv`
- 样本模式：`formal`
- 主样本数：`640`
- stress split 总样本数：`500`
- 每个 stress split 目标样本数：`100`
- 主样本输出：`outputs/mlise2026_qwen_final/runs/final_20260513_090357/tables/selected_main_subset_formal.csv`
- stress 样本输出：`outputs/mlise2026_qwen_final/runs/final_20260513_090357/tables/selected_stress_subset_formal_100.csv`

## Split 标签分布

| diagnostic_source   | dataset_variant   |   no |   yes |
|:--------------------|:------------------|-----:|------:|
| main                | main              |  320 |   320 |
| stress              | anticommonsense   |   50 |    50 |
| stress              | commonsense       |   50 |    50 |
| stress              | easy              |   50 |    50 |
| stress              | hard              |   50 |    50 |
| stress              | noncommonsense    |   50 |    50 |

## Query Type 标签分布

| diagnostic_source   | dataset_variant   | query_type         |   no |   yes |
|:--------------------|:------------------|:-------------------|-----:|------:|
| main                | main              | ate                |   50 |    50 |
| main                | main              | backadj            |   50 |    50 |
| main                | main              | correlation        |   50 |    50 |
| main                | main              | det-counterfactual |   30 |    30 |
| main                | main              | ett                |   30 |    30 |
| main                | main              | marginal           |   50 |    50 |
| main                | main              | nde                |   30 |    30 |
| main                | main              | nie                |   30 |    30 |
| stress              | anticommonsense   | ate                |    5 |     5 |
| stress              | anticommonsense   | backadj            |    5 |     5 |
| stress              | anticommonsense   | collider_bias      |    5 |     5 |
| stress              | anticommonsense   | correlation        |    5 |     5 |
| stress              | anticommonsense   | det-counterfactual |    5 |     5 |
| stress              | anticommonsense   | ett                |    5 |     5 |
| stress              | anticommonsense   | exp_away           |    5 |     5 |
| stress              | anticommonsense   | marginal           |    5 |     5 |
| stress              | anticommonsense   | nde                |    5 |     5 |
| stress              | anticommonsense   | nie                |    5 |     5 |
| stress              | commonsense       | ate                |    5 |     5 |
| stress              | commonsense       | backadj            |    5 |     5 |
| stress              | commonsense       | collider_bias      |    5 |     5 |
| stress              | commonsense       | correlation        |    5 |     5 |
| stress              | commonsense       | det-counterfactual |    5 |     5 |
| stress              | commonsense       | ett                |    5 |     5 |
| stress              | commonsense       | exp_away           |    5 |     5 |
| stress              | commonsense       | marginal           |    5 |     5 |
| stress              | commonsense       | nde                |    5 |     5 |
| stress              | commonsense       | nie                |    5 |     5 |
| stress              | easy              | ate                |    5 |     5 |
| stress              | easy              | backadj            |    5 |     5 |
| stress              | easy              | collider_bias      |    5 |     5 |
| stress              | easy              | correlation        |    5 |     5 |
| stress              | easy              | det-counterfactual |    5 |     5 |
| stress              | easy              | ett                |    5 |     5 |
| stress              | easy              | exp_away           |    5 |     5 |
| stress              | easy              | marginal           |    5 |     5 |
| stress              | easy              | nde                |    5 |     5 |
| stress              | easy              | nie                |    5 |     5 |
| stress              | hard              | ate                |    5 |     5 |
| stress              | hard              | backadj            |    5 |     5 |
| stress              | hard              | collider_bias      |    5 |     5 |
| stress              | hard              | correlation        |    5 |     5 |
| stress              | hard              | det-counterfactual |    5 |     5 |
| stress              | hard              | ett                |    5 |     5 |
| stress              | hard              | exp_away           |    5 |     5 |
| stress              | hard              | marginal           |    5 |     5 |
| stress              | hard              | nde                |    5 |     5 |
| stress              | hard              | nie                |    5 |     5 |
| stress              | noncommonsense    | ate                |    5 |     5 |
| stress              | noncommonsense    | backadj            |    5 |     5 |
| stress              | noncommonsense    | collider_bias      |    5 |     5 |
| stress              | noncommonsense    | correlation        |    5 |     5 |
| stress              | noncommonsense    | det-counterfactual |    5 |     5 |
| stress              | noncommonsense    | ett                |    5 |     5 |
| stress              | noncommonsense    | exp_away           |    5 |     5 |
| stress              | noncommonsense    | marginal           |    5 |     5 |
| stress              | noncommonsense    | nde                |    5 |     5 |
| stress              | noncommonsense    | nie                |    5 |     5 |
