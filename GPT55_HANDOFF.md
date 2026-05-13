# GPT-5.5 接手说明：MLISE 2026 当前状态

## 1. 当前唯一执行入口

当前 `MLISE 2026` 会议稿以以下文件为唯一正式执行方案：

[MLISE2026_FINAL_EXECUTION_PLAN.md](/Users/xiaokong/task/2026/IPM/MLISE2026_FINAL_EXECUTION_PLAN.md)

后续接手时，优先阅读该文件。旧版 scaling、diagnostic、enhancement 过程文档和旧实验结果已经从仓库中清理。所有实验需要按最终方案从零重新运行。

## 2. 当前仓库状态

已保留：

- `datasets/`：CLadder 数据集；
- `scripts/`：已有实验脚本，可作为最终脚本实现参考；
- `requirements-mlise-a100.txt`：服务器环境依赖；
- `AGENTS.md`：服务器路径、协作规则和安全边界；
- `MLISE2026_FINAL_EXECUTION_PLAN.md`：最终完整执行方案；
- `IP&M.md`、`IPM_rewrite.md`：长期项目参考。

已清理：

- `outputs/` 下所有旧结果；
- 旧 MLISE 过程方案；
- 旧 pilot 报告；
- 旧图表、旧表格、旧日志、旧自动报告。

## 3. 下一步工作

等待用户确认后再继续：

1. 实现 `scripts/mlise2026_qwen_final.py`。
2. 本地只做语法检查，不加载模型。
3. 提交并推送代码。
4. 服务器 `/data/kongyb/ipm` 拉取最新代码。
5. 按最终方案从零重新运行所有实验。

## 4. 实验边界

- 不更换模型家族。
- 不新增外部数据集。
- 三模型：`Qwen3-0.6B`、`Qwen3-4B`、`Qwen3-8B`。
- 主数据集：`CLadder full_v1.5_default`。
- stress split 使用 CLadder 现有五个 split。
- 图表使用英文。
- Markdown 报告、实验记录和行动记录使用中文。
- 正式实验只在服务器执行。
- 不写入或提交密码、token、密钥。

## 5. 主要实验模块

最终方案包含：

1. Overall performance。
2. Query type / rung 细粒度分析。
3. Correct Flip / Wrong Flip / SCCA / Signed CCC。
4. `nl -> nl_formal -> formula_only` 输入条件转移矩阵。
5. Formal component ablation：`nl_var_query`、`nl_var_graph`。
6. Stress robustness：per-split、mean、worst、std。
7. Bootstrap 95% CI 与 paired tests。
8. Qwen3-4B formal-to-natural residual patching。
9. Random patch control。

## 6. 重要提醒

正式论文正文不写执行过程，不写旧方案对比，不写“上一轮”或“与单纯比较模型规模不同”。论文叙事只围绕当前实验设计、结果、分析和结论展开。
