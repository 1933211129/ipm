# 实验与结论

## 实验设置

本文使用 CLadder `full_v1.5_default` 构建因果推理评测子集。主评测集包含 640 个样本，覆盖 `marginal`、`correlation`、`ate`、`backadj`、`det-counterfactual`、`ett`、`nie` 和 `nde` 八类 query type，并在每个 query type 内保持 yes/no 标签平衡。鲁棒性实验使用 CLadder 的 `commonsense`、`anticommonsense`、`noncommonsense`、`easy` 和 `hard` 五个 stress split，每个 split 抽取 100 个样本并保持标签平衡。

模型包括 Qwen3-0.6B、Qwen3-4B 和 Qwen3-8B。所有实验均采用确定性生成，不进行采样。输出解析优先匹配 `Final answer: yes/no`，其次匹配独立出现的 yes/no；无法解析的输出记为 invalid，并在 parse rate 中单独报告。

输入条件包括五种形式。`nl` 使用原始自然语言题干；`nl_var_query` 在自然语言题干后加入变量映射和 formal query；`nl_var_graph` 加入变量映射和 causal graph；`nl_formal` 同时加入变量映射、causal graph 和 formal query；`formula_only` 保留任务必要的事实条件、变量映射、因果图、formal query 与问题句。所有输入条件均不提供 oracle label 或完整推导步骤。

评价指标包括 Accuracy、Parse Rate、Strict Contrast Causal Consistency (CCC)、Correct Flip、Wrong Flip、Strict Correct Contrast Accuracy (SCCA)、Scaffold Gain、Rescue/Harm、输入条件转移轨迹和 stress robustness。CCC 在同一 story、query type 和 formal form 内构造 gold label 相反的严格对照 pair，并检验模型预测是否也发生翻转。Correct Flip 要求 pair 中两个样本均预测正确；Wrong Flip 表示 pair 中预测发生翻转但两个样本均预测错误。SCCA 定义为 Correct Flip 的平均值。本文还使用 bootstrap 估计 Accuracy、CCC 和条件差异的 95% 置信区间，并报告 McNemar 近似检验。

隐藏层分析在 Qwen3-4B 上进行。我们筛选 `nl` 条件预测错误而 `nl_formal` 条件预测正确的样本，使用 HuggingFace forward hook 进行 residual stream patching。具体而言，将 `nl_formal` 条件下每一层最后 token 的 residual 输出替换到同一样本的 `nl` 条件中，并记录 gold-label logit margin 的恢复量。为了排除任意形式输入 residual 的偶然影响，实验同时加入 random patch control，即使用另一样本的 `nl_formal` residual 替换当前样本的 `nl` residual。

## 实验结果

主结果显示，Qwen3-0.6B 在多数输入条件下接近标签平衡基线，`nl` 与 `nl_formal` 的 accuracy 均为 0.5000。Qwen3-4B 在 `nl` 条件下达到 0.5953，是主评测中的最高总体 accuracy；`nl_formal` 下降到 0.5688，`formula_only` 为 0.5641。Qwen3-8B 的 `nl` accuracy 为 0.5609，`nl_formal` 为 0.5453，`formula_only` 为 0.5531。两个较大模型均未从完整形式脚手架中获得稳定总体提升。

形式成分消融进一步表明，负效应并非来自所有形式信息。Qwen3-4B 的 `nl_var_graph` accuracy 为 0.5953，与 `nl` 持平；Qwen3-8B 的 `nl_var_graph` 也与 `nl` 持平，均为 0.5609。相反，加入 formal query 的 `nl_var_query` 使 Qwen3-4B 和 Qwen3-8B 分别下降 0.0266 和 0.0234。这说明 causal graph 成分本身没有表现出明显伤害，而 formal query 或完整形式包装更可能引入额外解析负担。

细粒度 query type 分析显示，模型错误具有明显结构性。ATE 是两个较大模型表现最好的类型，Qwen3-4B 和 Qwen3-8B 在 `nl` 条件下分别达到 0.7400 和 0.7600。Backadj、ett 以及部分 mediation/counterfactual 类型明显更弱。rung 维度上，Qwen3-4B 在 rung 2 表现较好，而 rung 3 明显下降；Qwen3-8B 也显示出高阶因果问题上的不稳定性。

CCC 分解揭示了总体一致性指标与正确性的张力。Qwen3-8B 在 `nl_formal` 条件下的 CCC 为 0.4721，高于 `nl` 条件的 0.4441，但 Correct Flip 只有 0.3128，Wrong Flip 达到 0.1592。更极端的是，Qwen3-8B 在 `det-counterfactual` 的 `nl_formal` 条件下 CCC 为 1.0000，但 Correct Flip 为 0，Wrong Flip 为 1.0000。这说明模型可以对严格对照变化产生预测翻转，但这种翻转并不一定是正确的因果判断。

Rescue/Harm 分析显示，形式脚手架在样本层面同时修复和破坏预测。Qwen3-4B 的 `nl_formal` 条件救回 26 个 `nl` 错误样本，但破坏 43 个 `nl` 正确样本；Qwen3-8B 救回 23 个样本，同时破坏 33 个样本。转移轨迹也显示，Qwen3-4B 的 C-C-C 比例为 47.5%，W-W-W 为 32.8%；Qwen3-8B 的 C-C-C 为 43.4%，W-W-W 为 33.1%。这表明相当一部分样本在输入条件变化下保持稳定，但不稳定迁移轨迹仍然构成形式提示效果不可靠的重要来源。

Stress split 结果显示，形式脚手架的平均收益很小，并伴随更大的跨 split 波动。Qwen3-4B 的 `nl_formal` 平均 stress accuracy 从 0.488 提高到 0.508，但 worst split 仅为 0.48，标准差从 0.0133 增至 0.0248。Qwen3-8B 的 `nl_formal` 平均值为 0.496，接近 `nl` 的 0.494，但 worst split 从 0.46 降至 0.45，标准差从 0.0206 增至 0.0463。因此，形式结构提示没有带来稳定鲁棒性提升。

Patching 实验产生 576 条 matched patching 结果和 576 条 random control 结果。Matched patching 的 mean absolute recovery 为 -0.0225，最大恢复量为 3.7188，mean normalized recovery 为 0.5157。Random control 的 mean absolute recovery 为 -0.0610。Matched-minus-random 的平均差为 0.0385，正差比例为 0.5069。该结果说明匹配形式输入 residual 在部分样本和层中包含可转移的答案方向信号，但这种信号较弱，且没有稳定转化为总体行为收益。

## 结论

实验结果表明，Qwen3 系列模型在 CLadder 因果推理任务上的困难不是单一的总体准确率不足，而表现为多层面的结构性不稳定。首先，错误在 query type 和 rung 上分布不均，ATE 相对容易，而 backadj、ett 和部分高阶 mediation/counterfactual 任务更容易失败。其次，严格对照一致性必须与正确性分开解释；较高的 CCC 可能包含大量错误翻转，不能直接作为正确因果推理能力的证据。第三，形式脚手架并不是稳定增益机制，它既能救回部分自然语言错误样本，也会破坏原本正确的样本。

形式成分消融进一步说明，不同形式信息的影响不同。Causal graph 成分在 Qwen3-4B 和 Qwen3-8B 上基本不降低总体 accuracy，而 formal query 或完整形式包装更容易造成性能下降。这一结果提示，模型并非完全无法利用结构信息，但当前形式提示可能引入了额外的符号解析负担，使其难以稳定转化为正确答案。

隐藏层 patching 提供了谨慎的机制证据。Matched patching 相比 random control 有小幅平均优势，并在个别层和样本上出现明显恢复，但总体均值接近零。因此，本文不将其解释为完整因果回路，而将其视为形式输入在局部表示中包含可转移答案方向信号的探索性证据。

本文的局限性包括：实验只覆盖 CLadder 一个数据集，不能代表所有真实因果问答场景；模型均来自 Qwen3 同一家族，不能解释跨架构差异；形式成分消融只覆盖变量映射、因果图和 formal query 的有限组合；patching 只在 Qwen3-4B 的 16 个样本上进行，适合作为机制探索而非强机制证明。后续工作可以扩展到更多因果基准、更多形式表示方式和更系统的表示层干预分析。
