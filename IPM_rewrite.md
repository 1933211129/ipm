# IP&M-Oriented Rewrite of the Study Idea

## 1. Recommended Title

**From Accuracy to Causal Consistency: An SCM-Based Framework for Evaluating Large Language Models on Causal Information Understanding**

### Alternative Titles

1. **Evaluating Causal Consistency in Large Language Models via SCM-Based Counterfactual Interventions**
2. **Beyond Static Accuracy: Measuring the Causal Robustness of Large Language Models in Information Understanding**
3. **A Counterfactual Evaluation Framework for Large Language Models on Causal Information Tasks**

## 2. Recommended Positioning for IP&M

这篇稿子更适合被定位为一篇 `methods + evaluation` 类型论文，而不是纯粹的机制可解释性论文。主线应当是：

- 提出一个基于结构因果模型（SCM）的动态评测框架，用于检验大语言模型在因果信息理解任务中的“因果一致性”。
- 通过系统性的事实/干预/反事实样本生成，揭示高准确率是否掩盖了浅层模式匹配。
- 在开放权重模型上加入小规模的 `activation patching` 分析，作为补充证据来解释模型在因果扰动下为何失败，而不是把白盒分析当成整篇文章的唯一核心。

换句话说，论文的中心问题不再是“模型内部病灶到底在哪”，而是“当信息条件发生受控变化时，模型是否还能保持符合因果结构的稳定理解与回答”。

## 3. Draft Abstract

**Abstract**

Large language models (LLMs) are increasingly used to interpret, summarize, and generate causal information in scientific, health, and public-facing communication. However, current evaluation practices rely heavily on static benchmark accuracy and provide limited evidence about whether model outputs remain causally coherent under controlled informational changes. This study proposes an SCM-based evaluation framework for causal information understanding in LLMs. Starting from predefined structural causal models, we generate matched sets of observational, interventional, and counterfactual prompts that preserve surface plausibility while varying the underlying causal structure. Based on these prompt sets, we evaluate whether model responses follow oracle-consistent changes rather than lexical or statistical shortcuts.

To move beyond conventional accuracy, we introduce a set of causal consistency metrics, including intervention consistency, counterfactual flip appropriateness, and explanation-outcome alignment. The framework is designed for information-centric scenarios such as scientific claims, public health communication, and event-based explanations, allowing us to examine both formal causal reasoning and domain-sensitive information understanding. We further compare model families, parameter scales, and prompting strategies to assess whether stronger benchmark performance corresponds to more stable causal behavior.

For a subset of open-weight models, we complement the behavioral evaluation with targeted activation patching to identify which internal components are most associated with successful or failed responses under causal perturbations. Rather than treating mechanistic analysis as the primary contribution, we use it to support a broader methodological claim: reliable evaluation of LLMs in information processing settings should test not only whether answers are correct, but whether they remain causally consistent when the informational world changes. The study contributes a reproducible evaluation framework, a new perspective on robustness in causal information tasks, and empirical evidence for distinguishing genuine causal sensitivity from shortcut-based performance.

**Keywords:** large language models; causal consistency; structural causal model; counterfactual intervention; information understanding; activation patching

## 4. Research Questions

### RQ1. 当信息场景中的关键因果变量发生受控干预时，LLM 的回答能否随之产生与 SCM 一致的变化？

这是整篇论文的主问题。重点不是模型“原来答对没有”，而是当我们对同一个因果结构做 `do()` 式干预后，模型输出是否会按照真实因果图发生应有的改变。

### RQ2. 传统准确率与“因果一致性”之间的关系是什么？

这个问题用来检验一个关键假设：高准确率不一定意味着高因果理解。模型可能在静态题上得分很高，但一旦输入条件改变，就暴露出依赖表面线索或语料先验的缺陷。

### RQ3. 模型家族、参数规模和提示策略是否会显著影响因果一致性？

这一问题用于支撑横向与纵向比较。横向比较不同模型家族，纵向比较相同家族内不同参数规模，同时比较 `zero-shot`、`chain-of-thought`、结构化因果提示等推理方式。

### RQ4. 在开放权重模型中，哪些内部组件与因果一致性失败最相关？

这是补充性研究问题。它不承担整篇论文的主贡献，但可以帮助解释：当模型在反事实输入下答错时，问题更可能出现在残差流、注意力输出还是 `MLP` 输出的哪一类组件上。

### RQ5. LLM 的因果一致性是否具有领域稳定性？

如果同一个模型在科学文本场景表现较好、但在公共健康或事件解释场景明显下降，那么说明其所谓“因果能力”可能高度依赖领域语料分布，而不是稳定的结构理解。

## 5. Experimental Design

### 5.1 Overall Design

建议采用“三层结构”的实验框架：

1. `SCM-native dataset construction`
2. `behavioral evaluation of causal consistency`
3. `targeted mechanistic analysis on a small open-model subset`

这样可以保证论文主线清楚，且工作量可控。

### 5.2 Data Construction Strategy

不建议从“任意现成题目”里反向抽取因果图，因为这一步本身就会变成另一个高风险研究问题。更稳妥的做法是：

- 以已有形式化因果基准为骨架，例如 `CLadder` 一类带有明确因果图和 oracle 答案的数据。
- 基于预定义的 `SCM` 模板，重新 verbalize 成更贴近 `IP&M` 的信息场景。
- 每个基础 `SCM` 实例生成一组配对样本，而不是只生成一道静态题。

建议的三类场景：

1. `scientific claims`
   例如研究摘要中的因果陈述、变量作用关系、干预结果解释。
2. `public health communication`
   例如健康建议、风险因素、干预措施与结果之间的关系。
3. `event-based information`
   例如新闻或叙事中的事件因果解释与反事实重写。

每个基础实例至少生成四种版本：

1. `observational version`
2. `interventional version`
3. `counterfactual version`
4. `lexical distractor version`

其中前三种对应真实因果变化，第四种只改变表面措辞、不改变底层因果结构，用来检测模型是否依赖词面线索。

### 5.3 Recommended Dataset Scale

为控制首篇论文的可执行性，建议先做一个中等规模版本：

- `300-500` 个基础 `SCM` 实例
- `3` 个信息场景
- 每个实例 `4-5` 个变体
- 总样本量控制在 `3,600-7,500` 条之间

另外建议进行两步质量控制：

1. 模板级自动校验：检查干预后答案是否与 oracle 一致。
2. 人工抽样校验：对 `10%-15%` 的样本进行语义自然性和因果正确性审核。

### 5.4 Models and Comparison Settings

建议把模型分为两组：

**主实验组：开放权重模型**

- 选择 `3` 个主流模型家族
- 每个家族选择 `2` 个相邻规模版本
- 保持参数量大致可比，避免完全不对称比较

例如可按“同家族不同规模 + 不同家族相近规模”的方式组织，但正文里不必过度强调某一具体厂商版本号，避免论文被版本迭代绑死。

**补充实验组：闭源模型**

- 只做行为层对比
- 不作为论文核心证据来源

这样既能体现前沿模型对比，又能保证论文的可复现性仍然建立在开放模型上。

### 5.5 Prompting Conditions

建议至少比较三种提示设置：

1. `zero-shot direct answer`
2. `chain-of-thought prompting`
3. `structured causal prompting`

其中第三种可以要求模型显式识别：

- 因果变量
- 干预对象
- 结果变化方向

这样可以检测“结构化提示是否真的提升因果一致性，而不只是提高措辞完整度”。

### 5.6 Evaluation Metrics

不建议把论文主指标写成 `ATE`，除非后续你能给出非常严格的 outcome 定义。更稳妥的指标体系如下：

1. `Accuracy`
   静态题目的基础正确率，用作参照。

2. `Intervention Consistency Score (ICS)`
   在干预版本中，模型回答是否按照 oracle 发生正确变化。

3. `Counterfactual Flip Appropriateness (CFA)`
   当 SCM 预期答案应改变时，模型是否发生“应当发生的翻转”；当 SCM 预期答案不应改变时，模型是否保持稳定。

4. `Explanation-Outcome Alignment (EOA)`
   模型给出的理由是否真正引用了被干预变量，并且解释方向与最终答案一致。

5. `Lexical Robustness Gap (LRG)`
   在词面变化但因果结构不变时，模型性能下降多少，用来估计其对表面线索的依赖程度。

对于开放权重模型，还可以增加：

6. `Logit Margin Shift (LMS)`
   在事实版本与干预版本之间，正确候选答案的 logit margin 如何变化。

### 5.7 Mechanistic Analysis as a Supporting Module

`activation patching` 不建议在所有模型、所有样本上穷举执行。更合理的设计是：

- 只选择 `1-2` 个开放权重模型
- 只分析一小批具有代表性的“事实正确但干预失败”的样本对
- 重点 patch 三类组件：
  - residual stream
  - attention output
  - MLP output

具体分析流程：

1. 对同一实例的 `clean` 与 `corrupted` prompt 分别前向传播。
2. 在 `corrupted` 运行中局部替换某层某位置的激活。
3. 观察 patch 后目标答案概率是否恢复。
4. 绘制 layer-wise heatmap，比较不同组件对恢复效果的贡献。

这一模块的目标不是声称“发现了模型因果推理的唯一病灶”，而是提供与行为指标相呼应的证据：当模型因果一致性失效时，哪类内部组件更可能与错误恢复相关。

### 5.8 Statistical Analysis

建议使用以下统计策略：

- 所有核心指标报告 `95% bootstrap confidence intervals`
- 比较不同模型/提示策略时使用配对检验
- 若样本跨多个领域，可使用 mixed-effects model，将“领域”和“模型家族”作为固定效应或分层因素

这样会让论文更像一篇成熟的评测研究，而不只是结果展示。

### 5.9 Expected Tables and Figures

建议至少准备以下图表：

1. 框架总览图：`SCM -> prompt generation -> evaluation -> patching analysis`
2. 主结果表：不同模型在 `Accuracy / ICS / CFA / EOA / LRG` 上的对比
3. 散点图：`Accuracy` 与 `ICS` 的关系，展示“高准确率不等于高因果一致性”
4. 分领域对比图：科学、健康、事件三类场景的表现差异
5. layer-wise heatmap：补充展示 patching 结果

## 6. Expected Contributions

建议把论文贡献收敛为以下三点：

1. 提出一个基于 `SCM` 的动态评测框架，用于系统检验 `LLM` 在因果信息理解中的一致性，而不再停留于静态准确率。
2. 引入一组适用于反事实评测的行为指标，用于区分“真正的因果敏感性”与“依赖表面线索的高分表现”。
3. 通过小规模白盒分析为行为结果提供解释性证据，说明模型在因果扰动下的失败可能对应于哪些内部组件。

## 7. Boundary Conditions and Writing Cautions

这部分在真正写稿时很重要，可以显著减少审稿人攻击点：

- 不要声称“从任意自然语言题目中自动抽取 SCM”。
- 不要把 `activation patching` 说成对“推理过程”的直接证明，应表述为对内部信息依赖模式的干预性分析。
- 不要把逻辑传递性直接等同于全部因果推理能力。
- 不要把闭源模型表现写成主要证据来源。
- 不要使用“首次揭示”“全面解释”这类高风险表述。

## 8. One-Sentence Paper Pitch

This paper proposes an SCM-based counterfactual evaluation framework to test whether LLMs remain causally consistent, rather than merely accurate, when information conditions are systematically changed.

