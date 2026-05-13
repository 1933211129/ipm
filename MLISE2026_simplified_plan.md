# MLISE 2026 简化投稿方案：基于 Qwen 家族的因果一致性与隐藏层机制验证

## 1. 文档目的

这份文档是在当前 `IP&M` 长线项目基础上，专门为一个更容易实现、篇幅更短、适合先投普通 EI 会议的版本而写。

目标会议暂定为 `MLISE 2026`，即 `2026 6th International Conference on Machine Learning and Intelligent Systems Engineering`。官网信息显示，会议地点为 Naples, Italy，时间为 `May 28-31, 2026`，投稿主题覆盖 machine learning、natural language processing、intelligent information systems、data mining、neural networks and applications 等方向。官网同时说明，论文页数要求为不少于 `4` 页且不超过 `12` 页，接收论文计划进入 IEEE Conference Proceedings，并提交 IEEE Xplore、EI Compendex 和 Scopus 检索。

因此，对我们来说比较合理的策略是：

- 不把这篇会议稿写成 `IP&M` 级别的完整评测框架。
- 不做多模型家族大规模横评。
- 不做复杂场景改写 benchmark。
- 聚焦一个清晰问题：**同一 Qwen 模型家族在因果信息理解任务上是否存在规模效应，以及隐藏层干预能否为失败样本提供初步机制解释。**

这篇会议稿的定位可以理解为：

> 一个面向 EI 会议的轻量实证研究：用 `CLadder` 测试 `Qwen3-0.6B / 4B / 8B` 的因果一致性，并用小规模 activation patching 熟悉和展示隐藏层机制分析。

---

## 2. 从原项目收缩到 EI 会议稿

### 2.1 原项目核心

当前完整项目的核心是：

> 用结构因果模型（SCM）组织评测样本，检验大语言模型在信息理解任务中的因果一致性，而不是只看单题准确率。

原项目包含三层：

1. `SCM-native benchmark / prompt source`
2. `causal consistency evaluation`
3. `targeted activation patching`

其中，`CLadder` 被用作当前阶段的 SCM 代理层，因为它已经提供了结构化因果问题、rung 分类、query type 和 oracle answer。

### 2.2 会议稿要主动砍掉的内容

为了让 6-9 页论文更稳，建议暂时砍掉以下内容：

- 不做 `scientific claims / public health / event explanation` 三类场景改写作为主实验。
- 不引入多个模型家族，比如 Llama、Mistral、Gemma 等。
- 不做复杂的 `EOA` 解释质量人工评估。
- 不声称提出一个全新的 benchmark。
- 不把 activation patching 写成强机制证明。

会议稿只保留最容易完成、最容易解释、最容易出图的部分：

- 一个模型家族：`Qwen3`
- 三个模型规模：`0.6B / 4B / 8B`
- 一个主数据集：`CLadder`
- 两个行为指标：`Accuracy` 和 `Pairwise Causal Consistency`
- 一个故事级指标：`Story All-Correct Rate`
- 一个轻量白盒模块：`residual stream activation patching`

### 2.3 简化后的论文主线

简化后的主线是：

> 如果模型真的学到了相对稳定的因果信息理解能力，那么同一因果故事下，不同 rung 或不同 query type 的回答应当随 oracle label 保持或翻转；如果只是表面模式匹配，那么即使单题 accuracy 接近可接受，跨题一致性也会很差。进一步地，我们用隐藏层 activation patching 检查部分失败样本中是否存在可恢复的内部因果信号。

这条主线足够清楚，也更适合 EI 会议。

---

## 3. 推荐论文题目

### 3.1 英文题目推荐

**Do Larger Language Models Become More Causally Consistent? A Lightweight Hidden-State Analysis of Qwen Models on CLadder**

### 3.2 备选题目

1. **Evaluating Causal Consistency in Qwen Models with Lightweight Activation Patching**
2. **From Accuracy to Causal Consistency: A Pilot Study of Qwen Models on Causal Reasoning Tasks**
3. **A Simple Behavioral and Hidden-State Analysis of Causal Reasoning in Qwen Language Models**

### 3.3 中文理解

这篇文章不是说“我们发明了新的因果推理算法”，而是说：

> 我们提出一个轻量实验流程，用来观察 Qwen 家族模型在因果问题上的规模效应，并用隐藏层干预解释一部分错误。

---

## 4. 适合 MLISE 2026 的论文定位

MLISE 的主题覆盖 machine learning、natural language processing、intelligent systems、neural networks and applications。我们的简化方案可以放在以下交叉点：

- `Natural Language Processing`
- `Machine Learning Model Evaluation`
- `Intelligent Information Systems`
- `Neural Networks and Applications`
- `Experimental Evaluations of Machine Learning`

投稿时不要把文章包装成图书情报学论文，而应包装成：

> A lightweight evaluation and analysis framework for causal reasoning behavior in open-source LLMs.

这样更贴合 `Machine Learning and Intelligent Systems Engineering`。

---

## 5. 简化版核心研究问题

建议会议稿只写三个研究问题。

### RQ1. Qwen 模型在 CLadder 因果推理任务上的准确率是否随模型规模提升？

对应实验：

- 比较 `Qwen3-0.6B`
- 比较 `Qwen3-4B`
- 比较 `Qwen3-8B`

核心图表：

- 不同模型规模的 overall accuracy
- 不同 rung 下的 accuracy
- 不同 query type 下的 accuracy heatmap

### RQ2. 模型规模提升是否带来更好的因果一致性，而不仅仅是更高 accuracy？

对应实验：

- 计算同一 `story_id` 下题对的 `Pairwise Causal Consistency (PCC)`
- 计算 `Story All-Correct Rate`

核心图表：

- `PCC` 随模型规模变化的柱状图
- `Accuracy` vs `PCC` 散点图
- `Story All-Correct Rate` 对比图

### RQ3. 对于典型失败样本，隐藏层 activation patching 是否能恢复正确答案倾向？

对应实验：

- 选取少量 clean/corrupted pair
- 对 residual stream 做 layer-wise patching
- 观察 `yes/no` logit margin 是否恢复

核心图表：

- layer-wise patching heatmap
- 1-2 个代表性 case 图

---

## 6. 方法机制：给会议稿用的简单版本

### 6.1 行为层机制

我们把每道题看成一个因果信息理解任务。

模型输入是自然语言问题，输出是 `yes/no`。普通评测只看模型答对没有，但这不够，因为模型可能靠表面词语、先验偏置或固定回答模式拿到部分分数。

因此我们增加一个跨题一致性思想：

> 同一个 `story_id` 下的问题共享相同或相关的因果结构。当 oracle label 应保持一致时，模型预测也应保持一致；当 oracle label 应翻转时，模型预测也应翻转。

这就是 `PCC` 的核心。

### 6.2 PCC 指标

对同一个 `story_id` 下的两个样本 `i` 和 `j`：

- 如果 `gold_i == gold_j`，则期望 `pred_i == pred_j`
- 如果 `gold_i != gold_j`，则期望 `pred_i != pred_j`

定义：

```text
PCC(i, j) = 1, if model prediction relation matches oracle relation
PCC(i, j) = 0, otherwise
```

最终对所有有效题对求平均。

直观解释：

- `Accuracy` 回答“模型单题答对了吗”
- `PCC` 回答“模型是否跟着因果结构稳定变化”

这就是文章最容易讲清楚的贡献点。

### 6.3 Story All-Correct Rate

对同一个 `story_id` 内被选中的多道题，只有全部答对才算这个 story 成功。

这个指标很严格，但很有解释力：

> 如果模型只是在若干题上碰巧答对，story-level 表现会很低；如果模型真的理解了同一故事背后的因果结构，这个指标应当随模型规模上升。

### 6.4 隐藏层 patching 机制

白盒部分只做最简单的 residual stream patching。

基本思想：

1. 找到一对相似问题：
   - `clean prompt`：模型答对
   - `corrupted prompt`：模型答错

2. 分别运行模型，保存每一层 residual stream activation。

3. 在运行 corrupted prompt 时，把某一层的 residual activation 替换成 clean prompt 对应层的 activation。

4. 看最终答案的 `yes/no` logit margin 是否向正确方向恢复。

如果某一层 patch 后恢复明显，说明该层可能携带与正确因果判断相关的信息。

会议稿里要谨慎表述：

> This does not prove that the model performs human-like causal reasoning. It only provides preliminary evidence that some hidden states contain recoverable signals associated with correct causal judgments.

---

## 7. 实验设计

### 7.1 模型选择

只选一个模型家族，建议：

| 模型 | 角色 | 预期作用 |
|---|---|---|
| `Qwen3-0.6B` | 小模型 baseline | 预计表现接近随机，可显示任务难度 |
| `Qwen3-4B` | 中等模型 | 观察是否开始出现准确率和 PCC 提升 |
| `Qwen3-8B` | 主模型 | 作为会议稿主结果模型，适合 A100 80G 跑完整实验 |

这样做有三个好处：

- 控制变量清楚：同一模型家族，只比较规模。
- 工程简单：tokenizer、prompt、加载方式基本一致。
- 论文叙事自然：研究问题就是 scaling effect on causal consistency。

### 7.2 数据集选择

主数据集使用：

```text
datasets/cladder/data/full_v1.5_default.csv
```

如果想更快跑通，也可以优先用：

```text
datasets/cladder/data/test-balanced-v1.5.csv
```

推荐会议稿主实验采用 `full_v1.5_default.csv` 中的分层抽样子集，而不是全量数据。

### 7.3 推荐样本规模

考虑到 EI 会议稿和 A100 资源，建议两阶段运行。

第一阶段：快速验证

| query type | 样本数 |
|---|---:|
| `marginal` | 40 |
| `correlation` | 40 |
| `ate` | 40 |
| `backadj` | 40 |
| `det-counterfactual` | 20 |
| `ett` | 20 |
| `nie` | 20 |
| `nde` | 20 |
| 合计 | 240 |

第二阶段：正式会议稿实验

| query type | 样本数 |
|---|---:|
| `marginal` | 100 |
| `correlation` | 100 |
| `ate` | 100 |
| `backadj` | 100 |
| `det-counterfactual` | 60 |
| `ett` | 60 |
| `nie` | 60 |
| `nde` | 60 |
| 合计 | 640 |

每个 query type 内尽量保持 `yes/no` 平衡。

### 7.4 Prompt 设置

为了简化会议稿，只保留两种 prompting：

1. `Direct`
2. `Structured`

不建议把 `CoT` 放进主实验，因为：

- 小模型容易输出冗长但无效的解释。
- 解析和比较更麻烦。
- 会议稿篇幅有限，解释不完三种 prompt 的差异。

`Direct` prompt：

```text
You are solving a formal causal reasoning question.
Return exactly one line in this format: Final answer: yes/no

Question:
{question}
```

`Structured` prompt：

```text
You are solving a causal reasoning question.
Identify the relevant causal relation and answer the question.
Return the final answer exactly in this format: Final answer: yes/no

Question:
{question}
```

### 7.5 输出解析

继续使用当前脚本中的正则解析逻辑：

- 优先提取 `Final answer: yes/no`
- 其次提取末尾独立 `yes/no`
- 解析失败则记为 `invalid`

EI 会议稿里建议不要依赖 DeepSeek 做答案归一化，除非只是作为失败兜底。主结果最好基于完全自动规则，审稿人更容易接受。

---

## 8. 白盒实验的最小可行方案

### 8.1 不建议做全量 patching

会议稿里不要做所有模型、所有组件、所有样本的完整 patching。

最小可行方案是：

- 行为评测：`Qwen3-0.6B / 4B / 8B`
- 白盒 patching：只选 `Qwen3-4B` 或 `Qwen3-8B`
- 组件：只做 `residual stream`
- 样本对：`8-16` 对
- 层数：全层扫描
- 指标：`yes/no logit margin recovery`

如果 `Qwen3-8B` 的 patching 内存压力较大，就优先用 `Qwen3-4B`。会议稿中可以解释为：hidden-state analysis is conducted on a representative mid-sized model due to computational cost。

### 8.2 pair 选择原则

优先选这样的样本对：

- 同一个 `story_id`
- 相同或相近的 `query_type`
- gold label 相反
- token 长度尽量接近
- 模型在一条上答对，在另一条上答错

当前脚本已经有类似逻辑，可以复用 `pick_patch_candidate_pairs`。

### 8.3 patching 指标

对 yes/no 任务，定义：

```text
logit_diff = logit(correct_answer) - logit(wrong_answer)
```

对于 corrupted prompt：

```text
recovery = patched_logit_diff - corrupted_logit_diff
```

如果 `recovery > 0`，说明 patch 让模型朝正确答案移动。

如果某些层的 recovery 明显更高，可以画成 heatmap。

### 8.4 会议稿里的解释方式

推荐写法：

> The patching results suggest that correct-answer information can be partially restored by replacing intermediate hidden states, indicating that some internal representations contain recoverable signals related to causal judgment.

不要写成：

> We discovered the causal reasoning circuit of Qwen.

后者太大，容易被审稿人质疑。

---

## 9. 预期图表

6-9 页会议稿建议控制在 `5` 张图表左右。

### Figure 1. Overall Framework

内容：

```text
CLadder SCM-style tasks
        ↓
Qwen3-0.6B / 4B / 8B
        ↓
Accuracy + PCC + Story All-Correct
        ↓
Residual Stream Patching on Representative Failures
```

这张图可以自己用 PowerPoint / draw.io / Mermaid 画。

### Figure 2. Accuracy by Model Size and Rung

横轴：

- `0.6B`
- `4B`
- `8B`

颜色：

- rung 1
- rung 2
- rung 3

用途：

- 展示模型规模提升是否带来单题准确率提升。

### Figure 3. PCC by Model Size

横轴：

- `0.6B`
- `4B`
- `8B`

纵轴：

- PCC

用途：

- 展示因果一致性是否随规模提升。

### Figure 4. Accuracy vs PCC

每个点代表一个模型或一个模型-prompt 组合。

用途：

- 说明 accuracy 和 causal consistency 不是完全等价的。

### Figure 5. Residual Stream Patching Heatmap

横轴：

- layer

纵轴：

- pair id 或 query type

颜色：

- recovery

用途：

- 展示隐藏层干预在哪些层产生较明显恢复信号。

---

## 10. 推荐表格

### Table 1. Dataset Statistics

包括：

- query type
- rung
- yes/no 数量
- 总样本数

### Table 2. Main Results

列：

- model
- prompt
- accuracy
- parse rate
- PCC
- Story All-Correct Rate

### Table 3. Patching Summary

列：

- model
- number of pairs
- positive recovery pair ratio
- mean recovery
- max recovery
- layer with highest recovery

---

## 11. 论文结构建议：6-9 页版本

### 1. Introduction

建议写 `1` 页左右。

核心内容：

- LLMs 越来越多用于复杂信息理解。
- 因果问题不仅要求答对单题，还要求在相关问题之间保持一致。
- 传统 accuracy 不足以揭示这种一致性。
- 本文用 Qwen 家族做一个轻量实证研究。

贡献写三点：

1. We evaluate causal consistency of Qwen models on CLadder beyond static accuracy.
2. We compare scaling behavior from `0.6B` to `8B` using accuracy, PCC, and story-level consistency.
3. We conduct lightweight residual-stream activation patching to inspect hidden-state signals in representative failures.

### 2. Related Work

建议写 `1` 页以内。

只写三类：

- LLM causal reasoning evaluation
- Consistency and robustness evaluation
- Mechanistic interpretability / activation patching

### 3. Method

建议写 `1.5-2` 页。

包含：

- Dataset and task formulation
- Prompting settings
- Metrics
- Patching method

重点把 `PCC` 写清楚。

### 4. Experiments

建议写 `1-1.5` 页。

包含：

- Models
- Dataset sampling
- Implementation details
- Hardware

### 5. Results and Analysis

建议写 `2-3` 页。

按 RQ 展开：

- RQ1: accuracy scaling
- RQ2: causal consistency scaling
- RQ3: hidden-state patching

### 6. Conclusion

建议半页。

强调：

- Qwen 模型规模可能改善因果一致性，但 consistency 与 accuracy 不是同一件事。
- hidden-state patching 可以作为理解失败样本的轻量工具。
- 未来工作会扩展到更多模型家族和更真实的信息场景。

---

## 12. 可以直接写进论文的摘要草稿

```text
Large language models are increasingly used to answer questions involving causal information, yet standard accuracy-based evaluation may overlook whether their predictions remain consistent across related causal conditions. This paper presents a lightweight empirical study of causal consistency in Qwen language models on the CLadder benchmark. We evaluate three model sizes from the same family, Qwen3-0.6B, Qwen3-4B, and Qwen3-8B, under direct and structured prompting settings. Beyond conventional accuracy, we compute pairwise causal consistency and story-level all-correct rate to examine whether model predictions change consistently with oracle causal labels across related questions. To provide preliminary insight into model internals, we further conduct residual-stream activation patching on representative failure pairs and measure whether replacing hidden states can recover the correct yes/no logit margin. The study offers a simple and reproducible workflow for evaluating causal reasoning behavior in open-source LLMs and demonstrates how behavioral consistency metrics can be combined with lightweight hidden-state analysis.
```

---

## 13. 当前脚本如何改成会议稿版本

当前主脚本是：

```text
scripts/qwen3_cladder_feasibility.py
```

建议不要直接大改原脚本，而是复制出一个会议稿版本：

```text
scripts/mlise2026_qwen_scaling.py
```

主要修改点：

### 13.1 支持多个模型路径

新增：

```python
MODEL_CONFIGS = [
    {"name": "qwen3_0_6b", "path": ROOT / "qwen3_0_6b"},
    {"name": "qwen3_4b", "path": ROOT / "qwen3_4b"},
    {"name": "qwen3_8b", "path": ROOT / "qwen3_8b"},
]
```

### 13.2 输出目录按会议稿区分

新增：

```python
OUT_DIR = ROOT / "outputs" / "mlise2026_qwen_scaling"
```

### 13.3 只保留两个 prompt mode

修改：

```python
PROMPT_MODES = ["direct", "structured"]
```

### 13.4 调整主样本规模

建议先跑 `240` 样本版本，确认无误后跑 `640` 样本版本。

### 13.5 patching 只对一个模型运行

新增：

```python
PATCH_MODEL_NAME = "qwen3_4b"
PATCH_COMPONENTS = ["residual"]
PATCH_PAIR_LIMIT = 16
```

### 13.6 输出会议稿报告

生成：

```text
MLISE2026_qwen_scaling_report.md
```

内容包括：

- 数据集统计
- 主结果表
- PCC 结果
- story all-correct 结果
- patching heatmap
- 初步论文结论

---

## 14. A100 上推荐运行顺序

### Step 1. 先复现本地 0.6B

目的：

- 确认环境、数据、模型路径都对。
- 确认 CUDA 下 TransformerLens 比 MPS 稳定。

命令示例：

```bash
conda activate ipm
python scripts/qwen3_cladder_feasibility.py
```

### Step 2. 跑会议稿快速版

目的：

- 三个模型都跑 `240` 样本。
- 暂时不开 patching 或只对 `0.6B` 开 patching。

预期产出：

```text
outputs/mlise2026_qwen_scaling/tables/main_eval.csv
outputs/mlise2026_qwen_scaling/tables/metrics_summary.csv
outputs/mlise2026_qwen_scaling/figures/
```

### Step 3. 跑会议稿正式版

目的：

- 三个模型都跑 `640` 样本。
- `Qwen3-4B` 或 `Qwen3-8B` 做 `8-16` 对 residual patching。

### Step 4. 开始写论文

优先把结果组织成：

- `Table 1`: dataset statistics
- `Table 2`: main results
- `Figure 2`: accuracy by rung
- `Figure 3`: PCC by model size
- `Figure 5`: patching heatmap

---

## 15. 成功标准

这篇 EI 会议稿不需要证明一个宏大的结论，只需要满足以下标准就有价值：

### 最低成功标准

- 三个 Qwen 模型全部能完成行为评测。
- 至少一个模型在 `PCC` 或 `Story All-Correct Rate` 上明显优于 `0.6B`。
- patching 至少在若干 pair 上出现正向 recovery。

### 理想成功标准

- `8B > 4B > 0.6B` 在 accuracy 上大体成立。
- `PCC` 的规模趋势比 accuracy 更有解释力。
- patching heatmap 显示中后层存在更明显恢复信号。

### 即使结果不好也能写的方向

如果 `4B` 和 `8B` 也没有明显提升，仍然可以写成：

> Larger parameter scale alone does not guarantee causal consistency. This suggests that causal information understanding requires targeted training or evaluation beyond generic scaling.

这也是一个合理的 EI 会议结论。

---

## 16. 这篇稿子的边界

写作时务必注意这些边界：

- 不声称模型具备真正的人类式因果推理。
- 不声称 patching 找到了完整 causal circuit。
- 不声称 `CLadder` 能覆盖所有真实世界因果信息任务。
- 不声称本文提出了新的大规模 benchmark。
- 不把 `0.6B` 的失败解释成 Qwen 家族整体失败。

更稳的表述是：

> This study provides a lightweight and reproducible workflow for combining causal consistency evaluation with preliminary hidden-state analysis.

---

## 17. 和 IP&M 长线项目的关系

这篇 MLISE 会议稿可以作为 IP&M 长线工作的前置版本。

会议稿解决三个小目标：

1. 熟悉多模型推理 pipeline。
2. 熟悉 hidden-state activation patching。
3. 得到一套可以展示的行为指标和图表。

后续 IP&M 版本再扩展：

- 多模型家族
- 正式信息场景改写
- 更严格的 SCM-based intervention generation
- 更系统的白盒机制分析
- 更完整的统计检验

换句话说，MLISE 稿子是“练兵 + pilot paper”，IP&M 稿子是“完整方法论文”。

---

## 18. 最推荐的下一步

建议下一步直接做三件事：

1. 新建 `scripts/mlise2026_qwen_scaling.py`，从当前脚本裁剪出多模型 scaling 版本。
2. 在 A100 上先跑 `Qwen3-0.6B / 4B / 8B` 的 `240` 样本快速版。
3. 根据结果决定 patching 放在 `4B` 还是 `8B` 上。

如果快速版中 `4B` 或 `8B` 已经明显改善 `PCC`，这篇 EI 会议稿就有一个很顺的故事：

> 模型规模会改善部分因果推理能力，但单题准确率不足以描述这种能力；PCC 和隐藏层 patching 能提供更细粒度的诊断。

如果快速版没有明显改善，也不算失败。那篇文章可以转向另一个同样清楚的结论：

> 通用 LLM 的规模扩展并不自动带来因果一致性，因果任务仍需要专门的评测和诊断工具。

这两个方向都能写，只是叙事重点不同。

---

## 19. 参考项目文件

当前目录中最重要的参考文件：

- `GPT55_HANDOFF.md`：完整项目交接说明。
- `IPM_rewrite.md`：面向 IP&M 的长线方案。
- `qwen3_cladder_feasibility_report.md`：`Qwen3-0.6B` 本地 pilot 报告。
- `scripts/qwen3_cladder_feasibility.py`：已经跑通的全流程脚本。
- `outputs/qwen3_cladder_feasibility/`：已有结果表和图。

会议稿下一步应该在这些文件基础上做减法，而不是重写一套完全不同的工程。

---

## 20. 外部信息来源

- MLISE 2026 官网：https://www.mlise.org/
- MLISE 2026 Submission 页面：https://mlise.org/Submission

