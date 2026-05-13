# MLISE 2026 论文实验与结论草稿

## 4 实验

### 4.1 实验设置

本文使用 CLadder `full_v1.5_default` 作为主评测数据集。CLadder 将自然语言因果问题与结构化因果形式相连接，每个样本包含自然语言题干、oracle yes/no 标签、rung、query type、story id、因果图和 formal query。主实验从 `full_v1.5_default` 中抽取 640 个样本，并在每个 query type 内保持 yes/no 标签平衡。具体而言，`marginal`、`correlation`、`ate` 和 `backadj` 各抽取 100 个样本，`det-counterfactual`、`ett`、`nie` 和 `nde` 各抽取 60 个样本。

为了检验模型在不同因果问题分布上的稳定性，实验还使用 CLadder 提供的五个 stress split：`commonsense`、`anticommonsense`、`noncommonsense`、`easy` 和 `hard`。每个 split 抽取 100 个样本，并保持标签平衡。stress split 仅用于补充分析，不作为主结果的唯一依据。

实验评估同一模型家族中的三个开源模型：Qwen3-0.6B、Qwen3-4B 和 Qwen3-8B。所有模型均从本地权重加载，使用贪心解码，最大生成长度设为 48 tokens。模型输出只采用自动规则解析：首先提取 `Final answer: yes/no` 形式的答案；若该形式不存在，则提取输出文本中的独立 yes/no；无法解析的输出记为 invalid。主实验不使用外部模型或人工修补答案。

每个主实验样本使用三种输入条件：

1. `nl`：只提供原始自然语言题干。
2. `nl_formal`：提供原始自然语言题干，并附加变量映射、因果图和 formal query。
3. `formula_only`：提供题干中的事实条件、变量映射、因果图、query type、rung、formal query 和问题句。

`nl_formal` 和 `formula_only` 均不提供 oracle 标签，也不提供 CLadder 中的完整推导步骤。`formula_only` 保留事实条件和概率信息，因为这些信息是因果判断所必需的；该条件的目的不是移除任务信息，而是降低原始叙事包装对模型判断的影响。

实验使用以下指标。`Accuracy` 衡量单题预测是否与 oracle 标签一致。`Parse Rate` 衡量输出能否被规则解析为 yes/no。`Strict Contrast Causal Consistency`（CCC）只统计严格对照样本对：两个样本具有相同的 `story_id`、`query_type` 和 `formal_form`，但 oracle 标签相反。若模型在该样本对上的预测也相反，则该 pair 记为一致。对于样本对 \((i,j)\)，当 \(y_i \ne y_j\) 时，CCC 定义为：

```text
CCC(i, j) = 1, if pred_i != pred_j
CCC(i, j) = 0, otherwise
```

最终 CCC 为所有有效严格对照样本对的平均值。该指标用于衡量模型是否能随最小因果对照发生预测翻转。除此之外，实验还计算 `Scaffold Gain`，即 `nl_formal` 或 `formula_only` 相对 `nl` 的 accuracy 差值；并计算 `Rescue/Harm`，其中 `rescue` 表示 `nl` 条件错误但 `nl_formal` 条件正确，`harm` 表示 `nl` 条件正确但 `nl_formal` 条件错误。

白盒分析在 Qwen3-4B 上进行。实验筛选自然语言条件回答错误、`nl_formal` 条件回答正确的样本，共得到 16 个样本。对于每个样本，先分别运行 `nl` 和 `nl_formal` 输入，并记录模型各层 residual stream 的最后 token 表示。随后在 `nl` 输入的前向传播中，将某一层最后 token 的 residual 输出替换为 `nl_formal` 条件下的对应表示，并计算 gold-label logit margin 的变化。设自然语言条件、形式脚手架条件和 patch 后的 gold-label margin 分别为 \(m_{nl}\)、\(m_{formal}\) 和 \(m_{patch}\)，则 absolute recovery 定义为：

```text
absolute recovery = m_patch - m_nl
```

normalized recovery 定义为：

```text
normalized recovery = (m_patch - m_nl) / (m_formal - m_nl)
```

该分析用于检查形式化输入条件下的隐藏状态是否包含可转移到自然语言条件的答案方向信号。

### 4.2 主实验结果

表 1 给出了主实验中的 accuracy、parse rate 和 CCC。所有模型在三种输入条件下的 parse rate 均为 1.0，说明模型输出格式稳定，主结果不受解析失败影响。

**表 1 主实验结果**

| Model | Input | N | Accuracy | Parse Rate | CCC |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | `nl` | 640 | 0.5000 | 1.0000 | 0.0000 |
| Qwen3-0.6B | `nl_formal` | 640 | 0.5000 | 1.0000 | 0.0000 |
| Qwen3-0.6B | `formula_only` | 640 | 0.5063 | 1.0000 | 0.0112 |
| Qwen3-4B | `nl` | 640 | 0.5953 | 1.0000 | 0.4078 |
| Qwen3-4B | `nl_formal` | 640 | 0.5688 | 1.0000 | 0.4078 |
| Qwen3-4B | `formula_only` | 640 | 0.5641 | 1.0000 | 0.3771 |
| Qwen3-8B | `nl` | 640 | 0.5594 | 1.0000 | 0.4553 |
| Qwen3-8B | `nl_formal` | 640 | 0.5406 | 1.0000 | 0.4581 |
| Qwen3-8B | `formula_only` | 640 | 0.5578 | 1.0000 | 0.4246 |

Qwen3-0.6B 在主实验中基本停留在随机水平附近。其 `nl` 和 `nl_formal` 条件 accuracy 均为 0.5000，CCC 为 0，说明该模型虽然能够稳定输出可解析答案，但没有形成可观察的严格对照翻转能力。`formula_only` 条件下 accuracy 仅为 0.5063，CCC 为 0.0112，提升幅度可以忽略。

Qwen3-4B 在 `nl` 条件下达到最高 accuracy，为 0.5953；`nl_formal` 和 `formula_only` 条件的 accuracy 分别下降到 0.5688 和 0.5641。对应的 Scaffold Gain 分别为 -0.0266 和 -0.0313。该结果表明，显式变量映射、因果图和 formal query 并未提高 Qwen3-4B 的单题准确率。Rescue/Harm 分析也显示，`nl_formal` 条件下 rescue 样本数为 26，harm 样本数为 43，net rescue rate 为 -0.0266。

Qwen3-8B 的 `nl` accuracy 为 0.5594，低于 Qwen3-4B；`nl_formal` 条件下降到 0.5406，`formula_only` 条件为 0.5578。其 `nl_formal` 和 `formula_only` 的 Scaffold Gain 分别为 -0.0188 和 -0.0016。Rescue/Harm 分析中，Qwen3-8B 的 rescue 样本数为 23，harm 样本数为 35，net rescue rate 为 -0.0188。由此可见，形式脚手架没有稳定改善 4B 或 8B 模型的单题预测。

虽然形式脚手架没有提升 accuracy，CCC 结果呈现出不同的结构。Qwen3-4B 在 `nl` 和 `nl_formal` 条件下的 CCC 均为 0.4078；Qwen3-8B 在 `nl` 条件下达到 0.4553，在 `nl_formal` 条件下达到 0.4581。相比之下，Qwen3-0.6B 的 CCC 接近 0。这说明较大模型在严格对照样本对上具有一定预测翻转能力，但这种能力并不必然转化为更高的单题 accuracy。换言之，模型可能在部分局部对照关系中捕捉到答案方向变化，却仍无法在完整样本分布上形成稳定、准确的因果判断。

### 4.3 Stress Split 结果

表 2 给出了五个 stress split 上的平均 accuracy。Qwen3-0.6B 在所有 stress split 上保持 0.5000，表现仍接近随机水平。Qwen3-4B 和 Qwen3-8B 的整体 accuracy 也接近 0.5，但 `nl_formal` 条件略高于 `nl` 条件：Qwen3-4B 从 0.4880 提升到 0.5080，Qwen3-8B 从 0.4920 提升到 0.5040。

**表 2 Stress split 平均结果**

| Model | Input | Mean Accuracy |
|---|---|---:|
| Qwen3-0.6B | `nl` | 0.5000 |
| Qwen3-0.6B | `nl_formal` | 0.5000 |
| Qwen3-4B | `nl` | 0.4880 |
| Qwen3-4B | `nl_formal` | 0.5080 |
| Qwen3-8B | `nl` | 0.4920 |
| Qwen3-8B | `nl_formal` | 0.5040 |

stress split 的结果说明，形式脚手架在分布压力条件下可能带来轻微平均改善，但改善幅度较小，不足以说明模型已形成稳定的因果推理能力。以 Qwen3-8B 为例，`commonsense` split 中 `nl_formal` accuracy 达到 0.5900，高于 `nl` 条件的 0.5200；但在 `anticommonsense`、`hard` 和 `noncommonsense` split 中，`nl_formal` 条件并未稳定优于 `nl` 条件。这表明模型表现对具体 split 较为敏感，形式脚手架的作用不是一致性的。

### 4.4 Formal-to-Natural Patching 结果

表 3 给出了 Qwen3-4B 上的 formal-to-natural residual patching 汇总结果。实验共分析 16 个自然语言条件失败、形式脚手架条件成功的样本，对每个样本扫描 36 层 residual stream，共得到 576 行逐层 patching 结果。

**表 3 Formal-to-natural patching 汇总结果**

| Model | Method | Rows | Samples | Mean Absolute Recovery | Max Absolute Recovery | Mean Normalized Recovery |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | `hf_last_token_formal_to_natural` | 576 | 16 | -0.0225 | 3.7188 | 0.5157 |

从全层平均 absolute recovery 看，patching 后的平均变化为 -0.0225，说明形式脚手架条件下的 residual 表示并不会在所有层上稳定提高自然语言条件下的 gold-label margin。然而，最大 absolute recovery 达到 3.7188，且 mean normalized recovery 为 0.5157，说明在部分样本和部分层中，形式脚手架条件的隐藏状态可以显著恢复自然语言失败样本的答案方向。

这一结果支持一种谨慎解释：形式化输入并没有在行为层稳定提高 accuracy，但其内部表示中仍可能包含与正确答案方向相关的局部信号。由于 patching 效果具有明显层间差异和样本差异，该结果不能被解释为完整的因果推理回路，只能作为隐藏状态中存在可转移答案信号的探索性证据。

### 4.5 结果讨论

综合主实验、stress split 和 patching 结果，可以得到三个观察。第一，Qwen3-0.6B 在 CLadder 的主实验和 stress split 中均接近随机水平，并且缺乏严格对照一致性。第二，Qwen3-4B 和 Qwen3-8B 在部分条件下表现出高于 0.6B 的 accuracy 和明显更高的 CCC，但两者的 accuracy 并不随参数规模单调提升。第三，显式形式脚手架没有稳定改善主实验 accuracy；在 4B 和 8B 上，它甚至带来小幅下降，但 stress split 中存在轻微平均改善。

这些结果表明，CLadder 上的模型表现不能仅由单题 accuracy 充分刻画。Qwen3-4B 和 Qwen3-8B 在严格对照 pair 上表现出一定预测翻转能力，但这种能力并不等同于全局稳定的因果判断。形式化结构信息也不能简单视为可靠增强信号：它可能在部分 split 或内部状态中提供帮助，但在主实验中没有形成一致的行为收益。

## 5 结论

本文在 CLadder 因果推理任务上评估了 Qwen3-0.6B、Qwen3-4B 和 Qwen3-8B，并从单题 accuracy、严格对照一致性、形式脚手架收益、stress split 稳定性和 residual stream patching 五个方面分析模型表现。实验结果显示，Qwen3-0.6B 基本停留在随机水平；Qwen3-4B 和 Qwen3-8B 具有更高的严格对照一致性，但单题 accuracy 仍然有限，且不呈现稳定单调变化。

实验还表明，显式提供变量映射、因果图和 formal query 并不能稳定提升主实验 accuracy。对于 Qwen3-4B 和 Qwen3-8B，`nl_formal` 条件下的 accuracy 反而低于原始自然语言条件。Rescue/Harm 分析进一步说明，形式脚手架带来的修复样本数量少于被破坏样本数量。因此，模型错误不能简单归因于缺少显式结构提示；即使给出形式化结构信息，模型仍可能无法稳定完成因果判断。

严格对照一致性提供了 accuracy 之外的证据。Qwen3-4B 和 Qwen3-8B 的 CCC 明显高于 Qwen3-0.6B，说明较大模型在部分最小对照样本中能够随 oracle 标签变化而翻转预测。然而，这种局部一致性没有充分转化为整体准确率，表明模型可能捕捉到部分答案方向变化，但尚未形成稳健的全局因果推理能力。

白盒 patching 结果显示，形式脚手架条件下的 residual stream 表示在部分层和部分样本中能够恢复自然语言失败样本的 gold-label margin。该现象说明模型隐藏状态中可能存在可转移的正确答案方向信号。但由于全层平均 recovery 接近 0，且 patching 效果具有较强样本差异，该结果应被解释为探索性证据，而不是完整机制解释。

总体而言，本文实验表明，开源语言模型在 CLadder 因果推理任务上的表现具有明显不稳定性。较大模型可以展现一定严格对照一致性，但单题准确率、形式脚手架收益和跨 split 稳定性仍然有限。未来工作可以扩展到其他模型家族、构造更细粒度的因果对照样本，并结合更系统的隐藏层干预方法，以区分自然语言理解、形式化因果运算和答案生成阶段各自造成的错误。
