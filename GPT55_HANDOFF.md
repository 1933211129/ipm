# GPT-5.5 接手说明：IP&M 因果一致性评测项目

> 2026-05-13 更新：`MLISE 2026` 短期会议稿已经从原来的 Qwen scaling 主线切换为诊断版主线。上一轮三模型 scaling 结果区分度不足，不能强写“规模效应”。当前优先阅读 [MLISE2026_diagnostic_plan.md](/Users/xiaokong/task/2026/IPM/MLISE2026_diagnostic_plan.md)，并使用 [scripts/mlise2026_qwen_diagnostic.py](/Users/xiaokong/task/2026/IPM/scripts/mlise2026_qwen_diagnostic.py) 在服务器 `/data/kongyb/ipm` 重新运行诊断实验。实验图表使用英文，Markdown 报告和实验记录使用中文。

## 1. 这份文档的用途

这是一份专门给下一个 `GPT-5.5` 窗口使用的 handoff 文档。

目标是让新的窗口在**不依赖当前对话上下文**的前提下，直接理解：

- 这个项目当前真正的研究核心是什么
- 我们提出的方法机制是什么
- 当前已经做到了哪一步
- 哪些结果可以直接复用
- 上传到 `80G A100` 服务器后，下一步应该优先做什么

请把这份文档和当前目录一起上传到服务器，新的窗口可以直接从这里继续。

---

## 2. 当前项目的核心研究问题

### 2.1 一句话概括

我们当前工作的核心，不是证明一个小模型已经会“因果推理”，而是验证这样一套研究范式是否成立：

> **能否用一个基于结构因果模型（SCM）的评测框架，系统检验大语言模型在信息理解任务中的因果一致性，而不是只看静态准确率。**

### 2.2 论文定位

这个项目不是纯粹的 mechanistic interpretability 论文，也不是普通 benchmark 跑分论文。

它更接近一篇面向 `IP&M` 的：

- `methods + evaluation`
- `LLM causal information understanding`
- `behavioral evaluation + supporting white-box evidence`

### 2.3 当前真正要回答的问题

当前工作想回答的是以下几个层次的问题：

1. **行为层：**
   当输入信息发生受控的因果变化时，模型输出是否也发生与 oracle 一致的变化？

2. **指标层：**
   `Accuracy` 是否会掩盖模型只是依赖表面模式，而没有形成稳定因果理解？

3. **迁移层：**
   这种评测是否能从形式化 benchmark 平滑迁移到更像 `IP&M` 的信息场景，比如 scientific claims / public health / event explanation？

4. **白盒层：**
   当模型因果判断失败时，能否用 `activation patching` 在内部组件上找到一定的“恢复信号”？

---

## 3. 我们提出的方法：核心机制是什么

## 3.1 方法总览

当前方法可以概括成三层：

1. `SCM-native benchmark / prompt source`
2. `causal consistency evaluation`
3. `targeted activation patching`

也就是说，我们不是只做“模型答对几题”，而是构造一条从**因果结构 -> 行为表现 -> 内部机制**的链路。

## 3.2 第一层：SCM 驱动的样本组织

当前我们并没有自己从自然语言中自动抽取 SCM。

为了降低研究风险，我们采用的是更稳妥的路线：

- 使用已有的形式化因果基准 `CLadder`
- 直接利用其中已经隐含的 causal structure / oracle answer
- 基于 `story_id`、`rung`、`query_type`、`formal_form` 重新组织评测样本

换句话说，目前我们把 `CLadder` 当作一个**可操作的 SCM 代理层**。

这比“从任意文本逆向构造因果图”更稳，更适合先把论文主线跑通。

## 3.3 第二层：从 Accuracy 到 Causal Consistency

这是当前方法最重要的部分。

我们不是只看单题准确率，而是增加了几类更贴近论文主线的指标：

- `Accuracy`
- `Parse Rate`
- `Pairwise Causal Consistency (PCC)`
- `Story All-Correct Rate`
- `Scenario Robustness Gap`

其中最关键的是：

### `PCC`

对同一 `story_id` 下不同 `rung` 的题对进行比较：

- 如果两个题目的 oracle label 相同，模型也应保持同号输出
- 如果两个题目的 oracle label 相反，模型应发生翻转

这比单题准确率更接近“模型是否真的跟着因果结构变化”。

### `Story All-Correct Rate`

对同一个故事下的一组题，要求模型全部答对。

这个指标能更直接暴露：

- 模型是否只是偶然答对个别题
- 还是在同一因果结构上真的保持了稳定理解

### `Scenario Robustness Gap`

把一小批形式化样本改写成更像：

- scientific claims
- public health communication
- event explanation

然后比较改写前后表现差异。

这个步骤的意义在于：

- 验证这套框架是否能从 formal benchmark 迁移到更像 `IP&M` 的信息场景
- 判断模型到底是靠结构理解，还是靠 benchmark 表面形式

## 3.4 第三层：定向 activation patching

这部分不是整篇论文的唯一核心，而是补充证据层。

当前实现逻辑是：

1. 在 `CLadder` 中寻找可配对样本：
   - 相同 `(story_id, query_type, formal_form)`
   - gold label 相反
   - token 长度一致

2. 先跑行为层结果

3. 在这些 pair 里筛选：
   - `clean`：模型答对的那条
   - `corrupted`：模型答错的那条

4. 对 `corrupted` 样本做 patching：
   - `residual`
   - `attention`
   - `mlp`

5. 观察 patch 后最终 `yes/no` 的 logit difference 是否向正确方向恢复

我们当前并不把这个解释成“模型推理过程的严格证明”，而是把它当作：

> 当行为层出现失败时，内部哪个组件更可能与恢复相关的探索性证据。

---

## 4. 当前已经完成的工作

## 4.1 已完成的工程实现

目前已经实现了完整的本地 pipeline，主脚本是：

[qwen3_cladder_feasibility.py](/Users/xiaokong/task/2026/IPM/scripts/qwen3_cladder_feasibility.py)

这份脚本已经能够完成：

- 加载本地 `Qwen3-0.6B`
- 读取 `CLadder`
- 构建 smoke 子集
- 构建 main 子集
- 构建 scenario source 子集
- 跑三种 prompting
- 自动解析 yes/no
- 在解析失败时调用 `DeepSeek` 做轻量归一化
- 调用 `DeepSeek` 做场景改写
- 计算行为层指标
- 做 patch candidate 筛选
- 用 `TransformerLens` 跑 patching
- 生成结果表
- 生成图
- 生成中文 Markdown 报告

## 4.2 已完成的本地实验

本地模型：

- `Qwen3-0.6B`
- 本地路径：`/Users/xiaokong/task/2026/IPM/qwen3_0_6b`

本地设备：

- `MacBook Air M4 24G`
- `MPS`

数据集：

- `CLadder full_v1.5_default`

实验规模：

- smoke：`48 × 3`
- main：`480 × 3`
- scenario source：`12 × 3`
- scenario rewrite：`36 × 3`
- patching：`12` 对 pair，`3` 类组件，`28` 层

## 4.3 当前最重要的实证结果

主结果报告在：

[qwen3_cladder_feasibility_report.md](/Users/xiaokong/task/2026/IPM/qwen3_cladder_feasibility_report.md)

最关键结论如下：

### 结论 A：流程可行

“可行”指的是：

- 整套流程可以跑通
- 可以生成有效结果表和图
- 可以做行为层、场景层、白盒层的联动分析

### 结论 B：Qwen3-0.6B 本身不行

这个结果很重要。

当前实验并不是证明 `Qwen3-0.6B` 有强因果能力，恰恰相反：

- `main` 集合上三种 prompting 准确率都在 `0.50` 附近
- `direct`：`480/480` 输出 `yes`
- `structured`：`480/480` 输出 `yes`
- `cot`：`468 yes / 12 no`

也就是说，这个模型在平衡子集上几乎退化成常数预测器。

这恰好说明：

> 我们的评测框架不是“帮模型刷出高分”，而是真的能暴露小模型的结构性缺陷。

### 结论 C：因果一致性指标是有信息量的

虽然 `Accuracy` 接近随机，

但：

- `PCC` 不完全等于 `0.5`
- `Story All-Correct Rate` 很低，仅 `0.0426`

这说明我们的指标层并不是冗余的。

它能把“偶然答对”和“跨题保持一致”区分开。

### 结论 D：场景迁移层也能跑

我们已经把部分样本改写成：

- scientific claims
- public health
- event explanation

结果显示：

- 改写流程能跑通
- 改写后整体 `parse rate = 1.0`
- 改写前后准确率差大多接近 `0`

这说明：

> 从 formal causal benchmark 迁移到更像 `IP&M` 的信息场景，是可操作的。

### 结论 E：白盒 patching 也能跑通

当前结果表明：

- `TransformerLens` 已经成功接上 `Qwen3-0.6B`
- `patching_results.csv` 已经生成
- `12` 对 pair 都观察到了正向恢复 signal

但必须加一句限制：

> 这批 patching 是在 `MPS` 上跑的，TransformerLens 官方对 `MPS` 后端给过 warning，所以它们目前只能视为探索性证据，不能当最终定论。

因此后续一定要在 `A100 CUDA` 上重跑白盒部分。

---

## 5. 当前目录结构

当前目录的关键结构如下：

```text
IPM/
├── IP&M.md
├── IPM_rewrite.md
├── GPT55_HANDOFF.md
├── qwen3_cladder_feasibility_report.md
├── call_deepseek.py
├── requirements-qwen3-py313.txt
├── qwen3_0_6b/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── ...
├── datasets/
│   └── cladder/
│       ├── README.md
│       ├── dataset_infos.json
│       └── data/
│           ├── full_v1.5_default.csv
│           ├── full_v1.csv
│           ├── test-balanced-v1.5.csv
│           └── ...
├── scripts/
│   └── qwen3_cladder_feasibility.py
└── outputs/
    └── qwen3_cladder_feasibility/
        ├── figures/
        │   ├── 01_subset_composition.png
        │   ├── 02_accuracy_by_rung.png
        │   ├── 03_accuracy_heatmap.png
        │   ├── 04_pcc.png
        │   ├── 05_scenario_robustness.png
        │   ├── 06_latency_parse.png
        │   ├── 07_patch_heatmap.png
        │   ├── 08_patch_case_1.png
        │   └── 08_patch_case_2.png
        └── tables/
            ├── selected_smoke_subset.csv
            ├── selected_main_subset.csv
            ├── selected_scenario_sources.csv
            ├── smoke_eval.csv
            ├── main_eval.csv
            ├── scenario_rewrites.csv
            ├── scenario_source_eval.csv
            ├── scenario_rewrite_eval.csv
            ├── all_eval_rows.csv
            ├── metrics_summary.csv
            ├── metrics_by_rung.csv
            ├── metrics_by_query.csv
            ├── metrics_pcc.csv
            ├── metrics_story_all_correct.csv
            ├── metrics_scenario.csv
            └── patching_results.csv
```

---

## 6. 各关键文件的作用

## 6.1 研究设计文件

### [IP&M.md](/Users/xiaokong/task/2026/IPM/IP&M.md)

最初的原始思路文档，偏“雄心版”，白盒味道更重。

### [IPM_rewrite.md](/Users/xiaokong/task/2026/IPM/IPM_rewrite.md)

已经重写成更像 `IP&M` 稿件的版本，建议新的窗口优先参考这份。

里面定义了：

- 论文定位
- 标题建议
- 摘要草稿
- RQ
- 方法框架

## 6.2 运行脚本

### [qwen3_cladder_feasibility.py](/Users/xiaokong/task/2026/IPM/scripts/qwen3_cladder_feasibility.py)

这是当前项目最核心的工程脚本。

它是一个“全流程脚本”，包含：

- 数据抽样
- 推理
- 解析
- 改写
- 指标统计
- patching
- 画图
- 生成报告

当前脚本已经支持：

- 每个 prompt mode 结束后 partial save
- 改写失败时 fallback template
- 通过 `TransformerLens` 跑 `Qwen3-0.6B`

## 6.3 辅助大模型调用

### [call_deepseek.py](/Users/xiaokong/task/2026/IPM/call_deepseek.py)

主要用于：

- 解析失败时的 yes/no 归一化
- scenario rewrite

当前并**没有**把 `DeepSeek` 当作主实验模型。

## 6.4 依赖文件

### [requirements-qwen3-py313.txt](/Users/xiaokong/task/2026/IPM/requirements-qwen3-py313.txt)

当前本地 pipeline 的依赖清单。

已经包含：

- `torch`
- `transformers`
- `datasets`
- `openai`
- `transformer-lens`
- `tabulate`

---

## 7. 当前输出结果如何读取

## 7.1 最重要的报告

### [qwen3_cladder_feasibility_report.md](/Users/xiaokong/task/2026/IPM/qwen3_cladder_feasibility_report.md)

这是目前对外可读性最强的总结材料。

如果新的窗口只看一份文件，先看这个。

## 7.2 关键结果表

### `main_eval.csv`

主行为评测逐条记录。

每行包含：

- 样本信息
- prompt mode
- raw output
- parsed label
- is_correct
- latency

### `metrics_summary.csv`

最顶层汇总表。

### `metrics_pcc.csv`

`PCC` 指标结果。

### `metrics_story_all_correct.csv`

故事级一致性结果。

### `metrics_scenario.csv`

改写前后鲁棒性结果。

### `patching_results.csv`

白盒 patching 结果表。

关键字段：

- `pair_id`
- `component`
- `layer`
- `clean_logit_diff`
- `corrupted_logit_diff`
- `patched_logit_diff`
- `recovery`
- `method`

---

## 8. 当前最重要的解释边界

这部分非常重要，新窗口一定不要忽略。

## 8.1 “可行”不等于“模型强”

当前结论中的“可行”，是指：

- 方法可行
- 工程可行
- 评测链可行

**不是**指：

- `Qwen3-0.6B` 表现很好
- 论文已经有强主结果

当前 `0.6B` 的主作用更像：

> 一个能跑通流程、同时又能暴露失败模式的 baseline。

## 8.2 当前场景改写层有 fallback

`scenario_rewrites.csv` 里的改写不是全部都来自高质量自由改写。

为了保证流程可运行，脚本在 `DeepSeek` 不稳定时使用了模板式 fallback。

因此当前场景层的意义更接近：

- 证明 pipeline 可以迁移
- 证明格式和数值约束能保持

而不是高质量正式 benchmark。

如果后续写论文，场景层数据还需要更认真打磨。

## 8.3 当前 patching 结果不是最终版

`patching_results.csv` 是真实跑出来的。

但因为它在 `MPS` 上完成，所以：

- 可以作为“这条白盒链能跑通”的证据
- 不建议作为论文定稿主图直接使用

后续必须在 `A100 CUDA` 上重跑。

---

## 9. 上传到 A100 之后，下一步最应该做什么

## 9.1 第一优先级：在 A100 上复现实验

建议新的窗口先做一次“服务器复现”，目标不是马上扩模型，而是确认当前 pipeline 在 CUDA 环境下稳定。

建议顺序：

1. 先把当前目录原样上传
2. 建新环境
3. 安装依赖
4. 确认本地模型路径和数据路径
5. 先重跑 `Qwen3-0.6B`
6. 检查：
   - 结果是否与本地大体一致
   - patching 是否更稳定
   - 整体耗时是否显著下降

## 9.2 第二优先级：扩模型规模

一旦 `0.6B` 在 A100 上复现实验成功，下一步最该做的是：

- 上更大规模模型
- 不要继续在 `0.6B` 上花太多时间

推荐优先顺序：

1. `Qwen3-4B`
2. `Qwen3-8B`
3. 再考虑加一个横向对照模型家族

原因很简单：

- 现在 `0.6B` 的结果已经说明流程可以跑
- 真正能写进论文主结果的，应该是更大模型的规模效应

## 9.3 第三优先级：把当前 pilot 升级成论文主实验

新的窗口在 A100 上最值得做的工作，不是推翻当前工程，而是把它升级成论文主实验版本。

推荐路线：

### 路线 A：保留当前 pipeline，扩成多模型主实验

- 继续使用当前 `CLadder` 分层采样逻辑
- 固定三种 prompting
- 先不增加太多新变量
- 重点做 model scaling / family comparison

这是最稳的路线。

### 路线 B：把 scenario layer 做成更像 IP&M 的正式子集

当前场景改写只是 pilot。

后续可以：

- 人工筛选一批更自然的 scientific/public health/event 样本
- 形成更可发表的“信息场景子基准”

### 路线 C：让白盒部分更聚焦

当前 patching 已经能跑，但后续不建议扩大到“全样本全组件穷举”。

更合理的是：

- 在更大模型上挑代表性 pair
- 只做最有信息量的组件
- 把白盒分析作为 supporting evidence

---

## 10. 建议新的 GPT-5.5 直接执行的任务顺序

如果我是新的窗口，我会按以下顺序接手：

1. 阅读：
   - `GPT55_HANDOFF.md`
   - `IPM_rewrite.md`
   - `qwen3_cladder_feasibility_report.md`

2. 检查服务器环境：
   - CUDA
   - PyTorch
   - `transformer-lens`
   - 本地模型路径是否一致

3. 先复现当前 `Qwen3-0.6B` 实验

4. 重点验证：
   - `patching_results.csv` 在 CUDA 上是否更稳定
   - 总耗时降低多少

5. 然后直接扩到：
   - `Qwen3-4B`
   - `Qwen3-8B`

6. 用同一套指标比较：
   - accuracy
   - PCC
   - story all-correct
   - scenario robustness
   - patch recovery

7. 开始为论文主稿组织：
   - methods
   - experiments
   - results

---

## 11. 推荐的服务器启动方式

以下只是建议，新窗口可以按服务器实际环境调整。

```bash
cd /path/to/IPM

conda create -n ipm python=3.13 -y
conda activate ipm

pip install -r requirements-qwen3-py313.txt
```

如果模型和数据仍然保持当前目录结构，可以直接运行：

```bash
python scripts/qwen3_cladder_feasibility.py
```

如果服务器上的模型路径或数据路径发生变化，新窗口需要优先修改：

- `MODEL_PATH`
- `DATA_PATH`
- `OUT_DIR`

它们都在：

[qwen3_cladder_feasibility.py](/Users/xiaokong/task/2026/IPM/scripts/qwen3_cladder_feasibility.py)

---

## 12. 当前最值得记住的三句话

1. **这套流程已经被证明能跑通。**
2. **`Qwen3-0.6B` 的表现接近随机，这不是坏消息，而是说明流程有诊断力。**
3. **真正的论文主结果，应该在 A100 上用更大模型继续做。**

---

## 13. 交接备注

当前窗口的使命已经基本完成。

已经完成的部分：

- 研究定位重写
- 本地 pipeline 实现
- 本地 pilot 实验
- 图表和中文报告
- 可继续扩展的 handoff 文档

因此，新的 `GPT-5.5` 窗口不需要再回头做“项目理解”工作，而应该直接从：

> **A100 上复现 + 扩模型 + 升级成论文主实验**

开始推进。
