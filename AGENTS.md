# 服务器链接

先连接：ssh kong@10.0.32.32,密码Kndev@6729

在10.0.32.32内再连接有显卡的机器，ssh kyb@10.3.35.21，密码133164

8张显卡均可以使用，用不了那么多可以只挑选一张使用。

模型文件在：

（1）/data/LLM/Qwen/Qwen3-8B

（2）/data/LLM/Qwen/Qwen3-0___6B

（3） /data/LLM/Qwen/Qwen3-4B

# 工作目录

本地Mac的工作目录：/Users/xiaokong/task/2026/IPM

服务器端Ubuntu的工作目录：/data/kongyb/ipm

# git相关操作

1.强制切换为 SSH 协议 (利用已配置的 SSH Key 免密操作)

git remote set-url origin git@github.com:1933211129/ipm.git

2. 锁定项目级身份 (确保 Commit 记录始终属于你)

git config user.name "1933211129"
git config user.email "k1933211129@163.com"

3. 规范同步指令

同步云端：git pull origin main

提交更新：git add . && git commit -m "msg" && git push

# 项目目标

当前项目的短期目标是完成 `MLISE 2026` 简化版实验，实现一篇 6-9 页 EI 会议论文所需的完整流程。

核心任务：

1. 基于 `CLadder` 构建因果推理评测子集。
2. 对 `Qwen3-0.6B / Qwen3-4B / Qwen3-8B` 做同家族规模对比。
3. 计算 `Accuracy`、`Pairwise Causal Consistency (PCC)`、`Story All-Correct Rate`。
4. 在一个代表性模型上做轻量隐藏层分析，优先做 `residual stream activation patching`。
5. 生成论文所需的结果表、图和中文/英文实验说明。

当前服务器阶段优先服务 `MLISE 2026`，不管长期的IP&M

# 必读文件

新窗口或新 Agent 接手后，必须优先阅读：

1. `GPT55_HANDOFF.md`
2. `MLISE2026_simplified_plan.md`
3. `qwen3_cladder_feasibility_report.md`
4. `scripts/qwen3_cladder_feasibility.py`

其中：

- `GPT55_HANDOFF.md` 是完整项目交接说明。
- `MLISE2026_simplified_plan.md` 是当前 EI 会议稿的主要执行方案。
- `qwen3_cladder_feasibility_report.md` 是本地 `Qwen3-0.6B` pilot 报告。
- `scripts/qwen3_cladder_feasibility.py` 是已经跑通的原始全流程脚本。

# 安全与危险操作边界

默认原则：只新增、追加、修改必要代码；不做任何删除类操作。

禁止操作：

1. 禁止执行 `rm`、`rm -rf`、`unlink`、`find -delete`。
2. 禁止执行 `git reset --hard`、`git clean -fd`、`git checkout -- <file>`。
3. 禁止删除数据集、模型权重、输出结果、日志、实验表格、图片。
4. 禁止覆盖已有重要结果文件，除非先备份或写入新的输出目录。
5. 禁止强推远程分支，除非用户明确要求。
6. 禁止在未确认的情况下修改服务器系统级配置。

允许操作：

1. 可以新增代码文件。
2. 可以新增输出目录。
3. 可以新增日志目录。
4. 可以修改项目内脚本以支持新的实验。
5. 可以安装缺失的 Python 依赖。
6. 可以创建新的 conda 环境。

# 敏感信息处理

本文件可能包含服务器访问信息。后续 Agent 必须遵守：

1. 不要在公开回答、论文、日志、README 或 issue 中复述服务器密码。
2. 不要把服务器密码写入实验脚本。
3. 不要把含有密钥、token、密码的日志提交到 Git。

# Python 与 Conda 环境

服务器上可以创建新的 conda 虚拟环境，不要求复用本地环境。

推荐环境名：

```bash
conda create -n ipm-mlise python=3.11 -y
conda activate ipm-mlise
```

说明：

- 服务器 A100 实验优先推荐 `Python 3.11`，通常比 `Python 3.13` 对 CUDA、PyTorch、TransformerLens、flash attention 等库更稳。
- 缺什么库可以直接安装，但安装前尽量先记录到新的 requirements 文件中，例如 `requirements-mlise-a100.txt`。

推荐先升级基础工具：

```bash
python -m pip install -U pip setuptools wheel
```

# 国内镜像源

为了提高下载速度，可以优先使用清华或阿里镜像。

pip 临时使用清华源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

pip 设置全局清华源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

pip 临时使用阿里源：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple package_name
```

conda 可以优先使用清华源。若服务器已有可用配置，不要反复覆盖；若下载很慢，再配置镜像。

# CUDA 与显卡使用

服务器有多张 GPU，可根据任务规模选择。

建议：

1. 行为评测可以先用单卡跑通。
2. `Qwen3-8B` 和 patching 可以使用显存更充足的 GPU。
3. 如果只需要一张卡，使用 `CUDA_VISIBLE_DEVICES` 指定，避免占用全部资源。
4. 如果要尽可能拉满速度，可以考虑多进程按模型或 prompt mode 拆分，但要保证输出目录互不冲突。

单卡示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/mlise2026_qwen_scaling.py
```

多任务示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/mlise2026_qwen_scaling.py --model qwen3_0_6b
CUDA_VISIBLE_DEVICES=1 python scripts/mlise2026_qwen_scaling.py --model qwen3_4b
CUDA_VISIBLE_DEVICES=2 python scripts/mlise2026_qwen_scaling.py --model qwen3_8b
```

# Git 协作流程

推荐协作方式：

1. 本地新增或修改代码。
2. 本地运行基本语法检查。
3. `git add`、`git commit`、`git push`。
4. 服务器执行 `git pull origin main`。
5. 服务器运行实验。
6. 服务器生成结果后，再把关键脚本、表格、图和报告同步回 Git。

提交前建议先看状态：

```bash
git status
git diff --stat
```

提交信息建议简洁明确，例如：

```bash
git add scripts/mlise2026_qwen_scaling.py MLISE2026_simplified_plan.md
git commit -m "Add MLISE Qwen scaling experiment pipeline"
git push
```

# 服务器长任务运行规范

耗时任务优先使用 `nohup`，并实时写入日志。

推荐目录：

```text
logs/
outputs/mlise2026_qwen_scaling/
```

运行示例：

```bash
mkdir -p logs
nohup python scripts/mlise2026_qwen_scaling.py > logs/mlise2026_qwen_scaling_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

查看日志：

```bash
tail -f logs/mlise2026_qwen_scaling_YYYYMMDD_HHMMSS.log
```

查看进程：

```bash
ps -ef | grep mlise2026_qwen_scaling
```

如果任务需要分模型运行，日志文件名必须包含模型名：

```bash
nohup python scripts/mlise2026_qwen_scaling.py --model qwen3_8b > logs/qwen3_8b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

# 输出目录规范

正式 MLISE 实验建议统一写入：

```text
outputs/mlise2026_qwen_scaling/
```

建议结构：

```text
outputs/mlise2026_qwen_scaling/
├── figures/
├── tables/
├── logs/
├── patching/
└── reports/
```

任何新实验都应该写入新的子目录，避免覆盖已有 `outputs/qwen3_cladder_feasibility/`。

建议每次实验保存：

1. `config.json`：模型、样本数、prompt、seed、设备。
2. `main_eval.csv`：逐条推理结果。
3. `metrics_summary.csv`：汇总指标。
4. `metrics_pcc.csv`：PCC 指标。
5. `metrics_story_all_correct.csv`：story 级指标。
6. `patching_results.csv`：隐藏层 patching 结果。
7. `run.log`：完整运行日志。
8. `report.md`：自动生成实验报告。

# 实验实现原则

优先复用当前已经跑通的脚本逻辑。

建议从：

```text
scripts/qwen3_cladder_feasibility.py
```

复制并裁剪出：

```text
scripts/mlise2026_qwen_scaling.py
```

会议稿版本应优先做到：

1. 支持多个模型路径。
2. 支持命令行参数选择模型。
3. 支持快速版 `240` 样本和正式版 `640` 样本。
4. 支持断点续跑，已有结果尽量不重复计算。
5. 支持每个模型单独输出结果，再统一聚合。
6. 白盒 patching 只在指定模型上运行，默认优先 `Qwen3-4B` 或 `Qwen3-8B`。

# 运行速度原则

计算资源充足时，可以优先用速度换时间，但要保证结果可复现。

建议：

1. 使用 batch inference。
2. 使用 `torch_dtype=bfloat16` 或模型支持的半精度。
3. 使用 `device_map=auto` 或单卡显式放置。
4. 行为评测和 patching 分开运行。
5. 不同模型可以并行跑，但输出目录必须分开。
6. 每个长任务都要写日志。

# MLISE 实验推荐顺序

第一步：服务器复现本地结果。

```bash
python scripts/qwen3_cladder_feasibility.py
```

第二步：创建会议稿脚本。

```text
scripts/mlise2026_qwen_scaling.py
```

第三步：快速版三模型行为评测。

```text
Qwen3-0.6B / Qwen3-4B / Qwen3-8B
240 samples
Direct + Structured prompts
```

第四步：正式版三模型行为评测。

```text
640 samples
Accuracy + PCC + Story All-Correct Rate
```

第五步：隐藏层 patching。

```text
Representative model: Qwen3-4B or Qwen3-8B
Component: residual stream
Pairs: 8-16
Metric: yes/no logit margin recovery
```

第六步：生成论文图表和报告。

```text
MLISE2026_qwen_scaling_report.md
figures/*.png
tables/*.csv
```

# 写作与报告规范

每次完成实验后，都要更新一个 markdown 报告，至少包括：

1. 实验时间。
2. 运行环境。
3. 模型路径。
4. 数据集路径。
5. 样本规模。
6. prompt 设置。
7. 主结果表。
8. 图表索引。
9. 初步结论。
10. 遇到的问题和下一步建议。

会议稿写作不要过度声称：

- 不要声称发现了完整 causal circuit。
- 不要声称模型具备人类式因果推理。
- 不要声称 `CLadder` 覆盖真实世界所有因果场景。
- 不要声称本文提出了大型新 benchmark。

# 依赖与可复现记录

每次新增依赖后，需要同步更新 requirements 文件。

建议服务器版本单独记录：

```text
requirements-mlise-a100.txt
```

可以使用：

```bash
pip freeze > requirements-mlise-a100-freeze.txt
```

但论文复现实验更推荐维护一个精简 requirements，而不是只依赖 freeze 全量文件。

# 异常处理

如果出现 CUDA OOM：

1. 降低 batch size。
2. 减少 max_new_tokens。
3. 先关闭 patching。
4. patching 改用 `Qwen3-4B`。
5. 按模型分开运行。

如果 TransformerLens 不支持某个模型：

1. 优先尝试 HuggingFace forward hooks。
2. 只做 residual stream patching。
3. 在报告中明确写为 fallback implementation。

如果输出解析失败率高：

1. 收紧 prompt。
2. 强制 `Final answer: yes/no`。
3. 先不要引入人工修补。
4. 必要时将 invalid 单独报告。

# 最重要的执行原则

1. 先跑通，再扩大。
2. 先行为评测，再白盒分析。
3. 先快速版，再正式版。
4. 所有长任务必须有日志。
5. 所有重要结果必须写入表格和报告。
6. 不删除，不覆盖，不强推。
7. 服务器负责跑实验，本地负责整理代码和论文。
8. 所有的markdown文字记录使用中文记录。
