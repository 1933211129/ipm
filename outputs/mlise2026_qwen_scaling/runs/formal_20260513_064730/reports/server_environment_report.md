# MLISE 2026 服务器环境检查记录

## 基本信息

- 记录时间：`2026-05-13 06:56:57`
- 主机名：`ubuntu`
- 系统：`Linux-6.8.0-111-generic-x86_64-with-glibc2.39`
- Python：`3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]`
- PyTorch：`2.11.0+cu130`
- CUDA可用：`True`
- CUDA_VISIBLE_DEVICES：``
- CUDA版本：`13.0`
- GPU数量：`8`
- GPU列表：`['NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA A100 80GB PCIe']`
- Transformers：`5.8.1`
- TransformerLens可导入：`True`

## 实验配置

- run_id：`formal_20260513_064730`
- stage：`aggregate`
- sample_mode：`formal`
- 数据集路径：`/data/kongyb/ipm/datasets/cladder/data/full_v1.5_default.csv`
- 输出根目录：`/data/kongyb/ipm/outputs/mlise2026_qwen_scaling`
- prompt 设置：`direct`、`structured`
- 解析规则：优先解析 `Final answer: yes/no`，其次解析独立 yes/no，失败记为 invalid。

## 模型路径

- Qwen3-0.6B：`/data/LLM/Qwen/Qwen3-0___6B`，存在：`True`
- Qwen3-4B：`/data/LLM/Qwen/Qwen3-4B`，存在：`True`
- Qwen3-8B：`/data/LLM/Qwen/Qwen3-8B`，存在：`True`