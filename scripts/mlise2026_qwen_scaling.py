#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
import random
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.utils import check_random_state

plt = None
sns = None
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
HookedTransformer = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "datasets" / "cladder" / "data" / "full_v1.5_default.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "mlise2026_qwen_scaling"
SEED = 42
PROMPT_MODES = ["direct", "structured"]

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "qwen3_0_6b": {
        "display_name": "Qwen3-0.6B",
        "path": "/data/LLM/Qwen/Qwen3-0___6B",
        "hf_name": "Qwen/Qwen3-0.6B",
        "size_order": "0.6B",
    },
    "qwen3_4b": {
        "display_name": "Qwen3-4B",
        "path": "/data/LLM/Qwen/Qwen3-4B",
        "hf_name": "Qwen/Qwen3-4B",
        "size_order": "4B",
    },
    "qwen3_8b": {
        "display_name": "Qwen3-8B",
        "path": "/data/LLM/Qwen/Qwen3-8B",
        "hf_name": "Qwen/Qwen3-8B",
        "size_order": "8B",
    },
}

SAMPLE_QUOTAS: dict[str, dict[str, int]] = {
    "quick": {
        "marginal": 40,
        "correlation": 40,
        "ate": 40,
        "backadj": 40,
        "det-counterfactual": 20,
        "ett": 20,
        "nie": 20,
        "nde": 20,
    },
    "formal": {
        "marginal": 100,
        "correlation": 100,
        "ate": 100,
        "backadj": 100,
        "det-counterfactual": 60,
        "ett": 60,
        "nie": 60,
        "nde": 60,
    },
}

MODEL_ORDER = ["qwen3_0_6b", "qwen3_4b", "qwen3_8b"]

YES_NO_RE = re.compile(r"final answer\s*:\s*(yes|no)\b", flags=re.IGNORECASE)
LAST_YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


@dataclass
class RunPaths:
    output_root: Path
    run_dir: Path
    table_dir: Path
    figure_dir: Path
    report_dir: Path
    log_dir: Path
    patch_dir: Path


@dataclass
class EvalConfig:
    stage: str
    model: str
    sample_mode: str
    run_id: str
    data_path: str
    output_root: str
    patch_model: str
    seed: int
    batch_size: int
    patch_batch_size: int
    max_new_tokens: int
    patch_pair_limit: int
    patch_candidate_pool: int
    resume: bool
    skip_figures: bool


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_model_runtime(import_transformer_lens: bool = False) -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer, HookedTransformer
    if torch is None:
        import torch as torch_module

        torch = torch_module
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        from transformers import AutoModelForCausalLM as hf_model_cls
        from transformers import AutoTokenizer as hf_tokenizer_cls

        AutoModelForCausalLM = hf_model_cls
        AutoTokenizer = hf_tokenizer_cls
    if import_transformer_lens and HookedTransformer is None:
        try:
            from transformer_lens import HookedTransformer as tl_model_cls  # type: ignore

            HookedTransformer = tl_model_cls
        except Exception:
            HookedTransformer = None


def optional_torch():
    global torch
    if torch is not None:
        return torch
    try:
        import torch as torch_module

        torch = torch_module
        return torch
    except Exception:
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module = optional_torch()
    if torch_module is None:
        return
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_yes_no(raw_output: str) -> str | None:
    if not raw_output:
        return None
    match = YES_NO_RE.search(raw_output)
    if match:
        return match.group(1).lower()
    matches = LAST_YES_NO_RE.findall(raw_output.lower())
    if matches:
        return matches[-1].lower()
    return None


def get_paths(output_root: Path, run_id: str) -> RunPaths:
    run_dir = output_root / "runs" / run_id
    return RunPaths(
        output_root=output_root,
        run_dir=run_dir,
        table_dir=run_dir / "tables",
        figure_dir=run_dir / "figures",
        report_dir=run_dir / "reports",
        log_dir=run_dir / "logs",
        patch_dir=run_dir / "patching",
    )


def ensure_dirs(paths: RunPaths) -> None:
    for path in [
        paths.run_dir,
        paths.table_dir,
        paths.figure_dir,
        paths.report_dir,
        paths.log_dir,
        paths.patch_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def model_table_dir(paths: RunPaths, model_key: str) -> Path:
    path = paths.table_dir / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_subset_path(paths: RunPaths, sample_mode: str) -> Path:
    return paths.table_dir / f"selected_subset_{sample_mode}.csv"


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["id"] = df["id"].astype(int)
    df["rung"] = df["rung"].astype(int)
    df["label"] = df["label"].str.lower().str.strip()
    return df


def balanced_sample(
    df: pd.DataFrame,
    n_total: int,
    seed: int,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if n_total % 2 != 0:
        raise ValueError(f"样本数必须为偶数，当前为 {n_total}")
    half = n_total // 2
    if len(df[df["label"] == "yes"]) < half or len(df[df["label"] == "no"]) < half:
        raise ValueError("当前 query type 的 yes/no 数量不足，无法完成平衡抽样")
    rng = check_random_state(seed)
    yes_df = df[df["label"] == "yes"].sample(n=half, random_state=rng)
    no_df = df[df["label"] == "no"].sample(n=half, random_state=rng)
    sampled = pd.concat([yes_df, no_df], ignore_index=True)
    if group_cols:
        return sampled.sort_values(group_cols + ["id"]).reset_index(drop=True)
    return sampled.sort_values(["id"]).reset_index(drop=True)


def select_main_subset(df: pd.DataFrame, sample_mode: str, seed: int) -> pd.DataFrame:
    quotas = SAMPLE_QUOTAS[sample_mode]
    chunks = []
    for idx, (query_type, n_total) in enumerate(quotas.items()):
        pool = df[df["query_type"] == query_type].copy()
        chunk = balanced_sample(
            pool,
            n_total=n_total,
            seed=seed + idx,
            group_cols=["story_id", "rung", "query_type", "label"],
        )
        chunks.append(chunk)
    subset = pd.concat(chunks, ignore_index=True)
    subset = subset.sort_values(["query_type", "label", "story_id", "rung", "id"]).reset_index(drop=True)
    subset["subset"] = sample_mode
    return subset


def dataset_statistics(subset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_query = (
        subset.groupby(["query_type", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    by_rung = (
        subset.groupby(["rung", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    by_query_rung = (
        subset.groupby(["query_type", "rung", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return {
        "dataset_stats_by_query": by_query,
        "dataset_stats_by_rung": by_rung,
        "dataset_stats_by_query_rung": by_query_rung,
    }


def save_sample_files(paths: RunPaths, subset: pd.DataFrame, sample_mode: str) -> None:
    subset.to_csv(selected_subset_path(paths, sample_mode), index=False)
    for name, table in dataset_statistics(subset).items():
        table.to_csv(paths.table_dir / f"{name}_{sample_mode}.csv", index=False)


def load_or_create_subset(paths: RunPaths, data_path: Path, sample_mode: str, seed: int) -> pd.DataFrame:
    path = selected_subset_path(paths, sample_mode)
    if path.exists():
        subset = pd.read_csv(path)
        subset["id"] = subset["id"].astype(int)
        subset["rung"] = subset["rung"].astype(int)
        subset["label"] = subset["label"].str.lower().str.strip()
        return subset
    df = load_dataset(data_path)
    subset = select_main_subset(df, sample_mode, seed)
    save_sample_files(paths, subset, sample_mode)
    return subset


def build_user_prompt(base_prompt: str, mode: str) -> str:
    if mode == "direct":
        return (
            "You are solving a formal causal reasoning question.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}"
        )
    if mode == "structured":
        return (
            "You are solving a causal reasoning question.\n"
            "Silently identify the relevant causal relation before answering.\n"
            "Do not write reasoning, variables, bullets, or any explanation.\n"
            "Return exactly one line and nothing else: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}"
        )
    raise ValueError(f"未知 prompt mode: {mode}")


def get_device() -> str:
    ensure_model_runtime()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def empty_device_cache() -> None:
    gc.collect()
    torch_module = optional_torch()
    if torch_module is not None and torch_module.cuda.is_available():
        torch.cuda.empty_cache()


class QwenEvaluator:
    def __init__(
        self,
        model_key: str,
        model_path: Path,
        hf_name: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        ensure_model_runtime()
        self.model_key = model_key
        self.model_path = str(model_path)
        self.hf_name = hf_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = get_device()
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在：{model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device == "cuda" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def close(self) -> None:
        del self.model
        del self.tokenizer
        empty_device_cache()

    def format_batch(self, prompts: list[str]) -> list[str]:
        formatted = []
        for prompt in prompts:
            try:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted.append(text)
        return formatted

    def generate_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        formatted = self.format_batch(prompts)
        encoded = self.tokenizer(formatted, padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        rows = []
        for seq in outputs:
            completion_tokens = seq[prompt_width:]
            raw_output = self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            rows.append(
                {
                    "raw_output": raw_output,
                    "input_tokens": prompt_width,
                    "output_tokens": int(completion_tokens.shape[0]),
                    "latency_sec": elapsed / max(1, len(prompts)),
                }
            )
        return rows


def run_eval(
    evaluator: QwenEvaluator,
    df: pd.DataFrame,
    model_key: str,
    sample_mode: str,
    run_id: str,
    modes: list[str],
    batch_size: int,
    save_path: Path,
    resume: bool,
) -> pd.DataFrame:
    existing = pd.DataFrame()
    completed: set[tuple[int, str]] = set()
    if resume and save_path.exists():
        existing = pd.read_csv(save_path)
        if not existing.empty:
            completed = set(zip(existing["sample_id"].astype(int), existing["prompt_mode"].astype(str)))
            print(f"[信息] resume 已读取 {len(existing)} 行：{save_path}", flush=True)
    elif save_path.exists() and not resume:
        raise FileExistsError(f"结果文件已存在，避免覆盖：{save_path}。如需续跑请加 --resume。")

    all_records: list[dict[str, Any]] = []
    for mode in modes:
        rows = [row for row in df.to_dict("records") if (int(row["id"]), mode) not in completed]
        print(f"[进度] model={model_key} sample={sample_mode} mode={mode} 待评测={len(rows)}", flush=True)
        batch_size_current = batch_size
        i = 0
        while i < len(rows):
            current_rows = rows[i : i + batch_size_current]
            prompts = [build_user_prompt(row["prompt"], mode) for row in current_rows]
            try:
                generated = evaluator.generate_batch(prompts)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message and batch_size_current > 1:
                    batch_size_current = max(1, batch_size_current // 2)
                    print(f"[警告] 显存不足，batch size 降为 {batch_size_current}", flush=True)
                    empty_device_cache()
                    continue
                raise

            for source_row, gen_row in zip(current_rows, generated):
                parsed = parse_yes_no(gen_row["raw_output"])
                record = {
                    "run_id": run_id,
                    "model": model_key,
                    "model_display_name": MODEL_CONFIGS[model_key]["display_name"],
                    "sample_mode": sample_mode,
                    "sample_id": int(source_row["id"]),
                    "story_id": source_row["story_id"],
                    "graph_id": source_row["graph_id"],
                    "rung": int(source_row["rung"]),
                    "query_type": source_row["query_type"],
                    "question_property": source_row["question_property"],
                    "formal_form": source_row["formal_form"],
                    "gold_label": source_row["label"],
                    "prompt_mode": mode,
                    "dataset_variant": "original_main",
                    "subset": source_row.get("subset", sample_mode),
                    **gen_row,
                }
                record["parsed_label"] = parsed if parsed in {"yes", "no"} else "invalid"
                record["is_invalid"] = record["parsed_label"] == "invalid"
                record["is_correct"] = int(record["parsed_label"] == record["gold_label"])
                all_records.append(record)

            i += len(current_rows)
            if all_records:
                combined = pd.concat([existing, pd.DataFrame(all_records)], ignore_index=True)
                combined.to_csv(save_path, index=False)

    if all_records:
        return pd.concat([existing, pd.DataFrame(all_records)], ignore_index=True)
    return existing


def compute_summary_metrics(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if eval_df.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "by_rung": empty, "by_query": empty, "label_distribution": empty}

    summary = (
        eval_df.groupby(["model", "model_display_name", "sample_mode", "prompt_mode"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            parse_rate=("is_invalid", lambda x: 1 - x.mean()),
            invalid_rate=("is_invalid", "mean"),
            latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )
    by_rung = (
        eval_df.groupby(["model", "model_display_name", "sample_mode", "prompt_mode", "rung"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            parse_rate=("is_invalid", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )
    by_query = (
        eval_df.groupby(["model", "model_display_name", "sample_mode", "prompt_mode", "query_type"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            parse_rate=("is_invalid", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )
    label_distribution = (
        eval_df.groupby(["model", "model_display_name", "prompt_mode", "parsed_label"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    return {
        "summary": summary,
        "by_rung": by_rung,
        "by_query": by_query,
        "label_distribution": label_distribution,
    }


def compute_pcc_tables(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pair_rows = []
    valid_labels = {"yes", "no"}
    for (model, display, sample_mode, mode), mode_df in eval_df.groupby(
        ["model", "model_display_name", "sample_mode", "prompt_mode"],
        dropna=False,
    ):
        for story_id, story_df in mode_df.groupby("story_id"):
            story_rows = story_df.to_dict("records")
            for left, right in itertools.combinations(story_rows, 2):
                transition = tuple(sorted((int(left["rung"]), int(right["rung"]))))
                if transition not in {(1, 2), (2, 3), (1, 3)}:
                    continue
                gold_same = left["gold_label"] == right["gold_label"]
                left_valid = left["parsed_label"] in valid_labels
                right_valid = right["parsed_label"] in valid_labels
                pred_same = left_valid and right_valid and left["parsed_label"] == right["parsed_label"]
                pair_rows.append(
                    {
                        "model": model,
                        "model_display_name": display,
                        "sample_mode": sample_mode,
                        "prompt_mode": mode,
                        "story_id": story_id,
                        "left_sample_id": int(left["sample_id"]),
                        "right_sample_id": int(right["sample_id"]),
                        "transition": f"{transition[0]}→{transition[1]}",
                        "gold_same": gold_same,
                        "pred_same": pred_same,
                        "has_invalid": not (left_valid and right_valid),
                        "is_consistent": int(gold_same == pred_same),
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        pcc_cols = ["model", "model_display_name", "sample_mode", "prompt_mode", "transition", "n", "pcc", "invalid_pair_rate"]
        overall_cols = ["model", "model_display_name", "sample_mode", "prompt_mode", "n", "pcc", "invalid_pair_rate"]
        return {
            "metrics_pcc_pairs": pair_df,
            "metrics_pcc": pd.DataFrame(columns=pcc_cols),
            "metrics_pcc_overall": pd.DataFrame(columns=overall_cols),
        }

    pcc = (
        pair_df.groupby(["model", "model_display_name", "sample_mode", "prompt_mode", "transition"])
        .agg(n=("story_id", "count"), pcc=("is_consistent", "mean"), invalid_pair_rate=("has_invalid", "mean"))
        .reset_index()
    )
    overall = (
        pair_df.groupby(["model", "model_display_name", "sample_mode", "prompt_mode"])
        .agg(n=("story_id", "count"), pcc=("is_consistent", "mean"), invalid_pair_rate=("has_invalid", "mean"))
        .reset_index()
    )
    return {"metrics_pcc_pairs": pair_df, "metrics_pcc": pcc, "metrics_pcc_overall": overall}


def compute_story_all_correct(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=["model", "model_display_name", "sample_mode", "prompt_mode", "n_stories", "story_all_correct_rate"]
        )
    story = (
        eval_df.groupby(["model", "model_display_name", "sample_mode", "prompt_mode", "story_id"], dropna=False)
        .agg(all_correct=("is_correct", "min"), n=("sample_id", "count"))
        .reset_index()
    )
    return (
        story.groupby(["model", "model_display_name", "sample_mode", "prompt_mode"], dropna=False)
        .agg(n_stories=("story_id", "count"), story_all_correct_rate=("all_correct", "mean"))
        .reset_index()
    )


def save_behavior_metrics(eval_df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    metrics = compute_summary_metrics(eval_df)
    for name, table in metrics.items():
        table.to_csv(out_dir / f"metrics_{name}.csv", index=False)
    pcc_tables = compute_pcc_tables(eval_df)
    for name, table in pcc_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    story = compute_story_all_correct(eval_df)
    story.to_csv(out_dir / "metrics_story_all_correct.csv", index=False)
    return {**metrics, **pcc_tables, "metrics_story_all_correct": story}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_environment_info() -> dict[str, Any]:
    torch_module = optional_torch()
    info: dict[str, Any] = {
        "记录时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "主机名": socket.gethostname(),
        "系统": platform.platform(),
        "Python": sys.version.replace("\n", " "),
        "PyTorch": torch_module.__version__ if torch_module is not None else "未安装或当前环境不可导入",
        "CUDA可用": bool(torch_module is not None and torch_module.cuda.is_available()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch_module is not None and torch_module.cuda.is_available():
        info["CUDA版本"] = torch_module.version.cuda
        info["GPU数量"] = torch_module.cuda.device_count()
        info["GPU列表"] = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    try:
        import transformers

        info["Transformers"] = transformers.__version__
    except Exception as exc:
        info["Transformers"] = f"无法读取：{exc}"
    try:
        from transformer_lens import HookedTransformer as _tl_model_cls  # type: ignore

        info["TransformerLens可导入"] = _tl_model_cls is not None
    except Exception:
        info["TransformerLens可导入"] = False
    return info


def write_environment_report(paths: RunPaths, config: EvalConfig) -> None:
    env = collect_environment_info()
    lines = [
        "# MLISE 2026 服务器环境检查记录",
        "",
        "## 基本信息",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- {key}：`{value}`")
    lines.extend(
        [
            "",
            "## 实验配置",
            "",
            f"- run_id：`{config.run_id}`",
            f"- stage：`{config.stage}`",
            f"- sample_mode：`{config.sample_mode}`",
            f"- 数据集路径：`{config.data_path}`",
            f"- 输出根目录：`{config.output_root}`",
            f"- prompt 设置：`direct`、`structured`",
            f"- 解析规则：优先解析 `Final answer: yes/no`，其次解析独立 yes/no，失败记为 invalid。",
            "",
            "## 模型路径",
            "",
        ]
    )
    for key in MODEL_ORDER:
        model_cfg = MODEL_CONFIGS[key]
        exists = Path(model_cfg["path"]).exists()
        lines.append(f"- {model_cfg['display_name']}：`{model_cfg['path']}`，存在：`{exists}`")
    (paths.report_dir / "server_environment_report.md").write_text("\n".join(lines), encoding="utf-8")


def model_keys_from_arg(model_arg: str) -> list[str]:
    if model_arg == "all":
        return MODEL_ORDER.copy()
    return [model_arg]


def run_sample_stage(paths: RunPaths, data_path: Path, sample_mode: str, seed: int) -> pd.DataFrame:
    df = load_dataset(data_path)
    subset = select_main_subset(df, sample_mode, seed)
    save_sample_files(paths, subset, sample_mode)
    report_lines = [
        "# MLISE 2026 抽样记录",
        "",
        f"- 抽样时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集路径：`{data_path}`",
        f"- 样本模式：`{sample_mode}`",
        f"- 总样本数：`{len(subset)}`",
        f"- story 数：`{subset['story_id'].nunique()}`",
        f"- 输出文件：`{selected_subset_path(paths, sample_mode)}`",
        "",
        "## Query Type 分布",
        "",
        dataset_statistics(subset)["dataset_stats_by_query"].to_markdown(index=False),
        "",
        "## Rung 分布",
        "",
        dataset_statistics(subset)["dataset_stats_by_rung"].to_markdown(index=False),
        "",
    ]
    (paths.report_dir / f"sample_report_{sample_mode}.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[完成] 抽样完成：{selected_subset_path(paths, sample_mode)}", flush=True)
    return subset


def run_behavior_stage(
    paths: RunPaths,
    args: argparse.Namespace,
    model_keys: list[str],
    config: EvalConfig,
) -> None:
    subset = load_or_create_subset(paths, Path(args.data_path), args.sample_mode, args.seed)
    for model_key in model_keys:
        model_cfg = MODEL_CONFIGS[model_key]
        out_dir = model_table_dir(paths, model_key)
        save_json(
            out_dir / "config.json",
            {
                "运行配置": asdict(config),
                "模型配置": model_cfg,
                "prompt_modes": PROMPT_MODES,
                "说明": "主实验只使用自动规则解析，不调用 DeepSeek 或其他外部模型修补答案。",
            },
        )
        evaluator = QwenEvaluator(
            model_key=model_key,
            model_path=Path(model_cfg["path"]),
            hf_name=model_cfg["hf_name"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            eval_path = out_dir / "main_eval.csv"
            eval_df = run_eval(
                evaluator=evaluator,
                df=subset,
                model_key=model_key,
                sample_mode=args.sample_mode,
                run_id=args.run_id,
                modes=PROMPT_MODES,
                batch_size=args.batch_size,
                save_path=eval_path,
                resume=args.resume,
            )
            save_behavior_metrics(eval_df, out_dir)
            write_model_behavior_report(paths, model_key, eval_df, out_dir)
        finally:
            evaluator.close()
        print(f"[完成] 行为评测完成：{model_key}", flush=True)


def _yes_no_token_ids(tokenizer: AutoTokenizer) -> tuple[int, int]:
    yes_id = None
    no_id = None
    for text in [" yes", "yes", " Yes", "Yes"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            yes_id = ids[0]
            break
    for text in [" no", "no", " No", "No"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            no_id = ids[0]
            break
    if yes_id is None or no_id is None:
        raise RuntimeError("无法定位单 token 的 yes/no id")
    return yes_id, no_id


def margin_from_logits(logits: torch.Tensor, yes_id: int, no_id: int, target_label: str) -> float:
    if target_label == "yes":
        return float(logits[0, -1, yes_id] - logits[0, -1, no_id])
    if target_label == "no":
        return float(logits[0, -1, no_id] - logits[0, -1, yes_id])
    raise ValueError(f"未知 target label：{target_label}")


def build_patch_candidates(
    df: pd.DataFrame,
    evaluator: QwenEvaluator,
    candidate_pool: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    tokenizer = evaluator.tokenizer
    grouped = df.groupby(["story_id", "query_type", "formal_form"], dropna=False)
    for _, group in grouped:
        if set(group["label"]) != {"yes", "no"}:
            continue
        token_groups: dict[int, list[dict[str, Any]]] = {}
        for row in group.to_dict("records"):
            prompt_text = build_user_prompt(row["prompt"], "direct")
            chat_text = evaluator.format_batch([prompt_text])[0]
            token_len = len(tokenizer(chat_text, add_special_tokens=False)["input_ids"])
            token_groups.setdefault(token_len, []).append(row | {"token_len": token_len})
        for token_len, token_rows in token_groups.items():
            labels = {item["label"] for item in token_rows}
            if labels != {"yes", "no"}:
                continue
            yes_row = next(item for item in token_rows if item["label"] == "yes")
            no_row = next(item for item in token_rows if item["label"] == "no")
            rows.append(
                {
                    "story_id": yes_row["story_id"],
                    "query_type": yes_row["query_type"],
                    "formal_form": yes_row["formal_form"],
                    "rung": int(yes_row["rung"]),
                    "token_len": int(token_len),
                    "yes_id": int(yes_row["id"]),
                    "yes_prompt": yes_row["prompt"],
                    "no_id": int(no_row["id"]),
                    "no_prompt": no_row["prompt"],
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    priority = candidates[candidates["rung"].isin([2, 3])].copy()
    if len(priority) >= candidate_pool:
        return priority.sample(n=candidate_pool, random_state=seed).reset_index(drop=True)
    extra = candidates[~candidates.index.isin(priority.index)].copy()
    needed = max(0, candidate_pool - len(priority))
    if needed > 0 and not extra.empty:
        extra = extra.sample(n=min(needed, len(extra)), random_state=seed)
    return pd.concat([priority, extra], ignore_index=True).reset_index(drop=True)


def select_clean_corrupted_pairs(
    evaluator: QwenEvaluator,
    candidates: pd.DataFrame,
    model_key: str,
    sample_mode: str,
    run_id: str,
    batch_size: int,
    pair_limit: int,
    patch_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    source_rows = []
    for _, row in candidates.iterrows():
        source_rows.extend(
            [
                {"id": int(row["yes_id"]), "prompt": row["yes_prompt"], "label": "yes"},
                {"id": int(row["no_id"]), "prompt": row["no_prompt"], "label": "no"},
            ]
        )
    source_df = pd.DataFrame(source_rows).drop_duplicates("id").copy()
    for column, value in {
        "story_id": "patch_candidate",
        "graph_id": "patch_candidate",
        "rung": 0,
        "query_type": "patch_candidate",
        "question_property": "patch_candidate",
        "formal_form": "patch_candidate",
        "subset": "patch_candidate",
    }.items():
        source_df[column] = value

    eval_path = patch_dir / "patch_candidate_eval.csv"
    candidate_eval = run_eval(
        evaluator=evaluator,
        df=source_df,
        model_key=model_key,
        sample_mode=sample_mode,
        run_id=run_id,
        modes=["direct"],
        batch_size=batch_size,
        save_path=eval_path,
        resume=True,
    )
    lookup = candidate_eval.set_index("sample_id")[["parsed_label", "is_correct", "raw_output"]].to_dict("index")

    selected: list[dict[str, Any]] = []
    for _, pair in candidates.iterrows():
        yes_meta = lookup.get(int(pair["yes_id"]))
        no_meta = lookup.get(int(pair["no_id"]))
        if not yes_meta or not no_meta:
            continue
        yes_correct = bool(yes_meta["is_correct"])
        no_correct = bool(no_meta["is_correct"])
        if yes_correct == no_correct:
            continue
        if yes_correct:
            clean_label = "yes"
            corrupted_label = "no"
            clean_id = int(pair["yes_id"])
            corrupted_id = int(pair["no_id"])
            clean_prompt = pair["yes_prompt"]
            corrupted_prompt = pair["no_prompt"]
        else:
            clean_label = "no"
            corrupted_label = "yes"
            clean_id = int(pair["no_id"])
            corrupted_id = int(pair["yes_id"])
            clean_prompt = pair["no_prompt"]
            corrupted_prompt = pair["yes_prompt"]
        selected.append(
            {
                "story_id": pair["story_id"],
                "query_type": pair["query_type"],
                "formal_form": pair["formal_form"],
                "rung": int(pair["rung"]),
                "token_len": int(pair["token_len"]),
                "clean_id": clean_id,
                "corrupted_id": corrupted_id,
                "clean_label": clean_label,
                "corrupted_gold_label": corrupted_label,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "clean_raw_output": yes_meta["raw_output"] if yes_correct else no_meta["raw_output"],
                "corrupted_raw_output": no_meta["raw_output"] if yes_correct else yes_meta["raw_output"],
            }
        )
        if len(selected) >= pair_limit:
            break
    return selected, candidate_eval


def try_load_hooked_transformer(evaluator: QwenEvaluator):
    ensure_model_runtime(import_transformer_lens=True)
    if HookedTransformer is None:
        return None, "transformer_lens_import_failed"
    try:
        tl_model = HookedTransformer.from_pretrained_no_processing(
            evaluator.hf_name,
            hf_model=evaluator.model,
            tokenizer=evaluator.tokenizer,
            device=evaluator.device,
            dtype="bfloat16" if evaluator.device == "cuda" else "float32",
            move_to_device=True,
            default_padding_side="left",
            trust_remote_code=True,
        )
        return tl_model, "transformer_lens"
    except Exception as exc:
        return None, f"transformer_lens_load_failed: {exc}"


def run_residual_patching(
    evaluator: QwenEvaluator,
    selected_pairs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, str]:
    if not selected_pairs:
        return pd.DataFrame(), "no_clean_corrupted_pairs"

    yes_id, no_id = _yes_no_token_ids(evaluator.tokenizer)
    tl_model, tl_status = try_load_hooked_transformer(evaluator)
    rows: list[dict[str, Any]] = []

    if tl_model is not None:
        for pair_idx, pair in enumerate(selected_pairs, start=1):
            clean_text = evaluator.format_batch([build_user_prompt(pair["clean_prompt"], "direct")])[0]
            corrupted_text = evaluator.format_batch([build_user_prompt(pair["corrupted_prompt"], "direct")])[0]
            clean_tokens = tl_model.to_tokens(clean_text).to(tl_model.cfg.device)
            corrupted_tokens = tl_model.to_tokens(corrupted_text).to(tl_model.cfg.device)

            with torch.inference_mode():
                clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
                corrupted_logits, _ = tl_model.run_with_cache(corrupted_tokens)

            clean_margin = margin_from_logits(clean_logits, yes_id, no_id, pair["clean_label"])
            corrupted_clean_target_margin = margin_from_logits(corrupted_logits, yes_id, no_id, pair["clean_label"])
            corrupted_gold_margin = margin_from_logits(corrupted_logits, yes_id, no_id, pair["corrupted_gold_label"])

            for layer in range(tl_model.cfg.n_layers):
                hook_name = f"blocks.{layer}.hook_resid_pre"

                def patch_hook(value, hook, clean_cache=clean_cache, hook_name=hook_name):
                    return clean_cache[hook_name]

                with torch.inference_mode():
                    patched_logits = tl_model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
                patched_clean_target_margin = margin_from_logits(
                    patched_logits,
                    yes_id,
                    no_id,
                    pair["clean_label"],
                )
                patched_corrupted_gold_margin = margin_from_logits(
                    patched_logits,
                    yes_id,
                    no_id,
                    pair["corrupted_gold_label"],
                )
                rows.append(
                    {
                        "pair_index": pair_idx,
                        "pair_id": f"{pair['clean_id']}->{pair['corrupted_id']}",
                        "story_id": pair["story_id"],
                        "query_type": pair["query_type"],
                        "rung": pair["rung"],
                        "component": "residual",
                        "layer": layer,
                        "clean_id": pair["clean_id"],
                        "corrupted_id": pair["corrupted_id"],
                        "clean_label": pair["clean_label"],
                        "corrupted_gold_label": pair["corrupted_gold_label"],
                        "clean_target_margin": clean_margin,
                        "corrupted_clean_target_margin": corrupted_clean_target_margin,
                        "patched_clean_target_margin": patched_clean_target_margin,
                        "recovery": patched_clean_target_margin - corrupted_clean_target_margin,
                        "corrupted_gold_margin": corrupted_gold_margin,
                        "patched_corrupted_gold_margin": patched_corrupted_gold_margin,
                        "corrupted_gold_recovery": patched_corrupted_gold_margin - corrupted_gold_margin,
                        "method": "transformer_lens_residual",
                    }
                )
            empty_device_cache()
        return pd.DataFrame(rows), tl_status

    hf_model = evaluator.model
    tok = evaluator.tokenizer
    device = evaluator.device

    def encode_prompt(prompt: str) -> dict[str, torch.Tensor]:
        text = evaluator.format_batch([build_user_prompt(prompt, "direct")])[0]
        encoded = tok(text, return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}

    for pair_idx, pair in enumerate(selected_pairs, start=1):
        clean_inputs = encode_prompt(pair["clean_prompt"])
        corrupted_inputs = encode_prompt(pair["corrupted_prompt"])
        if clean_inputs["input_ids"].shape[1] != corrupted_inputs["input_ids"].shape[1]:
            continue

        clean_cache: dict[int, torch.Tensor] = {}

        def save_layer_output(layer_idx: int):
            def hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                clean_cache[layer_idx] = hidden.detach().clone()
            return hook

        handles = [
            hf_model.model.layers[layer_idx].register_forward_hook(save_layer_output(layer_idx))
            for layer_idx in range(len(hf_model.model.layers))
        ]
        with torch.inference_mode():
            clean_outputs = hf_model(**clean_inputs)
        for handle in handles:
            handle.remove()

        with torch.inference_mode():
            corrupted_outputs = hf_model(**corrupted_inputs)

        clean_margin = margin_from_logits(clean_outputs.logits, yes_id, no_id, pair["clean_label"])
        corrupted_clean_target_margin = margin_from_logits(
            corrupted_outputs.logits,
            yes_id,
            no_id,
            pair["clean_label"],
        )
        corrupted_gold_margin = margin_from_logits(
            corrupted_outputs.logits,
            yes_id,
            no_id,
            pair["corrupted_gold_label"],
        )

        for layer_idx in range(len(hf_model.model.layers)):
            def patch_layer(_module, _inp, output, layer_idx=layer_idx):
                patched = clean_cache[layer_idx]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = hf_model.model.layers[layer_idx].register_forward_hook(patch_layer)
            with torch.inference_mode():
                patched_outputs = hf_model(**corrupted_inputs)
            handle.remove()

            patched_clean_target_margin = margin_from_logits(
                patched_outputs.logits,
                yes_id,
                no_id,
                pair["clean_label"],
            )
            patched_corrupted_gold_margin = margin_from_logits(
                patched_outputs.logits,
                yes_id,
                no_id,
                pair["corrupted_gold_label"],
            )
            rows.append(
                {
                    "pair_index": pair_idx,
                    "pair_id": f"{pair['clean_id']}->{pair['corrupted_id']}",
                    "story_id": pair["story_id"],
                    "query_type": pair["query_type"],
                    "rung": pair["rung"],
                    "component": "residual",
                    "layer": layer_idx,
                    "clean_id": pair["clean_id"],
                    "corrupted_id": pair["corrupted_id"],
                    "clean_label": pair["clean_label"],
                    "corrupted_gold_label": pair["corrupted_gold_label"],
                    "clean_target_margin": clean_margin,
                    "corrupted_clean_target_margin": corrupted_clean_target_margin,
                    "patched_clean_target_margin": patched_clean_target_margin,
                    "recovery": patched_clean_target_margin - corrupted_clean_target_margin,
                    "corrupted_gold_margin": corrupted_gold_margin,
                    "patched_corrupted_gold_margin": patched_corrupted_gold_margin,
                    "corrupted_gold_recovery": patched_corrupted_gold_margin - corrupted_gold_margin,
                    "method": "hf_forward_hook_residual",
                }
            )
        empty_device_cache()

    return pd.DataFrame(rows), tl_status


def run_patch_stage(paths: RunPaths, args: argparse.Namespace, config: EvalConfig) -> None:
    patch_model = args.patch_model
    model_cfg = MODEL_CONFIGS[patch_model]
    patch_out = paths.patch_dir / patch_model
    patch_out.mkdir(parents=True, exist_ok=True)
    if (patch_out / "patching_results.csv").exists() and not args.resume:
        raise FileExistsError(f"patching 结果已存在，避免覆盖：{patch_out / 'patching_results.csv'}")

    subset = load_or_create_subset(paths, Path(args.data_path), args.sample_mode, args.seed)
    full_df = load_dataset(Path(args.data_path))
    evaluator = QwenEvaluator(
        model_key=patch_model,
        model_path=Path(model_cfg["path"]),
        hf_name=model_cfg["hf_name"],
        batch_size=args.patch_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        candidates = build_patch_candidates(subset, evaluator, args.patch_candidate_pool, args.seed)
        candidate_source = "selected_subset"
        if candidates.empty or len(candidates) < args.patch_pair_limit:
            candidates = build_patch_candidates(full_df, evaluator, args.patch_candidate_pool, args.seed)
            candidate_source = "full_dataset"
        candidates.to_csv(patch_out / "patch_candidate_pairs.csv", index=False)
        if candidates.empty:
            patch_df = pd.DataFrame(
                columns=[
                    "model",
                    "model_display_name",
                    "sample_mode",
                    "candidate_source",
                    "pair_id",
                    "component",
                    "layer",
                    "recovery",
                    "method",
                ]
            )
            patch_df.to_csv(patch_out / "patching_results.csv", index=False)
            write_patch_report(paths, patch_model, patch_df, "no_candidate_pairs", candidate_source, [])
            return

        selected_pairs, candidate_eval = select_clean_corrupted_pairs(
            evaluator=evaluator,
            candidates=candidates,
            model_key=patch_model,
            sample_mode=args.sample_mode,
            run_id=args.run_id,
            batch_size=args.patch_batch_size,
            pair_limit=args.patch_pair_limit,
            patch_dir=patch_out,
        )
        pd.DataFrame(selected_pairs).to_csv(patch_out / "selected_clean_corrupted_pairs.csv", index=False)
        candidate_eval.to_csv(patch_out / "patch_candidate_eval.csv", index=False)

        patch_df, patch_status = run_residual_patching(evaluator, selected_pairs)
        patch_df["model"] = patch_model
        patch_df["model_display_name"] = model_cfg["display_name"]
        patch_df["sample_mode"] = args.sample_mode
        patch_df["candidate_source"] = candidate_source
        patch_df.to_csv(patch_out / "patching_results.csv", index=False)
        write_patch_report(paths, patch_model, patch_df, patch_status, candidate_source, selected_pairs)
    finally:
        evaluator.close()
    print(f"[完成] 白盒 patching 完成：{patch_model}", flush=True)


def read_model_eval_tables(paths: RunPaths) -> pd.DataFrame:
    frames = []
    for model_key in MODEL_ORDER:
        path = paths.table_dir / model_key / "main_eval.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_patch_tables(paths: RunPaths) -> pd.DataFrame:
    frames = []
    for model_key in MODEL_ORDER:
        path = paths.patch_dir / model_key / "patching_results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_results(paths: RunPaths, sample_mode: str, skip_figures: bool) -> None:
    eval_df = read_model_eval_tables(paths)
    if eval_df.empty:
        raise FileNotFoundError("未找到任何模型的 main_eval.csv，无法聚合。")

    aggregate_dir = paths.table_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(aggregate_dir / "all_main_eval.csv", index=False)

    metrics = save_behavior_metrics(eval_df, aggregate_dir)
    summary = metrics["summary"]
    pcc_overall = metrics["metrics_pcc_overall"]
    story = metrics["metrics_story_all_correct"]
    main_results = summary.merge(
        pcc_overall[["model", "prompt_mode", "pcc", "invalid_pair_rate"]],
        on=["model", "prompt_mode"],
        how="left",
    ).merge(
        story[["model", "prompt_mode", "story_all_correct_rate", "n_stories"]],
        on=["model", "prompt_mode"],
        how="left",
    )
    main_results["model_order"] = main_results["model"].map({key: idx for idx, key in enumerate(MODEL_ORDER)})
    main_results = main_results.sort_values(["model_order", "prompt_mode"]).drop(columns=["model_order"])
    main_results.to_csv(aggregate_dir / "aggregate_main_results.csv", index=False)

    patch_df = read_patch_tables(paths)
    if not patch_df.empty:
        patch_df.to_csv(aggregate_dir / "patching_results.csv", index=False)
        patch_summary = (
            patch_df.groupby(["model", "model_display_name"])
            .agg(
                n_rows=("pair_id", "count"),
                n_pairs=("pair_id", "nunique"),
                positive_pair_ratio=("recovery", lambda x: float((patch_df.loc[x.index].groupby("pair_id")["recovery"].max() > 0).mean())),
                mean_recovery=("recovery", "mean"),
                max_recovery=("recovery", "max"),
            )
            .reset_index()
        )
        patch_summary.to_csv(aggregate_dir / "patching_summary.csv", index=False)
    else:
        patch_summary = pd.DataFrame()

    figure_refs: list[tuple[str, str]] = []
    if not skip_figures:
        subset_path = selected_subset_path(paths, sample_mode)
        subset = pd.read_csv(subset_path) if subset_path.exists() else pd.DataFrame()
        figure_refs = make_aggregate_figures(paths, eval_df, metrics, main_results, patch_df, subset)

    write_final_report(paths, sample_mode, main_results, metrics, patch_df, patch_summary, figure_refs)
    print(f"[完成] 聚合报告已生成：{paths.report_dir / 'MLISE2026_qwen_scaling_report.md'}", flush=True)


def set_chinese_plot_style() -> None:
    global plt, sns
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    import seaborn as sns_module

    plt = plt_module
    sns = sns_module
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def make_aggregate_figures(
    paths: RunPaths,
    eval_df: pd.DataFrame,
    metrics: dict[str, pd.DataFrame],
    main_results: pd.DataFrame,
    patch_df: pd.DataFrame,
    subset: pd.DataFrame,
) -> list[tuple[str, str]]:
    set_chinese_plot_style()
    refs: list[tuple[str, str]] = []
    model_order_names = [MODEL_CONFIGS[key]["display_name"] for key in MODEL_ORDER]

    if not subset.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        composition = subset.groupby(["query_type", "label"]).size().reset_index(name="count")
        sns.barplot(data=composition, x="query_type", y="count", hue="label", ax=ax)
        ax.set_title("CLadder 主评测子集标签分布")
        ax.set_xlabel("Query Type")
        ax.set_ylabel("样本数")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = paths.figure_dir / "01_dataset_statistics.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("CLadder 主评测子集标签分布", rel_to_run(paths, path)))

    by_rung = metrics["by_rung"].copy()
    if not by_rung.empty:
        by_rung["model_display_name"] = pd.Categorical(
            by_rung["model_display_name"],
            categories=model_order_names,
            ordered=True,
        )
        grid = sns.catplot(
            data=by_rung,
            kind="bar",
            x="model_display_name",
            y="accuracy",
            hue="rung",
            col="prompt_mode",
            height=4,
            aspect=1.1,
        )
        grid.set_axis_labels("模型", "Accuracy")
        grid.set_titles("Prompt: {col_name}")
        grid.fig.suptitle("不同模型规模与 Rung 下的准确率", y=1.05)
        for ax in grid.axes.flat:
            ax.tick_params(axis="x", rotation=20)
        path = paths.figure_dir / "02_accuracy_by_model_and_rung.png"
        grid.fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(grid.fig)
        refs.append(("不同模型规模与 Rung 下的准确率", rel_to_run(paths, path)))

    pcc_overall = metrics["metrics_pcc_overall"].copy()
    if not pcc_overall.empty:
        pcc_overall["model_display_name"] = pd.Categorical(
            pcc_overall["model_display_name"],
            categories=model_order_names,
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pcc_overall, x="model_display_name", y="pcc", hue="prompt_mode", ax=ax)
        ax.set_title("不同模型规模下的 Pairwise Causal Consistency")
        ax.set_xlabel("模型")
        ax.set_ylabel("PCC")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = paths.figure_dir / "03_pcc_by_model.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("不同模型规模下的 Pairwise Causal Consistency", rel_to_run(paths, path)))

    if not main_results.empty and "pcc" in main_results:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=main_results,
            x="accuracy",
            y="pcc",
            hue="model_display_name",
            style="prompt_mode",
            s=100,
            ax=ax,
        )
        ax.set_title("Accuracy 与 PCC 的关系")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("PCC")
        fig.tight_layout()
        path = paths.figure_dir / "04_accuracy_vs_pcc.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Accuracy 与 PCC 的关系", rel_to_run(paths, path)))

    story = metrics["metrics_story_all_correct"].copy()
    if not story.empty:
        story["model_display_name"] = pd.Categorical(
            story["model_display_name"],
            categories=model_order_names,
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=story, x="model_display_name", y="story_all_correct_rate", hue="prompt_mode", ax=ax)
        ax.set_title("Story All-Correct Rate")
        ax.set_xlabel("模型")
        ax.set_ylabel("Story 全对率")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = paths.figure_dir / "05_story_all_correct.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Story All-Correct Rate", rel_to_run(paths, path)))

    if not patch_df.empty:
        heat_data = patch_df.groupby(["pair_id", "layer"])["recovery"].mean().reset_index()
        heat = heat_data.pivot(index="pair_id", columns="layer", values="recovery")
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(heat.index))))
        sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Residual Stream Patching 恢复热力图")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Clean→Corrupted Pair")
        fig.tight_layout()
        path = paths.figure_dir / "06_residual_patching_heatmap.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Residual Stream Patching 恢复热力图", rel_to_run(paths, path)))

    return refs


def rel_to_run(paths: RunPaths, path: Path) -> str:
    return str(path.relative_to(paths.run_dir))


def format_pct(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.1f}%"


def write_model_behavior_report(paths: RunPaths, model_key: str, eval_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = save_behavior_metrics(eval_df, out_dir)
    summary = metrics["summary"]
    pcc = metrics["metrics_pcc"]
    story = metrics["metrics_story_all_correct"]
    label_dist = metrics["label_distribution"]
    lines = [
        f"# {MODEL_CONFIGS[model_key]['display_name']} 行为评测记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型路径：`{MODEL_CONFIGS[model_key]['path']}`",
        f"- 样本数：`{eval_df['sample_id'].nunique()}`",
        f"- prompt 设置：`direct`、`structured`",
        f"- 解析规则：自动正则解析，不调用外部模型修补答案。",
        "",
        "## 主结果",
        "",
        summary.to_markdown(index=False) if not summary.empty else "暂无结果。",
        "",
        "## PCC",
        "",
        pcc.to_markdown(index=False) if not pcc.empty else "暂无 PCC 结果。",
        "",
        "## Story All-Correct Rate",
        "",
        story.to_markdown(index=False) if not story.empty else "暂无 Story 结果。",
        "",
        "## 输出标签分布",
        "",
        label_dist.to_markdown(index=False) if not label_dist.empty else "暂无标签分布。",
        "",
    ]
    (out_dir / "behavior_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_patch_report(
    paths: RunPaths,
    model_key: str,
    patch_df: pd.DataFrame,
    patch_status: str,
    candidate_source: str,
    selected_pairs: list[dict[str, Any]],
) -> None:
    patch_dir = paths.patch_dir / model_key
    if patch_df.empty:
        summary_text = "本轮未生成可分析的 patching 结果。"
        summary_table = pd.DataFrame()
    else:
        positive_pairs = int((patch_df.groupby("pair_id")["recovery"].max() > 0).sum())
        summary_text = (
            f"本轮共得到 `{patch_df['pair_id'].nunique()}` 对 clean/corrupted pair，"
            f"其中最大 recovery 为正的 pair 数为 `{positive_pairs}`。"
        )
        summary_table = (
            patch_df.groupby(["model", "model_display_name", "method"])
            .agg(
                n_rows=("pair_id", "count"),
                n_pairs=("pair_id", "nunique"),
                mean_recovery=("recovery", "mean"),
                max_recovery=("recovery", "max"),
                mean_corrupted_gold_recovery=("corrupted_gold_recovery", "mean"),
            )
            .reset_index()
        )
        summary_table.to_csv(patch_dir / "patching_summary.csv", index=False)

    lines = [
        f"# {MODEL_CONFIGS[model_key]['display_name']} Residual Stream Patching 记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型路径：`{MODEL_CONFIGS[model_key]['path']}`",
        f"- 候选来源：`{candidate_source}`",
        f"- TransformerLens 状态：`{patch_status}`",
        f"- 选中 pair 数：`{len(selected_pairs)}`",
        "",
        "## 结果概述",
        "",
        summary_text,
        "",
        "本实验报告中的 `recovery` 表示把 clean 样本的 residual stream 激活写入 corrupted 样本后，输出 logits 向 clean 样本正确答案方向移动的幅度。"
        "同时保留 `corrupted_gold_recovery` 字段，用于记录该 patch 对 corrupted 样本自身 gold label 的影响，避免过度解释。",
        "",
        "## 汇总表",
        "",
        summary_table.to_markdown(index=False) if not summary_table.empty else "暂无汇总表。",
        "",
    ]
    (patch_dir / "patching_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    paths: RunPaths,
    sample_mode: str,
    main_results: pd.DataFrame,
    metrics: dict[str, pd.DataFrame],
    patch_df: pd.DataFrame,
    patch_summary: pd.DataFrame,
    figure_refs: list[tuple[str, str]],
) -> None:
    env = collect_environment_info()
    pcc = metrics["metrics_pcc"]
    story = metrics["metrics_story_all_correct"]
    by_query = metrics["by_query"]
    label_dist = metrics["label_distribution"]

    lines = [
        "# MLISE 2026 Qwen 家族因果一致性实验报告",
        "",
        "## 1. 实验概述",
        "",
        f"- 实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本模式：`{sample_mode}`",
        "- 实验目标：比较 Qwen3 同一家族不同规模模型在 CLadder 因果推理任务上的准确率、因果一致性和 story 级稳定性，并在代表性模型上做轻量 residual stream activation patching。",
        "- 报告语言：中文。",
        "",
        "## 2. 服务器环境",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- {key}：`{value}`")
    lines.extend(
        [
            "",
            "## 3. 数据集与任务设置",
            "",
            "本实验使用 `CLadder full_v1.5_default` 中的分层平衡抽样子集。每道题是一个 yes/no 形式的因果推理问题，模型需要根据题干中的因果结构、概率信息或反事实条件给出最终判断。",
            "",
            "抽样时在每个 query type 内保持 yes/no 标签平衡。quick 版本用于服务器快速验证，formal 版本用于会议稿正式结果。",
            "",
            "## 4. 模型设置",
            "",
        ]
    )
    for model_key in MODEL_ORDER:
        model_cfg = MODEL_CONFIGS[model_key]
        lines.append(f"- {model_cfg['display_name']}：`{model_cfg['path']}`")
    lines.extend(
        [
            "",
            "## 5. Prompt 设置与解析规则",
            "",
            "主实验只保留两种 prompt：`direct` 和 `structured`。模型输出只使用自动规则解析：优先提取 `Final answer: yes/no`，其次提取输出文本中的独立 yes/no；无法解析时记为 invalid。主实验不调用 DeepSeek 或其他外部模型修补答案。",
            "",
            "## 6. 指标定义",
            "",
            "- `Accuracy`：单题预测是否等于 CLadder oracle label。",
            "- `Pairwise Causal Consistency (PCC)`：同一 story 下成对样本的预测关系是否与 oracle label 的保持或翻转关系一致。",
            "- `Story All-Correct Rate`：同一 story 中被抽中的题目是否全部答对。",
            "- `Residual patching recovery`：将 clean 样本 residual stream 激活写入 corrupted 样本后，clean 目标答案方向的 logit margin 变化。",
            "",
            "## 7. 主结果表",
            "",
            main_results.to_markdown(index=False) if not main_results.empty else "暂无主结果。",
            "",
            "## 8. PCC 结果",
            "",
            pcc.to_markdown(index=False) if not pcc.empty else "暂无 PCC 结果。",
            "",
            "## 9. Story All-Correct Rate",
            "",
            story.to_markdown(index=False) if not story.empty else "暂无 Story 结果。",
            "",
            "## 10. Query Type 结果",
            "",
            by_query.to_markdown(index=False) if not by_query.empty else "暂无 query type 结果。",
            "",
            "## 11. 输出标签分布",
            "",
            label_dist.to_markdown(index=False) if not label_dist.empty else "暂无标签分布。",
            "",
            "## 12. 白盒 Patching 结果",
            "",
        ]
    )
    if patch_df.empty:
        lines.append("当前 run 尚未生成 patching 结果。")
    else:
        lines.extend(
            [
                "本轮白盒分析只做 residual stream patching，作为行为结果之外的轻量内部信号检查。该结果不能解释为完整 causal circuit，只能说明部分隐藏状态中可能存在与答案方向相关的可恢复信号。",
                "",
                patch_summary.to_markdown(index=False) if not patch_summary.empty else "暂无 patching 汇总表。",
            ]
        )
    lines.extend(["", "## 13. 图表索引", ""])
    if figure_refs:
        for title, rel_path in figure_refs:
            lines.extend([f"### {title}", "", f"![{title}](../{rel_path})", ""])
    else:
        lines.append("当前未生成图表。")

    lines.extend(
        [
            "",
            "## 14. 中文实验描述草稿",
            "",
            "本文在 CLadder 因果推理基准上评估 Qwen3 家族模型的因果一致性。与只统计单题准确率不同，实验进一步计算同一 story 下不同 causal condition 之间的 Pairwise Causal Consistency，以及更严格的 Story All-Correct Rate。这样的设置可以检验模型是否只是偶然答对个别问题，还是能在共享因果结构的一组问题中保持稳定判断。",
            "",
            "隐藏层分析部分采用 residual stream activation patching。我们首先筛选 clean/corrupted 样本对，其中 clean 样本是模型答对的一侧，corrupted 样本是模型答错的一侧。随后在 corrupted 样本前向传播时，将某一层 residual stream 激活替换为 clean 样本对应层激活，并观察 yes/no logit margin 是否向 clean 样本正确答案方向恢复。该分析只作为探索性补充证据，不声称发现完整因果推理回路。",
            "",
            "## 15. 初步结论与下一步",
            "",
            "正式结论需要结合服务器完整结果填写。若更大模型在 Accuracy、PCC 或 Story All-Correct Rate 上显著高于 0.6B，可将论文主线写成模型规模带来部分因果一致性改善；若提升不明显，则可将主线转为通用规模扩展并不自动保证因果一致性，因果评测仍需要专门指标。",
            "",
            "下一步建议：检查 formal 版本三模型是否完整，核对 parse rate 和标签偏置；若 patching pair 不足，优先改用 Qwen3-8B 或扩大 candidate pool。",
            "",
        ]
    )
    (paths.report_dir / "MLISE2026_qwen_scaling_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLISE 2026 Qwen3 × CLadder 服务器端实验脚本")
    parser.add_argument("--stage", choices=["sample", "behavior", "patch", "aggregate", "all"], default="all")
    parser.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--sample-mode", choices=["quick", "formal"], default="quick")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--patch-model", choices=MODEL_ORDER, default="qwen3_4b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--patch-pair-limit", type=int, default=16)
    parser.add_argument("--patch-candidate-pool", type=int, default=48)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_id:
        args.run_id = f"{args.sample_mode}_{timestamp()}"
    seed_everything(args.seed)

    paths = get_paths(Path(args.output_root), args.run_id)
    ensure_dirs(paths)
    config = EvalConfig(
        stage=args.stage,
        model=args.model,
        sample_mode=args.sample_mode,
        run_id=args.run_id,
        data_path=args.data_path,
        output_root=args.output_root,
        patch_model=args.patch_model,
        seed=args.seed,
        batch_size=args.batch_size,
        patch_batch_size=args.patch_batch_size,
        max_new_tokens=args.max_new_tokens,
        patch_pair_limit=args.patch_pair_limit,
        patch_candidate_pool=args.patch_candidate_pool,
        resume=args.resume,
        skip_figures=args.skip_figures,
    )
    save_json(paths.run_dir / "config.json", asdict(config))
    write_environment_report(paths, config)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集不存在：{data_path}")

    model_keys = model_keys_from_arg(args.model)

    if args.stage in {"sample", "all"}:
        run_sample_stage(paths, data_path, args.sample_mode, args.seed)

    if args.stage in {"behavior", "all"}:
        run_behavior_stage(paths, args, model_keys, config)

    if args.stage in {"patch", "all"}:
        run_patch_stage(paths, args, config)

    if args.stage in {"aggregate", "all"}:
        aggregate_results(paths, args.sample_mode, args.skip_figures)


if __name__ == "__main__":
    main()
