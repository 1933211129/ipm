#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
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

import numpy as np
import pandas as pd

plt = None
sns = None
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "datasets" / "cladder" / "data"
DATA_PATH = DATA_DIR / "full_v1.5_default.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "mlise2026_qwen_final"
SEED = 42

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

MODEL_ORDER = ["qwen3_0_6b", "qwen3_4b", "qwen3_8b"]
MAIN_PROMPT_MODES = ["nl", "nl_formal", "formula_only"]
ABLATION_PROMPT_MODES = ["nl_var_query", "nl_var_graph"]
STRESS_PROMPT_MODES = ["nl", "nl_formal"]
PROMPT_LABELS = {
    "nl": "Natural Language",
    "nl_var_query": "NL + Variables + Formal Query",
    "nl_var_graph": "NL + Variables + Graph",
    "nl_formal": "NL + Formal Scaffold",
    "formula_only": "Formalized Input",
}
SCAFFOLD_MODES = ["nl_var_query", "nl_var_graph", "nl_formal", "formula_only"]
STRESS_SPLITS = {
    "commonsense": DATA_DIR / "test-commonsense-v1.5.csv",
    "anticommonsense": DATA_DIR / "test-anticommonsense-v1.5.csv",
    "noncommonsense": DATA_DIR / "test-noncommonsense-v1.5.csv",
    "easy": DATA_DIR / "test-easy-v1.5.csv",
    "hard": DATA_DIR / "test-hard-v1.5.csv",
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
class DiagnosticConfig:
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
    stress_sample_size: int
    patch_pair_limit: int
    bootstrap: int
    resume: bool
    skip_figures: bool


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_model_runtime() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer
    if torch is None:
        import torch as torch_module

        torch = torch_module
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        from transformers import AutoModelForCausalLM as hf_model_cls
        from transformers import AutoTokenizer as hf_tokenizer_cls

        AutoModelForCausalLM = hf_model_cls
        AutoTokenizer = hf_tokenizer_cls


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


def selected_main_path(paths: RunPaths, sample_mode: str) -> Path:
    return paths.table_dir / f"selected_main_subset_{sample_mode}.csv"


def selected_stress_path(paths: RunPaths, sample_mode: str, stress_sample_size: int) -> Path:
    return paths.table_dir / f"selected_stress_subset_{sample_mode}_{stress_sample_size}.csv"


def load_cladder_df(path: Path, dataset_variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(int)
    df["rung"] = df["rung"].astype(int)
    df["label"] = df["label"].str.lower().str.strip()
    if "question_property" not in df.columns:
        df["question_property"] = ""
    df["dataset_variant"] = dataset_variant
    return df


def balanced_label_sample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    if n_total % 2 != 0:
        raise ValueError(f"样本数必须为偶数，当前为 {n_total}")
    half = n_total // 2
    if len(df[df["label"] == "yes"]) < half or len(df[df["label"] == "no"]) < half:
        raise ValueError("yes/no 数量不足，无法完成平衡抽样")
    yes_df = df[df["label"] == "yes"].sample(n=half, random_state=seed)
    no_df = df[df["label"] == "no"].sample(n=half, random_state=seed + 1009)
    return pd.concat([yes_df, no_df], ignore_index=True)


def allocate_even_quotas(labels: list[str], total: int) -> dict[str, int]:
    """把 split 样本量分配到 query type，同时保持每个 query type 的 yes/no 平衡。"""
    if total % 2 != 0:
        raise ValueError(f"stress 样本数必须为偶数，当前为 {total}")
    if not labels:
        raise ValueError("没有可用于抽样的 query type")
    base_pairs = total // 2 // len(labels)
    remaining_pairs = total // 2 - base_pairs * len(labels)
    quotas: dict[str, int] = {}
    for idx, label in enumerate(labels):
        quotas[label] = 2 * (base_pairs + (1 if idx < remaining_pairs else 0))
    return quotas


def select_main_subset(df: pd.DataFrame, sample_mode: str, seed: int) -> pd.DataFrame:
    chunks = []
    for idx, (query_type, n_total) in enumerate(SAMPLE_QUOTAS[sample_mode].items()):
        pool = df[df["query_type"] == query_type].copy()
        chunk = balanced_label_sample(pool, n_total=n_total, seed=seed + idx)
        chunks.append(chunk)
    subset = pd.concat(chunks, ignore_index=True)
    subset = subset.sort_values(["query_type", "label", "story_id", "rung", "id"]).reset_index(drop=True)
    subset["subset"] = sample_mode
    subset["diagnostic_source"] = "main"
    return subset


def select_stress_subset(sample_mode: str, stress_sample_size: int, seed: int) -> pd.DataFrame:
    chunks = []
    for split_idx, (split_name, split_path) in enumerate(STRESS_SPLITS.items()):
        split_df = load_cladder_df(split_path, split_name)
        query_types = sorted(split_df["query_type"].dropna().unique())
        quotas = allocate_even_quotas(query_types, stress_sample_size)
        for query_idx, query_type in enumerate(query_types):
            pool = split_df[split_df["query_type"] == query_type].copy()
            chunk = balanced_label_sample(
                pool,
                n_total=quotas[query_type],
                seed=seed + split_idx * 100 + query_idx,
            )
            chunks.append(chunk)
    subset = pd.concat(chunks, ignore_index=True)
    subset = subset.sort_values(["dataset_variant", "query_type", "label", "story_id", "rung", "id"]).reset_index(drop=True)
    subset["subset"] = sample_mode
    subset["diagnostic_source"] = "stress"
    return subset


def dataset_statistics(subset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_split = (
        subset.groupby(["diagnostic_source", "dataset_variant", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    by_query = (
        subset.groupby(["diagnostic_source", "dataset_variant", "query_type", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    by_rung = (
        subset.groupby(["diagnostic_source", "dataset_variant", "rung", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return {
        "dataset_stats_by_split": by_split,
        "dataset_stats_by_query": by_query,
        "dataset_stats_by_rung": by_rung,
    }


def save_sample_files(paths: RunPaths, main_subset: pd.DataFrame, stress_subset: pd.DataFrame, args: argparse.Namespace) -> None:
    main_subset.to_csv(selected_main_path(paths, args.sample_mode), index=False)
    stress_subset.to_csv(selected_stress_path(paths, args.sample_mode, args.stress_sample_size), index=False)
    main_subset.to_csv(paths.table_dir / "selected_main_subset.csv", index=False)
    stress_subset.to_csv(paths.table_dir / "selected_stress_subset.csv", index=False)
    all_subset = pd.concat([main_subset, stress_subset], ignore_index=True)
    for name, table in dataset_statistics(all_subset).items():
        table.to_csv(paths.table_dir / f"{name}_{args.sample_mode}.csv", index=False)


def load_or_create_main_subset(paths: RunPaths, args: argparse.Namespace) -> pd.DataFrame:
    path = selected_main_path(paths, args.sample_mode)
    if path.exists():
        return load_cladder_df(path, "main")
    df = load_cladder_df(Path(args.data_path), "main")
    subset = select_main_subset(df, args.sample_mode, args.seed)
    subset.to_csv(path, index=False)
    return subset


def load_or_create_stress_subset(paths: RunPaths, args: argparse.Namespace) -> pd.DataFrame:
    path = selected_stress_path(paths, args.sample_mode, args.stress_sample_size)
    if path.exists():
        subset = pd.read_csv(path)
        subset["id"] = subset["id"].astype(int)
        subset["rung"] = subset["rung"].astype(int)
        subset["label"] = subset["label"].str.lower().str.strip()
        return subset
    subset = select_stress_subset(args.sample_mode, args.stress_sample_size, args.seed)
    subset.to_csv(path, index=False)
    return subset


def extract_scaffold(row: dict[str, Any]) -> dict[str, str]:
    lines = [line.strip() for line in str(row.get("reasoning", "")).splitlines() if line.strip()]
    variables = lines[0] if len(lines) >= 1 else ""
    graph = lines[1] if len(lines) >= 2 else ""
    formal_query = str(row.get("formal_form", "")).strip() or (lines[2] if len(lines) >= 3 else "")
    return {
        "variables": variables,
        "graph": graph,
        "formal_query": formal_query,
    }


def extract_question_sentence(prompt: str) -> str:
    prompt = str(prompt).strip()
    qidx = prompt.rfind("?")
    if qidx < 0:
        return prompt[-500:].strip()
    prefix = prompt[: qidx + 1]
    starts = [prefix.rfind(". "), prefix.rfind("\n"), prefix.rfind(": ")]
    start = max(starts)
    if start >= 0 and qidx - start < 500:
        return prefix[start + 1 :].strip()
    return prefix[-500:].strip()


def extract_condition_block(prompt: str) -> str:
    prompt = str(prompt).strip()
    question = extract_question_sentence(prompt)
    idx = prompt.rfind(question)
    if idx > 0:
        return prompt[:idx].strip()
    qidx = prompt.rfind("?")
    if qidx > 0:
        return prompt[:qidx].strip()
    return prompt


def build_user_prompt(row: dict[str, Any], mode: str) -> str:
    base_prompt = str(row["prompt"]).strip()
    scaffold = extract_scaffold(row)
    question = extract_question_sentence(base_prompt)
    condition_block = extract_condition_block(base_prompt)
    if mode == "nl":
        return (
            "You are answering a causal yes/no question.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}"
        )
    if mode == "nl_formal":
        return (
            "You are answering a causal yes/no question.\n"
            "Use the natural-language question and the provided causal scaffold.\n"
            "The scaffold contains only variables, graph structure, and query form; it does not contain the answer.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}\n\n"
            "Causal scaffold:\n"
            f"Variables: {scaffold['variables']}\n"
            f"Graph: {scaffold['graph']}\n"
            f"Formal query: {scaffold['formal_query']}"
        )
    if mode == "nl_var_query":
        return (
            "You are answering a causal yes/no question.\n"
            "Use the natural-language question, variable mapping, and formal query. No causal graph is provided.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}\n\n"
            "Formal components:\n"
            f"Variables: {scaffold['variables']}\n"
            f"Formal query: {scaffold['formal_query']}"
        )
    if mode == "nl_var_graph":
        return (
            "You are answering a causal yes/no question.\n"
            "Use the natural-language question, variable mapping, and causal graph. No formal query is provided.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}\n\n"
            "Formal components:\n"
            f"Variables: {scaffold['variables']}\n"
            f"Graph: {scaffold['graph']}"
        )
    if mode == "formula_only":
        return (
            "You are answering a formalized causal yes/no question.\n"
            "Use the factual conditions, variable map, causal graph, and formal query. Do not use any external knowledge.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Factual conditions:\n{condition_block}\n\n"
            f"Variables: {scaffold['variables']}\n"
            f"Causal graph: {scaffold['graph']}\n"
            f"Query type: {row.get('query_type', '')}\n"
            f"Rung: {row.get('rung', '')}\n"
            f"Formal query: {scaffold['formal_query']}\n\n"
            f"Question sentence:\n{question}"
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
        torch_module.cuda.empty_cache()


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


def run_condition_eval(
    evaluator: QwenEvaluator,
    df: pd.DataFrame,
    model_key: str,
    run_id: str,
    modes: list[str],
    batch_size: int,
    save_path: Path,
    resume: bool,
) -> pd.DataFrame:
    existing = pd.DataFrame()
    completed: set[tuple[str, int, str]] = set()
    if resume and save_path.exists():
        existing = pd.read_csv(save_path)
        if not existing.empty:
            completed = set(
                zip(
                    existing["dataset_variant"].astype(str),
                    existing["sample_id"].astype(int),
                    existing["prompt_mode"].astype(str),
                )
            )
            print(f"[信息] resume 已读取 {len(existing)} 行：{save_path}", flush=True)
    elif save_path.exists() and not resume:
        raise FileExistsError(f"结果文件已存在，避免覆盖：{save_path}。如需续跑请加 --resume。")

    all_records: list[dict[str, Any]] = []
    for mode in modes:
        source_rows = [
            row
            for row in df.to_dict("records")
            if (str(row["dataset_variant"]), int(row["id"]), mode) not in completed
        ]
        print(
            f"[进度] model={model_key} source={df['diagnostic_source'].iloc[0]} mode={mode} 待评测={len(source_rows)}",
            flush=True,
        )
        batch_size_current = batch_size
        i = 0
        while i < len(source_rows):
            current_rows = source_rows[i : i + batch_size_current]
            prompts = [build_user_prompt(row, mode) for row in current_rows]
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
                    "sample_id": int(source_row["id"]),
                    "story_id": source_row["story_id"],
                    "graph_id": source_row["graph_id"],
                    "rung": int(source_row["rung"]),
                    "query_type": source_row["query_type"],
                    "question_property": source_row.get("question_property", ""),
                    "formal_form": source_row["formal_form"],
                    "gold_label": source_row["label"],
                    "prompt_mode": mode,
                    "prompt_condition": PROMPT_LABELS.get(mode, mode),
                    "dataset_variant": source_row["dataset_variant"],
                    "diagnostic_source": source_row["diagnostic_source"],
                    "subset": source_row.get("subset", ""),
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
    return info


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无结果。"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```csv\n" + df.to_csv(index=False) + "```"


def compute_summary_metrics(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if eval_df.empty:
        empty = pd.DataFrame()
        return {
            "metrics_summary": empty,
            "metrics_by_query": empty,
            "metrics_by_rung": empty,
            "metrics_label_distribution": empty,
        }
    summary = (
        eval_df.groupby(
            ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode", "prompt_condition"],
            dropna=False,
        )
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            parse_rate=("is_invalid", lambda x: 1 - x.mean()),
            invalid_rate=("is_invalid", "mean"),
            latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )
    by_query = (
        eval_df.groupby(
            ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode", "query_type"],
            dropna=False,
        )
        .agg(n=("sample_id", "count"), accuracy=("is_correct", "mean"), parse_rate=("is_invalid", lambda x: 1 - x.mean()))
        .reset_index()
    )
    by_rung = (
        eval_df.groupby(
            ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode", "rung"],
            dropna=False,
        )
        .agg(n=("sample_id", "count"), accuracy=("is_correct", "mean"), parse_rate=("is_invalid", lambda x: 1 - x.mean()))
        .reset_index()
    )
    label_distribution = (
        eval_df.groupby(["model", "dataset_variant", "prompt_mode", "parsed_label"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    return {
        "metrics_summary": summary,
        "metrics_by_query": by_query,
        "metrics_by_rung": by_rung,
        "metrics_label_distribution": label_distribution,
    }


def compute_strict_ccc(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pair_rows = []
    valid_labels = {"yes", "no"}
    group_cols = ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"]
    for group_key, mode_df in eval_df.groupby(group_cols, dropna=False):
        model, display, source, variant, mode = group_key
        for contrast_key, contrast_df in mode_df.groupby(["story_id", "query_type", "formal_form"], dropna=False):
            rows = contrast_df.to_dict("records")
            for left, right in itertools.combinations(rows, 2):
                if left["gold_label"] == right["gold_label"]:
                    continue
                left_valid = left["parsed_label"] in valid_labels
                right_valid = right["parsed_label"] in valid_labels
                valid_pair = left_valid and right_valid
                pred_opposite = valid_pair and left["parsed_label"] != right["parsed_label"]
                correct_flip = (
                    valid_pair
                    and left["parsed_label"] == left["gold_label"]
                    and right["parsed_label"] == right["gold_label"]
                )
                wrong_flip = (
                    valid_pair
                    and left["parsed_label"] != left["gold_label"]
                    and right["parsed_label"] != right["gold_label"]
                )
                invariant_yes = valid_pair and left["parsed_label"] == right["parsed_label"] == "yes"
                invariant_no = valid_pair and left["parsed_label"] == right["parsed_label"] == "no"
                pair_rows.append(
                    {
                        "model": model,
                        "model_display_name": display,
                        "diagnostic_source": source,
                        "dataset_variant": variant,
                        "prompt_mode": mode,
                        "story_id": contrast_key[0],
                        "query_type": contrast_key[1],
                        "formal_form": contrast_key[2],
                        "left_sample_id": int(left["sample_id"]),
                        "right_sample_id": int(right["sample_id"]),
                        "has_invalid": not valid_pair,
                        "strict_contrast_consistent": int(pred_opposite),
                        "correct_flip": int(correct_flip),
                        "wrong_flip": int(wrong_flip),
                        "invariant_yes": int(invariant_yes),
                        "invariant_no": int(invariant_no),
                        "left_gold_label": left["gold_label"],
                        "right_gold_label": right["gold_label"],
                        "left_parsed_label": left["parsed_label"],
                        "right_parsed_label": right["parsed_label"],
                    }
                )
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        empty = pd.DataFrame()
        return {"metrics_ccc_pairs": empty, "metrics_ccc": empty, "metrics_ccc_by_query": empty}

    def aggregate_pairs(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        rows = []
        for key, group in frame.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            valid = group[~group["has_invalid"].astype(bool)]
            n_valid = len(valid)
            correct_rate = float(valid["correct_flip"].mean()) if n_valid else np.nan
            wrong_rate = float(valid["wrong_flip"].mean()) if n_valid else np.nan
            rows.append(
                {
                    **dict(zip(cols, key)),
                    "n_pairs": int(len(group)),
                    "n_valid_pairs": int(n_valid),
                    "strict_ccc": float(valid["strict_contrast_consistent"].mean()) if n_valid else np.nan,
                    "correct_flip_rate": correct_rate,
                    "wrong_flip_rate": wrong_rate,
                    "invariant_yes_rate": float(valid["invariant_yes"].mean()) if n_valid else np.nan,
                    "invariant_no_rate": float(valid["invariant_no"].mean()) if n_valid else np.nan,
                    "scca": correct_rate,
                    "signed_ccc": correct_rate - wrong_rate if n_valid else np.nan,
                    "invalid_pair_rate": float(group["has_invalid"].mean()) if len(group) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    ccc = aggregate_pairs(
        pair_df,
        ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"],
    )
    by_query = aggregate_pairs(
        pair_df,
        ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode", "query_type"],
    )
    return {"metrics_ccc_pairs": pair_df, "metrics_ccc": ccc, "metrics_ccc_by_query": by_query}


def compute_scaffold_gain(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base_cols = [col for col in summary.columns if col not in {"prompt_mode", "prompt_condition", "n", "accuracy", "parse_rate", "invalid_rate", "latency_sec"}]
    pivot = summary.pivot_table(index=base_cols, columns="prompt_mode", values="accuracy", aggfunc="first").reset_index()
    rows = []
    for _, row in pivot.iterrows():
        nl_acc = row.get("nl")
        for scaffold_mode in SCAFFOLD_MODES:
            if scaffold_mode not in row or pd.isna(row.get(scaffold_mode)) or pd.isna(nl_acc):
                continue
            rows.append(
                {
                    **{col: row[col] for col in base_cols},
                    "scaffold_mode": scaffold_mode,
                    "scaffold_condition": PROMPT_LABELS.get(scaffold_mode, scaffold_mode),
                    "nl_accuracy": nl_acc,
                    "scaffold_accuracy": row[scaffold_mode],
                    "scaffold_gain": row[scaffold_mode] - nl_acc,
                }
            )
    return pd.DataFrame(rows)


def compute_rescue_harm(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["model", "model_display_name", "diagnostic_source", "dataset_variant", "sample_id"]
    nl_df = eval_df[eval_df["prompt_mode"] == "nl"][key_cols + ["is_correct", "is_invalid"]].rename(
        columns={"is_correct": "nl_correct", "is_invalid": "nl_invalid"}
    )
    formal_df = eval_df[eval_df["prompt_mode"] == "nl_formal"][key_cols + ["is_correct", "is_invalid"]].rename(
        columns={"is_correct": "formal_correct", "is_invalid": "formal_invalid"}
    )
    merged = nl_df.merge(formal_df, on=key_cols, how="inner")
    for group_key, group in merged.groupby(["model", "model_display_name", "diagnostic_source", "dataset_variant"], dropna=False):
        model, display, source, variant = group_key
        valid = group[(~group["nl_invalid"].astype(bool)) & (~group["formal_invalid"].astype(bool))]
        nl_failures = int((valid["nl_correct"] == 0).sum())
        nl_successes = int((valid["nl_correct"] == 1).sum())
        rescue = int(((valid["nl_correct"] == 0) & (valid["formal_correct"] == 1)).sum())
        harm = int(((valid["nl_correct"] == 1) & (valid["formal_correct"] == 0)).sum())
        rows.append(
            {
                "model": model,
                "model_display_name": display,
                "diagnostic_source": source,
                "dataset_variant": variant,
                "n_valid_pairs": int(len(valid)),
                "rescue_count": rescue,
                "harm_count": harm,
                "rescue_rate_over_nl_failures": rescue / nl_failures if nl_failures else 0.0,
                "harm_rate_over_nl_successes": harm / nl_successes if nl_successes else 0.0,
                "net_rescue_rate": (rescue - harm) / len(valid) if len(valid) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_story_all_correct(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    story_items = (
        eval_df.groupby(
            ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode", "story_id"],
            dropna=False,
        )
        .agg(n_items=("sample_id", "count"), story_all_correct=("is_correct", "min"))
        .reset_index()
    )
    return (
        story_items.groupby(
            ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"],
            dropna=False,
        )
        .agg(
            n_stories=("story_id", "count"),
            mean_items_per_story=("n_items", "mean"),
            story_all_correct_rate=("story_all_correct", "mean"),
        )
        .reset_index()
    )


def compute_transition_patterns(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if eval_df.empty:
        empty = pd.DataFrame()
        return {"metrics_transition_patterns": empty, "metrics_transition_patterns_by_query": empty}
    required = ["nl", "nl_formal", "formula_only"]
    source = eval_df[eval_df["prompt_mode"].isin(required)].copy()
    key_cols = [
        "model",
        "model_display_name",
        "diagnostic_source",
        "dataset_variant",
        "sample_id",
        "query_type",
        "rung",
    ]
    pivot = source.pivot_table(index=key_cols, columns="prompt_mode", values="is_correct", aggfunc="first").reset_index()
    if any(mode not in pivot.columns for mode in required):
        empty = pd.DataFrame()
        return {"metrics_transition_patterns": empty, "metrics_transition_patterns_by_query": empty}
    pivot = pivot.dropna(subset=required)
    if pivot.empty:
        empty = pd.DataFrame()
        return {"metrics_transition_patterns": empty, "metrics_transition_patterns_by_query": empty}

    def cw(value: Any) -> str:
        return "C" if int(value) == 1 else "W"

    pivot["transition_pattern"] = pivot.apply(lambda row: "-".join(cw(row[mode]) for mode in required), axis=1)
    base_cols = ["model", "model_display_name", "diagnostic_source", "dataset_variant"]
    overall = pivot.groupby(base_cols + ["transition_pattern"], dropna=False).size().reset_index(name="n")
    totals = pivot.groupby(base_cols, dropna=False).size().reset_index(name="total")
    overall = overall.merge(totals, on=base_cols, how="left")
    overall["rate"] = overall["n"] / overall["total"]
    by_query = pivot.groupby(base_cols + ["query_type", "transition_pattern"], dropna=False).size().reset_index(name="n")
    query_totals = pivot.groupby(base_cols + ["query_type"], dropna=False).size().reset_index(name="total")
    by_query = by_query.merge(query_totals, on=base_cols + ["query_type"], how="left")
    by_query["rate"] = by_query["n"] / by_query["total"]
    return {"metrics_transition_patterns": overall, "metrics_transition_patterns_by_query": by_query}


def compute_stress_robustness(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    stress = eval_df[eval_df["diagnostic_source"] == "stress"].copy()
    if stress.empty:
        return pd.DataFrame()
    summary = compute_summary_metrics(stress)["metrics_summary"]
    rows = []
    base_cols = ["model", "model_display_name", "prompt_mode"]
    for key, group in summary.groupby(base_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        split_acc = {str(row["dataset_variant"]): float(row["accuracy"]) for _, row in group.iterrows()}
        acc_values = list(split_acc.values())
        row = {
            **dict(zip(base_cols, key)),
            **{f"{split}_accuracy": split_acc.get(split, np.nan) for split in STRESS_SPLITS},
            "mean_accuracy": float(np.mean(acc_values)) if acc_values else np.nan,
            "worst_split_accuracy": float(np.min(acc_values)) if acc_values else np.nan,
            "best_split_accuracy": float(np.max(acc_values)) if acc_values else np.nan,
            "std_across_splits": float(np.std(acc_values, ddof=0)) if acc_values else np.nan,
        }
        rows.append(row)
    robust = pd.DataFrame(rows)
    if robust.empty:
        return robust
    pivot = summary.pivot_table(
        index=["model", "model_display_name", "dataset_variant"],
        columns="prompt_mode",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    gain_rows = []
    for _, row in pivot.iterrows():
        if "nl" not in row or "nl_formal" not in row or pd.isna(row.get("nl")) or pd.isna(row.get("nl_formal")):
            continue
        gain_rows.append(
            {
                "model": row["model"],
                "model_display_name": row["model_display_name"],
                "dataset_variant": row["dataset_variant"],
                "nl_formal_gain": float(row["nl_formal"] - row["nl"]),
            }
        )
    gain_df = pd.DataFrame(gain_rows)
    if not gain_df.empty:
        gain_agg = (
            gain_df.groupby(["model", "model_display_name"], dropna=False)
            .agg(worst_split_scaffold_gain=("nl_formal_gain", "min"), mean_scaffold_gain=("nl_formal_gain", "mean"))
            .reset_index()
        )
        robust = robust.merge(gain_agg, on=["model", "model_display_name"], how="left")
    return robust


def bootstrap_mean_ci(values: Any, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if n_bootstrap <= 0 or len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for idx in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[idx] = sample.mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def bootstrap_diff_ci(left: Any, right: Any, n_bootstrap: int, seed: int) -> tuple[float, float]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    mask = ~np.isnan(left_arr) & ~np.isnan(right_arr)
    left_arr = left_arr[mask]
    right_arr = right_arr[mask]
    if n_bootstrap <= 0 or len(left_arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    indices = np.arange(len(left_arr))
    for idx in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        diffs[idx] = (right_arr[sample_idx] - left_arr[sample_idx]).mean()
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def compute_accuracy_bootstrap_ci(eval_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    if eval_df.empty or n_bootstrap <= 0:
        return pd.DataFrame()
    rows = []
    group_cols = ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"]
    for group_idx, (key, group) in enumerate(eval_df.groupby(group_cols, dropna=False)):
        if not isinstance(key, tuple):
            key = (key,)
        lo, hi = bootstrap_mean_ci(group["is_correct"].to_numpy(), n_bootstrap, seed + group_idx)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "metric": "accuracy",
                "estimate": float(group["is_correct"].mean()),
                "ci_low": lo,
                "ci_high": hi,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def compute_ccc_bootstrap_ci(pair_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    if pair_df.empty or n_bootstrap <= 0:
        return pd.DataFrame()
    rows = []
    group_cols = ["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"]
    metrics = {
        "strict_ccc": "strict_contrast_consistent",
        "correct_flip": "correct_flip",
        "wrong_flip": "wrong_flip",
        "scca": "correct_flip",
    }
    for group_idx, (key, group) in enumerate(pair_df.groupby(group_cols, dropna=False)):
        if not isinstance(key, tuple):
            key = (key,)
        valid = group[~group["has_invalid"].astype(bool)]
        if valid.empty:
            continue
        for metric_idx, (metric_name, col) in enumerate(metrics.items()):
            lo, hi = bootstrap_mean_ci(valid[col].to_numpy(), n_bootstrap, seed + group_idx * 17 + metric_idx)
            rows.append(
                {
                    **dict(zip(group_cols, key)),
                    "metric": metric_name,
                    "estimate": float(valid[col].mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_bootstrap": n_bootstrap,
                }
            )
        signed = valid["correct_flip"].to_numpy(dtype=float) - valid["wrong_flip"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(signed, n_bootstrap, seed + group_idx * 17 + 11)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "metric": "signed_ccc",
                "estimate": float(signed.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def mcnemar_p_value(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    statistic = (abs(b - c) - 1) ** 2 / total
    return float(math.erfc(math.sqrt(statistic / 2)))


def compute_paired_condition_tests(eval_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["model", "model_display_name", "diagnostic_source", "dataset_variant", "sample_id"]
    nl = eval_df[eval_df["prompt_mode"] == "nl"][key_cols + ["is_correct", "is_invalid"]].rename(
        columns={"is_correct": "nl_correct", "is_invalid": "nl_invalid"}
    )
    for mode_idx, mode in enumerate(SCAFFOLD_MODES):
        other = eval_df[eval_df["prompt_mode"] == mode][key_cols + ["is_correct", "is_invalid"]].rename(
            columns={"is_correct": "other_correct", "is_invalid": "other_invalid"}
        )
        if other.empty:
            continue
        merged = nl.merge(other, on=key_cols, how="inner")
        if merged.empty:
            continue
        for group_idx, (key, group) in enumerate(
            merged.groupby(["model", "model_display_name", "diagnostic_source", "dataset_variant"], dropna=False)
        ):
            if not isinstance(key, tuple):
                key = (key,)
            valid = group[(~group["nl_invalid"].astype(bool)) & (~group["other_invalid"].astype(bool))]
            if valid.empty:
                continue
            b = int(((valid["nl_correct"] == 1) & (valid["other_correct"] == 0)).sum())
            c = int(((valid["nl_correct"] == 0) & (valid["other_correct"] == 1)).sum())
            lo, hi = bootstrap_diff_ci(
                valid["nl_correct"].to_numpy(),
                valid["other_correct"].to_numpy(),
                n_bootstrap,
                seed + mode_idx * 101 + group_idx,
            )
            rows.append(
                {
                    **dict(zip(["model", "model_display_name", "diagnostic_source", "dataset_variant"], key)),
                    "comparison": f"nl_vs_{mode}",
                    "nl_accuracy": float(valid["nl_correct"].mean()),
                    "other_accuracy": float(valid["other_correct"].mean()),
                    "paired_accuracy_diff": float((valid["other_correct"] - valid["nl_correct"]).mean()),
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "mcnemar_b_nl_correct_other_wrong": b,
                    "mcnemar_c_nl_wrong_other_correct": c,
                    "mcnemar_p_approx": mcnemar_p_value(b, c),
                    "n_valid_pairs": int(len(valid)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def save_metrics(eval_df: pd.DataFrame, out_dir: Path, bootstrap: int = 0, seed: int = SEED) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_summary_metrics(eval_df)
    ccc = compute_strict_ccc(eval_df)
    summary = metrics["metrics_summary"]
    gain = compute_scaffold_gain(summary)
    gain_by_query = compute_scaffold_gain(metrics["metrics_by_query"])
    gain_by_rung = compute_scaffold_gain(metrics["metrics_by_rung"])
    rescue = compute_rescue_harm(eval_df)
    transition = compute_transition_patterns(eval_df)
    story = compute_story_all_correct(eval_df)
    stress_robustness = compute_stress_robustness(eval_df)
    accuracy_ci = compute_accuracy_bootstrap_ci(eval_df, bootstrap, seed)
    ccc_ci = compute_ccc_bootstrap_ci(ccc["metrics_ccc_pairs"], bootstrap, seed)
    paired_tests = compute_paired_condition_tests(eval_df, bootstrap, seed)
    all_metrics = {
        **metrics,
        **ccc,
        **transition,
        "metrics_scaffold_gain": gain,
        "metrics_scaffold_gain_by_query": gain_by_query,
        "metrics_scaffold_gain_by_rung": gain_by_rung,
        "metrics_rescue_harm": rescue,
        "metrics_story_all_correct": story,
        "metrics_stress_robustness": stress_robustness,
        "metrics_accuracy_bootstrap_ci": accuracy_ci,
        "metrics_ccc_bootstrap_ci": ccc_ci,
        "metrics_paired_condition_tests": paired_tests,
    }
    for name, table in all_metrics.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    if not ccc["metrics_ccc"].empty:
        ccc["metrics_ccc"].to_csv(out_dir / "metrics_pcc.csv", index=False)
    return all_metrics


def model_keys_from_arg(model_arg: str) -> list[str]:
    if model_arg == "all":
        return MODEL_ORDER.copy()
    return [model_arg]


def write_environment_report(paths: RunPaths, config: DiagnosticConfig) -> None:
    env = collect_environment_info()
    lines = [
        "# MLISE 2026 最终实验服务器环境记录",
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
            f"- 主数据集路径：`{config.data_path}`",
            f"- 输出根目录：`{config.output_root}`",
            f"- 主实验输入条件：`{', '.join(MAIN_PROMPT_MODES)}`",
            f"- stress split 输入条件：`{', '.join(STRESS_PROMPT_MODES)}`",
            "- 图表语言：英文。",
            "- Markdown 报告语言：中文。",
            "- 解析规则：优先解析 `Final answer: yes/no`，其次解析独立 yes/no，失败记为 invalid。",
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


def run_sample_stage(paths: RunPaths, args: argparse.Namespace) -> None:
    main_df = load_cladder_df(Path(args.data_path), "main")
    main_subset = select_main_subset(main_df, args.sample_mode, args.seed)
    stress_subset = select_stress_subset(args.sample_mode, args.stress_sample_size, args.seed)
    save_sample_files(paths, main_subset, stress_subset, args)
    all_subset = pd.concat([main_subset, stress_subset], ignore_index=True)
    stats = dataset_statistics(all_subset)
    lines = [
        "# MLISE 2026 最终实验抽样记录",
        "",
        f"- 抽样时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 主数据集路径：`{args.data_path}`",
        f"- 样本模式：`{args.sample_mode}`",
        f"- 主样本数：`{len(main_subset)}`",
        f"- stress split 总样本数：`{len(stress_subset)}`",
        f"- 每个 stress split 目标样本数：`{args.stress_sample_size}`",
        f"- 主样本输出：`{selected_main_path(paths, args.sample_mode)}`",
        f"- stress 样本输出：`{selected_stress_path(paths, args.sample_mode, args.stress_sample_size)}`",
        "",
        "## Split 标签分布",
        "",
        table_to_markdown(stats["dataset_stats_by_split"]),
        "",
        "## Query Type 标签分布",
        "",
        table_to_markdown(stats["dataset_stats_by_query"]),
        "",
    ]
    (paths.report_dir / "sample_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] 最终实验抽样完成：{paths.table_dir}", flush=True)


def run_behavior_stage(paths: RunPaths, args: argparse.Namespace, model_keys: list[str], config: DiagnosticConfig) -> None:
    subset = load_or_create_main_subset(paths, args)
    for model_key in model_keys:
        model_cfg = MODEL_CONFIGS[model_key]
        out_dir = model_table_dir(paths, model_key)
        save_json(out_dir / "config.json", {"运行配置": asdict(config), "模型配置": model_cfg})
        evaluator = QwenEvaluator(
            model_key=model_key,
            model_path=Path(model_cfg["path"]),
            hf_name=model_cfg["hf_name"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            eval_df = run_condition_eval(
                evaluator=evaluator,
                df=subset,
                model_key=model_key,
                run_id=args.run_id,
                modes=MAIN_PROMPT_MODES,
                batch_size=args.batch_size,
                save_path=out_dir / "main_eval.csv",
                resume=args.resume,
            )
            save_metrics(eval_df, out_dir, bootstrap=args.bootstrap, seed=args.seed)
            write_model_report(paths, model_key, out_dir, eval_df, "main")
        finally:
            evaluator.close()
        print(f"[完成] 主行为实验完成：{model_key}", flush=True)


def run_ablation_stage(paths: RunPaths, args: argparse.Namespace, model_keys: list[str], config: DiagnosticConfig) -> None:
    subset = load_or_create_main_subset(paths, args)
    for model_key in model_keys:
        model_cfg = MODEL_CONFIGS[model_key]
        out_dir = model_table_dir(paths, model_key)
        save_json(out_dir / "ablation_config.json", {"运行配置": asdict(config), "模型配置": model_cfg})
        evaluator = QwenEvaluator(
            model_key=model_key,
            model_path=Path(model_cfg["path"]),
            hf_name=model_cfg["hf_name"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            eval_df = run_condition_eval(
                evaluator=evaluator,
                df=subset,
                model_key=model_key,
                run_id=args.run_id,
                modes=ABLATION_PROMPT_MODES,
                batch_size=args.batch_size,
                save_path=out_dir / "ablation_eval.csv",
                resume=args.resume,
            )
            save_metrics(eval_df, out_dir / "ablation_metrics", bootstrap=args.bootstrap, seed=args.seed)
            write_model_report(paths, model_key, out_dir, eval_df, "ablation")
        finally:
            evaluator.close()
        print(f"[完成] 形式成分消融完成：{model_key}", flush=True)


def run_stress_stage(paths: RunPaths, args: argparse.Namespace, model_keys: list[str], config: DiagnosticConfig) -> None:
    subset = load_or_create_stress_subset(paths, args)
    for model_key in model_keys:
        model_cfg = MODEL_CONFIGS[model_key]
        out_dir = model_table_dir(paths, model_key)
        save_json(out_dir / "stress_config.json", {"运行配置": asdict(config), "模型配置": model_cfg})
        evaluator = QwenEvaluator(
            model_key=model_key,
            model_path=Path(model_cfg["path"]),
            hf_name=model_cfg["hf_name"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            eval_df = run_condition_eval(
                evaluator=evaluator,
                df=subset,
                model_key=model_key,
                run_id=args.run_id,
                modes=STRESS_PROMPT_MODES,
                batch_size=args.batch_size,
                save_path=out_dir / "stress_eval.csv",
                resume=args.resume,
            )
            save_metrics(eval_df, out_dir / "stress_metrics", bootstrap=args.bootstrap, seed=args.seed)
            write_model_report(paths, model_key, out_dir, eval_df, "stress")
        finally:
            evaluator.close()
        print(f"[完成] stress split 实验完成：{model_key}", flush=True)


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


def margin_from_logits(logits: Any, yes_id: int, no_id: int, target_label: str) -> float:
    if target_label == "yes":
        return float(logits[0, -1, yes_id] - logits[0, -1, no_id])
    if target_label == "no":
        return float(logits[0, -1, no_id] - logits[0, -1, yes_id])
    raise ValueError(f"未知 target label：{target_label}")


def select_patch_candidates(paths: RunPaths, args: argparse.Namespace, model_key: str) -> pd.DataFrame:
    eval_path = model_table_dir(paths, model_key) / "main_eval.csv"
    subset_path = selected_main_path(paths, args.sample_mode)
    if not eval_path.exists() or not subset_path.exists():
        return pd.DataFrame()
    eval_df = pd.read_csv(eval_path)
    subset = pd.read_csv(subset_path)
    key_cols = ["sample_id", "model", "model_display_name"]
    nl = eval_df[eval_df["prompt_mode"] == "nl"][key_cols + ["is_correct", "is_invalid", "parsed_label"]].rename(
        columns={
            "is_correct": "nl_correct",
            "is_invalid": "nl_invalid",
            "parsed_label": "nl_parsed_label",
        }
    )
    formal = eval_df[eval_df["prompt_mode"] == "nl_formal"][key_cols + ["is_correct", "is_invalid", "parsed_label"]].rename(
        columns={
            "is_correct": "formal_correct",
            "is_invalid": "formal_invalid",
            "parsed_label": "formal_parsed_label",
        }
    )
    merged = nl.merge(formal, on=key_cols, how="inner")
    candidates = merged[
        (merged["nl_correct"] == 0)
        & (merged["formal_correct"] == 1)
        & (~merged["nl_invalid"].astype(bool))
        & (~merged["formal_invalid"].astype(bool))
    ].copy()
    if candidates.empty:
        return candidates
    subset = subset.rename(columns={"id": "sample_id"})
    candidates = candidates.merge(subset, on="sample_id", how="left")
    candidates = candidates.sort_values(["query_type", "story_id", "sample_id"]).head(args.patch_pair_limit)
    return candidates.reset_index(drop=True)


def run_formal_to_natural_patching(evaluator: QwenEvaluator, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    yes_id, no_id = _yes_no_token_ids(evaluator.tokenizer)
    hf_model = evaluator.model
    tok = evaluator.tokenizer
    device = evaluator.device
    rows = []

    def encode_prompt(source_row: dict[str, Any], mode: str) -> dict[str, Any]:
        text = evaluator.format_batch([build_user_prompt(source_row, mode)])[0]
        encoded = tok(text, return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}

    for candidate_idx, row in enumerate(candidates.to_dict("records"), start=1):
        target_label = row["label"]
        natural_inputs = encode_prompt(row, "nl")
        formal_inputs = encode_prompt(row, "nl_formal")
        formal_cache: dict[int, Any] = {}

        def save_layer_output(layer_idx: int):
            def hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                formal_cache[layer_idx] = hidden.detach().clone()
            return hook

        handles = [
            hf_model.model.layers[layer_idx].register_forward_hook(save_layer_output(layer_idx))
            for layer_idx in range(len(hf_model.model.layers))
        ]
        with torch.inference_mode():
            formal_outputs = hf_model(**formal_inputs)
        for handle in handles:
            handle.remove()

        with torch.inference_mode():
            natural_outputs = hf_model(**natural_inputs)

        natural_margin = margin_from_logits(natural_outputs.logits, yes_id, no_id, target_label)
        formal_margin = margin_from_logits(formal_outputs.logits, yes_id, no_id, target_label)
        denom = formal_margin - natural_margin

        for layer_idx in range(len(hf_model.model.layers)):
            def patch_layer(_module, _inp, output, layer_idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                patched[:, -1:, :] = formal_cache[layer_idx][:, -1:, :]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = hf_model.model.layers[layer_idx].register_forward_hook(patch_layer)
            with torch.inference_mode():
                patched_outputs = hf_model(**natural_inputs)
            handle.remove()

            patched_margin = margin_from_logits(patched_outputs.logits, yes_id, no_id, target_label)
            absolute_recovery = patched_margin - natural_margin
            normalized_recovery = absolute_recovery / denom if abs(denom) > 1e-8 else None
            rows.append(
                {
                    "candidate_index": candidate_idx,
                    "sample_id": int(row["sample_id"]),
                    "story_id": row["story_id"],
                    "query_type": row["query_type"],
                    "rung": int(row["rung"]),
                    "gold_label": target_label,
                    "component": "residual",
                    "layer": layer_idx,
                    "natural_gold_margin": natural_margin,
                    "formal_gold_margin": formal_margin,
                    "patched_gold_margin": patched_margin,
                    "absolute_recovery": absolute_recovery,
                    "normalized_recovery": normalized_recovery,
                    "method": "hf_last_token_formal_to_natural",
                }
            )
        empty_device_cache()
    return pd.DataFrame(rows)


def run_random_patch_control(evaluator: QwenEvaluator, candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    if candidates.empty or len(candidates) < 2:
        return pd.DataFrame()
    yes_id, no_id = _yes_no_token_ids(evaluator.tokenizer)
    hf_model = evaluator.model
    tok = evaluator.tokenizer
    device = evaluator.device
    rng = random.Random(seed)
    candidate_records = candidates.to_dict("records")
    rows = []

    def encode_prompt(source_row: dict[str, Any], mode: str) -> dict[str, Any]:
        text = evaluator.format_batch([build_user_prompt(source_row, mode)])[0]
        encoded = tok(text, return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}

    for candidate_idx, row in enumerate(candidate_records, start=1):
        donors = [donor for donor in candidate_records if int(donor["sample_id"]) != int(row["sample_id"])]
        if not donors:
            continue
        donor = rng.choice(donors)
        target_label = row["label"]
        natural_inputs = encode_prompt(row, "nl")
        target_formal_inputs = encode_prompt(row, "nl_formal")
        donor_formal_inputs = encode_prompt(donor, "nl_formal")
        donor_cache: dict[int, Any] = {}

        def save_layer_output(layer_idx: int):
            def hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                donor_cache[layer_idx] = hidden.detach().clone()
            return hook

        handles = [
            hf_model.model.layers[layer_idx].register_forward_hook(save_layer_output(layer_idx))
            for layer_idx in range(len(hf_model.model.layers))
        ]
        with torch.inference_mode():
            _ = hf_model(**donor_formal_inputs)
        for handle in handles:
            handle.remove()

        with torch.inference_mode():
            natural_outputs = hf_model(**natural_inputs)
            target_formal_outputs = hf_model(**target_formal_inputs)

        natural_margin = margin_from_logits(natural_outputs.logits, yes_id, no_id, target_label)
        target_formal_margin = margin_from_logits(target_formal_outputs.logits, yes_id, no_id, target_label)
        denom = target_formal_margin - natural_margin

        for layer_idx in range(len(hf_model.model.layers)):
            def patch_layer(_module, _inp, output, layer_idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                patched[:, -1:, :] = donor_cache[layer_idx][:, -1:, :]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = hf_model.model.layers[layer_idx].register_forward_hook(patch_layer)
            with torch.inference_mode():
                patched_outputs = hf_model(**natural_inputs)
            handle.remove()

            patched_margin = margin_from_logits(patched_outputs.logits, yes_id, no_id, target_label)
            absolute_recovery = patched_margin - natural_margin
            normalized_recovery = absolute_recovery / denom if abs(denom) > 1e-8 else None
            rows.append(
                {
                    "candidate_index": candidate_idx,
                    "sample_id": int(row["sample_id"]),
                    "donor_sample_id": int(donor["sample_id"]),
                    "story_id": row["story_id"],
                    "query_type": row["query_type"],
                    "rung": int(row["rung"]),
                    "gold_label": target_label,
                    "component": "residual",
                    "layer": layer_idx,
                    "natural_gold_margin": natural_margin,
                    "target_formal_gold_margin": target_formal_margin,
                    "patched_gold_margin": patched_margin,
                    "absolute_recovery": absolute_recovery,
                    "normalized_recovery": normalized_recovery,
                    "method": "hf_last_token_random_formal_to_natural",
                }
            )
        empty_device_cache()
    return pd.DataFrame(rows)


def run_patch_stage(paths: RunPaths, args: argparse.Namespace, config: DiagnosticConfig) -> None:
    patch_model = args.patch_model
    model_cfg = MODEL_CONFIGS[patch_model]
    patch_out = paths.patch_dir / patch_model
    patch_out.mkdir(parents=True, exist_ok=True)
    result_path = patch_out / "formal_to_natural_patching_results.csv"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"patching 结果已存在，避免覆盖：{result_path}")
    candidates = select_patch_candidates(paths, args, patch_model)
    candidates.to_csv(patch_out / "formal_to_natural_patch_candidates.csv", index=False)
    if candidates.empty:
        pd.DataFrame().to_csv(result_path, index=False)
        write_patch_report(paths, patch_model, pd.DataFrame(), candidates)
        print(f"[警告] 未找到 natural 错、nl_formal 对的候选样本：{patch_model}", flush=True)
        return

    evaluator = QwenEvaluator(
        model_key=patch_model,
        model_path=Path(model_cfg["path"]),
        hf_name=model_cfg["hf_name"],
        batch_size=args.patch_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        patch_df = run_formal_to_natural_patching(evaluator, candidates)
        patch_df["model"] = patch_model
        patch_df["model_display_name"] = model_cfg["display_name"]
        patch_df["sample_mode"] = args.sample_mode
        patch_df.to_csv(result_path, index=False)
        write_patch_report(paths, patch_model, patch_df, candidates)
    finally:
        evaluator.close()
    print(f"[完成] formal-to-natural patching 完成：{patch_model}", flush=True)


def run_patch_control_stage(paths: RunPaths, args: argparse.Namespace, config: DiagnosticConfig) -> None:
    patch_model = args.patch_model
    model_cfg = MODEL_CONFIGS[patch_model]
    patch_out = paths.patch_dir / patch_model
    patch_out.mkdir(parents=True, exist_ok=True)
    result_path = patch_out / "random_patch_control_results.csv"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"random patch control 结果已存在，避免覆盖：{result_path}")
    candidates = select_patch_candidates(paths, args, patch_model)
    if candidates.empty or len(candidates) < 2:
        pd.DataFrame().to_csv(result_path, index=False)
        write_patch_control_report(paths, patch_model, pd.DataFrame(), candidates)
        print(f"[警告] random patch control 候选样本不足：{patch_model}", flush=True)
        return

    evaluator = QwenEvaluator(
        model_key=patch_model,
        model_path=Path(model_cfg["path"]),
        hf_name=model_cfg["hf_name"],
        batch_size=args.patch_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        control_df = run_random_patch_control(evaluator, candidates, args.seed)
        control_df["model"] = patch_model
        control_df["model_display_name"] = model_cfg["display_name"]
        control_df["sample_mode"] = args.sample_mode
        control_df.to_csv(result_path, index=False)
        write_patch_control_report(paths, patch_model, control_df, candidates)
    finally:
        evaluator.close()
    print(f"[完成] random patch control 完成：{patch_model}", flush=True)


def read_eval_tables(paths: RunPaths, filename: str) -> pd.DataFrame:
    frames = []
    for model_key in MODEL_ORDER:
        path = paths.table_dir / model_key / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_patch_tables(paths: RunPaths) -> pd.DataFrame:
    frames = []
    for model_key in MODEL_ORDER:
        path = paths.patch_dir / model_key / "formal_to_natural_patching_results.csv"
        if path.exists():
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_results(paths: RunPaths, args: argparse.Namespace) -> None:
    main_eval = read_eval_tables(paths, "main_eval.csv")
    ablation_eval = read_eval_tables(paths, "ablation_eval.csv")
    stress_eval = read_eval_tables(paths, "stress_eval.csv")
    if main_eval.empty and ablation_eval.empty and stress_eval.empty:
        raise FileNotFoundError("未找到 main_eval.csv、ablation_eval.csv 或 stress_eval.csv，无法聚合。")

    aggregate_dir = paths.table_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    if not main_eval.empty:
        main_eval.to_csv(aggregate_dir / "all_main_eval.csv", index=False)
    if not ablation_eval.empty:
        ablation_eval.to_csv(aggregate_dir / "all_ablation_eval.csv", index=False)
    if not stress_eval.empty:
        stress_eval.to_csv(aggregate_dir / "all_stress_eval.csv", index=False)

    all_eval = pd.concat([df for df in [main_eval, ablation_eval, stress_eval] if not df.empty], ignore_index=True)
    all_eval.to_csv(aggregate_dir / "all_final_eval.csv", index=False)

    metrics = save_metrics(all_eval, aggregate_dir, bootstrap=args.bootstrap, seed=args.seed)
    main_metrics = save_metrics(main_eval, aggregate_dir / "main_metrics", bootstrap=args.bootstrap, seed=args.seed) if not main_eval.empty else {}
    ablation_metrics = (
        save_metrics(ablation_eval, aggregate_dir / "ablation_metrics", bootstrap=args.bootstrap, seed=args.seed)
        if not ablation_eval.empty
        else {}
    )
    stress_metrics = save_metrics(stress_eval, aggregate_dir / "stress_metrics", bootstrap=args.bootstrap, seed=args.seed) if not stress_eval.empty else {}

    summary = metrics["metrics_summary"]
    ccc = metrics["metrics_ccc"]
    gain = metrics["metrics_scaffold_gain"]
    rescue = metrics["metrics_rescue_harm"]
    ccc_cols = [
        "model",
        "diagnostic_source",
        "dataset_variant",
        "prompt_mode",
        "strict_ccc",
        "correct_flip_rate",
        "wrong_flip_rate",
        "invariant_yes_rate",
        "invariant_no_rate",
        "scca",
        "signed_ccc",
        "invalid_pair_rate",
    ]
    if ccc.empty:
        ccc = pd.DataFrame(columns=ccc_cols)
    combined_results = summary.merge(ccc[ccc_cols], on=["model", "diagnostic_source", "dataset_variant", "prompt_mode"], how="left")
    combined_results["model_order"] = combined_results["model"].map({key: idx for idx, key in enumerate(MODEL_ORDER)})
    combined_results = combined_results.sort_values(["diagnostic_source", "dataset_variant", "model_order", "prompt_mode"])
    combined_results = combined_results.drop(columns=["model_order"])
    combined_results.to_csv(aggregate_dir / "aggregate_final_results.csv", index=False)
    main_results = combined_results[combined_results["diagnostic_source"] == "main"].copy()
    main_results.to_csv(aggregate_dir / "aggregate_main_results.csv", index=False)

    patch_df = read_patch_tables(paths)
    if not patch_df.empty:
        patch_df.to_csv(aggregate_dir / "formal_to_natural_patching_results.csv", index=False)
        patch_summary = (
            patch_df.groupby(["model", "model_display_name", "method"], dropna=False)
            .agg(
                n_rows=("sample_id", "count"),
                n_samples=("sample_id", "nunique"),
                mean_absolute_recovery=("absolute_recovery", "mean"),
                max_absolute_recovery=("absolute_recovery", "max"),
                mean_normalized_recovery=("normalized_recovery", "mean"),
            )
            .reset_index()
        )
        patch_summary.to_csv(aggregate_dir / "formal_to_natural_patching_summary.csv", index=False)
        layer_profile = (
            patch_df.groupby(["model", "model_display_name", "layer"], dropna=False)
            .agg(
                mean_recovery=("absolute_recovery", "mean"),
                median_recovery=("absolute_recovery", "median"),
                max_recovery=("absolute_recovery", "max"),
                positive_recovery_rate=("absolute_recovery", lambda x: float((x > 0).mean())),
                mean_normalized_recovery=("normalized_recovery", "mean"),
            )
            .reset_index()
        )
        layer_profile.to_csv(aggregate_dir / "patching_layer_profile.csv", index=False)
        layer_profile.sort_values(["model", "mean_recovery"], ascending=[True, False]).groupby("model").head(5).to_csv(
            aggregate_dir / "patching_top5_layers_by_mean.csv",
            index=False,
        )
    else:
        patch_summary = pd.DataFrame()
    patch_control_summaries = []
    for model_key in MODEL_ORDER:
        summary_df = summarize_patch_control(paths, model_key)
        if not summary_df.empty:
            patch_control_summaries.append(summary_df)
    patch_control_summary = pd.concat(patch_control_summaries, ignore_index=True) if patch_control_summaries else pd.DataFrame()
    if not patch_control_summary.empty:
        patch_control_summary.to_csv(aggregate_dir / "patch_control_summary.csv", index=False)

    figure_refs = []
    if not args.skip_figures:
        main_like_frames = [df for df in [main_eval, ablation_eval] if not df.empty]
        main_like_eval = pd.concat(main_like_frames, ignore_index=True) if main_like_frames else pd.DataFrame()
        figure_refs = make_figures(paths, main_like_eval, stress_eval, main_results, gain, rescue, patch_df)

    write_final_report(
        paths=paths,
        args=args,
        main_results=main_results,
        main_metrics=main_metrics,
        ablation_metrics=ablation_metrics,
        stress_metrics=stress_metrics,
        gain=gain,
        rescue=rescue,
        patch_df=patch_df,
        patch_summary=patch_summary,
        patch_control_summary=patch_control_summary,
        figure_refs=figure_refs,
    )
    print(f"[完成] 最终聚合报告已生成：{paths.report_dir / 'MLISE2026_qwen_final_report.md'}", flush=True)


def set_english_plot_style() -> None:
    global plt, sns
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    import seaborn as sns_module

    plt = plt_module
    sns = sns_module
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def rel_to_run(paths: RunPaths, path: Path) -> str:
    return str(path.relative_to(paths.run_dir))


def ordered_model_names(df: pd.DataFrame) -> list[str]:
    names = [MODEL_CONFIGS[key]["display_name"] for key in MODEL_ORDER]
    present = set(df["model_display_name"].dropna().unique()) if "model_display_name" in df.columns else set()
    return [name for name in names if name in present]


def make_figures(
    paths: RunPaths,
    main_eval: pd.DataFrame,
    stress_eval: pd.DataFrame,
    main_results: pd.DataFrame,
    gain: pd.DataFrame,
    rescue: pd.DataFrame,
    patch_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    set_english_plot_style()
    refs: list[tuple[str, str]] = []

    main_summary = main_results[main_results["diagnostic_source"] == "main"].copy()
    if not main_summary.empty:
        main_summary["model_display_name"] = pd.Categorical(
            main_summary["model_display_name"],
            categories=ordered_model_names(main_summary),
            ordered=True,
        )
        main_summary["prompt_condition"] = main_summary["prompt_mode"].map(PROMPT_LABELS)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=main_summary, x="model_display_name", y="accuracy", hue="prompt_condition", ax=ax)
        ax.set_title("Accuracy by Input Condition")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Input Condition", loc="best")
        fig.tight_layout()
        path = paths.figure_dir / "01_accuracy_by_input_condition.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Accuracy by Input Condition", rel_to_run(paths, path)))

    main_gain = gain[gain["diagnostic_source"] == "main"].copy() if not gain.empty else pd.DataFrame()
    if not main_gain.empty:
        main_gain["model_display_name"] = pd.Categorical(
            main_gain["model_display_name"],
            categories=ordered_model_names(main_gain),
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=main_gain, x="model_display_name", y="scaffold_gain", hue="scaffold_condition", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Scaffold Gain by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy Gain vs. Natural Language")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Scaffold")
        fig.tight_layout()
        path = paths.figure_dir / "02_scaffold_gain_by_model.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Scaffold Gain by Model", rel_to_run(paths, path)))

    main_ccc = main_summary.dropna(subset=["strict_ccc"]).copy() if not main_summary.empty else pd.DataFrame()
    if not main_ccc.empty:
        main_ccc["prompt_condition"] = main_ccc["prompt_mode"].map(PROMPT_LABELS)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=main_ccc, x="model_display_name", y="strict_ccc", hue="prompt_condition", ax=ax)
        ax.set_title("Strict Contrast Consistency")
        ax.set_xlabel("Model")
        ax.set_ylabel("CCC")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Input Condition")
        fig.tight_layout()
        path = paths.figure_dir / "03_strict_contrast_consistency.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Strict Contrast Consistency", rel_to_run(paths, path)))

    if not main_ccc.empty and {"correct_flip_rate", "wrong_flip_rate"}.issubset(main_ccc.columns):
        decomp = main_ccc[main_ccc["prompt_mode"].isin(["nl", "nl_formal", "formula_only"])].copy()
        if not decomp.empty:
            decomp_long = decomp.melt(
                id_vars=["model_display_name", "prompt_condition"],
                value_vars=["correct_flip_rate", "wrong_flip_rate"],
                var_name="flip_type",
                value_name="rate",
            )
            decomp_long["flip_type"] = decomp_long["flip_type"].map(
                {"correct_flip_rate": "Correct Flip", "wrong_flip_rate": "Wrong Flip"}
            )
            grid = sns.catplot(
                data=decomp_long,
                kind="bar",
                x="prompt_condition",
                y="rate",
                hue="flip_type",
                col="model_display_name",
                col_wrap=3,
                height=4,
                aspect=1.15,
            )
            grid.set_axis_labels("Input Condition", "Rate")
            grid.set_titles("{col_name}")
            grid.fig.suptitle("CCC Decomposition", y=1.03)
            for ax in grid.axes.flat:
                ax.tick_params(axis="x", rotation=25)
                ax.set_ylim(0, 1)
            path = paths.figure_dir / "04_ccc_decomposition.png"
            grid.fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(grid.fig)
            refs.append(("CCC Decomposition", rel_to_run(paths, path)))

    if not main_eval.empty:
        by_query = compute_summary_metrics(main_eval)["metrics_by_query"]
        query_main = by_query[by_query["diagnostic_source"] == "main"].copy()
        query_main = query_main[query_main["prompt_mode"].isin(["nl", "nl_formal", "formula_only"])]
        if not query_main.empty:
            query_main["condition_model"] = query_main["model_display_name"] + " | " + query_main["prompt_mode"].map(PROMPT_LABELS)
            heat = query_main.pivot_table(index="query_type", columns="condition_model", values="accuracy", aggfunc="first")
            fig, ax = plt.subplots(figsize=(14, 5))
            sns.heatmap(heat, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Accuracy by Query Type")
            ax.set_xlabel("Model and Input Condition")
            ax.set_ylabel("Query Type")
            fig.tight_layout()
            path = paths.figure_dir / "05_query_type_accuracy_heatmap.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            refs.append(("Accuracy by Query Type", rel_to_run(paths, path)))

        transitions = compute_transition_patterns(main_eval)["metrics_transition_patterns"]
        transitions = transitions[transitions["diagnostic_source"] == "main"].copy() if not transitions.empty else pd.DataFrame()
        if not transitions.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=transitions, x="model_display_name", y="rate", hue="transition_pattern", ax=ax)
            ax.set_title("Input Transition Patterns")
            ax.set_xlabel("Model")
            ax.set_ylabel("Rate")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=15)
            ax.legend(title="Pattern", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            path = paths.figure_dir / "06_input_transition_patterns.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            refs.append(("Input Transition Patterns", rel_to_run(paths, path)))

    if not stress_eval.empty:
        stress_summary = compute_summary_metrics(stress_eval)["metrics_summary"]
        stress_summary["prompt_condition"] = stress_summary["prompt_mode"].map(PROMPT_LABELS)
        stress_summary["model_display_name"] = pd.Categorical(
            stress_summary["model_display_name"],
            categories=ordered_model_names(stress_summary),
            ordered=True,
        )
        grid = sns.catplot(
            data=stress_summary,
            kind="bar",
            x="dataset_variant",
            y="accuracy",
            hue="prompt_condition",
            col="model_display_name",
            col_wrap=3,
            height=4,
            aspect=1.15,
        )
        grid.set_axis_labels("Stress Split", "Accuracy")
        grid.set_titles("{col_name}")
        grid.fig.suptitle("Stress Split Accuracy", y=1.03)
        for ax in grid.axes.flat:
            ax.tick_params(axis="x", rotation=25)
            ax.set_ylim(0, 1)
        path = paths.figure_dir / "07_stress_split_accuracy.png"
        grid.fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(grid.fig)
        refs.append(("Stress Split Accuracy", rel_to_run(paths, path)))

    main_rescue = rescue[rescue["diagnostic_source"] == "main"].copy() if not rescue.empty else pd.DataFrame()
    if not main_rescue.empty:
        long_rows = []
        for _, row in main_rescue.iterrows():
            long_rows.extend(
                [
                    {
                        "model_display_name": row["model_display_name"],
                        "type": "Rescue Rate",
                        "rate": row["rescue_rate_over_nl_failures"],
                    },
                    {
                        "model_display_name": row["model_display_name"],
                        "type": "Harm Rate",
                        "rate": row["harm_rate_over_nl_successes"],
                    },
                ]
            )
        rescue_long = pd.DataFrame(long_rows)
        rescue_long["model_display_name"] = pd.Categorical(
            rescue_long["model_display_name"],
            categories=ordered_model_names(rescue_long),
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=rescue_long, x="model_display_name", y="rate", hue="type", ax=ax)
        ax.set_title("Rescue vs. Harm Rate")
        ax.set_xlabel("Model")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="")
        fig.tight_layout()
        path = paths.figure_dir / "08_rescue_vs_harm_rate.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Rescue vs. Harm Rate", rel_to_run(paths, path)))

    if not patch_df.empty:
        layer_profile = patch_df.groupby(["model_display_name", "layer"], dropna=False)["absolute_recovery"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=layer_profile, x="layer", y="absolute_recovery", hue="model_display_name", marker="o", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Layer-wise Patching Recovery")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Absolute Recovery")
        fig.tight_layout()
        path = paths.figure_dir / "09_layerwise_patching_recovery.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Layer-wise Patching Recovery", rel_to_run(paths, path)))

        heat_data = patch_df.groupby(["sample_id", "layer"])["absolute_recovery"].mean().reset_index()
        heat = heat_data.pivot(index="sample_id", columns="layer", values="absolute_recovery")
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(heat.index))))
        sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Formal-to-Natural Patching Recovery")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Sample ID")
        fig.tight_layout()
        path = paths.figure_dir / "10_formal_to_natural_patching_recovery.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Formal-to-Natural Patching Recovery", rel_to_run(paths, path)))

    return refs


def write_model_report(paths: RunPaths, model_key: str, out_dir: Path, eval_df: pd.DataFrame, source_name: str) -> None:
    if source_name == "main":
        metrics_dir = out_dir
    else:
        metrics_dir = out_dir / f"{source_name}_metrics"
    metrics = save_metrics(eval_df, metrics_dir)
    lines = [
        f"# {MODEL_CONFIGS[model_key]['display_name']} 评测记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型路径：`{MODEL_CONFIGS[model_key]['path']}`",
        f"- 评测来源：`{source_name}`",
        f"- 样本数：`{eval_df['sample_id'].nunique()}`",
        "- 报告语言：中文。",
        "- 图表语言：英文。",
        "- 解析规则：自动正则解析，不调用外部模型修补答案。",
        "",
        "## Summary",
        "",
        table_to_markdown(metrics["metrics_summary"]),
        "",
        "## Strict Contrast Consistency",
        "",
        table_to_markdown(metrics["metrics_ccc"]),
        "",
        "## Scaffold Gain",
        "",
        table_to_markdown(metrics["metrics_scaffold_gain"]),
        "",
        "## Rescue / Harm",
        "",
        table_to_markdown(metrics["metrics_rescue_harm"]),
        "",
    ]
    (out_dir / f"{source_name}_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_patch_report(paths: RunPaths, model_key: str, patch_df: pd.DataFrame, candidates: pd.DataFrame) -> None:
    patch_dir = paths.patch_dir / model_key
    if patch_df.empty:
        summary = pd.DataFrame()
        summary_text = "本轮没有找到 natural prompt 错、formal scaffold prompt 对的样本，因此未生成可解释的 formal-to-natural patching 结果。"
    else:
        summary = (
            patch_df.groupby(["model", "model_display_name", "method"], dropna=False)
            .agg(
                n_rows=("sample_id", "count"),
                n_samples=("sample_id", "nunique"),
                mean_absolute_recovery=("absolute_recovery", "mean"),
                max_absolute_recovery=("absolute_recovery", "max"),
                mean_normalized_recovery=("normalized_recovery", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(patch_dir / "formal_to_natural_patching_summary.csv", index=False)
        positive_samples = int((patch_df.groupby("sample_id")["absolute_recovery"].max() > 0).sum())
        summary_text = (
            f"本轮共分析 `{patch_df['sample_id'].nunique()}` 个 natural-fail / scaffold-success 样本，"
            f"其中最大 absolute recovery 为正的样本数为 `{positive_samples}`。"
        )
    lines = [
        f"# {MODEL_CONFIGS[model_key]['display_name']} Formal-to-Natural Patching 记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型路径：`{MODEL_CONFIGS[model_key]['path']}`",
        f"- 候选样本数：`{len(candidates)}`",
        "- 方法：HuggingFace forward hook；将 `nl_formal` 条件下每层最后 token 的 residual 输出 patch 到 `nl` 条件。",
        "",
        "## 结果概述",
        "",
        summary_text,
        "",
        "## 汇总表",
        "",
        table_to_markdown(summary),
        "",
        "## 解释边界",
        "",
        "该结果只说明 formal scaffold 条件下的隐藏状态可能携带可转移的答案方向信号，不能解释为完整 causal circuit。",
        "",
    ]
    (patch_dir / "formal_to_natural_patching_report.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_patch_control(paths: RunPaths, model_key: str) -> pd.DataFrame:
    patch_dir = paths.patch_dir / model_key
    matched_path = patch_dir / "formal_to_natural_patching_results.csv"
    random_path = patch_dir / "random_patch_control_results.csv"
    frames = []
    for condition, path in [("matched", matched_path), ("random", random_path)]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["patch_condition"] = condition
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby(["model", "model_display_name", "patch_condition"], dropna=False)
        .agg(
            n_rows=("sample_id", "count"),
            n_samples=("sample_id", "nunique"),
            mean_absolute_recovery=("absolute_recovery", "mean"),
            median_absolute_recovery=("absolute_recovery", "median"),
            max_absolute_recovery=("absolute_recovery", "max"),
            positive_recovery_rate=("absolute_recovery", lambda x: float((x > 0).mean())),
            mean_normalized_recovery=("normalized_recovery", "mean"),
        )
        .reset_index()
    )
    matched = combined[combined["patch_condition"] == "matched"]
    random_control = combined[combined["patch_condition"] == "random"]
    if not matched.empty and not random_control.empty:
        key_cols = ["sample_id", "layer"]
        diff = matched[key_cols + ["absolute_recovery"]].merge(
            random_control[key_cols + ["absolute_recovery"]],
            on=key_cols,
            how="inner",
            suffixes=("_matched", "_random"),
        )
        if not diff.empty:
            diff["matched_minus_random"] = diff["absolute_recovery_matched"] - diff["absolute_recovery_random"]
            extra = {
                "model": model_key,
                "model_display_name": MODEL_CONFIGS[model_key]["display_name"],
                "patch_condition": "matched_minus_random",
                "n_rows": int(len(diff)),
                "n_samples": int(diff["sample_id"].nunique()),
                "mean_absolute_recovery": float(diff["matched_minus_random"].mean()),
                "median_absolute_recovery": float(diff["matched_minus_random"].median()),
                "max_absolute_recovery": float(diff["matched_minus_random"].max()),
                "positive_recovery_rate": float((diff["matched_minus_random"] > 0).mean()),
                "mean_normalized_recovery": np.nan,
            }
            summary = pd.concat([summary, pd.DataFrame([extra])], ignore_index=True)
    summary.to_csv(patch_dir / "patch_control_summary.csv", index=False)
    return summary


def write_patch_control_report(paths: RunPaths, model_key: str, control_df: pd.DataFrame, candidates: pd.DataFrame) -> None:
    patch_dir = paths.patch_dir / model_key
    summary = summarize_patch_control(paths, model_key)
    if control_df.empty:
        summary_text = "本轮 random patch control 没有生成可分析结果，主要原因是候选样本不足或 matched patching 尚未完成。"
    else:
        summary_text = (
            f"本轮 random patch control 使用 `{control_df['sample_id'].nunique()}` 个目标样本，"
            "将其他样本的 `nl_formal` residual patch 到当前样本的 `nl` 条件，用于检验 matched patching 的恢复是否只是任意形式输入 residual 的偶然效应。"
        )
    lines = [
        f"# {MODEL_CONFIGS[model_key]['display_name']} Random Patch Control 记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型路径：`{MODEL_CONFIGS[model_key]['path']}`",
        f"- 候选样本数：`{len(candidates)}`",
        "- 方法：随机选择另一样本的 `nl_formal` residual，patch 到当前样本的 `nl` prompt。",
        "",
        "## 结果概述",
        "",
        summary_text,
        "",
        "## Matched vs. Random 汇总",
        "",
        table_to_markdown(summary),
        "",
    ]
    (patch_dir / "random_patch_control_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    paths: RunPaths,
    args: argparse.Namespace,
    main_results: pd.DataFrame,
    main_metrics: dict[str, pd.DataFrame],
    ablation_metrics: dict[str, pd.DataFrame],
    stress_metrics: dict[str, pd.DataFrame],
    gain: pd.DataFrame,
    rescue: pd.DataFrame,
    patch_df: pd.DataFrame,
    patch_summary: pd.DataFrame,
    patch_control_summary: pd.DataFrame,
    figure_refs: list[tuple[str, str]],
) -> None:
    env = collect_environment_info()
    lines = [
        "# MLISE 2026 Qwen 因果推理最终实验报告",
        "",
        "## 1. 实验概述",
        "",
        f"- 实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- run_id：`{args.run_id}`",
        f"- 样本模式：`{args.sample_mode}`",
        "- 实验目标：在 CLadder 因果推理任务上评估 Qwen3 系列模型对自然语言题干、形式结构提示、严格对照样本和隐藏层表示的响应。",
        "- 图表标题、坐标轴、图例均使用英文，便于后续直接进入英文会议论文。",
        "- 本 Markdown 报告使用中文。",
        "",
        "## 2. 服务器环境",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- {key}：`{value}`")
    lines.extend(
        [
            "",
            "## 3. 数据集与输入条件",
            "",
            f"- 主数据集：`{args.data_path}`",
            "- 主实验样本：按 query type 分层抽样，并在每个 query type 内保持 yes/no 平衡。",
            f"- stress splits：`{', '.join(STRESS_SPLITS.keys())}`。",
            "- 主输入条件：`nl`、`nl_formal`、`formula_only`。",
            "- 消融输入条件：`nl_var_query`、`nl_var_graph`。",
            "- 解析规则：优先解析 `Final answer: yes/no`，其次解析独立 yes/no，失败记为 invalid。",
            "",
            "## 4. 模型路径",
            "",
        ]
    )
    for model_key in MODEL_ORDER:
        model_cfg = MODEL_CONFIGS[model_key]
        lines.append(f"- {model_cfg['display_name']}：`{model_cfg['path']}`")
    lines.extend(
        [
            "",
            "## 5. 指标定义",
            "",
            "- `Accuracy`：单题预测是否等于 CLadder oracle label。",
            "- `Strict Contrast Causal Consistency (CCC)`：同一 story、query type 和 formal form 下，gold 相反的 pair 是否也得到相反预测。",
            "- `Correct Flip / Wrong Flip / SCCA`：将 CCC 拆成正确翻转、错误翻转和严格正确对照准确率。",
            "- `Scaffold Gain`：各形式输入条件相对 `nl` 的 accuracy 增量。",
            "- `Rescue / Harm`：`nl` 错而 `nl_formal` 对记为 rescue；`nl` 对而 `nl_formal` 错记为 harm。",
            "- `Transition Pattern`：统计 `nl -> nl_formal -> formula_only` 的 C/W 正确性轨迹。",
            "- `Stress Robustness`：报告 split 级 accuracy、worst split 和跨 split 方差。",
            "- `Formal-to-Natural Patching Recovery`：把 `nl_formal` 条件的 residual 输出 patch 到 `nl` 条件后，gold-label logit margin 的变化。",
            "",
            "## 6. 聚合主结果",
            "",
            table_to_markdown(main_results),
            "",
            "## 7. 主实验 Scaffold Gain",
            "",
            table_to_markdown(gain[gain["diagnostic_source"] == "main"] if not gain.empty else pd.DataFrame()),
            "",
            "## 8. Query Type 与 Rung 细分",
            "",
            "### Query Type Accuracy",
            "",
            table_to_markdown(main_metrics.get("metrics_by_query", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "### Query Type Scaffold Gain",
            "",
            table_to_markdown(main_metrics.get("metrics_scaffold_gain_by_query", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "### Rung Accuracy",
            "",
            table_to_markdown(main_metrics.get("metrics_by_rung", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "## 9. CCC 正误分解",
            "",
            table_to_markdown(main_metrics.get("metrics_ccc", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "## 10. 输入条件转移矩阵",
            "",
            table_to_markdown(main_metrics.get("metrics_transition_patterns", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "## 11. Rescue / Harm",
            "",
            table_to_markdown(rescue[rescue["diagnostic_source"] == "main"] if not rescue.empty else pd.DataFrame()),
            "",
            "## 12. 形式成分消融",
            "",
            table_to_markdown(ablation_metrics.get("metrics_summary", pd.DataFrame()) if ablation_metrics else pd.DataFrame()),
            "",
            "## 13. Stress Split 鲁棒性",
            "",
            table_to_markdown(stress_metrics.get("metrics_summary", pd.DataFrame()) if stress_metrics else pd.DataFrame()),
            "",
            "### Worst-case 与方差",
            "",
            table_to_markdown(stress_metrics.get("metrics_stress_robustness", pd.DataFrame()) if stress_metrics else pd.DataFrame()),
            "",
            "## 14. Bootstrap 与配对检验",
            "",
            "### Accuracy Bootstrap CI",
            "",
            table_to_markdown(main_metrics.get("metrics_accuracy_bootstrap_ci", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "### Paired Condition Tests",
            "",
            table_to_markdown(main_metrics.get("metrics_paired_condition_tests", pd.DataFrame()) if main_metrics else pd.DataFrame()),
            "",
            "## 15. 白盒 Patching 结果",
            "",
        ]
    )
    if patch_df.empty:
        lines.append("当前 run 尚未生成可分析的 formal-to-natural patching 结果。")
    else:
        lines.extend(
            [
                "本轮白盒实验只作为探索性诊断：选择 natural prompt 错、formal scaffold prompt 对的样本，把 formal 条件下的 residual stream 输出 patch 到 natural 条件，观察 gold-label logit margin 是否恢复。",
                "",
                table_to_markdown(patch_summary),
            ]
        )
    lines.extend(
        [
            "",
            "### Random Patch Control",
            "",
            table_to_markdown(patch_control_summary),
            "",
            "## 16. 图表索引",
            "",
        ]
    )
    if figure_refs:
        for title, rel_path in figure_refs:
            lines.extend([f"### {title}", "", f"![{title}](../{rel_path})", ""])
    else:
        lines.append("当前未生成图表。")

    lines.extend(
        [
            "",
            "## 17. 中文论文实验描述草稿",
            "",
            "本文使用 CLadder `full_v1.5_default` 构建 640 条平衡评测子集，并在五个 stress split 上各抽取 100 条样本。模型包括 Qwen3-0.6B、Qwen3-4B 和 Qwen3-8B。输入条件包括原始自然语言题干、加入变量映射与形式查询的消融输入、加入变量映射与因果图的消融输入、完整形式脚手架输入，以及保留事实条件和形式结构的形式化输入。",
            "",
            "行为指标包括 Accuracy、Strict Contrast Causal Consistency、Correct Flip、Wrong Flip、SCCA、Scaffold Gain、Rescue/Harm、输入条件转移矩阵和 stress robustness。CCC 只统计 gold label 相反的严格对照 pair；Correct Flip 与 Wrong Flip 用于区分预测翻转是否同时保持正确方向；SCCA 进一步要求 pair 内两个样本都被正确预测。",
            "",
            "隐藏层分析采用 formal-to-natural residual stream patching。实验筛选自然语言条件回答错误、形式脚手架条件回答正确的样本，并把形式输入条件下每层最后 token 的 residual 输出 patch 到自然语言条件中，观察 gold-label logit margin 的恢复情况。随机 patch control 使用另一样本的形式输入 residual 作为对照，以检验恢复是否具有样本匹配性。该实验只作为机制探索，不声称发现完整因果回路。",
            "",
            "## 18. 初步结论与下一步建议",
            "",
            "正式结论需要结合本 run 的完整数值填写。重点检查三类证据：第一，错误是否集中在 counterfactual 或 mediation 类 query type；第二，CCC 的提升是否主要来自 Correct Flip 还是 Wrong Flip；第三，形式输入是否表现为稳定 rescue，还是在样本层面同时造成 rescue 与 harm。若 matched patching 明显强于 random control，可以将其写作形式输入在部分层中包含可转移答案方向信号的探索性证据。",
            "",
        ]
    )
    (paths.report_dir / "MLISE2026_qwen_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLISE 2026 Qwen3 × CLadder 最终服务器实验脚本")
    parser.add_argument(
        "--stage",
        choices=["sample", "behavior", "ablation", "stress", "analysis", "patch", "patch-control", "aggregate", "all"],
        default="all",
    )
    parser.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--sample-mode", choices=["quick", "formal"], default="formal")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--patch-model", choices=MODEL_ORDER, default="qwen3_4b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--stress-sample-size", type=int, default=100)
    parser.add_argument("--patch-pair-limit", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_id:
        args.run_id = f"final_{args.sample_mode}_{timestamp()}"
    seed_everything(args.seed)

    paths = get_paths(Path(args.output_root), args.run_id)
    ensure_dirs(paths)
    config = DiagnosticConfig(
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
        stress_sample_size=args.stress_sample_size,
        patch_pair_limit=args.patch_pair_limit,
        bootstrap=args.bootstrap,
        resume=args.resume,
        skip_figures=args.skip_figures,
    )
    save_json(paths.run_dir / "config.json", asdict(config))
    write_environment_report(paths, config)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"主数据集不存在：{data_path}")
    for split_name, split_path in STRESS_SPLITS.items():
        if not split_path.exists():
            raise FileNotFoundError(f"stress split 不存在：{split_name} -> {split_path}")

    model_keys = model_keys_from_arg(args.model)

    if args.stage in {"sample", "all"}:
        run_sample_stage(paths, args)

    if args.stage in {"behavior", "all"}:
        run_behavior_stage(paths, args, model_keys, config)

    if args.stage in {"ablation", "all"}:
        run_ablation_stage(paths, args, model_keys, config)

    if args.stage in {"stress", "all"}:
        run_stress_stage(paths, args, model_keys, config)

    if args.stage in {"patch", "all"}:
        run_patch_stage(paths, args, config)

    if args.stage in {"patch-control", "all"}:
        run_patch_control_stage(paths, args, config)

    if args.stage in {"aggregate", "analysis", "all"}:
        aggregate_results(paths, args)


if __name__ == "__main__":
    main()
