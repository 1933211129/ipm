#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mlise2026_qwen_final as base  # noqa: E402
import mlise2026_symbolic_intervention as symbolic  # noqa: E402
from mlise2026_binary_score import BinaryScorer  # noqa: E402


DEFAULT_RUN_ID = "final_20260513_090357"
PAPER_QUERY_TYPES = [
    "marginal",
    "correlation",
    "ate",
    "backadj",
    "det-counterfactual",
    "ett",
    "nie",
    "nde",
]
CONTROL_MODES = [
    "nl_binary_score",
    "symbolic_matched_binary_score",
    "symbolic_shuffled_binary_score",
]
CONTROL_LABELS = {
    "nl_binary_score": "NL Binary Score",
    "symbolic_matched_binary_score": "Matched Symbolic Binary Score",
    "symbolic_shuffled_binary_score": "Shuffled Symbolic Binary Score",
}


@dataclass
class ControlConfig:
    stage: str
    model: str
    run_id: str
    output_root: str
    sample_size: int
    seed: int
    batch_size: int
    bootstrap: int
    resume: bool


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def control_dir(paths: base.RunPaths) -> Path:
    path = paths.table_dir / "symbolic_controls"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_out_dir(paths: base.RunPaths, model_key: str) -> Path:
    path = paths.table_dir / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_control_path(paths: base.RunPaths, sample_size: int) -> Path:
    return control_dir(paths) / f"selected_symbolic_control_subset_{sample_size}.csv"


def balanced_extended_subset(sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size % (2 * len(PAPER_QUERY_TYPES)) != 0:
        raise ValueError(f"sample_size 必须能被 {2 * len(PAPER_QUERY_TYPES)} 整除，当前为 {sample_size}")
    df = base.load_cladder_df(base.DATA_PATH, "symbolic_control")
    df = df[df["query_type"].isin(PAPER_QUERY_TYPES)].copy()
    per_query = sample_size // len(PAPER_QUERY_TYPES)
    chunks = []
    for idx, query_type in enumerate(PAPER_QUERY_TYPES):
        pool = df[df["query_type"] == query_type].copy()
        chunk = base.balanced_label_sample(pool, per_query, seed + idx * 101)
        chunks.append(chunk)
    subset = pd.concat(chunks, ignore_index=True)
    subset = subset.sort_values(["query_type", "label", "story_id", "rung", "id"]).reset_index(drop=True)
    subset["subset"] = f"symbolic_control_{sample_size}"
    subset["diagnostic_source"] = "symbolic_control"
    subset["dataset_variant"] = "symbolic_control"
    return subset


def load_or_create_subset(paths: base.RunPaths, sample_size: int, seed: int) -> pd.DataFrame:
    path = selected_control_path(paths, sample_size)
    if path.exists():
        subset = pd.read_csv(path)
    else:
        subset = balanced_extended_subset(sample_size, seed)
        subset.to_csv(path, index=False)
    subset["id"] = subset["id"].astype(int)
    subset["rung"] = subset["rung"].astype(int)
    subset["label"] = subset["label"].str.lower().str.strip()
    subset["dataset_variant"] = "symbolic_control"
    subset["diagnostic_source"] = "symbolic_control"
    return subset


def build_donor_map(df: pd.DataFrame, seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    by_key: dict[tuple[str, int], list[int]] = {}
    for row in df.to_dict("records"):
        by_key.setdefault((str(row["query_type"]), int(row["rung"])), []).append(int(row["id"]))
    donor_map: dict[int, int] = {}
    rows_by_id = {int(row["id"]): row for row in df.to_dict("records")}
    for sample_id, row in rows_by_id.items():
        key = (str(row["query_type"]), int(row["rung"]))
        candidates = [sid for sid in by_key[key] if sid != sample_id]
        if not candidates:
            candidates = [sid for sid in df["id"].astype(int).tolist() if sid != sample_id]
        donor_map[sample_id] = rng.choice(candidates)
    return donor_map


def build_control_prompt(
    row: dict[str, Any],
    mode: str,
    donor_row: dict[str, Any] | None,
) -> tuple[str, str, int | None]:
    if mode == "nl_binary_score":
        return base.build_user_prompt(row, "nl"), "natural_language", None
    if mode == "symbolic_matched_binary_score":
        symbolic.INTERVENTION_MODE = "symbolic_solver_concise"
        prompt, trace_source = symbolic.build_symbolic_solver_prompt(row)
        return prompt, trace_source, None
    if mode == "symbolic_shuffled_binary_score":
        if donor_row is None:
            raise ValueError("shuffled control 需要 donor_row")
        decomposition, trace_source = symbolic.make_symbolic_decomposition(donor_row)
        prompt = (
            "You are a concise causal yes/no solver.\n"
            "Use the question to decide what yes/no means. Use the symbolic decomposition to compute the causal quantity.\n"
            "The decomposition below is an externally supplied symbolic trace. Output at most three short lines, "
            "and the last line must be exactly: Final answer: yes/no\n"
            "Do not add any text after the final answer.\n\n"
            f"Question:\n{str(row['prompt']).strip()}\n\n"
            "Symbolic decomposition:\n"
            f"{decomposition}\n\n"
            f"Query type: {row.get('query_type', '')}\n"
            f"Rung: {row.get('rung', '')}"
        )
        return prompt, f"shuffled_{trace_source}", int(donor_row["id"])
    raise ValueError(f"未知 control mode: {mode}")


def read_completed(path: Path, resume: bool) -> tuple[pd.DataFrame, set[tuple[int, str]]]:
    if not path.exists():
        return pd.DataFrame(), set()
    if not resume:
        raise FileExistsError(f"结果文件已存在，避免覆盖：{path}。如需续跑请加 --resume。")
    existing = pd.read_csv(path)
    completed = set(zip(existing["sample_id"].astype(int), existing["prompt_mode"].astype(str)))
    print(f"[信息] resume 已读取 {len(existing)} 行：{path}", flush=True)
    return existing, completed


def run_control_eval(
    scorer: BinaryScorer,
    subset: pd.DataFrame,
    model_key: str,
    run_id: str,
    sample_size: int,
    batch_size: int,
    save_path: Path,
    resume: bool,
    seed: int,
) -> pd.DataFrame:
    existing, completed = read_completed(save_path, resume)
    rows_by_id = {int(row["id"]): row for row in subset.to_dict("records")}
    donor_map = build_donor_map(subset, seed)
    records: list[dict[str, Any]] = []
    for mode in CONTROL_MODES:
        source_rows = [
            row
            for row in subset.to_dict("records")
            if (int(row["id"]), mode) not in completed
        ]
        print(f"[进度] model={model_key} mode={mode} 待评分={len(source_rows)}", flush=True)
        for start in range(0, len(source_rows), batch_size):
            current = source_rows[start : start + batch_size]
            built = []
            for row in current:
                donor_row = rows_by_id[donor_map[int(row["id"])]] if mode == "symbolic_shuffled_binary_score" else None
                built.append(build_control_prompt(row, mode, donor_row))
            prompts = [item[0] for item in built]
            trace_sources = [item[1] for item in built]
            donor_ids = [item[2] for item in built]
            scored = scorer.score_batch(prompts)
            for row, trace_source, donor_id, score_row in zip(current, trace_sources, donor_ids, scored):
                record = {
                    "run_id": run_id,
                    "model": model_key,
                    "model_display_name": base.MODEL_CONFIGS[model_key]["display_name"],
                    "sample_id": int(row["id"]),
                    "donor_sample_id": donor_id,
                    "story_id": row["story_id"],
                    "graph_id": row["graph_id"],
                    "rung": int(row["rung"]),
                    "query_type": row["query_type"],
                    "question_property": row.get("question_property", ""),
                    "formal_form": row["formal_form"],
                    "gold_label": row["label"],
                    "prompt_mode": mode,
                    "prompt_condition": CONTROL_LABELS[mode],
                    "dataset_variant": "symbolic_control",
                    "diagnostic_source": "symbolic_control",
                    "subset": f"symbolic_control_{sample_size}",
                    "trace_source": trace_source,
                    "raw_output": "",
                    "input_tokens": np.nan,
                    "output_tokens": 1,
                    **score_row,
                }
                record["is_invalid"] = False
                record["is_correct"] = int(record["parsed_label"] == record["gold_label"])
                records.append(record)
            if records:
                pd.concat([existing, pd.DataFrame(records)], ignore_index=True).to_csv(save_path, index=False)
    if records:
        return pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    return existing


def run_model(paths: base.RunPaths, args: argparse.Namespace, model_key: str) -> None:
    subset = load_or_create_subset(paths, args.sample_size, args.seed)
    out_dir = model_out_dir(paths, model_key)
    save_path = out_dir / f"symbolic_control_binary_score_{args.sample_size}.csv"
    model_cfg = base.MODEL_CONFIGS[model_key]
    scorer = BinaryScorer(model_key=model_key, model_path=Path(model_cfg["path"]), batch_size=args.batch_size)
    try:
        eval_df = run_control_eval(
            scorer=scorer,
            subset=subset,
            model_key=model_key,
            run_id=args.run_id,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            save_path=save_path,
            resume=args.resume,
            seed=args.seed,
        )
    finally:
        scorer.close()
    metrics_dir = out_dir / f"symbolic_control_binary_score_metrics_{args.sample_size}"
    base.save_metrics(eval_df, metrics_dir, bootstrap=args.bootstrap, seed=args.seed)
    save_json(
        out_dir / f"symbolic_control_binary_score_config_{args.sample_size}.json",
        {
            "run_id": args.run_id,
            "model": model_key,
            "sample_size": args.sample_size,
            "modes": CONTROL_MODES,
            "batch_size": args.batch_size,
            "bootstrap": args.bootstrap,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def read_control_tables(paths: base.RunPaths, sample_size: int) -> pd.DataFrame:
    frames = []
    for model_key in base.MODEL_ORDER:
        path = paths.table_dir / model_key / f"symbolic_control_binary_score_{sample_size}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_bootstrap(eval_df: pd.DataFrame, seed: int, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [
        ("symbolic_matched_binary_score", "nl_binary_score"),
        ("symbolic_matched_binary_score", "symbolic_shuffled_binary_score"),
        ("symbolic_shuffled_binary_score", "nl_binary_score"),
    ]
    for model, group in eval_df.groupby("model", dropna=False):
        for left, right in comparisons:
            ldf = group[group["prompt_mode"] == left][["sample_id", "is_correct"]].rename(columns={"is_correct": "left"})
            rdf = group[group["prompt_mode"] == right][["sample_id", "is_correct"]].rename(columns={"is_correct": "right"})
            merged = ldf.merge(rdf, on="sample_id", how="inner")
            if merged.empty:
                continue
            diff = (merged["left"].astype(float) - merged["right"].astype(float)).to_numpy()
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_bootstrap)])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "condition_a": left,
                    "condition_b": right,
                    "n": int(len(diff)),
                    "acc_a": float(merged["left"].mean()),
                    "acc_b": float(merged["right"].mean()),
                    "paired_gain": float(diff.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p_gain_le_0": float((boots <= 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def aggregate(paths: base.RunPaths, args: argparse.Namespace) -> None:
    eval_df = read_control_tables(paths, args.sample_size)
    if eval_df.empty:
        raise FileNotFoundError("未找到 symbolic control 结果，无法聚合。")
    aggregate_dir = paths.table_dir / "aggregate" / f"symbolic_control_{args.sample_size}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(aggregate_dir / "all_symbolic_control_binary_score.csv", index=False)
    metrics = base.save_metrics(eval_df, aggregate_dir / "metrics", bootstrap=args.bootstrap, seed=args.seed)
    summary = metrics["metrics_summary"].copy()
    summary.to_csv(aggregate_dir / "symbolic_control_summary.csv", index=False)
    by_query = metrics["metrics_by_query"].copy()
    by_query.to_csv(aggregate_dir / "symbolic_control_by_query.csv", index=False)
    paired = paired_bootstrap(eval_df, args.seed, max(1, args.bootstrap))
    paired.to_csv(aggregate_dir / "symbolic_control_paired_bootstrap.csv", index=False)
    lines = [
        "# MLISE 2026 Symbolic Control 实验结果",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本数：`{args.sample_size}`",
        "- 目标：检验 matched symbolic decomposition 是否优于 shuffled symbolic decomposition。",
        "",
        "## 总体结果",
        "",
        base.table_to_markdown(summary.round(4)),
        "",
        "## 配对 Bootstrap",
        "",
        base.table_to_markdown(paired.round(4)),
        "",
        "## Query Type 结果",
        "",
        base.table_to_markdown(by_query.round(4)),
        "",
    ]
    (paths.report_dir / f"MLISE2026_symbolic_control_{args.sample_size}_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLISE 2026 matched vs shuffled symbolic control binary scoring")
    parser.add_argument("--stage", choices=["sample", "run", "aggregate", "all"], default="all")
    parser.add_argument("--model", choices=base.MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", default=str(base.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.seed_everything(args.seed)
    paths = base.get_paths(Path(args.output_root), args.run_id)
    base.ensure_dirs(paths)
    save_json(paths.run_dir / f"symbolic_control_config_{args.sample_size}.json", asdict(ControlConfig(**vars(args))))
    if args.stage in {"sample", "all"}:
        subset = load_or_create_subset(paths, args.sample_size, args.seed)
        print(f"[完成] symbolic control 子集：{len(subset)} -> {selected_control_path(paths, args.sample_size)}", flush=True)
    if args.stage in {"run", "all"}:
        for model_key in base.model_keys_from_arg(args.model):
            run_model(paths, args, model_key)
            base.empty_device_cache()
            time.sleep(1)
    if args.stage in {"aggregate", "all"}:
        aggregate(paths, args)


if __name__ == "__main__":
    main()
