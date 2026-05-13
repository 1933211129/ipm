#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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


DEFAULT_RUN_ID = "final_20260513_090357"
CONTROL_DIR_NAME = "symbolic_control_2048"
MODELS = ["qwen3_4b", "qwen3_8b"]
POLICY_LABELS = {
    "always_nl": "Always NL",
    "always_symbolic": "Always Symbolic",
    "query_router": "Query Router",
    "confidence_by_query": "Confidence Router",
}


@dataclass
class RoutingConfig:
    run_id: str
    output_root: str
    splits: int
    test_size: float
    seed: int
    bootstrap: int


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def out_dir(paths: base.RunPaths) -> Path:
    path = paths.table_dir / "aggregate" / "adaptive_symbolic_routing"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_control(paths: base.RunPaths) -> pd.DataFrame:
    path = paths.table_dir / "aggregate" / CONTROL_DIR_NAME / "all_symbolic_control_binary_score.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 symbolic control 结果：{path}")
    df = pd.read_csv(path)
    df = df[df["model"].isin(MODELS)]
    df = df[df["prompt_mode"].isin(["nl_binary_score", "symbolic_matched_binary_score"])]
    return pivot_scores(df, symbolic_mode="symbolic_matched_binary_score")


def load_main(paths: base.RunPaths) -> pd.DataFrame:
    path = paths.table_dir / "aggregate" / "all_binary_score_eval.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 binary score 主评测结果：{path}")
    df = pd.read_csv(path)
    df = df[df["model"].isin(MODELS)]
    df = df[df["prompt_mode"].isin(["nl_binary_score", "symbolic_solver_concise_binary_score"])]
    return pivot_scores(df, symbolic_mode="symbolic_solver_concise_binary_score")


def pivot_scores(df: pd.DataFrame, symbolic_mode: str) -> pd.DataFrame:
    required = [
        "model",
        "model_display_name",
        "sample_id",
        "story_id",
        "graph_id",
        "query_type",
        "rung",
        "formal_form",
        "question_property",
        "gold_label",
    ]
    value_cols = ["parsed_label", "is_correct", "yes_no_margin"]
    pivot = (
        df.pivot_table(index=required, columns="prompt_mode", values=value_cols, aggfunc="first")
        .reset_index()
    )
    pivot.columns = [
        "_".join([str(item) for item in col if item != ""]) if isinstance(col, tuple) else col
        for col in pivot.columns
    ]
    rename = {
        f"parsed_label_{symbolic_mode}": "symbolic_prediction",
        f"is_correct_{symbolic_mode}": "symbolic_correct",
        f"yes_no_margin_{symbolic_mode}": "symbolic_margin",
        "parsed_label_nl_binary_score": "nl_prediction",
        "is_correct_nl_binary_score": "nl_correct",
        "yes_no_margin_nl_binary_score": "nl_margin",
    }
    pivot = pivot.rename(columns=rename)
    keep = required + [
        "nl_prediction",
        "nl_correct",
        "nl_margin",
        "symbolic_prediction",
        "symbolic_correct",
        "symbolic_margin",
    ]
    missing = [col for col in keep if col not in pivot.columns]
    if missing:
        raise ValueError(f"pivot 后缺少列：{missing}")
    pivot = pivot[keep].copy()
    pivot["sample_id"] = pivot["sample_id"].astype(int)
    pivot["rung"] = pivot["rung"].astype(int)
    pivot["nl_correct"] = pivot["nl_correct"].astype(int)
    pivot["symbolic_correct"] = pivot["symbolic_correct"].astype(int)
    pivot["margin_delta_abs"] = pivot["symbolic_margin"].abs() - pivot["nl_margin"].abs()
    return pivot


def fit_query_router(train: pd.DataFrame) -> set[str]:
    selected = set()
    for query_type, group in train.groupby("query_type", dropna=False):
        if group["symbolic_correct"].mean() >= group["nl_correct"].mean():
            selected.add(str(query_type))
    return selected


def fit_confidence_by_query(train: pd.DataFrame) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for query_type, group in train.groupby("query_type", dropna=False):
        delta = group["margin_delta_abs"].astype(float).to_numpy()
        candidates = np.unique(np.concatenate([np.quantile(delta, np.linspace(0, 1, 81)), [-np.inf, np.inf, 0.0]]))
        best_acc = -1.0
        best_tau = 0.0
        for tau in candidates:
            use_symbolic = delta > tau
            correct = np.where(use_symbolic, group["symbolic_correct"], group["nl_correct"]).astype(float)
            acc = float(correct.mean())
            if acc > best_acc:
                best_acc = acc
                best_tau = float(tau)
        thresholds[str(query_type)] = best_tau
    return thresholds


def fit_policies(train: pd.DataFrame) -> dict[str, Any]:
    return {
        "query_router": fit_query_router(train),
        "confidence_by_query": fit_confidence_by_query(train),
    }


def apply_policy(frame: pd.DataFrame, policy: str, params: dict[str, Any]) -> pd.DataFrame:
    if policy == "always_nl":
        use_symbolic = np.zeros(len(frame), dtype=bool)
    elif policy == "always_symbolic":
        use_symbolic = np.ones(len(frame), dtype=bool)
    elif policy == "query_router":
        selected = params.get("query_router", set())
        use_symbolic = frame["query_type"].astype(str).isin(selected).to_numpy()
    elif policy == "confidence_by_query":
        thresholds = params.get("confidence_by_query", {})
        use_symbolic = np.array(
            [
                row["margin_delta_abs"] > thresholds.get(str(row["query_type"]), np.inf)
                for _, row in frame.iterrows()
            ],
            dtype=bool,
        )
    else:
        raise ValueError(f"未知 routing policy: {policy}")
    out = frame[
        [
            "model",
            "model_display_name",
            "sample_id",
            "story_id",
            "graph_id",
            "query_type",
            "rung",
            "formal_form",
            "question_property",
            "gold_label",
        ]
    ].copy()
    out["prompt_mode"] = policy
    out["prompt_condition"] = POLICY_LABELS[policy]
    out["diagnostic_source"] = "adaptive_routing"
    out["dataset_variant"] = "adaptive_routing"
    out["selected_source"] = np.where(use_symbolic, "symbolic", "nl")
    out["parsed_label"] = np.where(use_symbolic, frame["symbolic_prediction"], frame["nl_prediction"])
    out["is_invalid"] = False
    out["is_correct"] = (out["parsed_label"] == out["gold_label"]).astype(int)
    out["symbolic_selection_rate"] = float(use_symbolic.mean()) if len(out) else np.nan
    return out


def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    return (
        pred.groupby(["eval_set", "model", "model_display_name", "prompt_mode", "prompt_condition"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            symbolic_selection_rate=("selected_source", lambda x: float((x == "symbolic").mean())),
        )
        .reset_index()
    )


def bootstrap_policy_gains(pred: pd.DataFrame, seed: int, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [
        ("query_router", "always_nl"),
        ("confidence_by_query", "always_nl"),
        ("always_symbolic", "always_nl"),
        ("query_router", "always_symbolic"),
        ("confidence_by_query", "always_symbolic"),
    ]
    for (eval_set, model, display), group in pred.groupby(["eval_set", "model", "model_display_name"], dropna=False):
        for left, right in comparisons:
            ldf = group[group["prompt_mode"] == left][["sample_id", "is_correct"]].rename(columns={"is_correct": "left"})
            rdf = group[group["prompt_mode"] == right][["sample_id", "is_correct"]].rename(columns={"is_correct": "right"})
            merged = ldf.merge(rdf, on="sample_id", how="inner")
            if merged.empty:
                continue
            diff = merged["left"].astype(float).to_numpy() - merged["right"].astype(float).to_numpy()
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_bootstrap)])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            rows.append(
                {
                    "eval_set": eval_set,
                    "model": model,
                    "model_display_name": display,
                    "condition_a": left,
                    "condition_b": right,
                    "n": int(len(diff)),
                    "acc_a": float(merged["left"].mean()),
                    "acc_b": float(merged["right"].mean()),
                    "paired_accuracy_diff": float(diff.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p_diff_le_0": float((boots <= 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def contrast_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    metrics = base.compute_strict_ccc(pred)
    return metrics["metrics_ccc"]


def repeated_group_splits(frame: pd.DataFrame, n_splits: int, test_size: float, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(frame["story_id"].astype(str).unique()))
    splits = []
    n_test = max(1, min(len(groups) - 1, int(round(len(groups) * test_size))))
    for _ in range(n_splits):
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        test_groups = set(shuffled[:n_test])
        is_test = frame["story_id"].astype(str).isin(test_groups).to_numpy()
        train_idx = np.where(~is_test)[0]
        test_idx = np.where(is_test)[0]
        splits.append((train_idx, test_idx))
    return splits


def run_crossval(control: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    contrast_rows = []
    policies = ["always_nl", "always_symbolic", "query_router", "confidence_by_query"]
    for model, model_df in control.groupby("model", dropna=False):
        splits = repeated_group_splits(model_df, args.splits, args.test_size, args.seed)
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            train = model_df.iloc[train_idx].copy()
            test = model_df.iloc[test_idx].copy()
            params = fit_policies(train)
            split_pred = []
            for policy in policies:
                current = apply_policy(test, policy, params)
                current["eval_set"] = "control_cv"
                current["split"] = split_idx
                split_pred.append(current)
                metric_rows.append(
                    {
                        "eval_set": "control_cv",
                        "split": split_idx,
                        "model": model,
                        "model_display_name": str(current["model_display_name"].iloc[0]),
                        "prompt_mode": policy,
                        "prompt_condition": POLICY_LABELS[policy],
                        "n": int(len(current)),
                        "accuracy": float(current["is_correct"].mean()),
                        "symbolic_selection_rate": float((current["selected_source"] == "symbolic").mean()),
                    }
                )
            ccc = contrast_metrics(pd.concat(split_pred, ignore_index=True))
            ccc["split"] = split_idx
            ccc["eval_set"] = "control_cv"
            contrast_rows.append(ccc)
    return pd.DataFrame(metric_rows), pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame()


def summarize_crossval(split_metrics: pd.DataFrame, baseline: str = "always_nl") -> pd.DataFrame:
    if split_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (model, display, policy, label), group in split_metrics.groupby(
        ["model", "model_display_name", "prompt_mode", "prompt_condition"],
        dropna=False,
    ):
        base_df = split_metrics[
            (split_metrics["model"] == model)
            & (split_metrics["prompt_mode"] == baseline)
        ][["split", "accuracy"]].rename(columns={"accuracy": "baseline_accuracy"})
        merged = group.merge(base_df, on="split", how="inner")
        gains = merged["accuracy"].to_numpy() - merged["baseline_accuracy"].to_numpy()
        rows.append(
            {
                "eval_set": "control_cv",
                "model": model,
                "model_display_name": display,
                "prompt_mode": policy,
                "prompt_condition": label,
                "n_splits": int(group["split"].nunique()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "accuracy_ci_low": float(group["accuracy"].quantile(0.025)),
                "accuracy_ci_high": float(group["accuracy"].quantile(0.975)),
                "mean_gain_vs_nl": float(gains.mean()),
                "gain_ci_low": float(np.quantile(gains, 0.025)),
                "gain_ci_high": float(np.quantile(gains, 0.975)),
                "p_gain_le_0": float((gains <= 0).mean()),
                "mean_symbolic_selection_rate": float(group["symbolic_selection_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def train_on_control_apply_to_main(control: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    policies = ["always_nl", "always_symbolic", "query_router", "confidence_by_query"]
    control_ids = set(control["sample_id"].astype(int).unique())
    frames = []
    for model, train in control.groupby("model", dropna=False):
        params = fit_policies(train)
        test_all = main[main["model"] == model].copy()
        test_nonoverlap = test_all[~test_all["sample_id"].astype(int).isin(control_ids)].copy()
        for eval_set, test in [("main_full_transfer", test_all), ("main_nonoverlap_transfer", test_nonoverlap)]:
            if test.empty:
                continue
            for policy in policies:
                current = apply_policy(test, policy, params)
                current["eval_set"] = eval_set
                current["split"] = -1
                frames.append(current)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_report(paths: base.RunPaths, summary: pd.DataFrame, gains: pd.DataFrame, cv_summary: pd.DataFrame, ccc: pd.DataFrame) -> None:
    lines = [
        "# MLISE 2026 自适应 Symbolic Routing 补实验",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 训练信号：2048 条 matched symbolic control 的 NL 与 symbolic binary score。",
        "- 测试设置：主评测 640 样本，以及去除 calibration 重叠样本后的 non-overlap 子集。",
        "- 路由策略：Query Router 按 query type 选择 NL 或 symbolic；Confidence Router 在每个 query type 内按 yes/no margin 差值选择输入源。",
        "",
        "## Transfer Accuracy",
        "",
        base.table_to_markdown(summary.round(4)),
        "",
        "## Transfer Paired Bootstrap",
        "",
        base.table_to_markdown(gains.round(4)),
        "",
        "## Control Cross-Validation",
        "",
        base.table_to_markdown(cv_summary.round(4)),
        "",
        "## Transfer Strict Contrast Metrics",
        "",
        base.table_to_markdown(ccc.round(4)),
        "",
    ]
    (paths.report_dir / "MLISE2026_adaptive_symbolic_routing_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive routing between NL and symbolic binary scores")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", default=str(base.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = base.get_paths(Path(args.output_root), args.run_id)
    base.ensure_dirs(paths)
    destination = out_dir(paths)
    save_json(paths.run_dir / "adaptive_symbolic_routing_config.json", asdict(RoutingConfig(**vars(args))))
    control = load_control(paths)
    main_df = load_main(paths)
    cv_split_metrics, cv_contrast = run_crossval(control, args)
    cv_summary = summarize_crossval(cv_split_metrics)
    transfer_pred = train_on_control_apply_to_main(control, main_df)
    transfer_summary = summarize_predictions(transfer_pred)
    transfer_gains = bootstrap_policy_gains(transfer_pred, args.seed, max(1, args.bootstrap))
    transfer_ccc = contrast_metrics(transfer_pred)
    cv_split_metrics.to_csv(destination / "routing_control_cv_split_metrics.csv", index=False)
    cv_contrast.to_csv(destination / "routing_control_cv_contrast_metrics.csv", index=False)
    cv_summary.to_csv(destination / "routing_control_cv_summary.csv", index=False)
    transfer_pred.to_csv(destination / "routing_transfer_predictions.csv", index=False)
    transfer_summary.to_csv(destination / "routing_transfer_summary.csv", index=False)
    transfer_gains.to_csv(destination / "routing_transfer_paired_bootstrap.csv", index=False)
    transfer_ccc.to_csv(destination / "routing_transfer_ccc.csv", index=False)
    write_report(paths, transfer_summary, transfer_gains, cv_summary, transfer_ccc)
    print(f"[完成] 自适应 symbolic routing 结果已写入：{destination}", flush=True)


if __name__ == "__main__":
    main()
