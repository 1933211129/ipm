#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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


INTERVENTION_LABELS = {
    "symbolic_solver": "Symbolic Solver",
    "symbolic_solver_concise": "Concise Symbolic Solver",
}
INTERVENTION_MODE = "symbolic_solver_concise"
INTERVENTION_LABEL = INTERVENTION_LABELS[INTERVENTION_MODE]
DEFAULT_RUN_ID = "final_20260513_090357"


@dataclass
class InterventionConfig:
    stage: str
    model: str
    run_id: str
    output_root: str
    seed: int
    batch_size: int
    max_new_tokens: int
    bootstrap: int
    resume: bool
    limit: int


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def model_table_dir(paths: base.RunPaths, model_key: str) -> Path:
    path = paths.table_dir / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_subset_path(paths: base.RunPaths) -> Path:
    direct = paths.table_dir / "selected_main_subset.csv"
    if direct.exists():
        return direct
    formal = paths.table_dir / "selected_main_subset_formal.csv"
    if formal.exists():
        return formal
    raise FileNotFoundError(f"没有找到主评测子集：{direct}")


def is_nan_line(line: str) -> bool:
    return line.strip().lower() in {"nan", "none", "null"}


def is_final_numeric_line(line: str) -> bool:
    text = line.strip()
    compact = text.replace("−", "-")
    if not text:
        return True
    if re.fullmatch(r"[01]", text):
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*(?:>|<|>=|<=|=)\s*0(?:\.0+)?", compact):
        return True
    if re.match(r"^[-+]?\d", compact) and "=" in compact and any(op in compact for op in ["+", "-", "*", "/"]):
        return True
    if re.match(r"^[A-Za-z]\w*\s*=", compact) and compact.count("=") >= 2:
        return True
    return False


def is_probability_fact(line: str) -> bool:
    return bool(re.search(r"\bP\(", line)) and "=" in line


def make_symbolic_decomposition(row: dict[str, Any]) -> tuple[str, str]:
    raw_lines = [line.strip() for line in str(row.get("reasoning", "")).splitlines() if line.strip()]
    raw_lines = [line for line in raw_lines if not is_nan_line(line)]
    kept: list[str] = []
    removed_final = 0
    for line in raw_lines:
        if is_final_numeric_line(line):
            removed_final += 1
            continue
        kept.append(line)

    scaffold = base.extract_scaffold(row)
    fallback_lines = [
        scaffold.get("variables", ""),
        scaffold.get("graph", ""),
        scaffold.get("formal_query", ""),
    ]
    fallback_lines = [line for line in fallback_lines if line and not is_nan_line(line)]

    has_formula_or_fact = any(is_probability_fact(line) or "E[" in line or "P(" in line for line in kept)
    has_deterministic_trace = (
        str(row.get("query_type", "")) == "det-counterfactual"
        and any("Solve for" in line for line in kept)
        and any("=" in line and "and" in line for line in kept)
    )
    if len(kept) < 3 or not (has_formula_or_fact or has_deterministic_trace):
        kept = fallback_lines
        source = "scaffold_only"
    else:
        source = "trace_without_final_result"

    normalized: list[str] = []
    for line in kept:
        if line not in normalized:
            normalized.append(line)
    if not normalized:
        normalized = ["No symbolic trace is available for this item."]
        source = "unavailable"
    return "\n".join(f"- {line}" for line in normalized), source


def build_symbolic_solver_prompt(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row["prompt"]).strip()
    decomposition, trace_source = make_symbolic_decomposition(row)
    if INTERVENTION_MODE == "symbolic_solver_concise":
        prompt = (
            "You are a concise causal yes/no solver.\n"
            "Use the question to decide what yes/no means. Use the symbolic decomposition to compute the causal quantity.\n"
            "The decomposition omits the final numeric comparison and the final answer.\n"
            "Output at most three short lines, and the last line must be exactly: Final answer: yes/no\n"
            "Do not add any text after the final answer.\n\n"
            f"Question:\n{question}\n\n"
            "Symbolic decomposition:\n"
            f"{decomposition}\n\n"
            f"Query type: {row.get('query_type', '')}\n"
            f"Rung: {row.get('rung', '')}"
        )
    else:
        prompt = (
            "You are solving a causal yes/no question with a symbolic causal decomposition.\n"
            "Use the natural-language question to decide what yes/no means. Use the decomposition to compute the causal quantity.\n"
            "The decomposition may include variable mapping, graph structure, formal query, formula templates, and probability facts. "
            "It intentionally omits the final numeric comparison and the final yes/no answer.\n"
            "Follow this protocol: identify the requested direction or threshold, compute the relevant quantity from the provided facts, "
            "compare it with the requested condition, and then answer.\n"
            "End with exactly one final line in this format: Final answer: yes/no\n\n"
            f"Question:\n{question}\n\n"
            "Symbolic causal decomposition:\n"
            f"{decomposition}\n\n"
            f"Query type: {row.get('query_type', '')}\n"
            f"Rung: {row.get('rung', '')}"
        )
    return prompt, trace_source


def read_existing_completed(save_path: Path, resume: bool) -> tuple[pd.DataFrame, set[tuple[str, int]]]:
    if not save_path.exists():
        return pd.DataFrame(), set()
    if not resume:
        raise FileExistsError(f"结果文件已存在，避免覆盖：{save_path}。如需续跑请加 --resume。")
    existing = pd.read_csv(save_path)
    completed = set(zip(existing["dataset_variant"].astype(str), existing["sample_id"].astype(int)))
    print(f"[信息] resume 已读取 {len(existing)} 行：{save_path}", flush=True)
    return existing, completed


def run_symbolic_solver_eval(
    evaluator: base.QwenEvaluator,
    df: pd.DataFrame,
    model_key: str,
    run_id: str,
    batch_size: int,
    save_path: Path,
    resume: bool,
) -> pd.DataFrame:
    existing, completed = read_existing_completed(save_path, resume)
    source_rows = [
        row
        for row in df.to_dict("records")
        if (str(row["dataset_variant"]), int(row["id"])) not in completed
    ]
    print(f"[进度] model={model_key} mode={INTERVENTION_MODE} 待评测={len(source_rows)}", flush=True)

    records: list[dict[str, Any]] = []
    batch_size_current = batch_size
    i = 0
    while i < len(source_rows):
        current_rows = source_rows[i : i + batch_size_current]
        built = [build_symbolic_solver_prompt(row) for row in current_rows]
        prompts = [item[0] for item in built]
        trace_sources = [item[1] for item in built]
        try:
            generated = evaluator.generate_batch(prompts)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message and batch_size_current > 1:
                batch_size_current = max(1, batch_size_current // 2)
                print(f"[警告] 显存不足，batch size 降为 {batch_size_current}", flush=True)
                base.empty_device_cache()
                continue
            raise

        for source_row, trace_source, gen_row in zip(current_rows, trace_sources, generated):
            parsed = base.parse_yes_no(gen_row["raw_output"])
            record = {
                "run_id": run_id,
                "model": model_key,
                "model_display_name": base.MODEL_CONFIGS[model_key]["display_name"],
                "sample_id": int(source_row["id"]),
                "story_id": source_row["story_id"],
                "graph_id": source_row["graph_id"],
                "rung": int(source_row["rung"]),
                "query_type": source_row["query_type"],
                "question_property": source_row.get("question_property", ""),
                "formal_form": source_row["formal_form"],
                "gold_label": source_row["label"],
                "prompt_mode": INTERVENTION_MODE,
                "prompt_condition": INTERVENTION_LABEL,
                "dataset_variant": source_row["dataset_variant"],
                "diagnostic_source": "main",
                "subset": source_row.get("subset", ""),
                "trace_source": trace_source,
                **gen_row,
            }
            record["parsed_label"] = parsed if parsed in {"yes", "no"} else "invalid"
            record["is_invalid"] = record["parsed_label"] == "invalid"
            record["is_correct"] = int(record["parsed_label"] == record["gold_label"])
            records.append(record)

        i += len(current_rows)
        if records:
            pd.concat([existing, pd.DataFrame(records)], ignore_index=True).to_csv(save_path, index=False)

    if records:
        return pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    return existing


def run_model(paths: base.RunPaths, args: argparse.Namespace, model_key: str) -> None:
    subset = pd.read_csv(selected_subset_path(paths))
    subset["id"] = subset["id"].astype(int)
    subset["rung"] = subset["rung"].astype(int)
    subset["label"] = subset["label"].str.lower().str.strip()
    subset["dataset_variant"] = subset.get("dataset_variant", "main")
    subset["diagnostic_source"] = "main"
    if args.limit > 0:
        subset = subset.head(args.limit).copy()
    out_dir = model_table_dir(paths, model_key)
    save_path = out_dir / f"{INTERVENTION_MODE}_eval.csv"
    model_cfg = base.MODEL_CONFIGS[model_key]
    evaluator = base.QwenEvaluator(
        model_key=model_key,
        model_path=Path(model_cfg["path"]),
        hf_name=model_cfg["hf_name"],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        eval_df = run_symbolic_solver_eval(
            evaluator=evaluator,
            df=subset,
            model_key=model_key,
            run_id=args.run_id,
            batch_size=args.batch_size,
            save_path=save_path,
            resume=args.resume,
        )
    finally:
        evaluator.close()
    metrics_dir = out_dir / f"{INTERVENTION_MODE}_metrics"
    base.save_metrics(eval_df, metrics_dir, bootstrap=args.bootstrap, seed=args.seed)
    save_json(
        out_dir / f"{INTERVENTION_MODE}_config.json",
        {
            "run_id": args.run_id,
            "model": model_key,
            "mode": INTERVENTION_MODE,
            "sample_n": int(len(subset)),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "bootstrap": args.bootstrap,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"[完成] {model_key} symbolic solver 结果：{save_path}", flush=True)


def read_model_tables(paths: base.RunPaths, filename: str) -> pd.DataFrame:
    frames = []
    for model_key in base.MODEL_ORDER:
        path = paths.table_dir / model_key / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_intervention_comparison(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = base.compute_summary_metrics(combined)
    ccc = base.compute_strict_ccc(combined)
    summary = metrics["metrics_summary"]
    ccc_summary = ccc["metrics_ccc"]
    merged = summary.merge(
        ccc_summary[
            [
                "model",
                "model_display_name",
                "diagnostic_source",
                "dataset_variant",
                "prompt_mode",
                "strict_ccc",
                "correct_flip_rate",
                "wrong_flip_rate",
                "scca",
                "signed_ccc",
            ]
        ],
        on=["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"],
        how="left",
    )
    keep_modes = ["nl", "nl_formal", INTERVENTION_MODE]
    merged = merged[merged["prompt_mode"].isin(keep_modes)].copy()
    model_order = {key: idx for idx, key in enumerate(base.MODEL_ORDER)}
    merged["model_order"] = merged["model"].map(model_order)
    merged["prompt_order"] = merged["prompt_mode"].map({"nl": 0, "nl_formal": 1, INTERVENTION_MODE: 2})
    merged = merged.sort_values(["model_order", "prompt_order"]).drop(columns=["model_order", "prompt_order"])

    rows = []
    for (model, display), group in merged.groupby(["model", "model_display_name"], dropna=False):
        acc = {row["prompt_mode"]: float(row["accuracy"]) for _, row in group.iterrows()}
        ccc_map = {row["prompt_mode"]: float(row["strict_ccc"]) for _, row in group.iterrows()}
        scca = {row["prompt_mode"]: float(row["scca"]) for _, row in group.iterrows()}
        rows.append(
            {
                "model": model,
                "model_display_name": display,
                "nl_accuracy": acc.get("nl", np.nan),
                "nl_formal_accuracy": acc.get("nl_formal", np.nan),
                "symbolic_solver_accuracy": acc.get(INTERVENTION_MODE, np.nan),
                "gain_vs_nl": acc.get(INTERVENTION_MODE, np.nan) - acc.get("nl", np.nan),
                "gain_vs_nl_formal": acc.get(INTERVENTION_MODE, np.nan) - acc.get("nl_formal", np.nan),
                "nl_strict_ccc": ccc_map.get("nl", np.nan),
                "symbolic_solver_strict_ccc": ccc_map.get(INTERVENTION_MODE, np.nan),
                "nl_scca": scca.get("nl", np.nan),
                "symbolic_solver_scca": scca.get(INTERVENTION_MODE, np.nan),
            }
        )
    comparison = pd.DataFrame(rows)

    by_query = metrics["metrics_by_query"]
    by_query = by_query[by_query["prompt_mode"].isin(keep_modes)].copy()
    query_rows = []
    for (model, display, query), group in by_query.groupby(["model", "model_display_name", "query_type"], dropna=False):
        acc = {row["prompt_mode"]: float(row["accuracy"]) for _, row in group.iterrows()}
        query_rows.append(
            {
                "model": model,
                "model_display_name": display,
                "query_type": query,
                "nl_accuracy": acc.get("nl", np.nan),
                "nl_formal_accuracy": acc.get("nl_formal", np.nan),
                "symbolic_solver_accuracy": acc.get(INTERVENTION_MODE, np.nan),
                "gain_vs_nl": acc.get(INTERVENTION_MODE, np.nan) - acc.get("nl", np.nan),
                "gain_vs_nl_formal": acc.get(INTERVENTION_MODE, np.nan) - acc.get("nl_formal", np.nan),
            }
        )
    by_query_comp = pd.DataFrame(query_rows)

    transition_rows = []
    key_cols = ["model", "model_display_name", "sample_id", "dataset_variant", "diagnostic_source"]
    nl = combined[combined["prompt_mode"] == "nl"][
        key_cols + ["query_type", "gold_label", "parsed_label", "is_correct", "is_invalid"]
    ].rename(
        columns={
            "parsed_label": "nl_parsed_label",
            "is_correct": "nl_correct",
            "is_invalid": "nl_invalid",
        }
    )
    sym = combined[combined["prompt_mode"] == INTERVENTION_MODE][key_cols + ["parsed_label", "is_correct", "is_invalid"]].rename(
        columns={
            "parsed_label": "symbolic_parsed_label",
            "is_correct": "symbolic_correct",
            "is_invalid": "symbolic_invalid",
        }
    )
    joined = nl.merge(sym, on=key_cols, how="inner")
    for (model, display), group in joined.groupby(["model", "model_display_name"], dropna=False):
        valid = group[(~group["nl_invalid"].astype(bool)) & (~group["symbolic_invalid"].astype(bool))]
        changed = valid["nl_parsed_label"] != valid["symbolic_parsed_label"]
        rescue = (valid["nl_correct"].astype(int).eq(0)) & (valid["symbolic_correct"].astype(int).eq(1))
        harm = (valid["nl_correct"].astype(int).eq(1)) & (valid["symbolic_correct"].astype(int).eq(0))
        transition_rows.append(
            {
                "model": model,
                "model_display_name": display,
                "n_valid": int(len(valid)),
                "prediction_change_count": int(changed.sum()),
                "prediction_change_rate": float(changed.mean()) if len(valid) else np.nan,
                "rescue_count": int(rescue.sum()),
                "harm_count": int(harm.sum()),
                "net_rescue": int(rescue.sum() - harm.sum()),
                "net_rescue_rate": float((rescue.sum() - harm.sum()) / len(valid)) if len(valid) else np.nan,
            }
        )
    transition = pd.DataFrame(transition_rows)
    return {
        "metrics_summary": merged,
        "metrics_ccc": ccc_summary,
        "intervention_comparison": comparison,
        "intervention_by_query": by_query_comp,
        "intervention_rescue_harm_vs_nl": transition,
        "metrics_accuracy_bootstrap_ci": base.compute_accuracy_bootstrap_ci(combined, n_bootstrap=1000, seed=base.SEED),
    }


def set_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def make_intervention_figures(paths: base.RunPaths, metrics: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    set_plot_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    refs: list[tuple[str, str]] = []
    fig_dir = paths.figure_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = metrics["metrics_summary"].copy()
    if not summary.empty:
        summary["prompt_condition"] = summary["prompt_mode"].map(
            {"nl": "Natural Language", "nl_formal": "Formal Scaffold", INTERVENTION_MODE: INTERVENTION_LABEL}
        )
        summary["model_display_name"] = pd.Categorical(
            summary["model_display_name"],
            categories=[base.MODEL_CONFIGS[key]["display_name"] for key in base.MODEL_ORDER],
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=summary, x="model_display_name", y="accuracy", hue="prompt_condition", ax=ax)
        ax.set_title("Accuracy with Symbolic Solver")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Input Condition", loc="best")
        fig.tight_layout()
        path = fig_dir / "11_symbolic_solver_accuracy.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Accuracy with Symbolic Solver", str(path.relative_to(paths.run_dir))))

    by_query = metrics["intervention_by_query"].copy()
    if not by_query.empty:
        plot_df = by_query.pivot(index="query_type", columns="model_display_name", values="gain_vs_nl").reset_index()
        melt = plot_df.melt(id_vars=["query_type"], var_name="Model", value_name="Gain vs Natural Language")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.barplot(data=melt, x="query_type", y="Gain vs Natural Language", hue="Model", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Symbolic Solver Gain by Query Type")
        ax.set_xlabel("Query Type")
        ax.set_ylabel("Accuracy Gain")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = fig_dir / "12_symbolic_solver_gain_by_query.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        refs.append(("Symbolic Solver Gain by Query Type", str(path.relative_to(paths.run_dir))))

    return refs


def table_md(df: pd.DataFrame) -> str:
    return base.table_to_markdown(df)


def write_intervention_report(paths: base.RunPaths, metrics: dict[str, pd.DataFrame], figure_refs: list[tuple[str, str]]) -> None:
    comp = metrics["intervention_comparison"].copy()
    summary = metrics["metrics_summary"].copy()
    by_query = metrics["intervention_by_query"].copy()
    transition = metrics["intervention_rescue_harm_vs_nl"].copy()
    ci = metrics["metrics_accuracy_bootstrap_ci"].copy()
    for frame in [comp, summary, by_query, transition, ci]:
        for col in frame.select_dtypes(include=[float]).columns:
            frame[col] = frame[col].round(4)
    lines = [
        "# MLISE 2026 符号分解辅助实验结果",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- run_id：`{paths.run_dir.name}`",
        "- 输入条件：`symbolic_solver`，即自然语言题干加去答案化的符号因果分解。",
        "- 图表标题与坐标轴使用英文；本报告正文使用中文。",
        "",
        "## 方法定义",
        "",
        "符号分解辅助输入保留自然语言问题，并附加变量映射、因果图、形式查询、公式模板和可用概率事实。生成输入时删除最终数值比较、最终符号判断和二值答案行；模型需要先识别问题要求的方向或阈值，再根据概率事实完成计算，最后输出 `Final answer: yes/no`。",
        "",
        "该输入不是把答案写入 prompt，而是把 CLadder 问题中本来隐含的因果计算对象显式化，目标是检验结构化中间表示能否把形式信息转化为更稳定的行为收益。",
        "",
        "## 总体结果",
        "",
        table_md(comp),
        "",
        "## 条件级指标",
        "",
        table_md(summary),
        "",
        "## Query Type 细分",
        "",
        table_md(by_query),
        "",
        "## 相对自然语言条件的样本迁移",
        "",
        table_md(transition),
        "",
        "## Accuracy Bootstrap 置信区间",
        "",
        table_md(ci[ci["prompt_mode"].isin(["nl", "nl_formal", INTERVENTION_MODE])]),
        "",
        "## 图表索引",
        "",
    ]
    if figure_refs:
        for title, rel_path in figure_refs:
            lines.extend([f"### {title}", "", f"![{title}](../{rel_path})", ""])
    else:
        lines.append("当前未生成图表。")
    lines.extend(
        [
            "",
            "## 论文结论写作要点",
            "",
            "若 `symbolic_solver` 相对 `nl` 或 `nl_formal` 获得正向提升，正文应将核心贡献写为：形式信息的有效性取决于中间表示是否把查询方向、估计量和概率事实组织成可执行的计算过程。相较于只加入图结构或形式查询，符号分解辅助输入直接减少了从叙事到因果估计量的映射负担，因此更可能产生稳定收益。",
            "",
            "若总体提升有限但在部分 query type 上明显提升，正文应避免写成全面有效，而应写成受任务类型约束的可行路径：符号分解在估计量明确、概率事实完整的因果问题中更有效；在 `backadj` 等缺少可用 trace 的类型上，收益受到输入结构质量限制。",
            "",
        ]
    )
    out_path = paths.report_dir / "MLISE2026_symbolic_intervention_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] 符号分解辅助报告：{out_path}", flush=True)


def aggregate(paths: base.RunPaths, args: argparse.Namespace) -> None:
    main_eval = read_model_tables(paths, "main_eval.csv")
    symbolic_eval = read_model_tables(paths, f"{INTERVENTION_MODE}_eval.csv")
    if main_eval.empty:
        raise FileNotFoundError("未找到 main_eval.csv，无法与自然语言基线比较。")
    if symbolic_eval.empty:
        raise FileNotFoundError(f"未找到 {INTERVENTION_MODE}_eval.csv，无法聚合干预实验。")
    combined = pd.concat(
        [
            main_eval[main_eval["prompt_mode"].isin(["nl", "nl_formal"])],
            symbolic_eval,
        ],
        ignore_index=True,
    )
    aggregate_dir = paths.table_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(aggregate_dir / f"all_{INTERVENTION_MODE}_eval.csv", index=False)
    metrics = compute_intervention_comparison(combined)
    metrics_dir = aggregate_dir / f"{INTERVENTION_MODE}_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, table in metrics.items():
        table.to_csv(metrics_dir / f"{name}.csv", index=False)
    figure_refs = make_intervention_figures(paths, metrics)
    write_intervention_report(paths, metrics, figure_refs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLISE 2026 符号因果分解辅助实验")
    parser.add_argument("--stage", choices=["run", "aggregate", "all"], default="all")
    parser.add_argument("--model", choices=base.MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=list(INTERVENTION_LABELS), default="symbolic_solver_concise")
    parser.add_argument("--output-root", default=str(base.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    global INTERVENTION_MODE, INTERVENTION_LABEL
    args = parse_args()
    INTERVENTION_MODE = args.mode
    INTERVENTION_LABEL = INTERVENTION_LABELS[args.mode]
    base.seed_everything(args.seed)
    paths = base.get_paths(Path(args.output_root), args.run_id)
    base.ensure_dirs(paths)
    config = InterventionConfig(
        stage=args.stage,
        model=args.model,
        run_id=args.run_id,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        bootstrap=args.bootstrap,
        resume=args.resume,
        limit=args.limit,
    )
    save_json(paths.run_dir / "symbolic_intervention_config.json", asdict(config))

    model_keys = base.model_keys_from_arg(args.model)
    if args.stage in {"run", "all"}:
        for model_key in model_keys:
            run_model(paths, args, model_key)
            base.empty_device_cache()
            time.sleep(1)
    if args.stage in {"aggregate", "all"}:
        aggregate(paths, args)


if __name__ == "__main__":
    main()
