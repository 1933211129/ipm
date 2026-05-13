#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


SCORE_MODES = ["nl", "nl_formal", "symbolic_solver_concise"]
MODE_LABELS = {
    "nl": "NL Binary Score",
    "nl_formal": "Formal Scaffold Binary Score",
    "symbolic_solver_concise": "Concise Symbolic Solver Binary Score",
}
DEFAULT_RUN_ID = "final_20260513_090357"


@dataclass
class BinaryScoreConfig:
    stage: str
    model: str
    run_id: str
    output_root: str
    modes: str
    seed: int
    batch_size: int
    bootstrap: int
    resume: bool
    limit: int


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_subset_path(paths: base.RunPaths) -> Path:
    direct = paths.table_dir / "selected_main_subset.csv"
    if direct.exists():
        return direct
    formal = paths.table_dir / "selected_main_subset_formal.csv"
    if formal.exists():
        return formal
    raise FileNotFoundError(f"没有找到主评测子集：{direct}")


def model_table_dir(paths: base.RunPaths, model_key: str) -> Path:
    path = paths.table_dir / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_prompt(row: dict[str, Any], mode: str) -> tuple[str, str]:
    if mode in {"nl", "nl_formal"}:
        return base.build_user_prompt(row, mode), "generation_prompt"
    if mode == "symbolic_solver_concise":
        symbolic.INTERVENTION_MODE = "symbolic_solver_concise"
        symbolic.INTERVENTION_LABEL = symbolic.INTERVENTION_LABELS["symbolic_solver_concise"]
        prompt, trace_source = symbolic.build_symbolic_solver_prompt(row)
        return prompt, trace_source
    raise ValueError(f"未知 scoring mode: {mode}")


class BinaryScorer:
    def __init__(self, model_key: str, model_path: Path, batch_size: int) -> None:
        base.ensure_model_runtime()
        self.model_key = model_key
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.device = base.get_device()
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在：{model_path}")

        self.tokenizer = base.AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = base.torch.bfloat16 if self.device == "cuda" else "auto"
        self.model = base.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        base.empty_device_cache()

    def format_prefix(self, prompt: str) -> str:
        try:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return formatted + "Final answer:"

    def score_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        torch = base.torch
        candidates = {"yes": " yes", "no": " no"}
        seq_records: list[dict[str, Any]] = []
        for prompt_idx, prompt in enumerate(prompts):
            prefix = self.format_prefix(prompt)
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            for label, suffix in candidates.items():
                cand_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
                full_ids = prefix_ids + cand_ids
                seq_records.append(
                    {
                        "prompt_idx": prompt_idx,
                        "label": label,
                        "ids": full_ids,
                        "candidate_start": len(prefix_ids),
                        "seq_len": len(full_ids),
                    }
                )

        max_len = max(item["seq_len"] for item in seq_records)
        pad_id = self.tokenizer.pad_token_id
        input_rows = []
        attention_rows = []
        for item in seq_records:
            pad_len = max_len - item["seq_len"]
            input_rows.append(item["ids"] + [pad_id] * pad_len)
            attention_rows.append([1] * item["seq_len"] + [0] * pad_len)

        input_ids = torch.tensor(input_rows, device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(attention_rows, device=self.device, dtype=torch.long)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        elapsed = time.perf_counter() - start

        scores_by_prompt: list[dict[str, Any]] = [
            {"yes_logprob": np.nan, "no_logprob": np.nan, "latency_sec": elapsed / max(1, len(prompts))}
            for _ in prompts
        ]
        log_probs = torch.log_softmax(logits, dim=-1)
        for row_idx, item in enumerate(seq_records):
            score = 0.0
            for pos in range(item["candidate_start"], item["seq_len"]):
                token_id = int(input_ids[row_idx, pos].item())
                score += float(log_probs[row_idx, pos - 1, token_id].item())
            scores_by_prompt[item["prompt_idx"]][f"{item['label']}_logprob"] = score

        rows = []
        for score_row in scores_by_prompt:
            yes_score = float(score_row["yes_logprob"])
            no_score = float(score_row["no_logprob"])
            pred = "yes" if yes_score >= no_score else "no"
            rows.append(
                {
                    "parsed_label": pred,
                    "yes_logprob": yes_score,
                    "no_logprob": no_score,
                    "yes_no_margin": yes_score - no_score,
                    "latency_sec": score_row["latency_sec"],
                }
            )
        return rows


def read_completed(save_path: Path, resume: bool) -> tuple[pd.DataFrame, set[tuple[str, int, str]]]:
    if not save_path.exists():
        return pd.DataFrame(), set()
    if not resume:
        raise FileExistsError(f"结果文件已存在，避免覆盖：{save_path}。如需续跑请加 --resume。")
    existing = pd.read_csv(save_path)
    completed = set(
        zip(
            existing["dataset_variant"].astype(str),
            existing["sample_id"].astype(int),
            existing["prompt_mode"].astype(str),
        )
    )
    print(f"[信息] resume 已读取 {len(existing)} 行：{save_path}", flush=True)
    return existing, completed


def run_binary_score(
    scorer: BinaryScorer,
    df: pd.DataFrame,
    model_key: str,
    run_id: str,
    modes: list[str],
    batch_size: int,
    save_path: Path,
    resume: bool,
) -> pd.DataFrame:
    existing, completed = read_completed(save_path, resume)
    records: list[dict[str, Any]] = []
    for mode in modes:
        source_rows = [
            row
            for row in df.to_dict("records")
            if (str(row["dataset_variant"]), int(row["id"]), f"{mode}_binary_score") not in completed
        ]
        print(f"[进度] model={model_key} mode={mode}_binary_score 待评分={len(source_rows)}", flush=True)
        i = 0
        while i < len(source_rows):
            current_rows = source_rows[i : i + batch_size]
            built = [build_prompt(row, mode) for row in current_rows]
            prompts = [item[0] for item in built]
            trace_sources = [item[1] for item in built]
            scored = scorer.score_batch(prompts)
            for source_row, trace_source, score_row in zip(current_rows, trace_sources, scored):
                prompt_mode = f"{mode}_binary_score"
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
                    "prompt_mode": prompt_mode,
                    "prompt_condition": MODE_LABELS[mode],
                    "dataset_variant": source_row["dataset_variant"],
                    "diagnostic_source": "main",
                    "subset": source_row.get("subset", ""),
                    "trace_source": trace_source,
                    "raw_output": "",
                    "input_tokens": np.nan,
                    "output_tokens": 1,
                    **score_row,
                }
                record["is_invalid"] = False
                record["is_correct"] = int(record["parsed_label"] == record["gold_label"])
                records.append(record)
            i += len(current_rows)
            if records:
                pd.concat([existing, pd.DataFrame(records)], ignore_index=True).to_csv(save_path, index=False)
    if records:
        return pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    return existing


def run_model(paths: base.RunPaths, args: argparse.Namespace, model_key: str, modes: list[str]) -> None:
    subset = pd.read_csv(selected_subset_path(paths))
    subset["id"] = subset["id"].astype(int)
    subset["rung"] = subset["rung"].astype(int)
    subset["label"] = subset["label"].str.lower().str.strip()
    subset["dataset_variant"] = subset.get("dataset_variant", "main")
    subset["diagnostic_source"] = "main"
    if args.limit > 0:
        subset = subset.head(args.limit).copy()
    out_dir = model_table_dir(paths, model_key)
    save_path = out_dir / "binary_score_eval.csv"
    model_cfg = base.MODEL_CONFIGS[model_key]
    scorer = BinaryScorer(
        model_key=model_key,
        model_path=Path(model_cfg["path"]),
        batch_size=args.batch_size,
    )
    try:
        eval_df = run_binary_score(
            scorer=scorer,
            df=subset,
            model_key=model_key,
            run_id=args.run_id,
            modes=modes,
            batch_size=args.batch_size,
            save_path=save_path,
            resume=args.resume,
        )
    finally:
        scorer.close()
    metrics_dir = out_dir / "binary_score_metrics"
    base.save_metrics(eval_df, metrics_dir, bootstrap=args.bootstrap, seed=args.seed)
    save_json(
        out_dir / "binary_score_config.json",
        {
            "run_id": args.run_id,
            "model": model_key,
            "modes": modes,
            "sample_n": int(len(subset)),
            "batch_size": args.batch_size,
            "bootstrap": args.bootstrap,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"[完成] {model_key} binary score 结果：{save_path}", flush=True)


def read_model_tables(paths: base.RunPaths, filename: str) -> pd.DataFrame:
    frames = []
    for model_key in base.MODEL_ORDER:
        path = paths.table_dir / model_key / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def comparison_tables(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = base.compute_summary_metrics(combined)["metrics_summary"]
    ccc = base.compute_strict_ccc(combined)["metrics_ccc"]
    ccc_cols = [
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
    merged = summary.merge(ccc[ccc_cols], on=["model", "model_display_name", "diagnostic_source", "dataset_variant", "prompt_mode"], how="left")
    keep = [
        "nl",
        "nl_formal",
        "symbolic_solver_concise",
        "nl_binary_score",
        "nl_formal_binary_score",
        "symbolic_solver_concise_binary_score",
    ]
    merged = merged[merged["prompt_mode"].isin(keep)].copy()
    model_order = {key: idx for idx, key in enumerate(base.MODEL_ORDER)}
    prompt_order = {mode: idx for idx, mode in enumerate(keep)}
    merged["model_order"] = merged["model"].map(model_order)
    merged["prompt_order"] = merged["prompt_mode"].map(prompt_order)
    merged = merged.sort_values(["model_order", "prompt_order"]).drop(columns=["model_order", "prompt_order"])

    rows = []
    for (model, display), group in merged.groupby(["model", "model_display_name"], dropna=False):
        acc = {row["prompt_mode"]: float(row["accuracy"]) for _, row in group.iterrows()}
        rows.append(
            {
                "model": model,
                "model_display_name": display,
                "nl_generation": acc.get("nl", np.nan),
                "nl_formal_generation": acc.get("nl_formal", np.nan),
                "symbolic_generation": acc.get("symbolic_solver_concise", np.nan),
                "nl_binary_score": acc.get("nl_binary_score", np.nan),
                "nl_formal_binary_score": acc.get("nl_formal_binary_score", np.nan),
                "symbolic_binary_score": acc.get("symbolic_solver_concise_binary_score", np.nan),
                "best_binary_gain_vs_nl": max(
                    acc.get("nl_binary_score", np.nan),
                    acc.get("nl_formal_binary_score", np.nan),
                    acc.get("symbolic_solver_concise_binary_score", np.nan),
                )
                - acc.get("nl", np.nan),
            }
        )
    comp = pd.DataFrame(rows)

    by_query = base.compute_summary_metrics(combined)["metrics_by_query"]
    by_query = by_query[by_query["prompt_mode"].isin(keep)].copy()
    query_rows = []
    for (model, display, query), group in by_query.groupby(["model", "model_display_name", "query_type"], dropna=False):
        acc = {row["prompt_mode"]: float(row["accuracy"]) for _, row in group.iterrows()}
        query_rows.append(
            {
                "model": model,
                "model_display_name": display,
                "query_type": query,
                "nl_generation": acc.get("nl", np.nan),
                "symbolic_generation": acc.get("symbolic_solver_concise", np.nan),
                "nl_binary_score": acc.get("nl_binary_score", np.nan),
                "nl_formal_binary_score": acc.get("nl_formal_binary_score", np.nan),
                "symbolic_binary_score": acc.get("symbolic_solver_concise_binary_score", np.nan),
            }
        )
    by_query_comp = pd.DataFrame(query_rows)
    return {
        "binary_score_summary": merged,
        "binary_score_comparison": comp,
        "binary_score_by_query": by_query_comp,
        "binary_score_accuracy_ci": base.compute_accuracy_bootstrap_ci(combined, n_bootstrap=1000, seed=base.SEED),
    }


def table_md(df: pd.DataFrame) -> str:
    return base.table_to_markdown(df)


def write_report(paths: base.RunPaths, tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# MLISE 2026 二分类似然打分实验结果",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- run_id：`{paths.run_dir.name}`",
        "- 方法：对 `yes` 与 `no` 两个候选 continuation 进行条件 log-likelihood 打分，选择分数更高者作为答案。",
        "- 该方法不依赖自由文本解析，所有样本的解析率为 100%。",
        "",
        "## 方法定义",
        "",
        "设输入提示为 p，候选答案集合为 A={yes,no}。二分类似然打分选择：",
        "",
        "$$",
        "\\hat{y}=\\arg\\max_{a\\in A}\\log P_\\theta(a\\mid p,\\text{``Final answer:''}).",
        "$$",
        "",
        "其中 p 可以是自然语言题干、形式脚手架题干，或符号分解辅助题干。该设置把答案空间固定为二元集合，减少自由生成中的格式漂移和冗余推理文本。",
        "",
        "## 总体比较",
        "",
    ]
    for key in ["binary_score_comparison", "binary_score_summary", "binary_score_by_query", "binary_score_accuracy_ci"]:
        frame = tables[key].copy()
        for col in frame.select_dtypes(include=[float]).columns:
            frame[col] = frame[col].round(4)
        lines.extend([f"## {key}", "", table_md(frame), ""])
    out_path = paths.report_dir / "MLISE2026_binary_score_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] 二分类似然打分报告：{out_path}", flush=True)


def aggregate(paths: base.RunPaths) -> None:
    main_eval = read_model_tables(paths, "main_eval.csv")
    symbolic_eval = read_model_tables(paths, "symbolic_solver_concise_eval.csv")
    score_eval = read_model_tables(paths, "binary_score_eval.csv")
    if main_eval.empty:
        raise FileNotFoundError("未找到 main_eval.csv。")
    if score_eval.empty:
        raise FileNotFoundError("未找到 binary_score_eval.csv。")
    frames = [main_eval[main_eval["prompt_mode"].isin(["nl", "nl_formal"])]]
    if not symbolic_eval.empty:
        frames.append(symbolic_eval)
    frames.append(score_eval)
    combined = pd.concat(frames, ignore_index=True)
    aggregate_dir = paths.table_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(aggregate_dir / "all_binary_score_eval.csv", index=False)
    tables = comparison_tables(combined)
    metrics_dir = aggregate_dir / "binary_score_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f"{name}.csv", index=False)
    write_report(paths, tables)


def parse_modes(modes_arg: str) -> list[str]:
    if modes_arg == "all":
        return SCORE_MODES.copy()
    modes = [item.strip() for item in modes_arg.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in SCORE_MODES]
    if unknown:
        raise ValueError(f"未知 scoring mode: {unknown}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLISE 2026 受约束 yes/no 二分类似然打分")
    parser.add_argument("--stage", choices=["run", "aggregate", "all"], default="all")
    parser.add_argument("--model", choices=base.MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", default=str(base.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--modes", default="all")
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.seed_everything(args.seed)
    modes = parse_modes(args.modes)
    paths = base.get_paths(Path(args.output_root), args.run_id)
    base.ensure_dirs(paths)
    save_json(
        paths.run_dir / "binary_score_config.json",
        asdict(
            BinaryScoreConfig(
                stage=args.stage,
                model=args.model,
                run_id=args.run_id,
                output_root=args.output_root,
                modes=",".join(modes),
                seed=args.seed,
                batch_size=args.batch_size,
                bootstrap=args.bootstrap,
                resume=args.resume,
                limit=args.limit,
            )
        ),
    )
    if args.stage in {"run", "all"}:
        for model_key in base.model_keys_from_arg(args.model):
            run_model(paths, args, model_key, modes)
            base.empty_device_cache()
            time.sleep(1)
    if args.stage in {"aggregate", "all"}:
        aggregate(paths)


if __name__ == "__main__":
    main()
