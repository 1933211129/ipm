#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mlise2026_qwen_final as base  # noqa: E402
import mlise2026_symbolic_intervention as symbolic  # noqa: E402


DEFAULT_RUN_ID = "final_20260513_090357"


@dataclass
class SymbolicPatchConfig:
    stage: str
    model: str
    run_id: str
    output_root: str
    seed: int
    patch_pair_limit: int
    resume: bool


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_dir(paths: base.RunPaths, model_key: str) -> Path:
    path = paths.patch_dir / model_key
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


def stratified_candidate_head(candidates: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if candidates.empty or len(candidates) <= limit:
        return candidates.reset_index(drop=True)
    shuffled_groups = []
    for group_idx, (query_type, group) in enumerate(candidates.groupby("query_type", sort=True)):
        shuffled = group.sample(frac=1.0, random_state=seed + group_idx * 997).reset_index(drop=True)
        shuffled_groups.append((query_type, shuffled))
    selected = []
    cursor = {query_type: 0 for query_type, _ in shuffled_groups}
    while len(selected) < limit:
        advanced = False
        for query_type, group in shuffled_groups:
            idx = cursor[query_type]
            if idx < len(group):
                selected.append(group.iloc[idx].to_dict())
                cursor[query_type] += 1
                advanced = True
                if len(selected) >= limit:
                    break
        if not advanced:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def select_symbolic_patch_candidates(paths: base.RunPaths, model_key: str, limit: int, seed: int) -> pd.DataFrame:
    main_path = paths.table_dir / model_key / "main_eval.csv"
    symbolic_path = paths.table_dir / model_key / "symbolic_solver_concise_eval.csv"
    subset_path = selected_subset_path(paths)
    if not main_path.exists() or not symbolic_path.exists():
        return pd.DataFrame()
    main_eval = pd.read_csv(main_path)
    sym_eval = pd.read_csv(symbolic_path)
    subset = pd.read_csv(subset_path).rename(columns={"id": "sample_id"})
    key = ["sample_id", "model", "model_display_name"]
    nl = main_eval[main_eval["prompt_mode"] == "nl"][key + ["is_correct", "is_invalid", "parsed_label"]].rename(
        columns={"is_correct": "nl_correct", "is_invalid": "nl_invalid", "parsed_label": "nl_parsed_label"}
    )
    sym = sym_eval[key + ["is_correct", "is_invalid", "parsed_label"]].rename(
        columns={
            "is_correct": "symbolic_correct",
            "is_invalid": "symbolic_invalid",
            "parsed_label": "symbolic_parsed_label",
        }
    )
    merged = nl.merge(sym, on=key, how="inner")
    candidates = merged[
        (merged["nl_correct"] == 0)
        & (merged["symbolic_correct"] == 1)
        & (~merged["nl_invalid"].astype(bool))
        & (~merged["symbolic_invalid"].astype(bool))
    ].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.merge(subset, on="sample_id", how="left")
    candidates = candidates.sort_values(["query_type", "story_id", "sample_id"]).reset_index(drop=True)
    return stratified_candidate_head(candidates, limit, seed)


def encode_prompt(evaluator: base.QwenEvaluator, source_row: dict[str, Any], prompt_kind: str) -> dict[str, Any]:
    if prompt_kind == "nl":
        prompt = base.build_user_prompt(source_row, "nl")
    elif prompt_kind == "symbolic":
        symbolic.INTERVENTION_MODE = "symbolic_solver_concise"
        prompt, _ = symbolic.build_symbolic_solver_prompt(source_row)
    else:
        raise ValueError(f"未知 prompt_kind: {prompt_kind}")
    text = evaluator.format_batch([prompt])[0]
    encoded = evaluator.tokenizer(text, return_tensors="pt")
    return {key: value.to(evaluator.device) for key, value in encoded.items()}


def cache_layer_outputs(evaluator: base.QwenEvaluator, inputs: dict[str, Any]) -> tuple[dict[int, Any], Any]:
    torch = base.torch
    hf_model = evaluator.model
    cache: dict[int, Any] = {}

    def save_layer_output(layer_idx: int):
        def hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[layer_idx] = hidden.detach().clone()

        return hook

    handles = [
        hf_model.model.layers[layer_idx].register_forward_hook(save_layer_output(layer_idx))
        for layer_idx in range(len(hf_model.model.layers))
    ]
    with torch.inference_mode():
        outputs = hf_model(**inputs)
    for handle in handles:
        handle.remove()
    return cache, outputs


def run_symbolic_to_natural_patching(evaluator: base.QwenEvaluator, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    torch = base.torch
    yes_id, no_id = base._yes_no_token_ids(evaluator.tokenizer)
    hf_model = evaluator.model
    rows = []
    for candidate_idx, row in enumerate(candidates.to_dict("records"), start=1):
        target_label = row["label"]
        natural_inputs = encode_prompt(evaluator, row, "nl")
        symbolic_inputs = encode_prompt(evaluator, row, "symbolic")
        symbolic_cache, symbolic_outputs = cache_layer_outputs(evaluator, symbolic_inputs)
        with torch.inference_mode():
            natural_outputs = hf_model(**natural_inputs)
        natural_margin = base.margin_from_logits(natural_outputs.logits, yes_id, no_id, target_label)
        symbolic_margin = base.margin_from_logits(symbolic_outputs.logits, yes_id, no_id, target_label)
        denom = symbolic_margin - natural_margin
        for layer_idx in range(len(hf_model.model.layers)):
            def patch_layer(_module, _inp, output, layer_idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                patched[:, -1:, :] = symbolic_cache[layer_idx][:, -1:, :]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = hf_model.model.layers[layer_idx].register_forward_hook(patch_layer)
            with torch.inference_mode():
                patched_outputs = hf_model(**natural_inputs)
            handle.remove()
            patched_margin = base.margin_from_logits(patched_outputs.logits, yes_id, no_id, target_label)
            absolute_recovery = patched_margin - natural_margin
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
                    "symbolic_gold_margin": symbolic_margin,
                    "patched_gold_margin": patched_margin,
                    "absolute_recovery": absolute_recovery,
                    "normalized_recovery": absolute_recovery / denom if abs(denom) > 1e-8 else None,
                    "method": "hf_last_token_symbolic_to_natural",
                }
            )
        base.empty_device_cache()
    return pd.DataFrame(rows)


def run_random_symbolic_patch_control(evaluator: base.QwenEvaluator, candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    if candidates.empty or len(candidates) < 2:
        return pd.DataFrame()
    torch = base.torch
    yes_id, no_id = base._yes_no_token_ids(evaluator.tokenizer)
    hf_model = evaluator.model
    rng = random.Random(seed)
    candidate_records = candidates.to_dict("records")
    rows = []
    for candidate_idx, row in enumerate(candidate_records, start=1):
        donors = [
            donor
            for donor in candidate_records
            if int(donor["sample_id"]) != int(row["sample_id"]) and donor["query_type"] != row["query_type"]
        ]
        if not donors:
            donors = [donor for donor in candidate_records if int(donor["sample_id"]) != int(row["sample_id"])]
        donor = rng.choice(donors)
        target_label = row["label"]
        natural_inputs = encode_prompt(evaluator, row, "nl")
        target_symbolic_inputs = encode_prompt(evaluator, row, "symbolic")
        donor_symbolic_inputs = encode_prompt(evaluator, donor, "symbolic")
        donor_cache, _ = cache_layer_outputs(evaluator, donor_symbolic_inputs)
        with torch.inference_mode():
            natural_outputs = hf_model(**natural_inputs)
            target_symbolic_outputs = hf_model(**target_symbolic_inputs)
        natural_margin = base.margin_from_logits(natural_outputs.logits, yes_id, no_id, target_label)
        target_symbolic_margin = base.margin_from_logits(target_symbolic_outputs.logits, yes_id, no_id, target_label)
        denom = target_symbolic_margin - natural_margin
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
            patched_margin = base.margin_from_logits(patched_outputs.logits, yes_id, no_id, target_label)
            absolute_recovery = patched_margin - natural_margin
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
                    "target_symbolic_gold_margin": target_symbolic_margin,
                    "patched_gold_margin": patched_margin,
                    "absolute_recovery": absolute_recovery,
                    "normalized_recovery": absolute_recovery / denom if abs(denom) > 1e-8 else None,
                    "method": "hf_last_token_random_symbolic_to_natural",
                }
            )
        base.empty_device_cache()
    return pd.DataFrame(rows)


def summarize(matched: pd.DataFrame, random_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    frames = []
    if not matched.empty:
        tmp = matched.copy()
        tmp["patch_condition"] = "matched_symbolic"
        frames.append(tmp)
    if not random_df.empty:
        tmp = random_df.copy()
        tmp["patch_condition"] = "random_symbolic"
        frames.append(tmp)
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
    if not matched.empty and not random_df.empty:
        keys = ["sample_id", "layer"]
        m = matched[keys + ["absolute_recovery"]].rename(columns={"absolute_recovery": "matched_recovery"})
        r = random_df[keys + ["absolute_recovery"]].rename(columns={"absolute_recovery": "random_recovery"})
        diff = m.merge(r, on=keys, how="inner")
        if not diff.empty:
            diff["absolute_recovery"] = diff["matched_recovery"] - diff["random_recovery"]
            extra = {
                "model": matched["model"].iloc[0],
                "model_display_name": matched["model_display_name"].iloc[0],
                "patch_condition": "matched_minus_random",
                "n_rows": int(len(diff)),
                "n_samples": int(diff["sample_id"].nunique()),
                "mean_absolute_recovery": float(diff["absolute_recovery"].mean()),
                "median_absolute_recovery": float(diff["absolute_recovery"].median()),
                "max_absolute_recovery": float(diff["absolute_recovery"].max()),
                "positive_recovery_rate": float((diff["absolute_recovery"] > 0).mean()),
                "mean_normalized_recovery": None,
            }
            summary = pd.concat([summary, pd.DataFrame([extra])], ignore_index=True)
    summary.to_csv(out_dir / "symbolic_patch_control_summary.csv", index=False)
    return summary


def run_patch(paths: base.RunPaths, args: argparse.Namespace, model_key: str) -> None:
    out_dir = patch_dir(paths, model_key)
    candidates = select_symbolic_patch_candidates(paths, model_key, args.patch_pair_limit, args.seed)
    candidates.to_csv(out_dir / "symbolic_to_natural_patch_candidates.csv", index=False)
    if candidates.empty:
        print(f"[警告] {model_key} 没有 nl 错、symbolic 对的候选样本。", flush=True)
        return
    model_cfg = base.MODEL_CONFIGS[model_key]
    evaluator = base.QwenEvaluator(
        model_key=model_key,
        model_path=Path(model_cfg["path"]),
        hf_name=model_cfg["hf_name"],
        batch_size=1,
        max_new_tokens=1,
    )
    try:
        matched_path = out_dir / "symbolic_to_natural_patching_results.csv"
        random_path = out_dir / "random_symbolic_patch_control_results.csv"
        if matched_path.exists() and not args.resume:
            raise FileExistsError(f"patching 结果已存在：{matched_path}")
        if random_path.exists() and not args.resume:
            raise FileExistsError(f"random control 结果已存在：{random_path}")
        matched = run_symbolic_to_natural_patching(evaluator, candidates)
        matched["model"] = model_key
        matched["model_display_name"] = model_cfg["display_name"]
        matched.to_csv(matched_path, index=False)
        random_df = run_random_symbolic_patch_control(evaluator, candidates, args.seed)
        random_df["model"] = model_key
        random_df["model_display_name"] = model_cfg["display_name"]
        random_df.to_csv(random_path, index=False)
    finally:
        evaluator.close()
    summary = summarize(matched, random_df, out_dir)
    layer_profile = (
        matched.groupby(["model", "model_display_name", "layer"], dropna=False)
        .agg(
            mean_recovery=("absolute_recovery", "mean"),
            median_recovery=("absolute_recovery", "median"),
            max_recovery=("absolute_recovery", "max"),
            positive_recovery_rate=("absolute_recovery", lambda x: float((x > 0).mean())),
            mean_normalized_recovery=("normalized_recovery", "mean"),
        )
        .reset_index()
    )
    layer_profile.to_csv(out_dir / "symbolic_patching_layer_profile.csv", index=False)
    layer_profile.sort_values("mean_recovery", ascending=False).head(8).to_csv(
        out_dir / "symbolic_patching_top_layers.csv",
        index=False,
    )
    lines = [
        "# MLISE 2026 symbolic-to-natural patching 结果",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：`{model_cfg['display_name']}`",
        f"- 候选样本：`{len(candidates)}`",
        "- 候选条件：`nl` 错，`symbolic_solver_concise` 对。",
        "",
        "## Control Summary",
        "",
        base.table_to_markdown(summary.round(4) if not summary.empty else summary),
        "",
        "## Top Layers",
        "",
        base.table_to_markdown(layer_profile.sort_values("mean_recovery", ascending=False).head(8).round(4)),
        "",
    ]
    (out_dir / "symbolic_to_natural_patching_report.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate(paths: base.RunPaths) -> None:
    frames = []
    summaries = []
    for model_key in base.MODEL_ORDER:
        out_dir = patch_dir(paths, model_key)
        result_path = out_dir / "symbolic_to_natural_patching_results.csv"
        summary_path = out_dir / "symbolic_patch_control_summary.csv"
        if result_path.exists():
            df = pd.read_csv(result_path)
            if not df.empty:
                frames.append(df)
        if summary_path.exists():
            sm = pd.read_csv(summary_path)
            if not sm.empty:
                summaries.append(sm)
    aggregate_dir = paths.table_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    if frames:
        all_patch = pd.concat(frames, ignore_index=True)
        all_patch.to_csv(aggregate_dir / "symbolic_to_natural_patching_results.csv", index=False)
    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        all_summary.to_csv(aggregate_dir / "symbolic_patch_control_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLISE 2026 symbolic-to-natural aligned patching")
    parser.add_argument("--stage", choices=["patch", "aggregate", "all"], default="all")
    parser.add_argument("--model", choices=base.MODEL_ORDER + ["all"], default="qwen3_8b")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", default=str(base.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--patch-pair-limit", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.seed_everything(args.seed)
    paths = base.get_paths(Path(args.output_root), args.run_id)
    base.ensure_dirs(paths)
    save_json(paths.run_dir / "symbolic_patching_config.json", asdict(SymbolicPatchConfig(**vars(args))))
    if args.stage in {"patch", "all"}:
        for model_key in base.model_keys_from_arg(args.model):
            run_patch(paths, args, model_key)
            base.empty_device_cache()
    if args.stage in {"aggregate", "all"}:
        aggregate(paths)


if __name__ == "__main__":
    main()
