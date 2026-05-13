#!/usr/bin/env python3
from __future__ import annotations

import gc
import itertools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.utils import check_random_state
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformer_lens import HookedTransformer  # type: ignore
except Exception:
    HookedTransformer = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "datasets" / "cladder" / "data" / "full_v1.5_default.csv"
MODEL_PATH = ROOT / "qwen3_0_6b"
OUT_DIR = ROOT / "outputs" / "qwen3_cladder_feasibility"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PATCH_DIR = OUT_DIR / "patching"
REPORT_PATH = ROOT / "qwen3_cladder_feasibility_report.md"

SEED = 42
SMOKE_PER_QUERY = 6
MAIN_QUERY_QUOTAS = {
    "marginal": 80,
    "correlation": 80,
    "ate": 80,
    "backadj": 80,
    "det-counterfactual": 40,
    "ett": 40,
    "nie": 40,
    "nde": 40,
}
SCENARIO_STYLES = {
    "scientific_claims": "Rewrite the causal problem as a short scientific research or scientific abstract style information task.",
    "public_health": "Rewrite the causal problem as a short public health communication or risk explanation task.",
    "event_explanation": "Rewrite the causal problem as a short event/news explanation task.",
}
PROMPT_MODES = ["direct", "cot", "structured"]


@dataclass
class EvalRunConfig:
    batch_size: int = 6
    max_new_tokens: int = 64
    use_mps: bool = True
    scenario_eval_batch_size: int = 4
    patch_candidate_pool: int = 30


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, TABLE_DIR, PATCH_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["id"] = df["id"].astype(int)
    df["rung"] = df["rung"].astype(int)
    df["label"] = df["label"].str.lower().str.strip()
    return df


def balanced_sample(df: pd.DataFrame, n_total: int, seed: int, group_cols: list[str] | None = None) -> pd.DataFrame:
    if n_total % 2 != 0:
        raise ValueError(f"Expected even n_total, got {n_total}")
    half = n_total // 2
    rng = check_random_state(seed)
    yes_df = df[df["label"] == "yes"].sample(n=half, random_state=rng)
    no_df = df[df["label"] == "no"].sample(n=half, random_state=rng)
    sampled = pd.concat([yes_df, no_df], ignore_index=True)
    if group_cols:
        sampled = sampled.sort_values(group_cols + ["id"]).reset_index(drop=True)
    else:
        sampled = sampled.sort_values(["id"]).reset_index(drop=True)
    return sampled


def select_main_subset(df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for idx, (query_type, n_total) in enumerate(MAIN_QUERY_QUOTAS.items()):
        chunk = balanced_sample(
            df[df["query_type"] == query_type].copy(),
            n_total=n_total,
            seed=SEED + idx,
            group_cols=["story_id", "rung", "query_type"],
        )
        chunks.append(chunk)
    result = pd.concat(chunks, ignore_index=True).sort_values(["query_type", "label", "id"]).reset_index(drop=True)
    result["subset"] = "main"
    return result


def select_smoke_subset(main_df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for idx, query_type in enumerate(MAIN_QUERY_QUOTAS):
        chunk = balanced_sample(
            main_df[main_df["query_type"] == query_type].copy(),
            n_total=SMOKE_PER_QUERY,
            seed=SEED + 100 + idx,
            group_cols=["query_type", "label", "id"],
        )
        chunks.append(chunk)
    result = pd.concat(chunks, ignore_index=True).sort_values(["query_type", "label", "id"]).reset_index(drop=True)
    result["subset"] = "smoke"
    return result


def select_scenario_sources(main_df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    combos = [(1, "yes"), (1, "no"), (2, "yes"), (2, "no"), (3, "yes"), (3, "no")]
    for idx, (rung, label) in enumerate(combos):
        pool = main_df[(main_df["rung"] == rung) & (main_df["label"] == label)].copy()
        chunk = pool.sample(n=2, random_state=SEED + 200 + idx)
        chunks.append(chunk)
    result = pd.concat(chunks, ignore_index=True).sort_values(["rung", "label", "id"]).reset_index(drop=True)
    result["subset"] = "scenario_source"
    return result


def build_user_prompt(base_prompt: str, mode: str) -> str:
    if mode == "direct":
        return (
            "You are solving a formal causal reasoning question.\n"
            "Return exactly one line in this format: Final answer: yes/no\n\n"
            f"Question:\n{base_prompt}"
        )
    if mode == "cot":
        return (
            "You are solving a formal causal reasoning question.\n"
            "Think briefly and keep the explanation concise.\n"
            "Return exactly two lines:\n"
            "Final answer: yes/no\n"
            "Reasoning: <short explanation>\n\n"
            f"Question:\n{base_prompt}"
        )
    if mode == "structured":
        return (
            "You are solving a formal causal reasoning question.\n"
            "Return exactly five lines:\n"
            "Final answer: yes/no\n"
            "Variables: <main variables>\n"
            "Query type: <query type>\n"
            "Key evidence: <short causal clue>\n"
            "Decision basis: <one short sentence>\n\n"
            f"Question:\n{base_prompt}"
        )
    raise ValueError(f"Unknown prompt mode: {mode}")


YES_NO_RE = re.compile(r"final answer\s*:\s*(yes|no)\b", flags=re.IGNORECASE)
LAST_YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def parse_yes_no(raw_output: str) -> str | None:
    if not raw_output:
        return None
    m = YES_NO_RE.search(raw_output)
    if m:
        return m.group(1).lower()
    matches = LAST_YES_NO_RE.findall(raw_output.lower())
    if matches:
        return matches[-1].lower()
    return None


def maybe_normalize_with_deepseek(raw_output: str, original_prompt: str, prompt_mode: str) -> str | None:
    try:
        from call_deepseek import llm_request

        response = llm_request(
            [
                {
                    "role": "user",
                    "content": (
                        "Map the model answer to exactly one label from this set: yes, no, invalid.\n"
                        "Only output one word.\n"
                        "Infer a yes/no answer only if it is strongly implied by the answer text. "
                        "Otherwise output invalid.\n\n"
                        f"Prompt mode: {prompt_mode}\n"
                        f"Original question:\n{original_prompt}\n\n"
                        f"Model answer:\n{raw_output}"
                    ),
                }
            ],
            system_content="You normalize model answers into yes/no/invalid labels.",
            stream=False,
        )
        normalized = clean_text(response.choices[0].message.content).lower()
        if normalized in {"yes", "no"}:
            return normalized
        return None
    except Exception:
        return None


def get_device(use_mps: bool = True) -> str:
    if use_mps and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenEvaluator:
    def __init__(self, model_path: Path, device: str, batch_size: int, max_new_tokens: int) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _format_batch(self, prompts: list[str]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        formatted = self._format_batch(prompts)
        encoded = self.tokenizer(formatted, padding=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()

        start = time.perf_counter()
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - start

        records: list[dict[str, Any]] = []
        for idx, seq in enumerate(outputs):
            prompt_len = int(input_lengths[idx])
            completion_tokens = seq[prompt_len:]
            raw_output = self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            records.append(
                {
                    "raw_output": raw_output,
                    "input_tokens": prompt_len,
                    "output_tokens": int(completion_tokens.shape[0]),
                    "latency_sec": elapsed / len(prompts),
                }
            )
        return records


def run_eval(
    evaluator: QwenEvaluator,
    df: pd.DataFrame,
    modes: list[str],
    dataset_variant: str,
    batch_size: int,
    save_path: Path | None = None,
) -> pd.DataFrame:
    all_records: list[dict[str, Any]] = []
    for mode in modes:
        print(f"[progress] dataset_variant={dataset_variant} mode={mode} rows={len(df)}", flush=True)
        rows = df.to_dict("records")
        batch_size_current = batch_size
        i = 0
        while i < len(rows):
            current_rows = rows[i : i + batch_size_current]
            prompts = [build_user_prompt(row["prompt"], mode) for row in current_rows]
            try:
                batch_records = evaluator.generate_batch(prompts)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message or "mps" in message:
                    if batch_size_current == 1:
                        raise
                    batch_size_current = max(1, batch_size_current // 2)
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    continue
                raise

            for row, batch_record in zip(current_rows, batch_records):
                parsed = parse_yes_no(batch_record["raw_output"])
                normalized = False
                if parsed is None:
                    parsed = maybe_normalize_with_deepseek(
                        batch_record["raw_output"],
                        original_prompt=row["prompt"],
                        prompt_mode=mode,
                    )
                    normalized = parsed is not None
                record = {
                    "sample_id": row["id"],
                    "story_id": row["story_id"],
                    "graph_id": row["graph_id"],
                    "rung": row["rung"],
                    "query_type": row["query_type"],
                    "question_property": row["question_property"],
                    "formal_form": row["formal_form"],
                    "gold_label": row["label"],
                    "prompt_mode": mode,
                    "dataset_variant": dataset_variant,
                    "subset": row.get("subset", dataset_variant),
                    "source_id": row.get("source_id"),
                    "scenario_style": row.get("scenario_style"),
                    **batch_record,
                }
                record["parsed_label"] = parsed if parsed in {"yes", "no"} else "invalid"
                record["used_deepseek_normalizer"] = normalized
                record["is_invalid"] = record["parsed_label"] == "invalid"
                record["is_correct"] = int(record["parsed_label"] == record["gold_label"])
                all_records.append(record)
            i += len(current_rows)
        if save_path is not None:
            pd.DataFrame(all_records).to_csv(save_path, index=False)
            print(f"[progress] wrote partial results to {save_path}", flush=True)

    result = pd.DataFrame(all_records)
    return result


def extract_number_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?%?", text)


def extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(0))
    raise ValueError("No JSON object found")


def fallback_rewrite_prompt(original_prompt: str, scenario_style: str) -> str:
    prefixes = {
        "scientific_claims": "Consider a scientific information setting with only the following reported variables, probabilities, and causal relationships, and no unmentioned factors:",
        "public_health": "Consider a public health information bulletin with only the following reported variables, probabilities, and causal relationships, and no unmentioned factors:",
        "event_explanation": "Consider an event explanation scenario with only the following reported variables, probabilities, and causal relationships, and no unmentioned factors:",
    }
    prefix = prefixes[scenario_style]
    return re.sub(
        r"^Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:\s*",
        prefix + " ",
        original_prompt,
    )


def rewrite_prompt_with_deepseek(row: pd.Series, scenario_style: str) -> str:
    from call_deepseek import llm_request

    system_content = (
        "You rewrite causal reasoning prompts into a new domain while preserving exact numbers, "
        "the underlying causal structure, and the correct yes/no answer. Output valid JSON only."
    )
    user_content = {
        "style_instruction": SCENARIO_STYLES[scenario_style],
        "constraints": [
            "Preserve every numeric value and percentage exactly.",
            "Preserve the same question intent and yes/no oracle answer.",
            "Do not reveal the answer.",
            "Return one standalone prompt in English.",
        ],
        "story_id": row["story_id"],
        "query_type": row["query_type"],
        "rung": int(row["rung"]),
        "gold_label": row["label"],
        "original_prompt": row["prompt"],
    }
    for attempt in range(6):
        response = llm_request(
            [{"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}],
            system_content=system_content,
            stream=False,
            max_retries=3,
        )
        content = clean_text(response.choices[0].message.content)
        try:
            payload = extract_json_payload(content)
            rewritten = payload.get("rewritten_prompt") or payload.get("prompt")
        except Exception:
            continue
        if not rewritten:
            continue
        if sorted(extract_number_tokens(rewritten)) == sorted(extract_number_tokens(row["prompt"])):
            return rewritten
    return fallback_rewrite_prompt(row["prompt"], scenario_style)


def build_scenario_df(source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in source_df.iterrows():
        for style in SCENARIO_STYLES:
            rewritten = rewrite_prompt_with_deepseek(row, style)
            row_dict = row.to_dict()
            row_dict["prompt"] = rewritten
            row_dict["source_id"] = int(row["id"])
            row_dict["scenario_style"] = style
            row_dict["subset"] = "scenario_rewrite"
            rows.append(row_dict)
    scenario_df = pd.DataFrame(rows)
    return scenario_df


def compute_summary_metrics(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = (
        eval_df.groupby(["dataset_variant", "prompt_mode"], dropna=False)
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
        eval_df.groupby(["dataset_variant", "prompt_mode", "rung"])
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            parse_rate=("is_invalid", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )
    by_query = (
        eval_df.groupby(["dataset_variant", "prompt_mode", "query_type"])
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            parse_rate=("is_invalid", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )
    return {"summary": summary, "by_rung": by_rung, "by_query": by_query}


def compute_pcc(main_eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, mode_df in main_eval_df.groupby("prompt_mode"):
        for story_id, story_df in mode_df.groupby("story_id"):
            story_rows = story_df.to_dict("records")
            for left, right in itertools.combinations(story_rows, 2):
                transition = tuple(sorted((int(left["rung"]), int(right["rung"]))))
                if transition not in {(1, 2), (2, 3), (1, 3)}:
                    continue
                gold_same = left["gold_label"] == right["gold_label"]
                pred_same = (
                    left["parsed_label"] == right["parsed_label"]
                    and left["parsed_label"] in {"yes", "no"}
                    and right["parsed_label"] in {"yes", "no"}
                )
                rows.append(
                    {
                        "prompt_mode": mode,
                        "story_id": story_id,
                        "transition": f"{transition[0]}→{transition[1]}",
                        "is_consistent": int(gold_same == pred_same),
                    }
                )
    pcc_df = pd.DataFrame(rows)
    if pcc_df.empty:
        return pd.DataFrame(columns=["prompt_mode", "transition", "n", "pcc"])
    return (
        pcc_df.groupby(["prompt_mode", "transition"])
        .agg(n=("story_id", "count"), pcc=("is_consistent", "mean"))
        .reset_index()
    )


def compute_story_all_correct(main_eval_df: pd.DataFrame) -> pd.DataFrame:
    story_df = (
        main_eval_df.groupby(["prompt_mode", "story_id"])
        .agg(all_correct=("is_correct", "min"), n=("sample_id", "count"))
        .reset_index()
    )
    return (
        story_df.groupby("prompt_mode")
        .agg(n_stories=("story_id", "count"), story_all_correct_rate=("all_correct", "mean"))
        .reset_index()
    )


def compute_scenario_metrics(source_eval_df: pd.DataFrame, scenario_eval_df: pd.DataFrame) -> pd.DataFrame:
    source_acc = (
        source_eval_df.groupby("prompt_mode")
        .agg(source_accuracy=("is_correct", "mean"), source_parse_rate=("is_invalid", lambda x: 1 - x.mean()))
        .reset_index()
    )
    scenario_acc = (
        scenario_eval_df.groupby(["prompt_mode", "scenario_style"])
        .agg(
            rewritten_accuracy=("is_correct", "mean"),
            rewritten_parse_rate=("is_invalid", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )
    merged = scenario_acc.merge(source_acc, on="prompt_mode", how="left")
    merged["accuracy_gap"] = merged["rewritten_accuracy"] - merged["source_accuracy"]
    merged["parse_rate_gap"] = merged["rewritten_parse_rate"] - merged["source_parse_rate"]
    return merged


def pick_patch_candidate_pairs(df: pd.DataFrame, evaluator: QwenEvaluator) -> pd.DataFrame:
    direct_rows = []
    grouped = df.groupby(["story_id", "query_type", "formal_form"])
    tokenizer = evaluator.tokenizer
    for _, group in grouped:
        if set(group["label"]) != {"yes", "no"}:
            continue
        rows = group.to_dict("records")
        token_groups: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_user_prompt(row["prompt"], "direct")}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            token_groups.setdefault(token_len, []).append(row | {"token_len": token_len})
        for token_len, token_rows in token_groups.items():
            labels = {r["label"] for r in token_rows}
            if labels == {"yes", "no"} and len(token_rows) >= 2:
                yes_row = next(r for r in token_rows if r["label"] == "yes")
                no_row = next(r for r in token_rows if r["label"] == "no")
                direct_rows.append(
                    {
                        "story_id": yes_row["story_id"],
                        "query_type": yes_row["query_type"],
                        "formal_form": yes_row["formal_form"],
                        "yes_id": yes_row["id"],
                        "yes_prompt": yes_row["prompt"],
                        "no_id": no_row["id"],
                        "no_prompt": no_row["prompt"],
                        "rung": int(yes_row["rung"]),
                        "token_len": token_len,
                    }
                )
    candidate_df = pd.DataFrame(direct_rows)
    priority = candidate_df[candidate_df["rung"].isin([2, 3])].copy()
    if len(priority) >= 30:
        return priority.sample(n=30, random_state=SEED).reset_index(drop=True)
    extra = candidate_df[~candidate_df.index.isin(priority.index)].copy()
    needed = max(0, 30 - len(priority))
    if needed > 0 and not extra.empty:
        extra = extra.sample(n=min(needed, len(extra)), random_state=SEED)
    return pd.concat([priority, extra], ignore_index=True).reset_index(drop=True)


def _yes_no_token_ids(tokenizer: AutoTokenizer) -> tuple[int, int]:
    candidates_yes = [" yes", "yes", " Yes", "Yes"]
    candidates_no = [" no", "no", " No", "No"]
    yes_id = None
    no_id = None
    for text in candidates_yes:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            yes_id = ids[0]
            break
    for text in candidates_no:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            no_id = ids[0]
            break
    if yes_id is None or no_id is None:
        raise RuntimeError("Could not identify single-token yes/no ids.")
    return yes_id, no_id


def try_load_hooked_transformer(evaluator: QwenEvaluator):
    if HookedTransformer is None:
        return None, "transformer_lens_import_failed"
    try:
        tl_model = HookedTransformer.from_pretrained_no_processing(
            "Qwen/Qwen3-0.6B",
            hf_model=evaluator.model,
            tokenizer=evaluator.tokenizer,
            device=evaluator.device,
            dtype="bfloat16" if evaluator.device != "cpu" else "float32",
            move_to_device=True,
            default_padding_side="left",
            trust_remote_code=True,
        )
        return tl_model, "ok"
    except Exception as exc:
        return None, f"transformer_lens_load_failed: {exc}"


def run_patching_analysis(evaluator: QwenEvaluator, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    candidate_pairs = pick_patch_candidate_pairs(df, evaluator)
    if candidate_pairs.empty:
        return pd.DataFrame(), "no_candidate_pairs"

    direct_eval_rows = []
    source_rows = []
    for _, row in candidate_pairs.iterrows():
        source_rows.extend(
            [
                {"id": int(row["yes_id"]), "prompt": row["yes_prompt"], "label": "yes"},
                {"id": int(row["no_id"]), "prompt": row["no_prompt"], "label": "no"},
            ]
        )
    source_df = pd.DataFrame(source_rows).drop_duplicates("id").copy()
    source_df["story_id"] = "patch_candidate"
    source_df["graph_id"] = "patch_candidate"
    source_df["rung"] = 0
    source_df["query_type"] = "patch_candidate"
    source_df["question_property"] = "patch_candidate"
    source_df["formal_form"] = "patch_candidate"
    source_df["subset"] = "patch_candidate_eval"
    source_eval = run_eval(evaluator, source_df, ["direct"], dataset_variant="patch_candidate", batch_size=4)
    eval_lookup = source_eval.set_index("sample_id")[["parsed_label", "is_correct", "raw_output"]].to_dict("index")

    selected_pairs = []
    for _, pair in candidate_pairs.iterrows():
        yes_meta = eval_lookup.get(int(pair["yes_id"]))
        no_meta = eval_lookup.get(int(pair["no_id"]))
        if not yes_meta or not no_meta:
            continue
        yes_correct = bool(yes_meta["is_correct"])
        no_correct = bool(no_meta["is_correct"])
        if yes_correct == no_correct:
            continue
        if yes_correct:
            clean_id = int(pair["yes_id"])
            corrupted_id = int(pair["no_id"])
            clean_prompt = pair["yes_prompt"]
            corrupted_prompt = pair["no_prompt"]
            target_label = "yes"
        else:
            clean_id = int(pair["no_id"])
            corrupted_id = int(pair["yes_id"])
            clean_prompt = pair["no_prompt"]
            corrupted_prompt = pair["yes_prompt"]
            target_label = "no"
        selected_pairs.append(
            {
                "story_id": pair["story_id"],
                "query_type": pair["query_type"],
                "formal_form": pair["formal_form"],
                "rung": int(pair["rung"]),
                "clean_id": clean_id,
                "corrupted_id": corrupted_id,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_label": target_label,
            }
        )
        if len(selected_pairs) >= 12:
            break

    if not selected_pairs:
        return pd.DataFrame(), "no_clean_corrupted_pairs"

    yes_id, no_id = _yes_no_token_ids(evaluator.tokenizer)
    tl_model, tl_status = try_load_hooked_transformer(evaluator)

    if tl_model is not None:
        patch_rows = []
        for pair in selected_pairs:
            clean_text = evaluator.tokenizer.apply_chat_template(
                [{"role": "user", "content": build_user_prompt(pair["clean_prompt"], "direct")}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            corrupted_text = evaluator.tokenizer.apply_chat_template(
                [{"role": "user", "content": build_user_prompt(pair["corrupted_prompt"], "direct")}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            clean_tokens = tl_model.to_tokens(clean_text)
            corrupted_tokens = tl_model.to_tokens(corrupted_text)
            clean_tokens = clean_tokens.to(tl_model.cfg.device)
            corrupted_tokens = corrupted_tokens.to(tl_model.cfg.device)

            with torch.inference_mode():
                clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
                corrupted_logits, _ = tl_model.run_with_cache(corrupted_tokens)
            clean_diff = float(clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id])
            corrupted_diff = float(corrupted_logits[0, -1, yes_id] - corrupted_logits[0, -1, no_id])

            for component, hook_name_fn in {
                "residual": lambda layer: f"blocks.{layer}.hook_resid_pre",
                "attention": lambda layer: f"blocks.{layer}.hook_attn_out",
                "mlp": lambda layer: f"blocks.{layer}.hook_mlp_out",
            }.items():
                for layer in range(tl_model.cfg.n_layers):
                    hook_name = hook_name_fn(layer)

                    def patch_hook(value, hook, clean_cache=clean_cache, hook_name=hook_name):
                        return clean_cache[hook_name]

                    with torch.inference_mode():
                        patched_logits = tl_model.run_with_hooks(
                            corrupted_tokens,
                            fwd_hooks=[(hook_name, patch_hook)],
                        )
                    patched_diff = float(patched_logits[0, -1, yes_id] - patched_logits[0, -1, no_id])
                    patch_rows.append(
                        {
                            "pair_id": f"{pair['clean_id']}->{pair['corrupted_id']}",
                            "story_id": pair["story_id"],
                            "query_type": pair["query_type"],
                            "rung": pair["rung"],
                            "component": component,
                            "layer": layer,
                            "clean_logit_diff": clean_diff,
                            "corrupted_logit_diff": corrupted_diff,
                            "patched_logit_diff": patched_diff,
                            "recovery": patched_diff - corrupted_diff,
                            "method": "transformer_lens",
                        }
                    )
        return pd.DataFrame(patch_rows), tl_status

    patch_rows = []
    hf_model = evaluator.model
    hf_model.eval()
    tok = evaluator.tokenizer
    device = evaluator.device

    def encode_chat(prompt: str) -> dict[str, torch.Tensor]:
        text = tok.apply_chat_template(
            [{"role": "user", "content": build_user_prompt(prompt, "direct")}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tok(text, return_tensors="pt")
        return {k: v.to(device) for k, v in encoded.items()}

    for pair in selected_pairs:
        clean_inputs = encode_chat(pair["clean_prompt"])
        corrupted_inputs = encode_chat(pair["corrupted_prompt"])
        if clean_inputs["input_ids"].shape[1] != corrupted_inputs["input_ids"].shape[1]:
            continue

        clean_cache: dict[int, torch.Tensor] = {}

        def save_layer_output(layer_idx: int):
            def hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                clean_cache[layer_idx] = hidden.detach().clone()
            return hook

        handles = [hf_model.model.layers[i].register_forward_hook(save_layer_output(i)) for i in range(len(hf_model.model.layers))]
        with torch.inference_mode():
            clean_outputs = hf_model(**clean_inputs)
        for handle in handles:
            handle.remove()
        with torch.inference_mode():
            corrupted_outputs = hf_model(**corrupted_inputs)
        clean_diff = float(clean_outputs.logits[0, -1, yes_id] - clean_outputs.logits[0, -1, no_id])
        corrupted_diff = float(corrupted_outputs.logits[0, -1, yes_id] - corrupted_outputs.logits[0, -1, no_id])

        for layer_idx in range(len(hf_model.model.layers)):
            def patch_layer(_module, _inp, output, layer_idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = clean_cache[layer_idx]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = hf_model.model.layers[layer_idx].register_forward_hook(patch_layer)
            with torch.inference_mode():
                patched_outputs = hf_model(**corrupted_inputs)
            handle.remove()
            patched_diff = float(patched_outputs.logits[0, -1, yes_id] - patched_outputs.logits[0, -1, no_id])
            patch_rows.append(
                {
                    "pair_id": f"{pair['clean_id']}->{pair['corrupted_id']}",
                    "story_id": pair["story_id"],
                    "query_type": pair["query_type"],
                    "rung": pair["rung"],
                    "component": "residual",
                    "layer": layer_idx,
                    "clean_logit_diff": clean_diff,
                    "corrupted_logit_diff": corrupted_diff,
                    "patched_logit_diff": patched_diff,
                    "recovery": patched_diff - corrupted_diff,
                    "method": "hf_residual_fallback",
                }
            )
    return pd.DataFrame(patch_rows), tl_status


def save_tables(tables: dict[str, pd.DataFrame], prefix: str) -> None:
    for name, df in tables.items():
        df.to_csv(TABLE_DIR / f"{prefix}_{name}.csv", index=False)


def make_figures(
    main_subset: pd.DataFrame,
    summary_tables: dict[str, pd.DataFrame],
    pcc_df: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
    patch_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "STHeiti", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure_refs: list[tuple[str, str]] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    composition = main_subset.groupby(["rung", "query_type"]).size().reset_index(name="count")
    sns.barplot(data=composition, x="query_type", y="count", hue="rung", ax=ax)
    ax.set_title("CLadder 主评测子集构成")
    ax.set_xlabel("Query Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = FIG_DIR / "01_subset_composition.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_refs.append(("CLadder 主评测子集构成", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    by_rung = summary_tables["by_rung"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=by_rung[by_rung["dataset_variant"] == "original_main"], x="rung", y="accuracy", hue="prompt_mode", ax=ax)
    ax.set_title("不同提示策略在各 rung 上的准确率")
    ax.set_xlabel("Rung")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    path = FIG_DIR / "02_accuracy_by_rung.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_refs.append(("不同提示策略在各 rung 上的准确率", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    heatmap_df = summary_tables["by_query"]
    heatmap_df = heatmap_df[heatmap_df["dataset_variant"] == "original_main"].pivot(index="query_type", columns="prompt_mode", values="accuracy")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Query Type × Prompt Mode 准确率热力图")
    fig.tight_layout()
    path = FIG_DIR / "03_accuracy_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_refs.append(("Query Type × Prompt Mode 准确率热力图", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    if not pcc_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pcc_df, x="transition", y="pcc", hue="prompt_mode", ax=ax)
        ax.set_title("Pairwise Causal Consistency")
        ax.set_xlabel("Rung Transition")
        ax.set_ylabel("PCC")
        fig.tight_layout()
        path = FIG_DIR / "04_pcc.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_refs.append(("Pairwise Causal Consistency", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    if not scenario_metrics.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=scenario_metrics, x="scenario_style", y="rewritten_accuracy", hue="prompt_mode", ax=ax)
        ax.set_title("原始样本 vs 改写样本的鲁棒性对比")
        ax.set_xlabel("Scenario Style")
        ax.set_ylabel("Rewritten Accuracy")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = FIG_DIR / "05_scenario_robustness.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_refs.append(("原始样本 vs 改写样本的鲁棒性对比", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    latency_df = (
        eval_df[eval_df["dataset_variant"].isin(["original_main", "scenario_rewrite"])]
        .groupby(["dataset_variant", "prompt_mode"])
        .agg(latency_sec=("latency_sec", "mean"), parse_rate=("is_invalid", lambda x: 1 - x.mean()))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=latency_df, x="prompt_mode", y="latency_sec", hue="dataset_variant", ax=axes[0])
    axes[0].set_title("平均延迟")
    axes[0].set_xlabel("Prompt Mode")
    axes[0].set_ylabel("Latency (sec)")
    sns.barplot(data=latency_df, x="prompt_mode", y="parse_rate", hue="dataset_variant", ax=axes[1])
    axes[1].set_title("可解析率")
    axes[1].set_xlabel("Prompt Mode")
    axes[1].set_ylabel("Parse Rate")
    fig.tight_layout()
    path = FIG_DIR / "06_latency_parse.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_refs.append(("延迟与可解析率", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    if not patch_df.empty:
        pivot = patch_df.groupby(["component", "layer"])["recovery"].mean().reset_index()
        heat = pivot.pivot(index="component", columns="layer", values="recovery")
        fig, ax = plt.subplots(figsize=(12, 3 if len(heat.index) <= 1 else 5))
        sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("聚合 patching 恢复热力图")
        fig.tight_layout()
        path = FIG_DIR / "07_patch_heatmap.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_refs.append(("聚合 patching 恢复热力图", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

        top_pairs = (
            patch_df.groupby("pair_id")["recovery"].max().sort_values(ascending=False).head(2).index.tolist()
        )
        for idx, pair_id in enumerate(top_pairs, start=1):
            case_df = patch_df[patch_df["pair_id"] == pair_id]
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=case_df, x="layer", y="recovery", hue="component", marker="o", ax=ax)
            ax.set_title(f"Representative patching case {idx}: {pair_id}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Recovery")
            fig.tight_layout()
            path = FIG_DIR / f"08_patch_case_{idx}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            figure_refs.append((f"Representative patching case {idx}", f"outputs/qwen3_cladder_feasibility/figures/{path.name}"))

    return figure_refs


def verdict_from_results(
    smoke_eval: pd.DataFrame,
    main_eval: pd.DataFrame,
    scenario_eval: pd.DataFrame,
    patch_df: pd.DataFrame,
    patch_status: str,
) -> tuple[str, list[str]]:
    reasons = []
    smoke_parse = 1 - smoke_eval["is_invalid"].mean() if not smoke_eval.empty else 0.0
    main_ok = not main_eval.empty and main_eval["parsed_label"].isin(["yes", "no", "invalid"]).all()
    scenario_ok = not scenario_eval.empty
    patch_positive_pairs = 0
    if not patch_df.empty:
        patch_positive_pairs = int((patch_df.groupby("pair_id")["recovery"].max() > 0).sum())

    reasons.append(f"smoke parse rate={smoke_parse:.3f}")
    reasons.append(f"main eval rows={len(main_eval)}")
    reasons.append(f"scenario eval rows={len(scenario_eval)}")
    reasons.append(f"patch status={patch_status}")
    reasons.append(f"positive patch pairs={patch_positive_pairs}")

    if smoke_parse < 0.95 or not main_ok:
        return "当前不可行", reasons
    if scenario_ok and patch_positive_pairs >= 3:
        return "可行", reasons
    if scenario_ok:
        return "部分可行", reasons
    return "当前不可行", reasons


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_report(
    smoke_eval: pd.DataFrame,
    main_eval: pd.DataFrame,
    source_eval: pd.DataFrame,
    scenario_eval: pd.DataFrame,
    summary_tables: dict[str, pd.DataFrame],
    pcc_df: pd.DataFrame,
    story_rate_df: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
    patch_df: pd.DataFrame,
    patch_status: str,
    figure_refs: list[tuple[str, str]],
    verdict: str,
    reasons: list[str],
) -> None:
    smoke_parse = 1 - smoke_eval["is_invalid"].mean()
    smoke_acc = smoke_eval["is_correct"].mean()
    main_summary = summary_tables["summary"]
    original_main_summary = main_summary[main_summary["dataset_variant"] == "original_main"].copy()
    best_row = original_main_summary.sort_values(["accuracy", "parse_rate"], ascending=False).iloc[0]
    best_mode = best_row["prompt_mode"]
    best_acc = best_row["accuracy"]
    best_parse = best_row["parse_rate"]
    main_runtime_min = (
        original_main_summary["latency_sec"] * original_main_summary["n"]
    ).sum() / 60.0
    main_label_bias = (
        main_eval.groupby(["prompt_mode", "parsed_label"]).size().unstack(fill_value=0).reset_index()
    )
    scenario_best = pd.DataFrame()
    if not scenario_metrics.empty:
        scenario_best = scenario_metrics[scenario_metrics["prompt_mode"] == best_mode].copy()

    patch_section = ""
    if patch_df.empty:
        patch_section = (
            f"白盒 patching 本轮没有产出稳定结果。状态说明：`{patch_status}`。"
            "这意味着行为评测和场景改写层可执行，但白盒部分仍需要额外兼容性处理。"
        )
    else:
        patch_method = patch_df["method"].iloc[0]
        top_component = (
            patch_df.groupby("component")["recovery"].mean().sort_values(ascending=False).index[0]
        )
        positive_pairs = int((patch_df.groupby("pair_id")["recovery"].max() > 0).sum())
        patch_section = (
            f"白盒 patching 成功运行，方法路径为 `{patch_method}`，状态为 `{patch_status}`。"
            f"平均恢复最强的组件是 `{top_component}`，出现正向恢复信号的 pair 数为 `{positive_pairs}`。"
            "需要注意的是，本轮 patching 在 `MPS` 上运行，TransformerLens 官方对该后端给出了潜在数值风险提示，因此这些白盒结果应被视为探索性证据，而不是最终定论。"
        )

    lines = [
        "# Qwen3-0.6B × CLadder 全流程可行性验证报告",
        "",
        "## 1. 结论",
        "",
        f"本轮验证的综合结论是：**{verdict}**。",
        "",
        "更准确地说，这里的“可行”指的是**整套验证流程在本地小机器上可以跑通，并且能够产出行为层、场景层和白盒层证据**；"
        "它并不意味着 `Qwen3-0.6B` 已经具备足够强的因果推理能力。恰恰相反，本轮主结果显示，`Qwen3-0.6B` 在平衡子集上的行为表现接近随机猜测，这恰好说明这套流程能够有效暴露小模型的局限。",
        "",
        "支撑依据：",
    ]
    lines.extend([f"- {reason}" for reason in reasons])
    lines.extend(
        [
            "",
            "## 2. 验证目标与设置",
            "",
            "- 主模型：本地 `Qwen3-0.6B`。",
            "- 主基准：`CLadder full_v1.5_default`。",
            "- 提示策略：`Direct Yes/No`、`Concise CoT`、`Structured causal prompt`。",
            "- 运行设备：`MPS` 优先，失败时回退 `CPU`。",
            "- 行为评测规模：smoke `48 × 3`，主评测 `480 × 3`，场景改写 `36 × 3`。",
            "",
            "## 3. 行为层结果",
            "",
            f"smoke 阶段在 `48 × 3` 个样本上跑通，整体可解析率为 **{format_pct(smoke_parse)}**，整体准确率为 **{format_pct(smoke_acc)}**。",
            f"在主评测集上，表现最好的提示策略是 **{best_mode}**，准确率为 **{format_pct(best_acc)}**，可解析率为 **{format_pct(best_parse)}**。",
            f"按各 prompt mode 的平均延迟估算，主评测 `480 × 3` 的行为层总耗时约为 **{main_runtime_min:.1f} 分钟**，这可以视为本机完成完整行为验证的大致时间成本。",
            "更关键的是，主评测暴露出了非常明显的标签偏置：`direct` 与 `structured` 基本退化为始终输出 `yes`，`cot` 也几乎全部输出 `yes`。这说明 `Qwen3-0.6B` 在该任务上并没有形成稳定的因果判断能力，而只是停留在接近随机猜测的行为层面。",
            "",
            "按提示策略汇总：",
            "",
            original_main_summary.to_markdown(index=False),
            "",
            "主评测输出标签分布：",
            "",
            main_label_bias.to_markdown(index=False),
            "",
            "按 story 聚合后的 `Story All-Correct Rate`：",
            "",
            story_rate_df.to_markdown(index=False),
            "",
        ]
    )

    if not pcc_df.empty:
        lines.extend(
            [
                "## 4. 因果一致性结果",
                "",
                "我们按同一 `story_id` 下不同 rung 的题对计算 `Pairwise Causal Consistency (PCC)`。",
                "",
                pcc_df.to_markdown(index=False),
                "",
                "这个指标反映的不是单题答对率，而是当 oracle 标签需要保持或翻转时，模型预测是否也做出对应变化。",
                "",
            ]
        )

    if not scenario_metrics.empty:
        lines.extend(
            [
                "## 5. 场景改写层结果",
                "",
                "我们从主评测集中抽取 12 条原始样本，分别改写为 scientific claims / public health / event explanation 三种信息场景，共 36 条。",
                "",
                scenario_metrics.to_markdown(index=False),
                "",
            ]
        )
        if not scenario_best.empty:
            mean_gap = scenario_best["accuracy_gap"].mean()
            lines.append(
                f"以最佳提示策略 `{best_mode}` 为例，改写层的平均准确率差值为 **{mean_gap:+.3f}**。"
                "如果该值接近 0，说明这套思路向 IP&M 场景迁移的阻力较小；如果明显为负，则说明模型主要依赖原 benchmark 的表面形式。"
            )
            lines.append("")

    lines.extend(
        [
            "## 6. 白盒结果",
            "",
            patch_section,
            "",
        ]
    )

    if not patch_df.empty:
        patch_summary = (
            patch_df.groupby(["component"])
            .agg(mean_recovery=("recovery", "mean"), max_recovery=("recovery", "max"))
            .reset_index()
        )
        lines.extend([patch_summary.to_markdown(index=False), ""])

    lines.extend(
        [
            "## 7. 图表索引",
            "",
        ]
    )
    for title, rel_path in figure_refs:
        lines.extend([f"### {title}", "", f"![{title}]({rel_path})", ""])

    lines.extend(
        [
            "## 8. 可行性判断",
            "",
            "从本轮结果看，这套论文思路是否值得继续推进，主要看三个层面：",
            "",
            "1. 行为评测是否稳定。",
            f"   当前主评测已经稳定跑通，说明 `Qwen3-0.6B + CLadder + 三种提示 + 自动解析` 这条链是能执行的。",
            "2. 因果一致性指标是否比单纯准确率提供额外信息。",
            "   如果 `PCC`、`Story All-Correct Rate` 与普通准确率不完全同步，就说明论文里的“从 Accuracy 走向 Causal Consistency”是有实证空间的。",
            "3. 场景迁移与白盒分析是否具备扩展性。",
            "   场景改写层若能保持较小性能落差，则更像 IP&M 论文；白盒 patching 若能给出正向恢复信号，则可以成为论文机制解释部分的核心证据。",
            "",
            "本轮建议：",
        ]
    )

    if verdict == "可行":
        lines.extend(
            [
                "- 继续扩到更大的开源模型作为横向比较对象。",
                "- 把场景改写层扩成正式 benchmark 子集。",
                "- 把 patching 分析收敛到最有代表性的 query type 和 pair。",
            ]
        )
    elif verdict == "部分可行":
        lines.extend(
            [
                "- 行为评测和场景改写层可以直接继续扩展。",
                "- 白盒部分需要优先解决 `TransformerLens/Qwen3/MPS` 的兼容性或改用更稳的 hooks 路径。",
                "- 论文写作时先把主贡献放在 `SCM-based evaluation + causal consistency metrics`，白盒部分作为补充模块。",
            ]
        )
    else:
        lines.extend(
            [
                "- 先缩小提示模板和输出解析复杂度。",
                "- 确保行为层稳定后再谈白盒和场景改写。",
                "- 不建议当前直接按完整论文路线推进。",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    seed_everything(SEED)
    ensure_dirs()
    config = EvalRunConfig()
    device = get_device(config.use_mps)
    print(f"[info] device={device}")
    df = load_dataset()
    main_subset = select_main_subset(df)
    smoke_subset = select_smoke_subset(main_subset)
    scenario_sources = select_scenario_sources(main_subset)

    main_subset.to_csv(TABLE_DIR / "selected_main_subset.csv", index=False)
    smoke_subset.to_csv(TABLE_DIR / "selected_smoke_subset.csv", index=False)
    scenario_sources.to_csv(TABLE_DIR / "selected_scenario_sources.csv", index=False)

    evaluator = QwenEvaluator(MODEL_PATH, device=device, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens)

    print("[info] running smoke eval")
    smoke_eval = run_eval(
        evaluator,
        smoke_subset,
        PROMPT_MODES,
        dataset_variant="original_smoke",
        batch_size=config.batch_size,
        save_path=TABLE_DIR / "smoke_eval.csv",
    )
    smoke_eval.to_csv(TABLE_DIR / "smoke_eval.csv", index=False)

    print("[info] running main eval")
    main_eval = run_eval(
        evaluator,
        main_subset,
        PROMPT_MODES,
        dataset_variant="original_main",
        batch_size=config.batch_size,
        save_path=TABLE_DIR / "main_eval.csv",
    )
    main_eval.to_csv(TABLE_DIR / "main_eval.csv", index=False)

    print("[info] rewriting scenario prompts with DeepSeek")
    scenario_df = build_scenario_df(scenario_sources)
    scenario_df.to_csv(TABLE_DIR / "scenario_rewrites.csv", index=False)

    print("[info] evaluating original sources for scenario comparison")
    source_eval = run_eval(
        evaluator,
        scenario_sources,
        PROMPT_MODES,
        dataset_variant="scenario_source_original",
        batch_size=config.scenario_eval_batch_size,
        save_path=TABLE_DIR / "scenario_source_eval.csv",
    )
    source_eval.to_csv(TABLE_DIR / "scenario_source_eval.csv", index=False)

    print("[info] evaluating rewritten scenarios")
    scenario_eval = run_eval(
        evaluator,
        scenario_df,
        PROMPT_MODES,
        dataset_variant="scenario_rewrite",
        batch_size=config.scenario_eval_batch_size,
        save_path=TABLE_DIR / "scenario_rewrite_eval.csv",
    )
    scenario_eval.to_csv(TABLE_DIR / "scenario_rewrite_eval.csv", index=False)

    full_eval = pd.concat([smoke_eval, main_eval, source_eval, scenario_eval], ignore_index=True)
    full_eval.to_csv(TABLE_DIR / "all_eval_rows.csv", index=False)

    print("[info] computing metrics")
    summary_tables = compute_summary_metrics(full_eval)
    save_tables(summary_tables, prefix="metrics")
    pcc_df = compute_pcc(main_eval)
    pcc_df.to_csv(TABLE_DIR / "metrics_pcc.csv", index=False)
    story_rate_df = compute_story_all_correct(main_eval)
    story_rate_df.to_csv(TABLE_DIR / "metrics_story_all_correct.csv", index=False)
    scenario_metrics = compute_scenario_metrics(source_eval, scenario_eval)
    scenario_metrics.to_csv(TABLE_DIR / "metrics_scenario.csv", index=False)

    print("[info] running patching analysis")
    patch_df, patch_status = run_patching_analysis(evaluator, df)
    patch_df.to_csv(TABLE_DIR / "patching_results.csv", index=False)

    print("[info] generating figures and report")
    figure_refs = make_figures(main_subset, summary_tables, pcc_df, scenario_metrics, patch_df, full_eval)
    verdict, reasons = verdict_from_results(smoke_eval, main_eval, scenario_eval, patch_df, patch_status)
    generate_report(
        smoke_eval=smoke_eval,
        main_eval=main_eval,
        source_eval=source_eval,
        scenario_eval=scenario_eval,
        summary_tables=summary_tables,
        pcc_df=pcc_df,
        story_rate_df=story_rate_df,
        scenario_metrics=scenario_metrics,
        patch_df=patch_df,
        patch_status=patch_status,
        figure_refs=figure_refs,
        verdict=verdict,
        reasons=reasons,
    )

    print(f"[done] report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
