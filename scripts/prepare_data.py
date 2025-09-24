#!/usr/bin/env python
"""Build ChatML-style QA data with the passage bundled into the user turn.

Outputs JSONL rows with:
  - instruction: prompt up to the assistant marker
  - output: assistant reply containing scratchpad and final answer markers

Supports SQuAD v1, SQuAD v2, and TriviaQA (rc).
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

# Logging helpers
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)

# ChatML templates
SYSTEM_PROMPT = (
    "You are a careful reading-comprehension assistant. "
    "Use ONLY the provided passage to answer the question."
)


def make_instruction(question: str, context: str) -> str:
    """Return a ChatML prompt with the question and passage in the user turn."""
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {question.strip()}\n\n"
        f"Context:\n{context.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant"
    )


def make_output(answer: str) -> str:
    """Format the assistant reply with a short scratchpad and final answer."""
    scratch = (
        "<scratchpad>"
        "Read the passage, locate the span that answers the question, and state it succinctly."
        "</scratchpad>"
    )
    final = f"<final>{answer.strip()}</final>"
    # Close the assistant turn so downstream tokenizers can rely on it.
    return scratch + "\n" + final + "\n<|im_end|>"


def create_record(question: str, answer: str, context: str) -> Dict[str, str]:
    return {
        "instruction": make_instruction(question, context),
        "output": make_output(answer),
    }


# ------------------------- Datasets -------------------------
def process_squad(limit: int) -> List[Dict[str, str]]:
    """Process SQuAD v1 into instruction/output pairs."""
    records: List[Dict[str, str]] = []
    ds = load_dataset("squad", split=f"train[:{limit}]")
    for ex in tqdm(ds, desc="SQuAD v1"):
        q = (ex.get("question") or "").strip()
        ctx = (ex.get("context") or "").strip()
        answers = ex.get("answers") or {}
        texts = answers.get("text") or []
        if q and ctx and texts:
            records.append(create_record(q, texts[0], ctx))
    return records


def process_squad_v2(limit: int) -> List[Dict[str, str]]:
    """Process SQuAD v2, skipping unanswerable examples."""
    records: List[Dict[str, str]] = []
    ds = load_dataset("squad_v2", split=f"train[:{limit}]")
    for ex in tqdm(ds, desc="SQuAD v2"):
        if ex.get("is_impossible"):
            continue
        q = (ex.get("question") or "").strip()
        ctx = (ex.get("context") or "").strip()
        answers = ex.get("answers") or {}
        texts = answers.get("text") or []
        if q and ctx and texts:
            records.append(create_record(q, texts[0], ctx))
    return records


def process_trivia_qa(limit: int) -> List[Dict[str, str]]:
    """Process TriviaQA (rc) and fall back to a guarded context when needed."""
    records: List[Dict[str, str]] = []
    ds = load_dataset("trivia_qa", "rc", split=f"train[:{limit}]")

    def extract_context(ex) -> str:
        # Prefer wiki passages; fall back to evidence or search snippets.
        ep = ex.get("entity_pages")
        if isinstance(ep, list) and ep:
            cand = ep[0]
            if isinstance(cand, dict):
                ctx = (cand.get("wiki_context") or cand.get("text") or cand.get("content") or "").strip()
                if ctx:
                    return ctx

        # 2) Try evidence (varies by sample)
        ev = ex.get("evidence")
        if isinstance(ev, list) and ev:
            if isinstance(ev[0], dict):
                ctx = (ev[0].get("text") or ev[0].get("value") or "").strip()
                if ctx:
                    return ctx
            elif isinstance(ev[0], str):
                ctx = ev[0].strip()
                if ctx:
                    return ctx

        # 3) Try search_results (can be [] / list[dict] / dict)
        sr = ex.get("search_results")
        if isinstance(sr, list) and sr:
            first = sr[0]
            if isinstance(first, dict):
                ctx = (first.get("description") or first.get("snippet") or first.get("title") or "").strip()
                if ctx:
                    return ctx
        elif isinstance(sr, dict):
            # take any dict/list value
            for v in sr.values():
                if isinstance(v, dict):
                    ctx = (v.get("description") or v.get("snippet") or v.get("title") or "").strip()
                    if ctx:
                        return ctx
                elif isinstance(v, list) and v:
                    if isinstance(v[0], dict):
                        ctx = (v[0].get("description") or v[0].get("snippet") or v[0].get("title") or "").strip()
                        if ctx:
                            return ctx

        # No usable context found.
        return ""

    for ex in tqdm(ds, desc="TriviaQA (rc)"):
        q = (ex.get("question") or "").strip()
        ans = ((ex.get("answer") or {}).get("value") or "").strip()
        if not (q and ans):
            continue

        ctx = extract_context(ex)
        if not ctx:
            ctx = "(No additional passage provided. Answer only if you are certain.)"

        records.append(create_record(q, ans, ctx))

    return records



# ------------------------- IO helpers -------------------------
def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------- Main -------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Prepare QA data in ChatML format.")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--train-size", type=int, default=20000, help="SQuAD v1 cap (others proportional)")
    p.add_argument("--val-fraction", type=float, default=0.02, help="Validation fraction [0,1)")
    p.add_argument("--include", nargs="+", default=["squad", "squad_v2", "trivia_qa"],
                   choices=["squad", "squad_v2", "trivia_qa"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear-cache", action="store_true",
                   help="If set, clears HF datasets cache (NOT recommended by default)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Optional datasets cache dir (dataset-specific preferred) ")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(args.verbose)
    random.seed(args.seed)

    if args.clear_cache:
        # Opt-in only
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        if cache_dir.exists():
            logger.warning(f"Clearing datasets cache at {cache_dir}")
            import shutil
            shutil.rmtree(cache_dir)

    all_records: List[Dict[str, str]] = []

    if "squad" in args.include:
        all_records += process_squad(limit=args.train_size)
    if "squad_v2" in args.include:
        # Use ~half the SQuAD v1 cap to keep mix balanced.
        all_records += process_squad_v2(limit=max(1, args.train_size // 2))
    if "trivia_qa" in args.include:
        all_records += process_trivia_qa(limit=max(1, args.train_size // 2))

    # Shuffle and split
    random.shuffle(all_records)
    n_total = len(all_records)
    n_val = int(n_total * args.val_fraction)
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    # Write
    out_dir = args.out_dir
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    meta_path = out_dir / "metadata.json"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    meta = {
        "num_total": n_total,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "include": args.include,
        "seed": args.seed,
        "system": SYSTEM_PROMPT,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Preview a couple samples
    logger.info(f"Wrote {len(train_records)} train / {len(val_records)} val to {out_dir}")
    for split_name, sample in (("train", train_records[0] if train_records else None),
                               ("val", val_records[0] if val_records else None)):
        if not sample:
            continue
        logger.info(f"Sample ({split_name}):\n"
                    f"--- instruction ---\n{sample['instruction'][:500]}\n"
                    f"--- output ---\n{sample['output'][:300]}")

    logger.info("Done. Next: pretokenize the JSONL with pretokenize_data.py")


if __name__ == "__main__":
    main()