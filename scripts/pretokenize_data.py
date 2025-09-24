#!/usr/bin/env python3
"""Pre-tokenize ChatML data for next-token prediction with boundary-aware masking."""

import argparse
from typing import List, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import os

ASSISTANT_MARKER = "<|im_start|>assistant"
END_MARKER = "<|im_end|>"

ADDITIONAL_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<scratchpad>",
    "</scratchpad>",
    "<final>",
    "</final>",
    "user",
    "assistant",
]


def find_sublist(haystack: List[int], needle: List[int]) -> int:
    """Return the index where `needle` starts within `haystack`, or -1."""
    if not needle or not haystack or len(needle) > len(haystack):
        return -1
    first = needle[0]
    max_i = len(haystack) - len(needle)
    for i in range(max_i + 1):
        if haystack[i] != first:
            continue
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def build_text(record: dict) -> str:
    """Join instruction and output as emitted by prepare_data.py."""
    instr = (record.get("instruction") or "").rstrip() + "\n"
    out = (record.get("output") or "").rstrip() + "\n"
    return instr + "\n" + out


def window_with_boundary(input_ids, boundary_ids, max_len, min_answer_tokens=8):
    pos = find_sublist(input_ids, boundary_ids)
    if pos == -1:
        return None
    n = len(input_ids)
    if n <= max_len:
        b_end = pos + len(boundary_ids)
        if n - b_end < min_answer_tokens:
            return None
        return input_ids, pos

    # Try to keep the boundary within the left quarter of the window to leave answer space.
    target_rel = max(len(boundary_ids) + 1, max_len // 4)
    start = max(0, min(pos - target_rel, n - max_len))
    win = input_ids[start:start + max_len]
    rel = find_sublist(win, boundary_ids)
    if rel != -1 and (len(win) - (rel + len(boundary_ids))) >= min_answer_tokens:
        return win, rel

    # fallback: boundary near left edge
    start = max(0, min(pos - 1, n - max_len))
    win = input_ids[start:start + max_len]
    rel = find_sublist(win, boundary_ids)
    if rel != -1 and (len(win) - (rel + len(boundary_ids))) >= min_answer_tokens:
        return win, rel

    return None




def tokenize_and_mask(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Optional[dict]:
    text = build_text(example)
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]

    boundary_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)

    win = window_with_boundary(input_ids, boundary_ids, max_length, min_answer_tokens=8)
    if win is None:
        return None
    window_ids, rel_boundary = win

    labels = list(window_ids)
    mask_upto = rel_boundary + len(boundary_ids)
    for i in range(mask_upto):
        labels[i] = -100

    return {
        "input_ids": window_ids,
        "attention_mask": [1] * len(window_ids),
        "labels": labels,
    }


def process_split(split_name: str, file_path: str, out_dir: str, tokenizer, max_length: int, num_proc: int, skip_if_no_boundary: bool):
    if not os.path.exists(file_path):
        print(f"⚠️  Skipping {split_name} — no file at {file_path}")
        return

    ds = load_dataset("json", data_files=file_path)

    def map_one(example):
        out = tokenize_and_mask(example, tokenizer, max_length)
        if out is None:
            return {}
        if sum(1 for t in out["labels"] if t != -100) < 8:
            return {}
        return out

    tokenized = ds["train"].map(
        map_one,
        remove_columns=ds["train"].column_names,
        desc=f"Tokenizing & masking {split_name}",
        num_proc=num_proc,
    )

    tokenized = tokenized.filter(
        lambda x: "input_ids" in x and len(x["input_ids"]) > 0,
        desc=f"Dropping rows without boundary ({split_name})",
    )

    split_out = os.path.join(out_dir, f"{split_name}_tokenized")
    tokenized.save_to_disk(split_out)
    print(f"✅ Saved {split_name} dataset to: {split_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default="data/train.jsonl",
                   help="Training file (default: data/train.jsonl)")
    p.add_argument("--val_file", type=str, default="data/val.jsonl",
                   help="Validation file (default: data/val.jsonl)")
    p.add_argument("--out_dir", type=str, default="data",
                   help="Base output directory (default: data/)")
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/phi-4-mini-reasoning",
        help="Base model/tokenizer to use (default: microsoft/phi-4-mini-reasoning)",
    )
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--skip_if_no_boundary", action="store_true",
                   help="Skip rows missing the assistant marker")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # Ensure all special tokens exist
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    process_split("train", args.train_file, args.out_dir, tokenizer, args.max_length, args.num_proc, args.skip_if_no_boundary)
    process_split("val", args.val_file, args.out_dir, tokenizer, args.max_length, args.num_proc, args.skip_if_no_boundary)


if __name__ == "__main__":
    main()
