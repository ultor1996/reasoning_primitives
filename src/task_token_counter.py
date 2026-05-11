"""
check_token_lengths.py — print token lengths for all dataset JSON files.

Usage
-----
python check_token_lengths.py
python check_token_lengths.py --data-dir /path/to/data
python check_token_lengths.py --model allenai/OLMo-Hybrid-Instruct-SFT-7B
"""

import argparse
import glob
import json
import os

from transformers import AutoTokenizer


def check_file(path: str, tokenizer, sample_m_values: list[int] | None = None) -> dict:
    print(f"\nFile: {os.path.basename(path)}")
    print("-" * 60)

    data = json.load(open(path))
    if not data:
        print("  (empty)")
        return {}

    # Get all unique (m, n) pairs in this file
    all_pairs = sorted({(s["m"], s["n"]) for s in data if "m" in s and "n" in s})

    max_tokens = 0
    total_samples = len(data)
    shot = data[0].get("shot", "zero") if data else "zero"
    task = data[0].get("task", "unknown") if data else "unknown"

    for m_val, n_val in all_pairs:
        samples = [s for s in data if s.get("m") == m_val and s.get("n") == n_val]
        if not samples:
            continue
        sample = samples[0]
        system_prompt = sample.get("system_prompt", "")
        prompt = sample["prompt"]
        full_text = system_prompt + "\n" + prompt if system_prompt else prompt
        tokens = tokenizer(full_text, return_tensors="pt")
        n_tokens = tokens.input_ids.shape[1]
        max_tokens = max(max_tokens, n_tokens)
        shot = sample.get("shot", "zero")
        print(f"  m={m_val:<6} n={n_val:<6} tokens: {n_tokens}  (shot={shot})")

    n_pairs = len(all_pairs)
    print(f"  → {total_samples} total samples | {n_pairs} (m,n) pairs | max tokens: {max_tokens} | shot: {shot}")

    return {
        "task": task,
        "shot": shot,
        "total_samples": total_samples,
        "n_pairs": n_pairs,
        "max_tokens": max_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Print token lengths for dataset JSON files."
    )
    parser.add_argument(
        "--data-dir",
        default="/lustre/mlnvme/data/srawat_hpc-reasoning_primitivs/reasoning_primitives/data",
        help="Directory containing dataset JSON files.",
    )
    parser.add_argument(
        "--model",
        default="allenai/OLMo-3-7B-Instruct",
        help="HuggingFace model name to use for tokenization (default: allenai/OLMo-3-7B-Instruct).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Filter to files matching this task name (e.g. olmo_original, dyck).",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Tokenizer loaded.")

    # Find all JSON files in data dir
    pattern = os.path.join(args.data_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No JSON files found in {args.data_dir}")
        return

    # Filter by task if requested
    if args.task:
        files = [f for f in files if os.path.basename(f).startswith(args.task)]

    if not files:
        print(f"No files found matching task '{args.task}'")
        return

    print(f"\nFound {len(files)} file(s) in {args.data_dir}")

    all_stats = []
    for path in files:
        stats = check_file(path, tokenizer)
        if stats:
            stats["file"] = os.path.basename(path)
            all_stats.append(stats)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'File':<55} {'Task':<15} {'Shot':<6} {'Samples':>8} {'Pairs':>6} {'MaxTok':>8}")
    print("-" * 70)
    for s in all_stats:
        print(f"  {s['file']:<53} {s['task']:<15} {s['shot']:<6} {s['total_samples']:>8} {s['n_pairs']:>6} {s['max_tokens']:>8}")

    # Per-task totals
    from collections import defaultdict
    task_totals = defaultdict(lambda: {"samples": 0, "files": 0})
    for s in all_stats:
        key = f"{s['task']} ({s['shot']})"
        task_totals[key]["samples"] += s["total_samples"]
        task_totals[key]["files"]   += 1

    print("\nPer-task totals:")
    for key, vals in sorted(task_totals.items()):
        print(f"  {key:<35} {vals['files']:>3} file(s)   {vals['samples']:>8} samples")

    print("\nDone.")

if __name__ == "__main__":
    main()