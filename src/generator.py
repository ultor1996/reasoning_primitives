"""
generator.py — generate evaluation datasets.

Usage
-----
python generator.py \\
    --task collisions \\
    --n-samples 100 \\
    --difficulties 2 4 8 16 \\
    --output-dir /home/sr/Desktop/code/physics_mamba/data \\
    [--output collisions_n100.json]  # optional filename override
    [--seed 42]

For each difficulty level, *n_samples* independent prompts are generated.
The output is a single JSON file (list of sample dicts) that inference.py
can consume directly.

Each sample contains:
  task          : str   task name
  difficulty    : int   difficulty level used
  system_prompt : str   the task system prompt (ready to pass to the model)
  prompt        : str   the user-facing question
  correct_option: str   ground-truth label (A/B/C/D)
  option_A … option_D : str | null
  metadata      : dict  task-specific metadata (e.g. num_particles)
"""

import argparse
import json
import os
import random
import sys
import time

from templates import get_task, list_tasks
from utils import save_json


def generate_dataset(
    task_name: str,
    n_samples: int,
    difficulties: list[int],
    seed: int = 42,
    csv_path: str | None = None,
) -> list[dict]:
    """
    Generate *n_samples* per difficulty level.

    *csv_path* is forwarded to the `astro` / `olmo3` task generators.
    Returns a flat list of sample dicts.
    """
    task = get_task(task_name, csv_path=csv_path)
    rng = random.Random(seed)
    samples = []

    for diff in difficulties:
        print(f"  Generating {n_samples} samples at difficulty={diff} …", flush=True)
        generated = 0
        attempts = 0
        max_attempts = n_samples * 20  # guard against pathological failures

        while generated < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                sample = task.generate_sample(difficulty=diff, rng=rng)
            except Exception as e:
                print(f"    [warn] generation failed: {e}", file=sys.stderr)
                continue

            sample.update(
                {
                    "task": task_name,
                    "difficulty": diff,
                    "system_prompt": task.system_prompt,
                    "sample_id": len(samples),
                }
            )
            samples.append(sample)
            generated += 1

        if generated < n_samples:
            print(
                f"  [warn] Only generated {generated}/{n_samples} samples "
                f"at difficulty={diff} (too many failures).",
                file=sys.stderr,
            )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation prompts for a named task."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list_tasks(),
        help="Task to generate (e.g. collisions, astro, olmo3).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per difficulty level (default: 100).",
    )
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Difficulty levels to generate (default: 4 8 16).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write the output JSON into. "
            "Defaults to a 'data/' subfolder next to generator.py."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output filename (basename only, or full path). "
            "Defaults to <task>_diff<d>_n<n>.json inside --output-dir."
        ),
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help=(
            "Path to the exoplanet CSV (for astro / olmo3 tasks). "
            "Overrides the EXOPLANETS_CSV env var and the compiled-in default."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve output filename
    if args.output is None:
        diff_str = "_".join(str(d) for d in args.difficulties)
        filename = f"{args.task}_diff{diff_str}_n{args.n_samples}.json"
    else:
        filename = args.output

    # If the user gave a full path, use it directly; otherwise put it in output_dir
    if os.path.isabs(filename) or os.sep in filename:
        output_path = filename
    else:
        output_path = os.path.join(args.output_dir, filename)

    print(f"Task        : {args.task}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Samples/diff: {args.n_samples}")
    print(f"Seed        : {args.seed}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Output file : {output_path}")
    print()

    t0 = time.time()
    samples = generate_dataset(
        task_name=args.task,
        n_samples=args.n_samples,
        difficulties=args.difficulties,
        seed=args.seed,
        csv_path=args.csv_path,
    )

    save_json(samples, output_path)
    elapsed = time.time() - t0
    print(f"\nDone — {len(samples)} samples written to {output_path}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
