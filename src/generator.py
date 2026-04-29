"""
generator.py — generate evaluation datasets.

Usage
-----
python generator.py \\
    --task collisions \\
    --n-samples 100 \\
    --m 4 8 16 \\
    --n 4 8 16 \\
    [--output-dir /path/to/data] \\
    [--output my_dataset.json] \\
    [--seed 42]

m and n meaning per task:
  collisions    : m = number of particles,  n = number of collision steps
  astro         : m = number of table rows, n = number of swaps
  olmo_original : m = bit-array size,       n = number of swap lines

A sample is generated for every (m, n) pair in the cartesian product of
--m and --n values.  n_samples independent prompts are generated per pair.
"""

import argparse
import os
import random
import sys
import time

from templates import get_task, list_tasks
from utils import save_json


def generate_dataset(
    task_name: str,
    n_samples: int,
    m_values: list[int],
    n_values: list[int],
    seed: int = 42,
    csv_path: str | None = None,
) -> list[dict]:
    """
    Generate *n_samples* per (m, n) pair.
    Returns a flat list of sample dicts.
    """
    task = get_task(task_name, csv_path=csv_path)
    rng = random.Random(seed)
    samples = []

    for m in m_values:
        for n in n_values:
            print(f"  Generating {n_samples} samples at m={m}, n={n} …", flush=True)
            generated = 0
            attempts = 0
            max_attempts = n_samples * 20

            while generated < n_samples and attempts < max_attempts:
                attempts += 1
                try:
                    sample = task.generate_sample(m=m, n=n, rng=rng)
                except Exception as e:
                    print(f"    [warn] generation failed: {e}", file=sys.stderr)
                    continue

                sample.update(
                    {
                        "task": task_name,
                        "m": m,
                        "n": n,
                        "system_prompt": task.system_prompt,
                        "sample_id": len(samples),
                    }
                )
                samples.append(sample)
                generated += 1

            if generated < n_samples:
                print(
                    f"  [warn] Only generated {generated}/{n_samples} samples "
                    f"at m={m}, n={n} (too many failures).",
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
        help="Task to generate (collisions, astro, olmo_original).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per (m, n) pair (default: 100).",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help=(
            "Values of m to generate. "
            "collisions=num_particles, astro=num_rows, olmo_original=bit_array_size. "
            "(default: 4 8 16)"
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Values of n to generate. "
            "collisions=num_steps, astro=num_swaps, olmo_original=num_swap_lines. "
            "Defaults to same values as --m if not set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the output JSON into. Defaults to ../data/ relative to this script.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename override. Defaults to <task>_m<vals>_n<vals>_s<n_samples>.json.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to exoplanet CSV (astro task only). Overrides EXOPLANETS_CSV env var.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    # n defaults to same as m if not provided
    if args.n is None:
        args.n = args.m

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve output filename
    if args.output is None:
        m_str = "_".join(str(v) for v in args.m)
        n_str = "_".join(str(v) for v in args.n)
        filename = f"{args.task}_m{m_str}_n{n_str}_s{args.n_samples}.json"
    else:
        filename = args.output

    if os.path.isabs(filename) or os.sep in filename:
        output_path = filename
    else:
        output_path = os.path.join(args.output_dir, filename)

    print(f"Task       : {args.task}")
    print(f"m values   : {args.m}")
    print(f"n values   : {args.n}")
    print(f"Samples/(m,n): {args.n_samples}")
    print(f"Seed       : {args.seed}")
    print(f"Output dir : {args.output_dir}")
    print(f"Output file: {output_path}")
    print()

    t0 = time.time()
    samples = generate_dataset(
        task_name=args.task,
        n_samples=args.n_samples,
        m_values=args.m,
        n_values=args.n,
        seed=args.seed,
        csv_path=args.csv_path,
    )

    save_json(samples, output_path)
    elapsed = time.time() - t0
    print(f"\nDone — {len(samples)} samples written to {output_path}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()