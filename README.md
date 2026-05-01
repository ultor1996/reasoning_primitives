# State-Based Recall Evaluation Suite

A pipeline for generating, running, and evaluating LLM benchmarks focused on **state tracking** and **state-based recall** — the reasoning primitives studied in [OLMo Hybrid (Merrill et al., 2026)](https://arxiv.org/abs/2604.03444).

---

## Tasks

| Name | Description | `m` | `n` |
|---|---|---|---|
| `collisions` | Track particle velocities through elastic collisions | number of particles | number of collision steps |
| `astro` | Track variable→planet mappings through swap operations over an exoplanet table | number of table rows | number of swaps |
| `olmo_original` | State-based recall from the OLMo Hybrid paper: track 5 pointer variables through swaps, then index into a bit array | bit-array size | number of swap lines |
| `dyck` | Track a bracket stack through a Dyck expression and identify the correct closing token at a masked position | stack depth at query position | sequence length |

All tasks output **4-option multiple choice (A/B/C/D)**.

---

## Repository structure

```
├── src/
│   ├── utils.py          # Shared helpers (JSON I/O, answer parsing, vLLM loading)
│   ├── templates.py      # Task definitions — all prompts and generators live here
│   ├── generator.py      # Generate evaluation datasets (no GPU needed)
│   ├── inference.py      # Run a model over a dataset (requires GPU + vLLM)
│   ├── inference.sh      # SLURM job script for HPC clusters
│   ├── eval.py           # Compute accuracy from inference output
│   └── paper_plots.py    # Generate publication figures
├── data/                 # Generated datasets (auto-created)
└── results/              # Inference and eval outputs
```

---

## Installation

### Requirements

- Python 3.10+
- For **generation and evaluation only** (no GPU needed):

```bash
pip install json-repair
```

- For **inference** (GPU required):

```bash
pip install vllm transformers accelerate torch json-repair
```

### Clone the repo

```bash
git clone <your-repo-url>
cd <repo-name>
```

### Exoplanet CSV (for `astro` task only)

The `astro` task requires an exoplanet CSV file placed in the same folder as `templates.py`, or set the path explicitly:

```bash
# Option 1 — place the file next to templates.py (default)
cp exoplanets.csv src/

# Option 2 — environment variable
export EXOPLANETS_CSV=/path/to/exoplanets.csv

# Option 3 — pass it at generation time
python generator.py --task astro --csv-path /path/to/exoplanets.csv ...
```

The CSV must contain these columns:
`Planet`, `Host Star`, `Orbital Period (days)`, `Planet Radius (Earth radii)`, `Planet Mass (Earth masses)`, `Equilibrium Temp (K)`, `Semi-Major Axis (AU)`, `Eccentricity`, `Stellar Temp (K)`, `Stellar Radius (Solar radii)`, `Stellar Mass (Solar masses)`

---

## Step 1 — Generate a dataset

Run locally (no GPU needed). Output goes to `data/` by default (one level up from `src/`).

Use `--mode` to control how `m` and `n` are combined:
- `zip` *(default)* — pairs them together: `(4,4)`, `(8,8)`, `(16,16)`
- `cartesian` — all combinations: `(4,4)`, `(4,8)`, `(8,4)`, `(8,8)`

```bash
# zip mode (default) — only (4,4),(8,8),(16,16),(32,32)
python generator.py \
    --task collisions \
    --n-samples 100 \
    --m 4 8 16 32

# cartesian mode — all 16 combinations of m and n
python generator.py \
    --task collisions \
    --n-samples 100 \
    --m 4 8 16 32 \
    --n 4 8 16 32 \
    --mode cartesian

# olmo_original — different m and n ranges in cartesian mode
python generator.py \
    --task olmo_original \
    --n-samples 100 \
    --m 16 32 64 \
    --n 4 8 16 \
    --mode cartesian

# Astro task with explicit CSV path
python generator.py \
    --task astro \
    --n-samples 100 \
    --m 4 8 16 \
    --csv-path /path/to/exoplanets.csv

# dyck — cartesian mode recommended to separate stack depth vs sequence length
python generator.py \
    --task dyck \
    --n-samples 100 \
    --m 1 2 4 8 16 \
    --n 8 16 32 64 128 \
    --mode cartesian
```

> **Note for `dyck`:** `m` is stack depth at the query position and `n` is sequence
> length. Always keep `n >= 4*m` so the sequence is long enough to reach the target
> depth — e.g. pair `m=8` with `n=32` or longer. If `n` is too small the generator
> will silently use `target_depth * 4` as the actual sequence length instead.
> Cartesian mode is recommended over zip so stack depth and sequence length can be
> varied independently.

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--task` | required | `collisions`, `astro`, `olmo_original`, or `dyck` |
| `--n-samples` | `100` | Samples per `(m, n)` pair |
| `--m` | `4 8 16` | Space-separated list of `m` values |
| `--n` | same as `--m` | Space-separated list of `n` values (defaults to `--m` if omitted) |
| `--mode` | `zip` | `zip` = pair m and n together, `cartesian` = all combinations |
| `--output-dir` | `../data/` relative to script | Directory to save the JSON file |
| `--output` | auto-named | Override the output filename |
| `--csv-path` | env var / script-relative default | Path to exoplanet CSV (`astro` only) |
| `--seed` | `42` | Random seed for reproducibility |

Output is a single JSON file, e.g. `data/collisions_m4_8_16_n4_8_16_s100.json`.

---

## Step 2 — Run inference (HPC / GPU)

### On a local GPU

```bash
python inference.py \
    --input  data/collisions_m4_8_16_n4_8_16_s100.json \
    --model  allenai/Olmo-3-7B-Think \
    --output results/collisions_olmo3think.json
```

### On a SLURM cluster

```bash
sbatch inference.sh \
    --input  /path/to/data/collisions_m4_8_16_n4_8_16_s100.json \
    --model  allenai/Olmo-3-7B-Think \
    --output /path/to/results/collisions_olmo3think.json
```

Before submitting, update this variable at the top of `inference.sh` to match your cluster:

```bash
HF_CACHE="/path/to/.cache/huggingface"
```

**All inference options:**

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to generator output JSON |
| `--model` | required | HuggingFace model name or local path |
| `--output` | auto-named | Output JSON path |
| `--batch-size` | `8` | vLLM generation batch size |
| `--max-model-len` | `16000` | Max context + generation length |
| `--tensor-parallel` | `1` | Number of GPUs for tensor parallelism |

The output file is the input JSON with three extra fields added to every sample: `raw_output`, `completion` (thinking trace stripped), and `reasoning` (content of `<think>…</think>` if present).

---

## Step 3 — Evaluate

```bash
python eval.py \
    --input  results/collisions_olmo3think.json \
    --output scores/collisions_olmo3think_eval.json
```

Prints a summary to stdout and writes a JSON file with overall accuracy and a per `(m, n)` breakdown.

Add `--no-samples` to omit per-sample detail and keep the output file small.

---

## Step 4 — Plot results

```bash
python paper_plots.py \
    --inputs scores/*_eval.json \
    --output-dir figures/
```

Produces three figures (line plot, bar chart, heatmap) in both `.pdf` and `.png`.

---

## End-to-end example

```bash
# 1. Generate
python generator.py --task olmo_original --n-samples 100 --m 4 8 16 32 64

# 2. Copy to cluster and run inference
scp data/olmo_original_m4_8_16_32_64_n4_8_16_32_64_s100.json marvin:/path/to/data/
sbatch inference.sh \
    --input  /path/to/data/olmo_original_m4_8_16_32_64_n4_8_16_32_64_s100.json \
    --model  allenai/Olmo-Hybrid-Think-SFT-7B \
    --output /path/to/results/sbr_hybrid_think.json

# 3. Copy results back and evaluate
scp marvin:/path/to/results/sbr_hybrid_think.json results/
python eval.py --input results/sbr_hybrid_think.json

# 4. Plot
python paper_plots.py --inputs scores/*_eval.json --output-dir figures/
```
---

## Adding a new task

All tasks are defined in `templates.py`. To add one:

1. Write a generator function `my_task_generator(m: int, n: int, rng: random.Random) -> dict` that returns a dict with keys `prompt`, `correct_option`, `option_A`–`option_D`, and `metadata`.
2. Add a system prompt string.
3. Register it in `_REGISTRY` at the bottom of the file.

That's it — `generator.py`, `inference.py`, and `eval.py` will all pick it up automatically.

---

## Citation

If you use the `olmo_original` task, please cite:

```bibtex
@article{merrill2026olmohybrid,
  title   = {OLMo Hybrid: From Theory to Practice and Back},
  author  = {Merrill, William and Li, Yanhong and Romero, Tyler and others},
  journal = {arXiv preprint arXiv:2604.03444},
  year    = {2026}
}
```