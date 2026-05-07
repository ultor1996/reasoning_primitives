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
| `dag_arithmetic` | Trace cascading addition/subtraction computations through a DAG with cross-layer dependencies and report the final value of a queried variable | variables per layer (width) | number of layers (depth) |

All tasks output **4-option multiple choice (A/B/C/D)**.

---

## Repository structure

```
├── src/
│   ├── utils.py                 # Shared helpers (JSON I/O, answer parsing, vLLM loading)
│   ├── templates.py             # Task definitions — all prompts and generators live here
│   ├── generator.py             # Generate evaluation datasets (no GPU needed)
│   ├── inference.py             # Run a model over a dataset (requires GPU + vLLM)
│   ├── inference.sh             # SLURM job script for HPC clusters
│   ├── eval.py                  # Compute accuracy from inference output
│   ├── paper_plots.py           # Generate publication figures
│   └── check_token_lengths.py   # Check prompt token lengths across difficulty levels
├── data/                        # Generated datasets (auto-created)
└── results/                     # Inference and eval outputs
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

# dag_arithmetic — cartesian mode recommended to separate width vs depth
python generator.py \
    --task dag_arithmetic \
    --n-samples 100 \
    --m 2 4 8 16 \
    --n 2 4 8 16 \
    --mode cartesian
```

> **Note for `dyck`:** `m` is stack depth at the query position and `n` is sequence
> length. Always keep `n >= 4*m` so the sequence is long enough to reach the target
> depth — e.g. pair `m=8` with `n=32` or longer. If `n` is too small the generator
> will silently use `target_depth * 4` as the actual sequence length instead.
> Cartesian mode is recommended over zip so stack depth and sequence length can be
> varied independently.

> **Note for `dag_arithmetic`:** `m` is the number of variables per layer (width) and
> `n` is the number of computation layers (depth). Operations are limited to `+` and `-`.
> Cartesian mode is recommended because width and depth are independent difficulty axes.
> With 50% probability, nodes reference variables from non-adjacent earlier layers,
> meaning a computation in layer 9 may depend on a value from layer 2 — the model must
> hold many intermediate values in memory simultaneously rather than just the previous layer.
>
> **Difficulty knobs** (tunable in `_dag_generator` in `templates.py`):
> - `op1` distant-layer probability (default `0.5`): controls how often the first operand
>   references a non-adjacent layer. Higher = more long-range dependencies = harder.
>   Range: `0.0` (always previous layer, easiest) to `1.0` (always distant, hardest).
> - `op2` variable probability (default `0.5`): controls how often the second operand is
>   a variable vs a small constant (1–5). Higher = more variable references = harder.
>   Range: `0.0` (always constant) to `1.0` (always a variable from any previous layer).
>
> Example — harder task (more cross-layer dependencies):
> ```python
> if layer >= 3 and rng.random() < 0.6:   # op1: 60% distant
> if rng.random() < 0.7:                  # op2: 70% variable
> ```
> Example — easier task (more linear flow):
> ```python
> if layer >= 3 and rng.random() < 0.2:   # op1: 20% distant
> if rng.random() < 0.3:                  # op2: 30% variable
> ```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--task` | required | `collisions`, `astro`, `olmo_original`, `dyck`, or `dag_arithmetic` |
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

## Step 1b — Check prompt token lengths

Before running inference, use `check_token_lengths.py` to verify that prompt lengths
at each difficulty level fit within your model's context window:

```bash
# Check all files in the default data directory
python check_token_lengths.py

# Check only olmo_original files
python check_token_lengths.py --task olmo_original

# Use a different model for tokenization
python check_token_lengths.py --model allenai/OLMo-Hybrid-Instruct-SFT-7B

# Use a different data directory
python check_token_lengths.py --data-dir /path/to/data
```

This prints `m`, `n`, and token count for every difficulty level in every matching file.

**Setting `--max-model-len` for inference:**
- For **instruct models**: use `largest_prompt_tokens + 256`
- For **thinking models**: use `largest_prompt_tokens + 4000` (thinking traces are long)
- Make sure the total stays within the model's context window (OLMo models: 32k)

> **Example:** At `m=2500` the prompt is ~27,580 tokens.
> Instruct models: `27580 + 256 = 27836` → use `--max-model-len 28000`
> Thinking models: `27580 + 4000 = 31580` → use `--max-model-len 32000`

---

## Step 2 — Run inference (HPC / GPU)

### On a local GPU

```bash
python inference.py \
    --input  data/collisions_m4_8_16_n4_8_16_s100.json \
    --model  allenai/OLMo-3-7B-Think \
    --output results/collisions_olmo3think.json
```

### On a SLURM cluster

```bash
sbatch inference.sh \
    --input  /path/to/data/collisions_m4_8_16_n4_8_16_s100.json \
    --model  allenai/OLMo-3-7B-Think \
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
| `--max-tokens` | `512` | Max tokens the model generates per sample |
| `--tensor-parallel` | `1` | Number of GPUs for tensor parallelism |

> **Thinking models** need a larger `--max-tokens` budget to accommodate the reasoning
> trace before the final JSON answer. Use `--max-tokens 4000` for thinking models vs
> `--max-tokens 256` for instruct models. If jobs OOM, reduce `--batch-size` to `4` or `2`.

---

## Step 3 — Evaluate

```bash
python eval.py \
    --input  results/collisions_olmo3think.json \
    --output scores/collisions_olmo3think_eval.json
```

Prints a summary to stdout and writes a JSON file with:
- overall accuracy and **overall parsed weighted accuracy (PWA)**
- a per `(m, n)` breakdown with both `accuracy` and `parsed_weighted_accuracy`

> **Parsed weighted accuracy** = `accuracy × (n_scored / n_total)` per `(m, n)` combo.
> It penalises difficulty levels where the model frequently failed to produce valid JSON,
> giving a more conservative estimate of true performance.

Add `--no-samples` to omit per-sample detail and keep the output file small.

To run eval on all inference files in a directory at once:

```bash
for f in ../results/*.json; do
    [[ "$f" == *_eval.json ]] && continue
    python eval.py --input "$f" --output "${f%.json}_eval.json"
done
```

---

## Step 4 — Plot results

```bash
python paper_plots.py \
    --inputs scores/*_eval.json \
    --output-dir figures/
```

Produces three figures in both `.pdf` and `.png`:
- `accuracy_line_<task>_<models>` — accuracy per model across difficulty levels
- `pwa_line_<task>_<models>` — parsed weighted accuracy per model across difficulty levels
- `parse_rate_line_<task>_<models>` — fraction of successfully parsed responses per model across difficulty levels

The task name and model names are automatically detected from the eval JSON files and
included in the output filenames. When multiple eval files are provided for the same
model (e.g. from different random seeds), the plots automatically show the mean across
seeds with a shaded region indicating ±1 standard deviation.

**All plot options:**

| Flag | Default | Description |
|---|---|---|
| `--inputs` | required | Paths or globs to eval JSON files |
| `--output-dir` | `figures/` | Directory to write figures |
| `--title` | `""` | Optional figure title suffix |
| `--max-m` | `None` | Cap plots at this `m` value — useful when models have different context limits and you want a fair comparison (e.g. `--max-m 2048` excludes larger difficulties) |
| `--task` | auto-detected | Override the task name in output filenames |

> **Example use case for `--max-m`:** If instruct models were run on data up to `m=2500`
> but thinking models can only fit up to `m=2048` within their context window, use
> `--max-m 2048` so all models are compared on the same difficulty levels.

```bash
# Compare all models fairly up to m=2048
python paper_plots.py \
    --inputs scores/*_eval.json \
    --output-dir figures/ \
    --max-m 2048
```

> **Multiple seeds:** To get statistically more robust results, generate datasets with
> multiple random seeds, run inference and eval on each, then pass all eval files
> together. The plotting script automatically averages across seeds and shows ±1 std
> as a shaded band:
> ```bash
> # Generate with 3 seeds
> python generator.py --task olmo_original --n-samples 1000 --m 4 8 16 32 64 --seed 42
> python generator.py --task olmo_original --n-samples 1000 --m 4 8 16 32 64 --seed 123
> python generator.py --task olmo_original --n-samples 1000 --m 4 8 16 32 64 --seed 456
>
> # After inference and eval on all 3:
> python paper_plots.py --inputs scores/olmo_original_*_eval.json --output-dir figures/
> ```

---

## End-to-end example

```bash
# 1. Generate
python generator.py --task olmo_original --n-samples 100 --m 4 8 16 32 64

# 2. Check token lengths
python check_token_lengths.py --task olmo_original

# 3. Copy to cluster and run inference
scp data/olmo_original_m4_8_16_32_64_n4_8_16_32_64_s100.json marvin:/path/to/data/

# Instruct model
sbatch inference.sh \
    --input  /path/to/data/olmo_original_m4_8_16_32_64_n4_8_16_32_64_s100.json \
    --model  allenai/OLMo-3-7B-Instruct \
    --output /path/to/results/olmo_original_olmo3_instruct.json \
    --max-model-len 28000 --max-tokens 256

# Thinking model
sbatch inference.sh \
    --input  /path/to/data/olmo_original_m4_8_16_32_64_n4_8_16_32_64_s100.json \
    --model  allenai/OLMo-3-7B-Think \
    --output /path/to/results/olmo_original_olmo3_think.json \
    --max-model-len 32000 --max-tokens 4000

# 4. Copy results back and evaluate
scp marvin:/path/to/results/*.json results/
python eval.py --input results/olmo_original_olmo3_instruct.json
python eval.py --input results/olmo_original_olmo3_think.json

# 5. Plot — cap at m=2048 for fair comparison across all models
python paper_plots.py \
    --inputs scores/*_eval.json \
    --output-dir figures/ \
    --max-m 2048
```


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