"""
templates.py — task definitions for the three benchmark tasks.

Exoplanet CSV
-------------
The `astro` and `olmo3` tasks load planet data from a CSV file.
Set the path via the EXOPLANETS_CSV environment variable, or pass it
explicitly when calling get_task():

    task = get_task("astro", csv_path="/path/to/exoplanets.csv")

The CSV must have at minimum these columns (matching your confirmed_planets.csv):
    Planet, Host Star, Orbital Period (days), Planet Radius (Earth radii),
    Planet Mass (Earth masses), Equilibrium Temp (K), Semi-Major Axis (AU),
    Eccentricity, Stellar Temp (K), Stellar Radius (Solar radii),
    Stellar Mass (Solar masses)

Each task exposes:
  SYSTEM_PROMPT   : str
  generate_sample : (difficulty: int, rng: random.Random) -> dict
                    Returns a dict with at least:
                      "prompt"        : str   (user-facing question)
                      "correct_option": str   (one of A/B/C/D)
                      "option_A" …    "option_D" : str
                      "metadata"      : dict  (difficulty breakdown, etc.)

Supported task names (strings):
  "collisions"  — elastic-collision velocity-tracking
  "astro"       — exoplanet table state-based recall
  "olmo3"       — OLMo-3 style orbital-period / planet swap task
                  (same dataset, smaller table, A/B only — kept for
                   backward-compatibility with older eval files)

Usage example
-------------
from templates import get_task
task = get_task("collisions")
sample = task.generate_sample(difficulty=8)
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Exoplanet CSV loader
# ---------------------------------------------------------------------------

# Columns we keep from the CSV (must all be present)
_CSV_COLUMNS = [
    "Planet",
    "Host Star",
    "Orbital Period (days)",
    "Planet Radius (Earth radii)",
    "Planet Mass (Earth masses)",
    "Equilibrium Temp (K)",
    "Semi-Major Axis (AU)",
    "Eccentricity",
    "Stellar Temp (K)",
    "Stellar Radius (Solar radii)",
    "Stellar Mass (Solar masses)",
]

# Default CSV path — override with the EXOPLANETS_CSV env var or by passing
# csv_path= to get_task().
_DEFAULT_CSV_PATH = os.environ.get(
    "EXOPLANETS_CSV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "exoplanets.csv"),
)

# Module-level cache so we only read the file once per process
_planets_cache: dict[str, list[dict]] = {}


def _load_planets(csv_path: str) -> list[dict]:
    """
    Load and return a list of planet dicts from *csv_path*.

    Numeric columns are cast to float where possible; rows with a missing
    Planet name or Orbital Period are dropped.  The result is cached by path.
    """
    if csv_path in _planets_cache:
        return _planets_cache[csv_path]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Exoplanet CSV not found: {csv_path}\n"
            "Set the EXOPLANETS_CSV environment variable or pass csv_path= to get_task()."
        )

    planets: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        # Strip whitespace from header names
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]

        for raw in reader:
            row = {k.strip(): v.strip() for k, v in raw.items()}

            # Must have a planet name and a valid orbital period
            name = row.get("Planet", "").strip()
            period_str = row.get("Orbital Period (days)", "").strip()
            if not name or not period_str:
                continue

            planet: dict = {}
            for col in _CSV_COLUMNS:
                val = row.get(col, "")
                try:
                    planet[col] = float(val) if val else None
                except ValueError:
                    planet[col] = val  # keep as string (e.g. "Planet" name)
            # Overwrite Planet / Host Star as plain strings
            planet["Planet"]    = name
            planet["Host Star"] = row.get("Host Star", "").strip()
            planets.append(planet)

    if not planets:
        raise ValueError(f"No valid rows found in {csv_path}")

    _planets_cache[csv_path] = planets
    return planets


# ============================================================================
# Base dataclass
# ============================================================================

@dataclass
class TaskTemplate:
    name: str
    system_prompt: str
    # callable(difficulty, rng) -> sample dict
    _generator: Callable = field(repr=False)

    def generate_sample(self, m: int, n: int, rng: random.Random | None = None) -> dict:
        """
        Generate one sample.

        m : first axis of difficulty (particles / rows / bit-array size)
        n : second axis of difficulty (collision steps / swaps / swap lines)
        """
        if rng is None:
            rng = random.Random()
        return self._generator(m, n, rng)


# ============================================================================
# helpers
# ============================================================================

def _get_particle_names(n: int) -> list[str]:
    """A, B, …, Z, AA, AB, …"""
    names = []
    for i in range(n):
        name = ""
        x = i
        while True:
            name = chr(ord("A") + (x % 26)) + name
            x = x // 26 - 1
            if x < 0:
                break
        names.append(name)
    return names


def _get_variable_names(n: int) -> list[str]:
    """a, b, …, z, aa, ab, …"""
    names = []
    for i in range(n):
        name = ""
        x = i
        while True:
            name = chr(ord("a") + (x % 26)) + name
            x = x // 26 - 1
            if x < 0:
                break
        names.append(name)
    return names


def _build_mc_options(correct_value, all_values: list, rng: random.Random, n_options: int = 4):
    """
    Build a shuffled multiple-choice option dict {label: value} and return
    (options_dict, correct_label).  Returns None if not enough wrong candidates.
    """
    wrong = [v for v in set(all_values) if v != correct_value]
    if len(wrong) < n_options - 1:
        return None, None
    wrong_chosen = rng.sample(wrong, n_options - 1)
    values = [correct_value] + wrong_chosen
    rng.shuffle(values)
    labels = ["A", "B", "C", "D"][:n_options]
    options = dict(zip(labels, values))
    correct_label = next(lbl for lbl, val in options.items() if val == correct_value)
    return options, correct_label


# ============================================================================
# Task: collisions
# ============================================================================

_COLLISIONS_SYSTEM_PROMPT = """You are a strict state-tracking engine for collision systems.

Task:
- You are given particles with initial velocities.
- You are given a sequence of pairwise collisions.

Core rule (MUST be applied exactly):
- When two equal-mass particles collide, they EXCHANGE velocities.
- This is equivalent to swapping their velocity values.

Reasoning requirements:
- Maintain an explicit mapping: particle → velocity.
- Apply each collision in order.
- After each collision, update BOTH particles' velocities.
- Do NOT skip steps.
- Do NOT infer physics beyond the given rule.

Output requirements:
- Return EXACTLY one JSON object.
- No extra text.

Format:
{
  "answer": "A | B | C | D"
}
"""


def _collisions_generator(m: int, n: int, rng: random.Random) -> dict:
    """
    m = number of particles (minimum 4 for 4-option MC).
    n = number of collision steps.
    """
    num_particles = max(4, m)
    num_steps = max(1, n)

    particles = _get_particle_names(num_particles)
    velocity_pool = range(1, max(1001, num_particles * 2))
    velocities = rng.sample(velocity_pool, num_particles)
    initial = dict(zip(particles, velocities))

    state = dict(initial)
    steps = []
    for _ in range(num_steps):
        for _ in range(1000):
            a, b = rng.sample(particles, 2)
            if not steps or steps[-1] != (b, a):
                break
        steps.append((a, b))
        state[a], state[b] = state[b], state[a]

    query = rng.choice(particles)
    correct_value = state[query]

    options, correct_label = _build_mc_options(correct_value, velocities, rng)
    if options is None:
        return _collisions_generator(m, n, random.Random(rng.randint(0, 2**31)))

    # Build prompt
    lines = [
        "# Physics Collision Task\n",
        "## Problem\n",
        "Consider a one-dimensional system where all particles move along a line.\n",
        "**Key rule:**",
        "- When two equal-mass particles collide elastically, they exchange velocities.\n",
        "### Initial velocities",
    ]
    for p in particles:
        lines.append(f"- {p} = {initial[p]}")
    lines.append("\n### Collisions")
    for i, (a, b) in enumerate(steps, 1):
        lines.append(f"{i}. {a} collides with {b}")
    lines.append(f"\n### Question\nWhat is the velocity of particle {query} after all collisions?\n")
    lines.append("### Options")
    for lbl in ["A", "B", "C", "D"]:
        lines.append(f"{lbl}) {options[lbl]}")

    return {
        "prompt": "\n".join(lines),
        "correct_option": correct_label,
        "option_A": str(options["A"]),
        "option_B": str(options["B"]),
        "option_C": str(options["C"]),
        "option_D": str(options["D"]),
        "metadata": {
            "num_particles": num_particles,
            "num_steps": num_steps,
            "m": num_particles,
            "n": num_steps,
            "question_particle": query,
            "correct_answer": correct_value,
        },
    }


# ============================================================================
# Task: astro  (exoplanet table state-based recall)
# ============================================================================

_ASTRO_SYSTEM_PROMPT = """You are a precise reasoning assistant.

You will be given:
1. A table of exoplanet data
2. Variable assignments mapping variable names to column values
3. One or more swap operations (Python-style simultaneous assignment)
4. A multiple-choice question asking which option matches the final value

Trace the swaps carefully and determine the correct option.

Output requirements:
- Return EXACTLY one JSON object.
- No extra text.

Format:
{
  "answer": "A | B | C | D"
}
"""

def _astro_generator(m: int, n: int, rng: random.Random, csv_path: str = _DEFAULT_CSV_PATH) -> dict:
    """
    m = number of table rows shown (minimum 4).
    n = number of swap steps.
    Planets are sampled WITHOUT replacement from the CSV.
    """
    num_rows = max(4, m)
    num_swaps = max(1, n)

    planets = _load_planets(csv_path)
    if len(planets) < num_rows:
        raise ValueError(
            f"CSV has only {len(planets)} valid rows but m={m} "
            f"requires {num_rows}."
        )
    chosen = rng.sample(planets, num_rows)

    target_col = "Orbital Period (days)"
    retrieve_col = "Planet"

    var_names = _get_variable_names(num_rows)
    values = [p[target_col] for p in chosen]
    retrieve_values = [p[retrieve_col] for p in chosen]

    # Build initial variable mapping
    var_map = dict(zip(var_names, list(range(num_rows))))  # var -> index into chosen

    # Simulate swaps
    state = list(range(num_rows))  # state[i] = current index into chosen for var i
    swap_steps = []
    for _ in range(num_swaps):
        for _ in range(1000):
            i, j = rng.sample(range(num_rows), 2)
            if not swap_steps or swap_steps[-1] != (j, i):
                break
        swap_steps.append((i, j))
        state[i], state[j] = state[j], state[i]

    # Query
    query_var_idx = rng.randrange(num_rows)
    query_var = var_names[query_var_idx]
    correct_retrieve = retrieve_values[state[query_var_idx]]

    # Build options
    all_retrieve = retrieve_values
    options, correct_label = _build_mc_options(correct_retrieve, all_retrieve, rng)
    if options is None:
        return _astro_generator(m, n, random.Random(rng.randint(0, 2**31)), csv_path)

    # Markdown table header — use all available CSV columns
    cols = ["Planet", "Host Star", "Orbital Period (days)", "Planet Radius (Earth radii)",
            "Planet Mass (Earth masses)", "Equilibrium Temp (K)", "Semi-Major Axis (AU)",
            "Eccentricity", "Stellar Temp (K)", "Stellar Radius (Solar radii)",
            "Stellar Mass (Solar masses)"]
    # Only keep cols that are actually present in the data
    cols = [c for c in cols if chosen[0].get(c) is not None]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = [header, sep]
    for p in chosen:
        rows.append("| " + " | ".join(str(p.get(c, "")) for c in cols) + " |")

    vars_str = ", ".join(var_names)
    vals_str = ", ".join(str(v) for v in values)

    lines = [
        "\n".join(rows),
        "",
        f"Consider the following {target_col}: {vars_str} = {vals_str}",
        "",
        "Consider the following swapping:",
    ]
    for i, j in swap_steps:
        vi, vj = var_names[i], var_names[j]
        lines.append(f"- {vi}, {vj} = {vj}, {vi}")
    lines.append("")
    lines.append(f"The {retrieve_col} with the {target_col} = {query_var} is:")
    lines.append("")
    lines.append("### Options")
    for lbl in ["A", "B", "C", "D"]:
        lines.append(f"{lbl}) {options[lbl]}")

    return {
        "prompt": "\n".join(lines),
        "correct_option": correct_label,
        "option_A": str(options["A"]),
        "option_B": str(options["B"]),
        "option_C": str(options["C"]),
        "option_D": str(options["D"]),
        "metadata": {
            "num_rows": num_rows,
            "num_swaps": num_swaps,
            "m": num_rows,
            "n": num_swaps,
            "query_variable": query_var,
        },
    }


# ============================================================================
# Task: olmo_original
#
# From Merrill et al. (2026) "OLMo Hybrid: From Theory to Practice and Back"
# Figure 5 / Appendix C.1.
#
# Structure (next-token prediction framed as 4-option MC):
#
#   bits = [0, 1, 0, 0, ...]          # m bits
#   a, b, c, d, e = 36, 23, 12, 2, 56 # 0 to m-1  (5 variables)
#   a, c = c, e                        # swap line 1
#   ...                                # n swap lines total
#   assert bits[a] == _                # answer: 0 or 1
#
# Difficulty parameter = n (number of swap lines) = m (bit-array size),
# matching the paper's default n = m sweep.
# The 5 pointer variables are always named a, b, c, d, e (as in the paper).
# Answer is 4-option MC over {0, 1} — but since the answer space is binary,
# we add two plausible distractors drawn as wrong-bit values so the format
# stays consistent with the rest of the pipeline (A/B/C/D).
# ============================================================================

_SBR_SYSTEM_PROMPT = """You are a strict code-execution engine.

Task:
- You are given a bit array and five pointer variables (a, b, c, d, e).
- You are given a sequence of simultaneous swap assignments.
- You must track the pointer values through every swap and then look up the
  correct bit in the array.

Rules:
- Each swap line uses Python simultaneous assignment: x, y = y, x
- Apply every swap in order; do NOT skip any.
- After all swaps, evaluate bits[<queried variable>].

Output requirements:
- Return EXACTLY one JSON object, no other text.

Format:
{
  "answer": "A | B | C | D"
}
"""

# Number of pointer variables — fixed at 5 to match the paper.
_SBR_NUM_VARS = 5
_SBR_VAR_NAMES = ["a", "b", "c", "d", "e"]


def _sbr_generator(m: int, n: int, rng: random.Random) -> dict:
    """
    State-Based Recall task (Merrill et al. 2026, Figure 5).

    m = bit-array size (minimum 5 to hold 5 distinct pointer values).
    n = number of swap lines.
    """
    m = max(5, m)
    n = max(1, n)

    # --- bit array ---
    bits = [rng.randint(0, 1) for _ in range(m)]

    # --- pointer initialisation: 5 distinct indices in [0, m-1] ---
    if m < _SBR_NUM_VARS:
        # fallback: allow repeats when m is tiny (shouldn't happen with min=4)
        pointers = [rng.randrange(m) for _ in range(_SBR_NUM_VARS)]
    else:
        pointers = rng.sample(range(m), _SBR_NUM_VARS)

    # --- simulate n swap lines ---
    # Each swap is a simultaneous assignment over two of the 5 variables.
    state = list(pointers)   # current pointer values
    swap_lines = []
    for _ in range(n):
        for _ in range(1000):   # avoid immediate undo
            i, j = rng.sample(range(_SBR_NUM_VARS), 2)
            if not swap_lines or swap_lines[-1] != (j, i):
                break
        swap_lines.append((i, j))
        state[i], state[j] = state[j], state[i]

    # --- query ---
    query_idx = rng.randrange(_SBR_NUM_VARS)
    query_var = _SBR_VAR_NAMES[query_idx]
    final_ptr = state[query_idx]
    correct_bit = bits[final_ptr]

    # --- 4-option MC ---
    # Answer space is {0, 1}.  To fill 4 options without repetition we add
    # two "distractor" integers that are clearly wrong (2, 3) — this is a
    # deliberate design choice so the task remains a genuine binary lookup
    # while keeping the MC format consistent with the rest of the pipeline.
    wrong_options = [1 - correct_bit, 2, 3]
    rng.shuffle(wrong_options)
    option_values = [correct_bit] + wrong_options
    rng.shuffle(option_values)

    labels = ["A", "B", "C", "D"]
    options = dict(zip(labels, option_values))
    correct_label = next(lbl for lbl, val in options.items() if val == correct_bit)

    # --- build prompt (code-like, matching paper Figure 5) ---
    bits_str = "[" + ", ".join(str(b) for b in bits) + "]"
    ptr_vals_str = ", ".join(str(p) for p in pointers)
    var_names_str = ", ".join(_SBR_VAR_NAMES)

    lines = [
        f"bits = {bits_str}  # {m} bits",
        f"{var_names_str} = {ptr_vals_str}  # 0 to {m - 1}",
    ]
    for i, j in swap_lines:
        vi, vj = _SBR_VAR_NAMES[i], _SBR_VAR_NAMES[j]
        lines.append(f"{vi}, {vj} = {vj}, {vi}")
    lines.append(f"assert bits[{query_var}] == _  # 0 or 1")
    lines.append("")
    lines.append("### Options")
    for lbl in labels:
        lines.append(f"{lbl}) {options[lbl]}")

    return {
        "prompt": "\n".join(lines),
        "correct_option": correct_label,
        "option_A": str(options["A"]),
        "option_B": str(options["B"]),
        "option_C": str(options["C"]),
        "option_D": str(options["D"]),
        "metadata": {
            "m": m,
            "n": n,
            "query_variable": query_var,
            "final_pointer": final_ptr,
            "correct_bit": correct_bit,
        },
    }

# ============================================================================
# Task: dyck
#
# Standard Dyck language with fixed vocab: ( ) [ ] { }
# Strict LIFO nesting.
#
# m = stack depth at the query position (working memory pressure)
# n = sequence length (how far back the model must read)
#
# The prompt gives a complete sequence with one closer masked as _.
# The model must identify the correct closing token from 4 options.
# ============================================================================

_DYCK_SYSTEM_PROMPT = """You are a strict language validator for Dyck expressions.

Rules:
- A Dyck expression uses bracket pairs: ( ), [ ], { }
- Every opening bracket must be closed by its exact matching closer.
- Brackets must be closed in the correct order (last opened = first closed).

Task:
- You are given a Dyck expression with one token masked as _.
- You must determine what token _ must be to keep the expression valid.

Output requirements:
- Return EXACTLY one JSON object, no other text.

Format:
{
  "answer": "A | B | C | D"
}
"""

_DYCK_OPENERS = ["(", "[", "{"]
_DYCK_CLOSERS = [")", "]", "}"]
_DYCK_MATCH   = {"(": ")", "[": "]", "{": "}"}


def _dyck_generator(m: int, n: int, rng: random.Random) -> dict:
    """
    Standard Dyck language task.

    m = stack depth at the query position (working memory pressure).
    n = sequence length (how far back the model must read).
    """
    target_depth = max(1, m)
    seq_len      = max(target_depth * 4, n + (n % 2))  # ensure even and long enough

    openers = _DYCK_OPENERS[:3]
    matcher = _DYCK_MATCH

    for attempt in range(10000):
        sequence = []
        stack    = []

        while len(sequence) < seq_len:
            remaining  = seq_len - len(sequence)
            must_close = len(stack) >= remaining
            must_open  = len(stack) == 0

            if must_close:
                sequence.append(matcher[stack[-1]])
                stack.pop()
            elif must_open:
                b = rng.choice(openers)
                sequence.append(b)
                stack.append(b)
            else:
                if rng.random() < 0.5:
                    b = rng.choice(openers)
                    sequence.append(b)
                    stack.append(b)
                else:
                    sequence.append(matcher[stack[-1]])
                    stack.pop()

        # Find closer positions where stack depth == target_depth just before close
        candidate_positions = []
        sim_stack = []
        for i, token in enumerate(sequence):
            if token in openers:
                sim_stack.append(token)
            else:
                if len(sim_stack) == target_depth:
                    candidate_positions.append(i)
                if sim_stack:
                    sim_stack.pop()

        if candidate_positions:
            query_pos     = rng.choice(candidate_positions)
            correct_token = sequence[query_pos]
            break
    else:
        raise RuntimeError(
            f"Could not generate dyck sample with "
            f"target_depth={target_depth}, seq_len={seq_len} after 10000 attempts."
        )

    # --- Build MC options ---
    all_tokens   = _DYCK_OPENERS + _DYCK_CLOSERS
    wrong_tokens = [t for t in all_tokens if t != correct_token]
    wrong_chosen = rng.sample(wrong_tokens, 3)

    option_values = [correct_token] + wrong_chosen
    rng.shuffle(option_values)
    labels  = ["A", "B", "C", "D"]
    options = dict(zip(labels, option_values))
    correct_label = next(lbl for lbl, val in options.items() if val == correct_token)

    # --- Build prompt ---
    display            = list(sequence)
    display[query_pos] = "_"
    expr_str           = " ".join(display)

    # Reconstruct stack at query position
    sim_stack = []
    for i in range(query_pos):
        t = sequence[i]
        if t in openers:
            sim_stack.append(t)
        else:
            if sim_stack:
                sim_stack.pop()

    lines = [
        f"Expression ({seq_len} tokens): {expr_str}",
        "",
        "Bracket pairs: ( )  [ ]  { }",
        "Every opener must be closed by its matching closer in the correct order.",
        "",
        f"What token must replace _ at position {query_pos + 1}?",
        "",
        "### Options",
    ]
    for lbl in labels:
        lines.append(f"{lbl}) {options[lbl]}")

    return {
        "prompt": "\n".join(lines),
        "correct_option": correct_label,
        "option_A": str(options["A"]),
        "option_B": str(options["B"]),
        "option_C": str(options["C"]),
        "option_D": str(options["D"]),
        "metadata": {
            "m": target_depth,
            "n": seq_len,
            "stack_depth_at_query": target_depth,
            "sequence_length": seq_len,
            "query_position": query_pos,
            "correct_token": correct_token,
            "stack_at_query": list(sim_stack),
            "sequence": "".join(sequence),
        },
    }
# Add this block right before the Registry section:

# ============================================================================
# Task: dag_arithmetic
# ============================================================================

_DAG_SYSTEM_PROMPT = """You are a strict arithmetic computation engine.

Task:
- You are given a set of input variables with integer values.
- You are given a sequence of computation steps organized in layers.
- Each step computes a new variable from one or two previous variables
  using addition or subtraction only.
- You must trace every computation in order and track all variable values.

Rules:
- Apply every step in order. Do NOT skip any.
- Use integer arithmetic throughout.
- After all steps, report the value of the queried variable.

Output requirements:
- Return EXACTLY one JSON object, no other text.

Format:
{
  "answer": "A | B | C | D"
}
"""

_DAG_OPS = ["+", "-"]


def _dag_generator(m: int, n: int, rng: random.Random) -> dict:
    """
    DAG arithmetic task with cross-layer dependencies.

    m = number of variables per layer (width, minimum 2).
    n = number of computation layers (depth, minimum 2).
    """
    m = max(2, m)
    n = max(2, n)

    def var_name(layer: int, idx: int) -> str:
        if layer == 0:
            name = ""
            x = idx
            while True:
                name = chr(ord("a") + (x % 26)) + name
                x = x // 26 - 1
                if x < 0:
                    break
            return name
        return f"v{layer}_{idx}"

    # Layer 0: input variables
    layer_vars = [[var_name(0, i) for i in range(m)]]
    values = {v: rng.randint(1, 20) for v in layer_vars[0]}

    all_steps = []

    for layer in range(1, n + 1):
        curr_layer = []
        for idx in range(m):
            result_var = var_name(layer, idx)
            curr_layer.append(result_var)

            op = rng.choice(_DAG_OPS)

            # All variables available from any previous layer
            all_prev_vars = [v for past_layer in layer_vars for v in past_layer]

            # 40% chance to pick from a distant layer (2+ layers back) when deep enough
            if layer >= 3 and rng.random() < 0.4:
                distant_vars = [v for past_layer in layer_vars[:-1] for v in past_layer]
                op1 = rng.choice(distant_vars)
            else:
                op1 = rng.choice(layer_vars[-1])

            # Second operand: any previous variable or a small constant
            if rng.random() < 0.5:
                op2_var = rng.choice(all_prev_vars)
                v1, v2 = values[op1], values[op2_var]
                op2_is_var = True
            else:
                op2_var = None
                v1 = values[op1]
                v2 = rng.randint(1, 5)
                op2_is_var = False

            if op == "+":   result = v1 + v2
            elif op == "-": result = v1 - v2

            values[result_var] = result
            all_steps.append((result_var, op, op1, op2_var, v2, op2_is_var))

        layer_vars.append(curr_layer)

    # Query from final layer
    query_var = rng.choice(layer_vars[-1])
    correct_value = values[query_var]

    # MC options
    all_values = list(set(values.values()))
    wrong_values = [v for v in all_values if v != correct_value]
    for delta in [1, -1, 2, -2, 3, -3, 5, 10, -5, -10]:
        if len(wrong_values) >= 3:
            break
        c = correct_value + delta
        if c not in wrong_values and c != correct_value:
            wrong_values.append(c)
    wrong_chosen = rng.sample(wrong_values, 3)
    option_values = [correct_value] + wrong_chosen
    rng.shuffle(option_values)
    labels = ["A", "B", "C", "D"]
    options = dict(zip(labels, option_values))
    correct_label = next(lbl for lbl, val in options.items() if val == correct_value)

    # Build prompt
    lines = [
        f"DAG arithmetic computation ({m} variables wide, {n} layers deep)",
        "",
        "Input variables:",
    ]
    for v in layer_vars[0]:
        lines.append(f"  {v} = {values[v]}")
    lines.append("")
    lines.append("Computation steps:")

    step_num = 1
    for layer in range(1, n + 1):
        lines.append(f"\n  Layer {layer}:")
        for idx in range(m):
            result_var, op, op1, op2_var, const, op2_is_var = all_steps[(layer - 1) * m + idx]
            expr = f"{op1} {op} {op2_var}" if op2_is_var else f"{op1} {op} {const}"
            lines.append(f"  Step {step_num:>3}: {result_var} = {expr}")
            step_num += 1

    lines.append("")
    lines.append(f"What is the value of {query_var} after all computations?")
    lines.append("")
    lines.append("### Options")
    for lbl in labels:
        lines.append(f"{lbl}) {options[lbl]}")

    return {
        "prompt": "\n".join(lines),
        "correct_option": correct_label,
        "option_A": str(options["A"]),
        "option_B": str(options["B"]),
        "option_C": str(options["C"]),
        "option_D": str(options["D"]),
        "metadata": {
            "m": m,
            "n": n,
            "width": m,
            "depth": n,
            "total_steps": m * n,
            "query_variable": query_var,
            "correct_value": correct_value,
        },
    }
# ============================================================================
# Registry
# ============================================================================

_REGISTRY: dict[str, TaskTemplate] = {
    "collisions": TaskTemplate(
        name="collisions",
        system_prompt=_COLLISIONS_SYSTEM_PROMPT,
        _generator=_collisions_generator,
    ),
    "astro": TaskTemplate(
        name="astro",
        system_prompt=_ASTRO_SYSTEM_PROMPT,
        _generator=_astro_generator,
    ),
    "olmo_original": TaskTemplate(
        name="olmo_original",
        system_prompt=_SBR_SYSTEM_PROMPT,
        _generator=_sbr_generator,
    ),
    "dyck": TaskTemplate(
    name="dyck",
    system_prompt=_DYCK_SYSTEM_PROMPT,
    _generator=_dyck_generator,
    ),
    "dag_arithmetic": TaskTemplate(
        name="dag_arithmetic",
        system_prompt=_DAG_SYSTEM_PROMPT,
        _generator=_dag_generator,
    ),
}


def get_task(name: str, csv_path: str | None = None) -> TaskTemplate:
    """
    Return the TaskTemplate for *name*.

    For the `astro` and `olmo3` tasks, *csv_path* overrides the default
    exoplanet CSV path (which can also be set via the EXOPLANETS_CSV env var).
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown task '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    template = _REGISTRY[name]

    # For planet tasks, wrap the generator to inject csv_path
    if name == "astro" and csv_path is not None:
        base_gen = template._generator
        def _gen_with_path(m, n, rng, _gen=base_gen, _path=csv_path):
            return _gen(m, n, rng, csv_path=_path)
        template = TaskTemplate(
            name=template.name,
            system_prompt=template.system_prompt,
            _generator=_gen_with_path,
        )

    return template


def list_tasks() -> list[str]:
    """Return the names of all registered tasks."""
    return sorted(_REGISTRY.keys())