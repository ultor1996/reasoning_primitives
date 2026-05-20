"""
utils.py — shared helpers used across the pipeline.

Covers:
  - JSON I/O (load / save with optional json_repair)
  - GPU / model utilities (vLLM load, free)
  - Answer normalisation
  - JSON extraction from raw model output
  - Model-name helpers
"""

import gc
import json
import os
import re

# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_json(path: str):
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str, indent: int = 2):
    """Serialise *obj* to *path* as pretty-printed JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Answer helpers
# ---------------------------------------------------------------------------

def normalize_answer(raw_answer) -> str | None:
    """
    Normalise a raw model answer to one of {A, B, C, D} or None.

    Accepts:
      - "A", "b", "C" (plain letter, any case)
      - "A) some text …"   (letter + closing paren + optional text)
    """
    if raw_answer is None:
        return None

    s = str(raw_answer).strip().upper()

    if s in {"A", "B", "C", "D"}:
        return s

    m = re.fullmatch(r"([ABCD])\)\s*.*", s, re.DOTALL)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# JSON extraction from raw generation
# ---------------------------------------------------------------------------

# def extract_json(raw_output: str):
#     """
#     Try to parse / repair a JSON object from raw model output.

#     Uses json_repair (must be installed) so clean JSON is also handled.

#     Returns
#     -------
#     (parsed : dict | None, was_repaired : bool, error : str | None)
#     """
#     try:
#         from json_repair import repair_json  # optional dependency
#         result = repair_json(raw_output, return_objects=True)
#         if isinstance(result, dict):
#             return result, False, None
#         if isinstance(result, list):
#             for item in result:
#                 if isinstance(item, dict) and any(
#                     k in item for k in ("answer", "Answer", "ANSWER")
#                 ):
#                     return item, True, None
#             if result and isinstance(result[0], dict):
#                 return result[0], True, None
#         return None, True, f"Unexpected repair_json return type: {type(result)}"
#     except ImportError:
#         pass

#     # Fallback: simple regex scan for {...}
#     for m in re.finditer(r"\{[^{}]*\}", raw_output, re.DOTALL):
#         try:
#             return json.loads(m.group()), False, None
#         except json.JSONDecodeError:
#             pass
#     return None, True, "No valid JSON found"

def extract_json(raw_output: str):
    try:
        from json_repair import repair_json
        result = repair_json(raw_output, return_objects=True)
        if isinstance(result, dict):
            return result, False, None
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and any(
                    k in item for k in ("answer", "Answer", "ANSWER")
                ):
                    return item, True, None
            if result and isinstance(result[0], dict):
                return result[0], True, None
        return None, True, f"Unexpected repair_json return type: {type(result)}"
    except ImportError:
        pass
    except (ValueError, RecursionError):
        # json_repair exceeded recursion depth on deeply nested/long output
        # fall through to regex fallback below
        pass

    # Fallback: simple regex scan for {...}
    for m in re.finditer(r"\{[^{}]*\}", raw_output, re.DOTALL):
        try:
            return json.loads(m.group()), False, None
        except json.JSONDecodeError:
            pass
    return None, True, "No valid JSON found"


def parse_reasoning_trace(raw_output: str):
    """
    Split a <think>…</think> reasoning trace from the final answer.

    Returns
    -------
    (completion : str, reasoning : str)
    Where *completion* is the text AFTER </think> (stripped),
    and *reasoning* is the content inside the tags (or "" if absent).
    """
    m = re.search(r"<think>(.*?)</think>(.*)", raw_output, re.DOTALL)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    return raw_output.strip(), ""


# ---------------------------------------------------------------------------
# Model-name helpers
# ---------------------------------------------------------------------------

def safe_model_name(model_name: str) -> str:
    """Convert a HuggingFace model id into a filesystem-safe string."""
    return model_name.replace("/", "__").replace(".", "_")


def is_large_model(model_name: str, threshold_b: float = 7.0) -> bool:
    """Return True when the model name contains a parameter count ≥ threshold_b B."""
    nums = re.findall(r"(\d+(?:\.\d+)?)B", model_name, re.IGNORECASE)
    return any(float(n) >= threshold_b for n in nums)


def model_uses_qwen_thinking(model_name: str) -> bool:
    return "qwen/" in model_name.lower()


# ---------------------------------------------------------------------------
# vLLM model management
# ---------------------------------------------------------------------------

def load_vllm_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 16_000,
):
    """
    Load a model with vLLM.

    GPU memory utilisation is chosen automatically based on model size.
    """
    import torch
    from vllm import LLM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if "Qwen3.5" in model_name:
        gpu_mem = 0.70
    elif is_large_model(model_name):
        gpu_mem = 0.85
    else:
        gpu_mem = 0.90

    print(f"  Loading {model_name} (gpu_mem={gpu_mem}, max_len={max_model_len}) …")
    # model = LLM(
    #     model=model_name,
    #     dtype=dtype,
    #     tensor_parallel_size=tensor_parallel_size,
    #     gpu_memory_utilization=gpu_mem,
    #     max_model_len=max_model_len,
    #     trust_remote_code=True,
    #     enforce_eager=True,
    #     limit_mm_per_prompt={"image": 0, "video": 0},
    # )
    kwargs = dict(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if "hybrid" in model_name.lower():
        kwargs["mamba_ssm_cache_dtype"] = "float32"

    model = LLM(**kwargs)
    return model


def free_vllm_model(model):
    """Aggressively release GPU memory after a vLLM model is no longer needed."""
    import torch

    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        for _ in range(3):
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
