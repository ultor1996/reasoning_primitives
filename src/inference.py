"""
inference.py — run a model over a prompt dataset produced by generator.py.

Usage
-----
python inference.py \\
    --input  data/collisions_diff4_8_16_n100.json \\
    --model  allenai/Olmo-3-7B-Think \\
    --output results/collisions_olmo3think.json \\
    [--batch-size 8] \\
    [--max-model-len 16000] \\
    [--tensor-parallel 1]

Output format
-------------
The output file is the input list with two extra keys added to every sample:
  raw_output : str   the model's raw text generation
  completion : str   raw_output with any <think>…</think> block stripped
  reasoning  : str   content of <think>…</think> (or "" if absent)

Offline mode
------------
Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in the environment if the
cluster has no internet access (already exported by inference.sh).
"""

import argparse
import json
import os
import sys
import time

from utils import (
    extract_json,
    free_vllm_model,
    load_json,
    load_vllm_model,
    model_uses_qwen_thinking,
    parse_reasoning_trace,
    save_json,
)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, sample: dict, model_name: str) -> str:
    """
    Apply the chat template to produce a tokenised prompt string.

    The system prompt and user content are taken from *sample*.
    For Qwen thinking models we enable the thinking mode flag.
    """
    system_prompt = sample.get("system_prompt", "")
    user_content = sample["prompt"]

    # Append MC options block if options are stored separately
    if sample.get("option_A"):
        option_lines = []
        for lbl in ["A", "B", "C", "D"]:
            val = sample.get(f"option_{lbl}")
            if val is not None:
                option_lines.append(f"{lbl}) {val}")
        if option_lines:
            user_content = (
                user_content
                + "\n\nOptions:\n"
                + "\n".join(option_lines)
                + '\n\nReturn only JSON: {"answer": "A"}'
            )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if model_uses_qwen_thinking(model_name):
        kwargs["enable_thinking"] = True

    return tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    samples: list[dict],
    model_name: str,
    batch_size: int = 8,
    max_model_len: int = 16_000,
    max_tokens: int = 512,
    tensor_parallel: int = 1,
) -> list[dict]:
    """
    Run vLLM inference over all samples and return an augmented list.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_vllm_model(
        model_name,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_model_len,
    )

    # Read temperature from model's generation_config.json (vLLM default behaviour)
    default_sampling_params = model.llm_engine.model_config.get_diff_sampling_param()
    temperature = default_sampling_params.get("temperature", 1.0)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else [],
    )
    print(f"Using sampling parameters: {sampling_params}")

    # Build all prompts up front
    prompts = [build_prompt(tokenizer, s, model_name) for s in samples]

    augmented = list(samples)  # shallow copy so we don't mutate the input list
    for i in range(len(augmented)):
        augmented[i] = dict(augmented[i])  # make each sample mutable

    print(f"Running inference on {len(samples)} samples (batch_size={batch_size}) …")
    t0 = time.time()

    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch_prompts = prompts[start:end]

        print(f"  Batch {start}:{end} / {len(samples)}", flush=True)
        outputs = model.generate(batch_prompts, sampling_params)

        for i, output in enumerate(outputs):
            global_idx = start + i
            raw = output.outputs[0].text.strip()
            completion, reasoning = parse_reasoning_trace(raw)
            augmented[global_idx]["raw_output"] = raw
            augmented[global_idx]["completion"] = completion
            augmented[global_idx]["reasoning"] = reasoning
            augmented[global_idx]["model_name"] = model_name

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    free_vllm_model(model)
    return augmented


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run model inference over a generator.py JSON dataset."
    )
    parser.add_argument("--input",  required=True, help="Path to input JSON (from generator.py).")
    parser.add_argument("--model",  required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--output", default=None,  help="Output JSON path.")
    parser.add_argument("--batch-size",      type=int, default=8)
    parser.add_argument("--max-model-len",   type=int, default=16_000)
    parser.add_argument("--max-tokens",      type=int, default=512,
                        help="Max tokens the model generates per sample (default: 512).")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    args = parser.parse_args()

    if args.output is None:
        from utils import safe_model_name
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{stem}__{safe_model_name(args.model)}.json"

    print(f"Input  : {args.input}")
    print(f"Model  : {args.model}")
    print(f"Output : {args.output}")
    print()

    samples = load_json(args.input)
    print(f"Loaded {len(samples)} samples.")

    results = run_inference(
        samples=samples,
        model_name=args.model,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel=args.tensor_parallel,
    )

    save_json(results, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()