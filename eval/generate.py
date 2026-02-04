"""
Step 2: Run vLLM offline inference on prepared benchmark JSONL.

Reads prepared JSONL from step 1, runs vLLM offline inference,
and writes output JSONL with model responses.

Each output row is the same as input with `response` added:
  {
    "prompt": "...",
    "ground_truth": "...",
    "dataset": "math500",
    "metadata": { ... },
    "response": "Let me solve this step by step...\n$\\boxed{42}$"
  }
"""

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_prompts(jsonl_path: str) -> list[dict]:
    """Read the JSONL file from step 1 and return list of row dicts."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_llm(cfg: dict) -> LLM:
    """Build a vllm.LLM instance from the config dict.

    All parameters in cfg["vllm"] are passed directly to vLLM.
    """
    return LLM(model=cfg["model_name"], **cfg["vllm"])


def build_sampling_params(cfg: dict, stop_token_ids: list[int], mode: str = "greedy") -> SamplingParams:
    """Build vllm.SamplingParams from the config dict.

    Args:
        cfg: Eval config dict
        stop_token_ids: List of stop token IDs
        mode: "greedy" for single deterministic completion, "scaling" for n stochastic completions

    Returns:
        SamplingParams configured for the specified mode

    For greedy mode, cfg["sampling"] is used with temperature forced to 0.
    For scaling mode, cfg["scaling"] is passed directly.
    """
    if mode == "greedy":
        # Use sampling config but force greedy decoding
        params = {**cfg["sampling"], "temperature": 0.0, "n": 1, "stop_token_ids": stop_token_ids}
        return SamplingParams(**params)
    else:  # scaling
        params = {**cfg["scaling"], "stop_token_ids": stop_token_ids}
        return SamplingParams(**params)


def generate(llm: LLM, sampling_params: SamplingParams, rows: list[dict]) -> list[dict]:
    """Run inference and attach response field to each row."""
    prompts = [row["prompt"] for row in rows]
    outputs = llm.generate(prompts, sampling_params)

    for row, output in zip(rows, outputs):
        row["response"] = output.outputs[0].text

    return rows


def save_results(rows: list[dict], output_path: str) -> None:
    """Write JSONL with original fields + response."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} results to {output_path}")


def get_stop_token_ids(model_name: str) -> list[int]:
    """Get stop token IDs from the tokenizer (eos + <|im_end|>)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        if im_end_id not in stop_ids:
            stop_ids.append(im_end_id)

    print(f"Stop token IDs: {stop_ids}")
    return stop_ids


def main(input_path: str, config_path: str, output_dir: str | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config["eval"]

    if output_dir is None:
        output_dir = cfg["output_dir"]

    # Derive output filename from input
    input_stem = Path(input_path).stem
    output_path = Path(output_dir) / f"{input_stem}_responses.jsonl"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {cfg['model_name']}")

    rows = load_prompts(input_path)
    print(f"Loaded {len(rows)} prompts")

    stop_token_ids = get_stop_token_ids(cfg["model_name"])
    llm = build_llm(cfg)
    sampling_params = build_sampling_params(cfg, stop_token_ids)

    print(f"Running inference on {len(rows)} prompts...")
    rows = generate(llm, sampling_params, rows)

    save_results(rows, output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference on prepared JSONL")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to prepared JSONL (e.g. eval/data/math500.jsonl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: from config)",
    )
    args = parser.parse_args()
    main(args.input, args.config, args.output_dir)
