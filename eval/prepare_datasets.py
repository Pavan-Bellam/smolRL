"""
Step 1: Prepare benchmark datasets.

Downloads MATH-500, GSM8K, and AIME-2024 from HuggingFace,
formats prompts in Qwen chat template, and saves as JSONL.

Each output row:
  {
    "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{question}\n...<|im_end|>\n<|im_start|>assistant\n",
    "ground_truth": "<final answer string>",
    "dataset": "math500",
    "metadata": { ... }
  }
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


QWEN_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

BENCHMARKS = {
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "hf_config": None,
        "split": "test",
        "question_col": "problem",
        "answer_col": "answer",
    },
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "question_col": "question",
        "answer_col": "answer",
    },
    "aime24": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "hf_config": None,
        "split": "train",
        "question_col": "problem",
        "answer_col": "answer",
    },
}


def parse_gsm8k_gt(answer_text: str) -> str:
    """Extract the final number after #### in GSM8K answers."""
    return answer_text.split("####")[-1].strip().replace(",", "")


def prepare_benchmark(name: str, cfg: dict, output_dir: Path) -> int:
    print(f"\n{'='*50}")
    print(f"Preparing {name}")
    print(f"{'='*50}")

    if cfg["hf_config"]:
        ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_name"], split=cfg["split"])

    print(f"  Loaded {len(ds)} examples from {cfg['hf_name']}")

    output_path = output_dir / f"{name}.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            question = example[cfg["question_col"]]
            raw_answer = example[cfg["answer_col"]]

            # Parse ground truth
            if name == "gsm8k":
                ground_truth = parse_gsm8k_gt(raw_answer)
            else:
                ground_truth = str(raw_answer).strip()

            # Format prompt
            prompt = QWEN_TEMPLATE.format(question=question)

            # Build metadata
            metadata = {"index": i}
            if "level" in example:
                metadata["level"] = example["level"]
            if "subject" in example:
                metadata["subject"] = example["subject"]
            if "solution" in example:
                metadata["solution"] = example["solution"]

            row = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "dataset": name,
                "metadata": metadata,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} examples to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Which benchmarks to prepare (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/data",
        help="Output directory for JSONL files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for name in args.benchmarks:
        cfg = BENCHMARKS[name]
        total += prepare_benchmark(name, cfg, output_dir)

    print(f"\nDone. {total} total examples across {len(args.benchmarks)} benchmarks.")


if __name__ == "__main__":
    main()
