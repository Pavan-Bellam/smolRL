"""
Eval pipeline entry point — connects steps 1-4.

Usage:
    python eval/run.py
    python eval/run.py --benchmarks math500 gsm8k
    python eval/run.py --skip-prepare --skip-generate
    python eval/run.py --no-wandb
"""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run the full eval pipeline (steps 1-4)")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["math500", "gsm8k", "aime24"],
        help="Benchmark names to evaluate (default: math500 gsm8k aime24)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: from config)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip step 1 (reuse existing data JSONL)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip step 2 (reuse existing response JSONL)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging regardless of config",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    data_dir = Path(eval_cfg.get("data_dir", "eval/data"))
    output_dir = Path(args.output_dir) if args.output_dir else Path(eval_cfg.get("output_dir", "eval/outputs"))

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add eval/ to sys.path so score.py can import grader
    eval_dir = Path(__file__).resolve().parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    for benchmark in args.benchmarks:
        print(f"\n{'#' * 60}")
        print(f"  Benchmark: {benchmark}")
        print(f"{'#' * 60}")

        data_path = data_dir / f"{benchmark}.jsonl"
        response_path = output_dir / f"{benchmark}_responses.jsonl"
        scored_path = output_dir / f"{benchmark}_scored.jsonl"

        # Step 1: Prepare datasets
        if not args.skip_prepare:
            from prepare_datasets import BENCHMARKS, prepare_benchmark

            if benchmark not in BENCHMARKS:
                print(f"Unknown benchmark: {benchmark}, skipping.")
                continue

            prepare_benchmark(benchmark, BENCHMARKS[benchmark], data_dir)
        else:
            print(f"\n[skip-prepare] Reusing {data_path}")
            if not data_path.exists():
                print(f"  ERROR: {data_path} not found. Run without --skip-prepare first.")
                continue

        # Step 2: Generate responses
        if not args.skip_generate:
            from generate import main as generate_main

            generate_main(str(data_path), args.config, str(output_dir))
        else:
            print(f"\n[skip-generate] Reusing {response_path}")
            if not response_path.exists():
                print(f"  ERROR: {response_path} not found. Run without --skip-generate first.")
                continue

        # Step 3: Score responses
        from score import load_responses, score_all, compute_stats, save_results, print_report

        print(f"\nScoring {response_path}...")
        rows = load_responses(str(response_path))
        rows = score_all(rows)
        stats = compute_stats(rows)
        save_results(rows, str(scored_path))
        print_report(stats, benchmark)

        # Step 4: W&B logging
        if not args.no_wandb:
            wandb_cfg = eval_cfg.get("wandb", {})
            if wandb_cfg.get("enabled", True):
                from report import log_to_wandb

                log_to_wandb(stats, config, benchmark, str(scored_path))
            else:
                print("W&B logging disabled in config.")
        else:
            print("W&B logging skipped (--no-wandb).")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
