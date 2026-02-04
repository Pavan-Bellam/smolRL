"""
Eval pipeline entry point — connects steps 1-4.

Pipeline order:
  1. Prepare all benchmarks
  2. Generate all (single vLLM initialization)
  3. Score all
  4. Report all to W&B

Supports two modes:
  - greedy: Single deterministic completion per problem (default)
  - scaling: n stochastic completions for test-time compute scaling

Usage:
    python eval/run.py
    python eval/run.py --mode scaling --benchmarks math500
    python eval/run.py --benchmarks math500 gsm8k
    python eval/run.py --skip-prepare --skip-generate
    python eval/run.py --no-wandb
"""

import argparse
import sys
import time
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["greedy", "scaling"],
        default=None,
        help="Eval mode: greedy (single completion) or scaling (n completions). Overrides config.",
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

    # Determine eval mode (CLI overrides config)
    mode = args.mode if args.mode else eval_cfg.get("mode", "greedy")
    print(f"\nEval mode: {mode}")

    # Add eval/ to sys.path so we can import modules
    eval_dir = Path(__file__).resolve().parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    # Build paths for all benchmarks (different output paths for scaling mode)
    benchmark_paths = {}
    for benchmark in args.benchmarks:
        if mode == "scaling":
            benchmark_paths[benchmark] = {
                "data": data_dir / f"{benchmark}.jsonl",
                "response": output_dir / f"{benchmark}_scaling_responses.jsonl",
                "scored": output_dir / f"{benchmark}_scaling_scored.jsonl",
            }
        else:
            benchmark_paths[benchmark] = {
                "data": data_dir / f"{benchmark}.jsonl",
                "response": output_dir / f"{benchmark}_responses.jsonl",
                "scored": output_dir / f"{benchmark}_scored.jsonl",
            }

    # =========================================================================
    # Step 1: Prepare all datasets
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 1: Prepare datasets")
    print(f"{'=' * 60}")

    if not args.skip_prepare:
        from prepare_datasets import BENCHMARKS, prepare_benchmark

        for benchmark in args.benchmarks:
            if benchmark not in BENCHMARKS:
                print(f"Unknown benchmark: {benchmark}, skipping.")
                continue
            prepare_benchmark(benchmark, BENCHMARKS[benchmark], data_dir)
    else:
        print("[skip-prepare] Reusing existing data files")
        for benchmark in args.benchmarks:
            data_path = benchmark_paths[benchmark]["data"]
            if not data_path.exists():
                print(f"  ERROR: {data_path} not found. Run without --skip-prepare first.")
                args.benchmarks.remove(benchmark)

    # =========================================================================
    # Step 2: Generate responses (single vLLM initialization)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 2: Generate responses")
    print(f"{'=' * 60}")

    # Collect generation stats for W&B logging (empty if --skip-generate)
    all_gen_stats = {}

    if not args.skip_generate:
        from generate import (
            build_llm,
            build_sampling_params,
            compute_generation_stats,
            get_stop_token_ids,
            load_prompts,
            print_generation_report,
            save_results,
        )

        # Initialize vLLM once
        print(f"\nInitializing vLLM with model: {eval_cfg['model_name']}")
        llm = build_llm(eval_cfg)
        stop_token_ids = get_stop_token_ids(eval_cfg["model_name"])
        sampling_params = build_sampling_params(eval_cfg, stop_token_ids, mode=mode)

        if mode == "scaling":
            n = eval_cfg["scaling"]["n"]
            print(f"Scaling mode: generating {n} completions per prompt")

        # Generate for each benchmark
        for benchmark in args.benchmarks:
            paths = benchmark_paths[benchmark]
            print(f"\n--- Generating for {benchmark} ---")

            rows = load_prompts(str(paths["data"]))
            print(f"Loaded {len(rows)} prompts")

            prompts = [row["prompt"] for row in rows]

            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - start

            gen_stats = compute_generation_stats(outputs, elapsed)
            print_generation_report(gen_stats)
            all_gen_stats[benchmark] = gen_stats

            if mode == "scaling":
                # Extract all completions with logprobs
                for row, output in zip(rows, outputs):
                    row["completions"] = [
                        {
                            "text": c.text,
                            "cumulative_logprob": c.cumulative_logprob,
                            "finish_reason": c.finish_reason,
                            "num_tokens": len(c.token_ids),
                        }
                        for c in output.outputs
                    ]
            else:
                # Greedy: single response
                for row, output in zip(rows, outputs):
                    row["response"] = output.outputs[0].text

            save_results(rows, str(paths["response"]))

        # Clean up vLLM to free GPU memory
        del llm
        print("\nvLLM instance released.")
    else:
        print("[skip-generate] Reusing existing response files")
        for benchmark in args.benchmarks:
            response_path = benchmark_paths[benchmark]["response"]
            if not response_path.exists():
                print(f"  ERROR: {response_path} not found. Run without --skip-generate first.")
                args.benchmarks.remove(benchmark)

    # =========================================================================
    # Step 3: Score all responses
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 3: Score responses")
    print(f"{'=' * 60}")

    all_stats = {}

    if mode == "scaling":
        from score_scaling import (
            get_k_values,
            load_scaling_responses,
            score_all_scaling,
            compute_scaling_stats,
            save_scaling_results,
            print_scaling_report,
        )

        n = eval_cfg["scaling"]["n"]
        k_values = get_k_values(n)
        print(f"k values: {k_values}")

        for benchmark in args.benchmarks:
            paths = benchmark_paths[benchmark]
            print(f"\n--- Scoring {benchmark} (scaling) ---")

            rows = load_scaling_responses(str(paths["response"]))
            rows = score_all_scaling(rows, k_values)
            stats = compute_scaling_stats(rows, k_values)
            save_scaling_results(rows, str(paths["scored"]))
            print_scaling_report(stats, benchmark)

            all_stats[benchmark] = stats
    else:
        from score import load_responses, score_all, compute_stats, save_results as save_scored, print_report

        for benchmark in args.benchmarks:
            paths = benchmark_paths[benchmark]
            print(f"\n--- Scoring {benchmark} ---")

            rows = load_responses(str(paths["response"]))
            rows = score_all(rows)
            stats = compute_stats(rows)
            save_scored(rows, str(paths["scored"]))
            print_report(stats, benchmark)

            all_stats[benchmark] = stats

    # =========================================================================
    # Step 4: Report all to W&B
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 4: Report to W&B")
    print(f"{'=' * 60}")

    if not args.no_wandb:
        wandb_cfg = eval_cfg.get("wandb", {})
        if wandb_cfg.get("enabled", True):
            from report import log_eval_results

            print(f"\nLogging {len(all_stats)} benchmark(s) to W&B...")
            log_eval_results(all_stats, config, mode, all_gen_stats)
        else:
            print("W&B logging disabled in config.")
    else:
        print("W&B logging skipped (--no-wandb).")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
