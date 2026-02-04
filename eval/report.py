"""
Step 4: Log evaluation results to Weights & Biases.

Creates a single W&B run with clean tables for easy comparison:
- Greedy mode: results table with benchmark, accuracy, correct, total, null_rate
- Scaling mode: results table with benchmark, strategy, k, accuracy, correct, total

Also logs level/subject breakdowns for MATH-500 style benchmarks.
"""

import wandb


def build_run_config(config: dict, mode: str, benchmarks: list[str]) -> dict:
    """Build a flat config dict for wandb.config.

    Args:
        config: Full config dict (parsed config.yaml)
        mode: "greedy" or "scaling"
        benchmarks: List of benchmark names being evaluated

    Returns:
        Flat dict for W&B run config
    """
    eval_cfg = config["eval"]

    flat = {
        "model_name": eval_cfg["model_name"],
        "mode": mode,
        "benchmarks": ", ".join(benchmarks),
    }

    # Add vLLM config
    for key, value in eval_cfg.get("vllm", {}).items():
        flat[f"vllm/{key}"] = value

    if mode == "greedy":
        for key, value in eval_cfg.get("sampling", {}).items():
            flat[f"sampling/{key}"] = value
    else:
        for key, value in eval_cfg.get("scaling", {}).items():
            flat[f"scaling/{key}"] = value

    return flat


def log_eval_results(
    all_stats: dict[str, dict],
    config: dict,
    mode: str,
    all_gen_stats: dict[str, dict] | None = None,
) -> None:
    """Log evaluation results to W&B in a single run.

    Creates clean tables for easy comparison across benchmarks.

    Args:
        all_stats: Dict mapping benchmark name -> stats dict
            For greedy: stats from score.compute_stats()
            For scaling: stats from score_scaling.compute_scaling_stats()
        config: Full config dict (parsed config.yaml)
        mode: "greedy" or "scaling"
        all_gen_stats: Optional dict mapping benchmark name -> generation stats
            (from generate.compute_generation_stats). Includes time and throughput.
    """
    if all_gen_stats is None:
        all_gen_stats = {}
    eval_cfg = config["eval"]
    wandb_cfg = eval_cfg.get("wandb", {})

    if not wandb_cfg.get("enabled", True):
        print("W&B logging disabled in config, skipping.")
        return

    if not all_stats:
        print("No stats to log.")
        return

    project = wandb_cfg.get("project", "mathsmall-eval")
    entity = wandb_cfg.get("entity")

    # Build run name: model + mode
    model_name = eval_cfg["model_name"]
    model_short = model_name.split("/")[-1]
    run_name = f"{model_short}_{mode}"

    benchmarks = list(all_stats.keys())
    run_config = build_run_config(config, mode, benchmarks)

    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=run_config,
    )

    if mode == "greedy":
        _log_greedy_results(all_stats, all_gen_stats)
    else:
        _log_scaling_results(all_stats, all_gen_stats)

    wandb.finish()
    print(f"W&B run logged: {run_name}")


def _log_greedy_results(all_stats: dict[str, dict], all_gen_stats: dict[str, dict]) -> None:
    """Log greedy mode results as clean tables."""
    # Main results table
    results_table = wandb.Table(columns=[
        "benchmark", "accuracy", "correct", "total", "null_rate",
        "time_s", "throughput_tok_s", "resp_len_mean", "resp_len_median", "resp_len_p90"
    ])

    for benchmark, stats in all_stats.items():
        gen = all_gen_stats.get(benchmark, {})
        results_table.add_data(
            benchmark,
            stats["accuracy"],
            stats["correct"],
            stats["total"],
            stats["null_rate"],
            gen.get("elapsed_seconds"),
            gen.get("throughput_output"),
            stats["response_length"]["mean"],
            stats["response_length"]["median"],
            stats["response_length"]["p90"],
        )

    wandb.log({"results": results_table})

    # Log summary metrics for easy dashboard access
    for benchmark, stats in all_stats.items():
        metrics = {
            f"{benchmark}/accuracy": stats["accuracy"],
            f"{benchmark}/null_rate": stats["null_rate"],
        }
        # Add generation metrics if available
        gen = all_gen_stats.get(benchmark, {})
        if gen:
            metrics[f"{benchmark}/time_s"] = gen.get("elapsed_seconds")
            metrics[f"{benchmark}/throughput_tok_s"] = gen.get("throughput_output")
        wandb.log(metrics)

    # Level breakdown table (for MATH-500 style benchmarks)
    level_rows = []
    for benchmark, stats in all_stats.items():
        if "per_level" in stats:
            for level, data in stats["per_level"].items():
                level_rows.append((benchmark, level, data["accuracy"], data["correct"], data["total"]))

    if level_rows:
        level_table = wandb.Table(columns=["benchmark", "level", "accuracy", "correct", "total"])
        for row in level_rows:
            level_table.add_data(*row)
        wandb.log({"level_breakdown": level_table})

    # Subject breakdown table
    subject_rows = []
    for benchmark, stats in all_stats.items():
        if "per_subject" in stats:
            for subject, data in stats["per_subject"].items():
                subject_rows.append((benchmark, subject, data["accuracy"], data["correct"], data["total"]))

    if subject_rows:
        subject_table = wandb.Table(columns=["benchmark", "subject", "accuracy", "correct", "total"])
        for row in subject_rows:
            subject_table.add_data(*row)
        wandb.log({"subject_breakdown": subject_table})


def _log_scaling_results(all_stats: dict[str, dict], all_gen_stats: dict[str, dict]) -> None:
    """Log scaling mode results as clean tables."""
    # Log generation stats table first (one row per benchmark)
    if all_gen_stats:
        gen_table = wandb.Table(columns=[
            "benchmark", "time_s", "throughput_tok_s", "input_tokens", "output_tokens", "completions"
        ])
        for benchmark, gen in all_gen_stats.items():
            gen_table.add_data(
                benchmark,
                gen.get("elapsed_seconds"),
                gen.get("throughput_output"),
                gen.get("input_tokens"),
                gen.get("output_tokens"),
                gen.get("num_completions"),
            )
        wandb.log({"generation_stats": gen_table})

    # Main scaling results table: benchmark, strategy, k, accuracy, correct, total
    results_table = wandb.Table(columns=[
        "benchmark", "strategy", "k", "accuracy", "correct", "total"
    ])

    strategy_display = {
        "naive": "Naive Majority",
        "weighted": "Weighted Vote",
        "smv": "Shortest Majority",
        "smv_weighted": "Shortest+Weighted",
    }

    for benchmark, stats in all_stats.items():
        k_values = stats["k_values"]
        strategies = stats["strategies"]

        for strategy_name, strategy_stats in strategies.items():
            display_name = strategy_display.get(strategy_name, strategy_name)
            for k in k_values:
                k_key = f"k{k}"
                k_stats = strategy_stats[k_key]
                results_table.add_data(
                    benchmark,
                    display_name,
                    k,
                    k_stats["accuracy"],
                    k_stats["correct"],
                    k_stats["total"],
                )

    wandb.log({"scaling_results": results_table})

    # Summary table: best accuracy per benchmark per strategy (at max k)
    summary_table = wandb.Table(columns=["benchmark", "strategy", "best_k", "accuracy"])

    for benchmark, stats in all_stats.items():
        k_values = stats["k_values"]
        max_k = max(k_values)
        strategies = stats["strategies"]

        for strategy_name, strategy_stats in strategies.items():
            display_name = strategy_display.get(strategy_name, strategy_name)
            best_acc = strategy_stats[f"k{max_k}"]["accuracy"]
            summary_table.add_data(benchmark, display_name, max_k, best_acc)

    wandb.log({"scaling_summary": summary_table})

    # Log flat metrics for dashboard charts
    for benchmark, stats in all_stats.items():
        k_values = stats["k_values"]
        max_k = max(k_values)
        strategies = stats["strategies"]

        # Log best accuracy (at max k) for each strategy
        for strategy_name, strategy_stats in strategies.items():
            wandb.log({
                f"{benchmark}/{strategy_name}/accuracy": strategy_stats[f"k{max_k}"]["accuracy"],
            })


# Keep old functions for backwards compatibility but mark as deprecated
def log_to_wandb(stats: dict, config: dict, dataset_name: str, scored_path: str) -> None:
    """DEPRECATED: Use log_eval_results() instead.

    This function is kept for backwards compatibility but creates
    a separate run per benchmark which is harder to compare.
    """
    print("WARNING: log_to_wandb() is deprecated. Use log_eval_results() instead.")
    log_eval_results({dataset_name: stats}, config, mode="greedy", all_gen_stats=None)


def log_scaling_to_wandb(stats: dict, config: dict, dataset_name: str, scored_path: str) -> None:
    """DEPRECATED: Use log_eval_results() instead.

    This function is kept for backwards compatibility but creates
    a separate run per benchmark which is harder to compare.
    """
    print("WARNING: log_scaling_to_wandb() is deprecated. Use log_eval_results() instead.")
    log_eval_results({dataset_name: stats}, config, mode="scaling", all_gen_stats=None)
