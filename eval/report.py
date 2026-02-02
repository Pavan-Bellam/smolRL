"""
Step 4: Log evaluation results to Weights & Biases.

Logs summary metrics (accuracy, null rate, response length stats),
per-level/subject breakdowns, and eval config to a W&B run.
"""

import wandb


def flatten_config(eval_cfg: dict, dataset_name: str) -> dict:
    """Flatten the eval config section into a flat dict for wandb.config."""
    flat = {
        "model_name": eval_cfg["model_name"],
        "dataset": dataset_name,
    }

    for key, value in eval_cfg.get("sampling", {}).items():
        flat[f"sampling/{key}"] = value

    for key, value in eval_cfg.get("vllm", {}).items():
        flat[f"vllm/{key}"] = value

    return flat


def log_to_wandb(
    stats: dict,
    config: dict,
    dataset_name: str,
    scored_path: str,
) -> None:
    """Log evaluation results to W&B.

    Args:
        stats: Output of score.compute_stats() — accuracy, null rate, breakdowns, etc.
        config: Full config dict (parsed config.yaml).
        dataset_name: Benchmark name (e.g. "math500").
        scored_path: Path to the scored JSONL file (for reference in the run).
    """
    eval_cfg = config["eval"]
    wandb_cfg = eval_cfg.get("wandb", {})

    if not wandb_cfg.get("enabled", True):
        print("W&B logging disabled in config, skipping.")
        return

    project = wandb_cfg.get("project", "mathsmall-eval")
    entity = wandb_cfg.get("entity")  # None = default entity

    # Build run name: short model name + dataset
    model_name = eval_cfg["model_name"]
    model_short = model_name.split("/")[-1]
    run_name = f"{model_short}_{dataset_name}"

    flat_config = flatten_config(eval_cfg, dataset_name)
    flat_config["scored_path"] = scored_path

    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=flat_config,
    )

    # Flat summary metrics
    metrics = {
        "accuracy": stats["accuracy"],
        "correct": stats["correct"],
        "total": stats["total"],
        "null_predictions": stats["null_predictions"],
        "null_rate": stats["null_rate"],
        "response_length/mean": stats["response_length"]["mean"],
        "response_length/median": stats["response_length"]["median"],
        "response_length/p90": stats["response_length"]["p90"],
    }

    # Per-level metrics
    if "per_level" in stats:
        for level, data in stats["per_level"].items():
            metrics[f"level/{level}/accuracy"] = data["accuracy"]

    # Per-subject metrics
    if "per_subject" in stats:
        for subject, data in stats["per_subject"].items():
            metrics[f"subject/{subject}/accuracy"] = data["accuracy"]

    wandb.log(metrics)

    # Log per-level breakdown as a W&B Table
    if "per_level" in stats:
        level_table = wandb.Table(columns=["level", "correct", "total", "accuracy"])
        for level, data in stats["per_level"].items():
            level_table.add_data(level, data["correct"], data["total"], data["accuracy"])
        wandb.log({"level_breakdown": level_table})

    # Log per-subject breakdown as a W&B Table
    if "per_subject" in stats:
        subject_table = wandb.Table(columns=["subject", "correct", "total", "accuracy"])
        for subject, data in stats["per_subject"].items():
            subject_table.add_data(subject, data["correct"], data["total"], data["accuracy"])
        wandb.log({"subject_breakdown": subject_table})

    wandb.finish()
    print(f"W&B run logged: {run_name}")
