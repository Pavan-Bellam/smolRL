"""GRPO training entry point for MathSmall."""

import argparse
import logging
import os
from datetime import datetime

import yaml
import datasets
from accelerate import PartialState
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)

from .rewards import accuracy_reward, compute_metrics

# Global storage for metrics (updated by reward function, read by trainer)
_step_metrics = {}


def accuracy_reward_with_metrics(completions, answer, **kwargs):
    """Wrapper that computes reward AND logs detailed metrics to W&B."""
    global _step_metrics

    # Compute actual rewards
    rewards = accuracy_reward(completions, answer, **kwargs)

    # Compute detailed metrics for logging
    metrics = compute_metrics(completions, answer)
    _step_metrics.update(metrics)

    # Log to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, commit=False)
    except ImportError:
        pass

    return rewards


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_run_name(model_name: str) -> str:
    """Generate automatic run name from model name and timestamp."""
    # Extract short model name (e.g., "Qwen/Qwen2.5-7B" -> "qwen2.5-7b")
    short_name = model_name.split("/")[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{short_name}_{timestamp}"


def build_dataset(parquet_path: str) -> datasets.Dataset:
    """Load parquet dataset with pre-formatted prompts."""
    ds = datasets.Dataset.from_parquet(parquet_path)

    # Validate required columns
    required_cols = ["prompt", "answer"]
    missing = [c for c in required_cols if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {ds.column_names}")

    return ds


def log_sample_example(dataset: datasets.Dataset, logger):
    """Log a single example to verify data format."""
    example = dataset[0]
    prompt = example["prompt"]
    answer = example["answer"]

    # Extract text from prompt (handles chat format)
    if isinstance(prompt, list):
        prompt_text = prompt[0]["content"] if prompt else "<empty>"
    else:
        prompt_text = str(prompt)

    logger.info("=" * 60)
    logger.info("SAMPLE EXAMPLE (index 0):")
    logger.info("=" * 60)
    logger.info("PROMPT FORMAT: %s", type(prompt).__name__)
    logger.info("PROMPT TEXT:\n%s", prompt_text[:2000] + "..." if len(prompt_text) > 2000 else prompt_text)
    logger.info("-" * 60)
    logger.info("ANSWER: %s", answer)
    logger.info("=" * 60)


def log_main(msg, *args):
    """Log only on the main process."""
    state = PartialState()
    if state.is_main_process:
        logger.info(msg, *args)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)["train"]
    log_main("Loaded config from %s", args.config)
    log_main("Model: %s", cfg["model_name"])

    # W&B setup
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = wandb_cfg["project"]
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]
    # Auto-generate run name if not specified
    run_name = wandb_cfg.get("run_name") or generate_run_name(cfg["model_name"])
    os.environ["WANDB_NAME"] = run_name
    log_main("W&B run name: %s", run_name)

    # Build GRPOConfig from merged config sections
    grpo_args = {}
    grpo_args["output_dir"] = cfg["output_dir"]
    grpo_args.update(cfg.get("grpo", {}))
    grpo_args.update(cfg.get("vllm", {}))
    grpo_args.update(cfg.get("training", {}))
    grpo_args.update(cfg.get("logging", {}))

    training_args = GRPOConfig(**grpo_args)
    log_main("Built GRPOConfig: output_dir=%s, num_generations=%s, lr=%s",
             training_args.output_dir,
             training_args.num_generations,
             training_args.learning_rate)

    # Load dataset
    log_main("Loading dataset from %s", cfg["dataset"])
    dataset = build_dataset(cfg["dataset"])
    log_main("Dataset loaded: %d examples", len(dataset))
    log_main("Dataset columns: %s", dataset.column_names)

    # Log sample example on main process
    state = PartialState()
    if state.is_main_process:
        log_sample_example(dataset, logger)

    # Create trainer
    log_main("Initializing GRPOTrainer with reward function: accuracy_reward (with metrics logging)")
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        reward_funcs=[accuracy_reward_with_metrics],
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    if args.resume:
        log_main("Resuming training from %s", args.resume)
    else:
        log_main("Starting training from scratch")
    trainer.train(resume_from_checkpoint=args.resume)
    log_main("Training complete, saving model to %s", cfg["output_dir"])
    trainer.save_model(cfg["output_dir"])
    log_main("Model saved")


if __name__ == "__main__":
    main()
