"""Reward functions and metrics for GRPO training."""

import re
import sys
import os

# Add project root to path so eval.grader can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.grader import math_equal
from .answer_extraction import extract_answer, strip_string


def compute_metrics(completions, answer):
    """Compute detailed metrics for logging (not used as rewards).

    Returns:
        dict with:
        - accuracy: fraction of correct answers
        - format_rate: fraction with \\boxed{}
        - accuracy_given_format: accuracy among those with \\boxed{}
        - accuracy_without_format: accuracy among those without \\boxed{}
    """
    boxed_pattern = re.compile(r"\\boxed\s*\{")

    correct = 0
    has_format = 0
    correct_with_format = 0
    correct_without_format = 0
    total_with_format = 0
    total_without_format = 0

    for completion, gt in zip(completions, answer):
        # Handle completion format
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = str(completion)

        # Check format
        has_boxed = bool(boxed_pattern.search(text))
        if has_boxed:
            has_format += 1
            total_with_format += 1
        else:
            total_without_format += 1

        # Check correctness
        pred = extract_answer(text)
        gt_clean = strip_string(str(gt))
        is_correct = pred != "" and math_equal(pred, gt_clean)

        if is_correct:
            correct += 1
            if has_boxed:
                correct_with_format += 1
            else:
                correct_without_format += 1

    n = len(completions)
    return {
        "metrics/accuracy": correct / n if n > 0 else 0.0,
        "metrics/format_rate": has_format / n if n > 0 else 0.0,
        "metrics/accuracy_given_format": (
            correct_with_format / total_with_format if total_with_format > 0 else 0.0
        ),
        "metrics/accuracy_without_format": (
            correct_without_format / total_without_format if total_without_format > 0 else 0.0
        ),
    }


def accuracy_reward(completions, answer, **kwargs):
    """Score completions by comparing extracted answer to ground truth.

    Uses multi-tier answer extraction with fallback chain:
    1. minerva style ("final answer is $...$")
    2. \\boxed{...}
    3. "the answer is" / "he answer is"
    4. "final answer is"
    5. Last number in text (fallback)

    This ensures models get credit for correct answers regardless of format.
    Returns 1.0 if the answer matches, 0.0 otherwise.

    Args:
        completions: List of generated texts (strings from TRL)
        answer: List of ground truth answers from dataset
        **kwargs: Additional columns from dataset (unused)

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    # Validate inputs
    if answer is None:
        raise ValueError("accuracy_reward requires 'answer' column in dataset")

    ground_truths = answer
    if len(completions) != len(ground_truths):
        raise ValueError(
            f"Mismatch: {len(completions)} completions vs {len(ground_truths)} answers"
        )

    rewards = []
    for completion, gt in zip(completions, ground_truths):
        # TRL passes completions as strings; handle legacy chat format just in case
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = str(completion)

        pred = extract_answer(text)  # Multi-tier extraction with fallback
        if pred == "":
            rewards.append(0.0)
            continue

        gt_clean = strip_string(str(gt))
        rewards.append(1.0 if math_equal(pred, gt_clean) else 0.0)

    return rewards


def format_reward(completions, **kwargs):
    """Score completions based on whether they contain a \\boxed{...} answer.

    Returns 1.0 if \\boxed{} is present, 0.0 otherwise.

    NOTE: Not used in training (per SimpleRL-Zoo paper recommendation).
    Format rewards hinder exploration for base models.
    """
    pattern = re.compile(r"\\boxed\s*\{")
    rewards = []
    for completion in completions:
        # TRL passes completions as strings; handle legacy chat format just in case
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = str(completion)
        rewards.append(1.0 if pattern.search(text) else 0.0)
    return rewards
