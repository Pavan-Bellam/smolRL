"""
Scoring orchestrator for test-time compute scaling evaluation.

Scores completions using multiple voting strategies at various k values,
computing accuracy curves for each strategy.
"""

import json
from pathlib import Path

from tqdm import tqdm

from grader import strip_string, math_equal
from voting import (
    extract_answers,
    group_by_equivalence,
    naive_majority_vote,
    logprob_weighted_vote,
    shortest_majority_vote,
    shortest_weighted_vote,
)


def get_k_values(n: int) -> list[int]:
    """Get k values based on n. Only includes values <= n.

    Args:
        n: Number of samples per problem

    Returns:
        List of k values to evaluate
    """
    all_k = [1, 4, 8, 12, 16, 24, 32]
    return [k for k in all_k if k <= n]


STRATEGIES = {
    "naive": naive_majority_vote,
    "weighted": logprob_weighted_vote,
    "smv": shortest_majority_vote,
    "smv_weighted": shortest_weighted_vote,
}


def score_row_scaling(row: dict, k_values: list[int]) -> dict:
    """Score one row with all strategies at all k values.

    Args:
        row: Dict with completions list and ground_truth
        k_values: List of k values to evaluate

    Returns:
        Row augmented with voting_results dict
    """
    completions = row["completions"]
    ground_truth = strip_string(row["ground_truth"])

    # Extract answers from all completions (once)
    extracted = extract_answers(completions)

    # Pre-compute groups for efficiency (used by naive and weighted)
    groups, null_group = group_by_equivalence(extracted)

    # Score with each strategy at each k
    voting_results = {}
    for strategy_name, strategy_fn in STRATEGIES.items():
        voting_results[strategy_name] = {}
        for k in k_values:
            # SMV strategies sort internally, others use pre-computed groups
            if strategy_name in ("smv", "smv_weighted"):
                answer, meta = strategy_fn(extracted, k)
            else:
                answer, meta = strategy_fn(extracted, k, groups)

            correct = answer is not None and math_equal(answer, ground_truth, timeout=True)
            voting_results[strategy_name][f"k{k}"] = {
                "answer": answer,
                "correct": correct,
                "metadata": meta,
            }

    return {
        **row,
        "voting_results": voting_results,
    }


def score_all_scaling(rows: list[dict], k_values: list[int]) -> list[dict]:
    """Score all rows for scaling evaluation.

    Args:
        rows: List of row dicts with completions
        k_values: List of k values to evaluate

    Returns:
        List of scored rows
    """
    scored = []
    for row in tqdm(rows, desc="Scoring (scaling)"):
        scored.append(score_row_scaling(row, k_values))
    return scored


def compute_scaling_stats(rows: list[dict], k_values: list[int]) -> dict:
    """Aggregate accuracy at each k for each strategy.

    Args:
        rows: List of scored rows
        k_values: List of k values

    Returns:
        Dict with total, per-strategy accuracy at each k
    """
    total = len(rows)

    stats = {
        "total": total,
        "k_values": k_values,
        "strategies": {},
    }

    for strategy_name in STRATEGIES:
        stats["strategies"][strategy_name] = {}
        for k in k_values:
            correct = sum(
                1 for r in rows
                if r["voting_results"][strategy_name][f"k{k}"]["correct"]
            )
            accuracy = correct / total if total > 0 else 0.0
            stats["strategies"][strategy_name][f"k{k}"] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }

    return stats


def load_scaling_responses(jsonl_path: str) -> list[dict]:
    """Load scaling responses JSONL."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_scaling_results(rows: list[dict], output_path: str) -> None:
    """Save scored scaling results to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_scaling_report(stats: dict, dataset_name: str) -> None:
    """Pretty-print scaling report to console."""
    print(f"\n{'=' * 78}")
    print(f"  Scaling Report: {dataset_name}  (n={max(stats['k_values'])})")
    print(f"{'=' * 78}")
    print(f"  Total problems: {stats['total']}")
    print()

    k_values = stats["k_values"]

    # Header
    header = "  Strategy             " + "".join(f"k={k:<6}" for k in k_values)
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Strategy names for display
    strategy_display = {
        "naive": "Naive Majority",
        "weighted": "Weighted Vote",
        "smv": "Shortest Majority",
        "smv_weighted": "Shortest+Weighted",
    }

    for strategy_name in STRATEGIES:
        row_str = f"  {strategy_display[strategy_name]:<20}"
        for k in k_values:
            acc = stats["strategies"][strategy_name][f"k{k}"]["accuracy"]
            row_str += f" {acc:>5.1%} "
        print(row_str)

    print(f"{'=' * 78}")
