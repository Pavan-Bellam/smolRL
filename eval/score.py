"""
Step 3: Score model responses against ground truth.

Provides functions to extract \\boxed{} answers from model responses,
compare against ground truth using math_equal, compute summary stats,
and save scored JSONL.

Scored row format (input fields + scoring fields):
  {
    "prompt": "...",
    "ground_truth": "\\frac{1}{2}",
    "dataset": "math500",
    "metadata": { ... },
    "response": "Let me solve this...\\n$\\boxed{\\frac{1}{2}}$",
    "pred_answer": "\\frac{1}{2}",
    "correct": true
  }
"""

import json
import statistics
from pathlib import Path

from tqdm import tqdm

from grader import extract_raw_boxed, strip_string, math_equal


def load_responses(jsonl_path: str) -> list[dict]:
    """Read JSONL from step 2 (has prompt, ground_truth, dataset, metadata, response)."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(row: dict) -> dict:
    """Score a single row by extracting the predicted answer and comparing to GT.

    Returns the row augmented with pred_answer, correct, and null_pred.
    """
    # Extract predicted answer from model response
    raw_pred = extract_raw_boxed(row["response"])
    null_pred = raw_pred is None
    pred_answer = strip_string(raw_pred) if raw_pred is not None else None

    # Normalize ground truth
    gt = strip_string(row["ground_truth"])

    # Compare
    correct = False
    if pred_answer is not None:
        correct = math_equal(pred_answer, gt, timeout=True)

    return {
        **row,
        "pred_answer": pred_answer,
        "correct": correct,
        "null_pred": null_pred,
    }


def score_all(rows: list[dict]) -> list[dict]:
    """Score all rows with a progress bar."""
    scored = []
    for row in tqdm(rows, desc="Scoring"):
        scored.append(score_row(row))
    return scored


def compute_stats(rows: list[dict]) -> dict:
    """Compute summary statistics from scored rows."""
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    null_preds = sum(1 for r in rows if r["null_pred"])

    # Response length stats
    response_lengths = [len(r["response"]) for r in rows]
    sorted_lengths = sorted(response_lengths)

    stats = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "null_predictions": null_preds,
        "null_rate": null_preds / total if total > 0 else 0.0,
        "response_length": {
            "mean": statistics.mean(response_lengths) if response_lengths else 0,
            "median": statistics.median(response_lengths) if response_lengths else 0,
            "p90": sorted_lengths[int(len(sorted_lengths) * 0.9)] if sorted_lengths else 0,
        },
    }

    # Per-level breakdown (MATH-500)
    levels = {}
    for r in rows:
        level = r.get("metadata", {}).get("level")
        if level is not None:
            levels.setdefault(level, {"total": 0, "correct": 0})
            levels[level]["total"] += 1
            if r["correct"]:
                levels[level]["correct"] += 1
    if levels:
        stats["per_level"] = {
            k: {**v, "accuracy": v["correct"] / v["total"]}
            for k, v in sorted(levels.items())
        }

    # Per-subject breakdown (MATH-500)
    subjects = {}
    for r in rows:
        subject = r.get("metadata", {}).get("subject")
        if subject is not None:
            subjects.setdefault(subject, {"total": 0, "correct": 0})
            subjects[subject]["total"] += 1
            if r["correct"]:
                subjects[subject]["correct"] += 1
    if subjects:
        stats["per_subject"] = {
            k: {**v, "accuracy": v["correct"] / v["total"]}
            for k, v in sorted(subjects.items())
        }

    return stats


def save_results(rows: list[dict], output_path: str):
    """Write scored JSONL (all original fields + pred_answer, correct)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            # Drop null_pred from output (internal bookkeeping only)
            out = {k: v for k, v in row.items() if k != "null_pred"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def print_report(stats: dict, dataset_name: str):
    """Pretty-print scoring summary."""
    print(f"\n{'=' * 60}")
    print(f"  Scoring Report: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:         {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    print(f"  Null predictions: {stats['null_predictions']}/{stats['total']} ({stats['null_rate']:.1%})")
    print(f"  Response length:  mean={stats['response_length']['mean']:.0f}  "
          f"median={stats['response_length']['median']:.0f}  "
          f"p90={stats['response_length']['p90']}")

    if "per_level" in stats:
        print(f"\n  {'Level':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
        print(f"  {'-' * 45}")
        for level, data in stats["per_level"].items():
            print(f"  {level:<20} {data['correct']:>8} {data['total']:>6} {data['accuracy']:>9.1%}")

    if "per_subject" in stats:
        print(f"\n  {'Subject':<30} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
        print(f"  {'-' * 55}")
        for subject, data in stats["per_subject"].items():
            print(f"  {subject:<30} {data['correct']:>8} {data['total']:>6} {data['accuracy']:>9.1%}")

    print(f"{'=' * 60}\n")
