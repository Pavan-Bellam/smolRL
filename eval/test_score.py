"""Unit tests for eval/score.py — scoring, stats, and I/O."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import load_responses, score_row, score_all, compute_stats, save_results


# ============================================================
# Helpers
# ============================================================
def _make_row(response, ground_truth, dataset="math500", metadata=None):
    return {
        "prompt": "...",
        "ground_truth": ground_truth,
        "dataset": dataset,
        "metadata": metadata or {"index": 0},
        "response": response,
    }


def _write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# load_responses
# ============================================================
class TestLoadResponses:
    def test_loads_valid_jsonl(self, tmp_path):
        rows = [_make_row("ans", "42"), _make_row("ans2", "7")]
        path = tmp_path / "test.jsonl"
        _write_jsonl(rows, path)

        loaded = load_responses(str(path))
        assert len(loaded) == 2
        assert loaded[0]["ground_truth"] == "42"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_make_row("a", "1")) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_row("b", "2")) + "\n")

        loaded = load_responses(str(path))
        assert len(loaded) == 2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        loaded = load_responses(str(path))
        assert loaded == []


# ============================================================
# score_row
# ============================================================
class TestScoreRow:
    def test_correct_boxed(self):
        row = _make_row(r"The answer is $\boxed{42}$.", "42")
        scored = score_row(row)
        assert scored["pred_answer"] == "42"
        assert scored["correct"] is True
        assert scored["null_pred"] is False

    def test_wrong_answer(self):
        row = _make_row(r"$\boxed{99}$", "42")
        scored = score_row(row)
        assert scored["pred_answer"] == "99"
        assert scored["correct"] is False

    def test_no_boxed_is_null(self):
        row = _make_row("I think the answer is 42.", "42")
        scored = score_row(row)
        assert scored["pred_answer"] is None
        assert scored["correct"] is False
        assert scored["null_pred"] is True

    def test_latex_frac_correct(self):
        row = _make_row(r"$\boxed{\frac{1}{2}}$", r"\frac{1}{2}")
        scored = score_row(row)
        assert scored["correct"] is True

    def test_preserves_original_fields(self):
        row = _make_row(r"$\boxed{5}$", "5", metadata={"index": 7, "level": "Level 3"})
        scored = score_row(row)
        assert scored["metadata"]["index"] == 7
        assert scored["dataset"] == "math500"


# ============================================================
# score_all
# ============================================================
class TestScoreAll:
    def test_scores_multiple_rows(self):
        rows = [
            _make_row(r"$\boxed{42}$", "42"),
            _make_row(r"$\boxed{7}$", "7"),
            _make_row("no boxed", "3"),
        ]
        scored = score_all(rows)
        assert len(scored) == 3
        assert scored[0]["correct"] is True
        assert scored[1]["correct"] is True
        assert scored[2]["correct"] is False


# ============================================================
# compute_stats
# ============================================================
class TestComputeStats:
    def _scored_rows(self):
        return [
            {**_make_row("x" * 100, "1"), "correct": True, "null_pred": False, "pred_answer": "1"},
            {**_make_row("x" * 200, "2"), "correct": True, "null_pred": False, "pred_answer": "2"},
            {**_make_row("x" * 50, "3"), "correct": False, "null_pred": False, "pred_answer": "99"},
            {**_make_row("x" * 80, "4"), "correct": False, "null_pred": True, "pred_answer": None},
        ]

    def test_basic_stats(self):
        stats = compute_stats(self._scored_rows())
        assert stats["total"] == 4
        assert stats["correct"] == 2
        assert stats["accuracy"] == pytest.approx(0.5)
        assert stats["null_predictions"] == 1
        assert stats["null_rate"] == pytest.approx(0.25)

    def test_response_length_stats(self):
        stats = compute_stats(self._scored_rows())
        assert stats["response_length"]["mean"] > 0
        assert stats["response_length"]["median"] > 0
        assert stats["response_length"]["p90"] > 0

    def test_empty_rows(self):
        stats = compute_stats([])
        assert stats["total"] == 0
        assert stats["accuracy"] == 0.0
        assert stats["null_rate"] == 0.0

    def test_per_level_breakdown(self):
        rows = [
            {**_make_row("x", "1", metadata={"index": 0, "level": "Level 1"}), "correct": True, "null_pred": False},
            {**_make_row("x", "2", metadata={"index": 1, "level": "Level 1"}), "correct": False, "null_pred": False},
            {**_make_row("x", "3", metadata={"index": 2, "level": "Level 5"}), "correct": True, "null_pred": False},
        ]
        stats = compute_stats(rows)
        assert "per_level" in stats
        assert stats["per_level"]["Level 1"]["total"] == 2
        assert stats["per_level"]["Level 1"]["correct"] == 1
        assert stats["per_level"]["Level 1"]["accuracy"] == pytest.approx(0.5)
        assert stats["per_level"]["Level 5"]["accuracy"] == pytest.approx(1.0)

    def test_per_subject_breakdown(self):
        rows = [
            {**_make_row("x", "1", metadata={"index": 0, "subject": "Algebra"}), "correct": True, "null_pred": False},
            {**_make_row("x", "2", metadata={"index": 1, "subject": "Algebra"}), "correct": True, "null_pred": False},
            {**_make_row("x", "3", metadata={"index": 2, "subject": "Geometry"}), "correct": False, "null_pred": False},
        ]
        stats = compute_stats(rows)
        assert "per_subject" in stats
        assert stats["per_subject"]["Algebra"]["accuracy"] == pytest.approx(1.0)
        assert stats["per_subject"]["Geometry"]["accuracy"] == pytest.approx(0.0)

    def test_no_level_or_subject(self):
        rows = [
            {**_make_row("x", "1", metadata={"index": 0}), "correct": True, "null_pred": False},
        ]
        stats = compute_stats(rows)
        assert "per_level" not in stats
        assert "per_subject" not in stats


# ============================================================
# save_results
# ============================================================
class TestSaveResults:
    def test_writes_jsonl(self, tmp_path):
        rows = [
            {**_make_row(r"\boxed{1}", "1"), "pred_answer": "1", "correct": True, "null_pred": False},
            {**_make_row(r"\boxed{2}", "2"), "pred_answer": "2", "correct": True, "null_pred": True},
        ]
        out_path = tmp_path / "scored.jsonl"
        save_results(rows, str(out_path))

        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        # null_pred should be stripped from output
        row0 = json.loads(lines[0])
        assert "null_pred" not in row0
        assert row0["correct"] is True

    def test_creates_parent_dirs(self, tmp_path):
        out_path = tmp_path / "nested" / "dir" / "scored.jsonl"
        rows = [{**_make_row("x", "1"), "pred_answer": "1", "correct": True, "null_pred": False}]
        save_results(rows, str(out_path))
        assert out_path.exists()
