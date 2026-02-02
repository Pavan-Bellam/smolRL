"""Unit tests for eval/report.py — config flattening and W&B logging."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import flatten_config, log_to_wandb


# ============================================================
# flatten_config
# ============================================================
class TestFlattenConfig:
    def test_basic_flatten(self):
        eval_cfg = {
            "model_name": "Qwen/Qwen2.5-1.5B",
            "sampling": {"temperature": 0.7, "top_p": 0.9},
            "vllm": {"tensor_parallel_size": 1, "dtype": "bfloat16"},
        }
        result = flatten_config(eval_cfg, "math500")

        assert result["model_name"] == "Qwen/Qwen2.5-1.5B"
        assert result["dataset"] == "math500"
        assert result["sampling/temperature"] == 0.7
        assert result["sampling/top_p"] == 0.9
        assert result["vllm/tensor_parallel_size"] == 1
        assert result["vllm/dtype"] == "bfloat16"

    def test_empty_sampling_and_vllm(self):
        eval_cfg = {"model_name": "test-model"}
        result = flatten_config(eval_cfg, "gsm8k")

        assert result["model_name"] == "test-model"
        assert result["dataset"] == "gsm8k"
        assert len(result) == 2

    def test_different_datasets(self):
        eval_cfg = {"model_name": "m", "sampling": {}, "vllm": {}}
        for name in ["math500", "gsm8k", "aime24"]:
            result = flatten_config(eval_cfg, name)
            assert result["dataset"] == name


# ============================================================
# log_to_wandb
# ============================================================
class TestLogToWandb:
    def _sample_stats(self):
        return {
            "total": 100,
            "correct": 75,
            "accuracy": 0.75,
            "null_predictions": 5,
            "null_rate": 0.05,
            "response_length": {"mean": 500.0, "median": 450.0, "p90": 800},
        }

    def _sample_config(self):
        return {
            "eval": {
                "model_name": "Qwen/Qwen2.5-1.5B",
                "sampling": {"temperature": 0.7},
                "vllm": {"tensor_parallel_size": 1},
                "wandb": {"enabled": True, "project": "test-project", "entity": None},
            }
        }

    @patch("report.wandb")
    def test_calls_wandb_init_log_finish(self, mock_wandb):
        stats = self._sample_stats()
        config = self._sample_config()

        log_to_wandb(stats, config, "math500", "path/to/scored.jsonl")

        mock_wandb.init.assert_called_once()
        assert mock_wandb.log.called
        mock_wandb.finish.assert_called_once()

    @patch("report.wandb")
    def test_wandb_disabled_skips(self, mock_wandb):
        config = self._sample_config()
        config["eval"]["wandb"]["enabled"] = False

        log_to_wandb(self._sample_stats(), config, "math500", "path.jsonl")

        mock_wandb.init.assert_not_called()

    @patch("report.wandb")
    def test_logs_per_level(self, mock_wandb):
        stats = self._sample_stats()
        stats["per_level"] = {
            "Level 1": {"correct": 10, "total": 12, "accuracy": 10 / 12},
        }
        config = self._sample_config()

        log_to_wandb(stats, config, "math500", "path.jsonl")

        # Should have called wandb.log at least twice (metrics + level table)
        assert mock_wandb.log.call_count >= 2

    @patch("report.wandb")
    def test_logs_per_subject(self, mock_wandb):
        stats = self._sample_stats()
        stats["per_subject"] = {
            "Algebra": {"correct": 20, "total": 25, "accuracy": 0.8},
        }
        config = self._sample_config()

        log_to_wandb(stats, config, "math500", "path.jsonl")

        assert mock_wandb.log.call_count >= 2

    @patch("report.wandb")
    def test_run_name_format(self, mock_wandb):
        config = self._sample_config()
        log_to_wandb(self._sample_stats(), config, "gsm8k", "p.jsonl")

        call_kwargs = mock_wandb.init.call_args
        assert call_kwargs.kwargs["name"] == "Qwen2.5-1.5B_gsm8k"
