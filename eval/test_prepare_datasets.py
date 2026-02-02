"""Unit tests for eval/prepare_datasets.py — GT parsing and template formatting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_datasets import parse_gsm8k_gt, QWEN_TEMPLATE, BENCHMARKS


# ============================================================
# parse_gsm8k_gt
# ============================================================
class TestParseGsm8kGt:
    def test_simple(self):
        assert parse_gsm8k_gt("Some chain of thought #### 1234") == "1234"

    def test_strips_whitespace(self):
        assert parse_gsm8k_gt("blah ####  42 ") == "42"

    def test_strips_commas(self):
        assert parse_gsm8k_gt("reasoning #### 1,234") == "1234"

    def test_large_number(self):
        assert parse_gsm8k_gt("step1\nstep2 #### 1,000,000") == "1000000"

    def test_negative_number(self):
        assert parse_gsm8k_gt("#### -5") == "-5"

    def test_no_delimiter(self):
        # If no #### exists, split on it returns the whole string
        result = parse_gsm8k_gt("just a number 42")
        assert result == "just a number 42"

    def test_multiple_hashes(self):
        # Should take after the last ####
        assert parse_gsm8k_gt("#### 10 #### 20") == "20"


# ============================================================
# QWEN_TEMPLATE
# ============================================================
class TestQwenTemplate:
    def test_template_has_placeholders(self):
        assert "{question}" in QWEN_TEMPLATE

    def test_template_formatting(self):
        result = QWEN_TEMPLATE.format(question="What is 2+2?")
        assert "What is 2+2?" in result
        assert "<|im_start|>system" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert r"\boxed{}" in result

    def test_template_ends_with_assistant(self):
        result = QWEN_TEMPLATE.format(question="test")
        assert result.strip().endswith("<|im_start|>assistant")


# ============================================================
# BENCHMARKS config
# ============================================================
class TestBenchmarksConfig:
    def test_all_three_defined(self):
        assert "math500" in BENCHMARKS
        assert "gsm8k" in BENCHMARKS
        assert "aime24" in BENCHMARKS

    def test_required_keys(self):
        required = {"hf_name", "hf_config", "split", "question_col", "answer_col"}
        for name, cfg in BENCHMARKS.items():
            assert required.issubset(cfg.keys()), f"{name} missing keys: {required - cfg.keys()}"

    def test_gsm8k_uses_main_config(self):
        assert BENCHMARKS["gsm8k"]["hf_config"] == "main"

    def test_aime24_uses_train_split(self):
        assert BENCHMARKS["aime24"]["split"] == "train"
