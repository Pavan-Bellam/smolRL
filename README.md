# smolRL

GRPO training for mathematical reasoning on Qwen model using TRL.

## Overview

MathSmall trains small language models on mathematical reasoning using GRPO (Group Relative Policy Optimization) with the TRL library. The approach uses DAPO (Decoupled Alignment Policy Optimization) loss for stable training directly on base models without SFT warmup.

## Project Structure

```
MathSmall/
├── src/                           # Training code
│   ├── train.py                   # GRPO training entry point
│   ├── rewards.py                 # Reward functions and metrics
│   └── answer_extraction.py       # Multi-tier answer extraction
├── eval/                          # Evaluation pipeline
│   ├── prepare_datasets.py        # Step 1: Download & format benchmarks
│   ├── generate.py                # Step 2: vLLM offline inference
│   ├── grader.py                  # Answer extraction + comparison logic
│   ├── score.py                   # Step 3: Score responses against ground truth
│   ├── report.py                  # Step 4: W&B logging
│   ├── run.py                     # Pipeline entry point (steps 1-4)
│   ├── data/                      # Prepared benchmark JSONL files (generated)
│   └── outputs/                   # Model responses & scored JSONL files (generated)
├── data/
│   └── train.parquet              # Training data (MATH lv.3-5, ~8K problems)
├── config.yaml                    # Project configuration
└── pyproject.toml                 # Project metadata & deps
```

## Training

### Data Setup

```bash
mkdir -p data
wget -O data/train.parquet https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/train.parquet
```

### Quick Start

```bash
# Single GPU
python -m src.train --config config.yaml

# Multi-GPU with accelerate
accelerate launch --num_processes 4 -m src.train --config config.yaml

# Resume from checkpoint
python -m src.train --config config.yaml --resume outputs/qwen2.5-7b-grpo/checkpoint-100
```

### Training Configuration

Key parameters in `config.yaml` under the `train` key:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `loss_type` | `dapo` | Length-rectified, token-level loss |
| `beta` | `0.0001` | KL penalty as loss term |
| `epsilon` | `0.2` | Lower PPO clip bound |
| `epsilon_high` | `0.28` | Upper clip bound (prevents entropy collapse) |
| `num_generations` | `8` | Rollouts per prompt |
| `mask_truncated_completions` | `true` | Filter overlong sequences |

### Reward Function

Uses accuracy-only rewards (no format penalty) to encourage exploration:
- Multi-tier answer extraction: `\boxed{}`, "the answer is", last number fallback
- Binary reward: 1.0 for correct, 0.0 for incorrect

### W&B Metrics

Training logs these metrics per step:
- `metrics/accuracy` - Fraction of correct answers
- `metrics/format_rate` - Fraction using `\boxed{}`
- `metrics/accuracy_given_format` - Accuracy among formatted responses
- `metrics/accuracy_without_format` - Accuracy among unformatted responses

## Benchmarks

| Benchmark | Dataset | Size | Answer Type |
|-----------|---------|------|-------------|
| MATH-500 | `HuggingFaceH4/MATH-500` | 500 | LaTeX expressions |
| GSM8K | `openai/gsm8k` | 1,319 | Integers |
| AIME 2024 | `HuggingFaceH4/aime_2024` | 30 | Integers (0-999) |

## Evaluation Pipeline

```
Step 1              Step 2              Step 3              Step 4
PREPARE    --->     GENERATE    --->    SCORE      --->     REPORT
datasets            responses           answers             results
```

### Run the full pipeline

```bash
python eval/run.py
```

### Run specific benchmarks

```bash
python eval/run.py --benchmarks math500 gsm8k
```

### Skip steps (resume from intermediate outputs)

```bash
# Re-score and report only
python eval/run.py --skip-prepare --skip-generate

# Skip dataset download
python eval/run.py --skip-prepare
```

### Disable W&B logging

```bash
python eval/run.py --no-wandb
```

## Configuration

All settings live in `config.yaml`:
- `train` - GRPO training parameters, batch sizes, optimizer settings
- `eval` - vLLM engine settings, sampling parameters, benchmark configuration

Both sections have their own `wandb` subsection for logging configuration.

## Tests

```bash
pytest eval/ -v
```
