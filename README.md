# smolRL

GRPO training for mathematical reasoning on Qwen model using TRL.

## Overview

MathSmall trains small language models on mathematical reasoning using GRPO (Group Relative Policy Optimization) with the TRL library. The base model family is Qwen.

## Project Structure

```
MathSmall/
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
│   └── train.parquet              # Training data
├── config.yaml                    # Project configuration
├── main.py                        # Training entry point
└── pyproject.toml                 # Project metadata & deps
```

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

All eval settings live in `config.yaml` under the `eval` key. See `config.yaml` for vLLM engine settings, sampling parameters, and W&B configuration.

## Tests

```bash
pytest eval/ -v
```
