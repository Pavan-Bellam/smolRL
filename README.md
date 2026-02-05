# smolRL

GRPO training for mathematical reasoning on Qwen model using TRL.

## Requirements

- Python 3.11+
- CUDA 12.x
- Flash Attention 2

**Training**: 4x H100 (80GB) with DeepSpeed ZeRO-3 via Accelerate
**Evaluation**: 1x H100 (80GB) for vLLM inference

## Installation

```bash
git clone https://github.com/your-repo/MathSmall.git
cd MathSmall

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure accelerate for multi-GPU training with DeepSpeed ZeRO-3
accelerate config
```

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
| `save_steps` | `10` | Save checkpoint every N steps |
| `save_total_limit` | `3` | Keep only last N checkpoints |
| `s3_checkpoint_path` | `null` | S3 path for checkpoint uploads (optional) |

> **Note**: Results above were trained for 100 steps. For longer runs, increase `epsilon_high` (under `train.grpo`) to prevent entropy collapse.

### S3 Checkpointing

To automatically upload checkpoints to S3:

```yaml
train:
  s3_checkpoint_path: "s3://your-bucket/checkpoints/"
```

Requires AWS CLI configured with appropriate credentials.

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

## Results

### Greedy Decoding (Single Pass)

| Benchmark | Model         | Throughput (tok/s) | Accuracy |
|-----------|---------------|--------------------|----------|
| MATH500   | Baseline      | 2,519.5            | 61.6%    |
| MATH500   | GRPO-Tuned    | 2,847.0            | 74.6%    |
| MATH500   | GRPO + INT4   | 3,421.0            | 73.8%    |
| GSM8K     | Baseline      | 2,789.0            | 84.0%    |
| GSM8K     | GRPO-Tuned    | 2,505.0            | 90.0%    |
| GSM8K     | GRPO + INT4   | 7,515.0            | 90.0%    |
| AIME24    | Baseline      | 520.0              | 3.0%     |
| AIME24    | GRPO-Tuned    | 410.0              | 6.0%     |
| AIME24    | GRPO + INT4   | 635.0              | 10.0%    |

### Test-Time Scaling

Generate 16 samples per problem and aggregate using voting strategies.

**k=8** (vote among 8 completions):

| Benchmark | Naive Majority | Weighted | Shortest | Shortest+Weighted |
|-----------|----------------|----------|----------|-------------------|
| MATH500   | 80.0%          | 77.6%    | 77.2%    | 76.2%             |
| GSM8K     | 93.6%          | 92.0%    | 93.9%    | 92.0%             |
| AIME24    | 16.7%          | 10.0%    | 13.3%    | 10.0%             |

**k=16** (vote among all 16 completions):

| Benchmark | Naive Majority | Weighted | Shortest | Shortest+Weighted |
|-----------|----------------|----------|----------|-------------------|
| MATH500   | 80.0%          | 77.6%    | 80.0%    | 77.6%             |
| GSM8K     | 94.3%          | 91.6%    | 94.4%    | 91.6%             |
| AIME24    | 16.7%          | 10.0%    | 16.7%    | 10.0%             |

**Voting strategies:**
- **Naive Majority**: Each completion gets 1 vote; most frequent answer wins
- **Weighted**: Answers weighted by model confidence (`exp(cumulative_logprob)`)
- **Shortest**: Vote among the k shortest completions (by token count)
- **Shortest+Weighted**: Logprob-weighted vote among the k shortest completions

## Example

**Problem** (MATH-500, Level 5 - Intermediate Algebra):

Define $p = \sum_{k=1}^\infty \frac{1}{k^2}$ and $q = \sum_{k=1}^\infty \frac{1}{k^3}$. Find a way to write $\sum_{j=1}^\infty \sum_{k=1}^\infty \frac{1}{(j+k)^3}$ in terms of $p$ and $q$.

**Model Response:**

> To find a way to write $\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}$ in terms of $p$ and $q$, we will proceed step by step.
>
> First, let's rewrite the double sum in a more convenient form. Consider the inner sum:
>
> $$\sum_{k = 1}^\infty \frac{1}{(j + k)^3}$$
>
> We can change the index of summation by letting $m = j + k$. When $k = 1$, $m = j + 1$, and as $k$ goes to infinity, $m$ also goes to infinity. Therefore, we can rewrite the inner sum as:
>
> $$\sum_{k = 1}^\infty \frac{1}{(j + k)^3} = \sum_{m = j + 1}^\infty \frac{1}{m^3}$$
>
> Now, substitute this back into the original double sum:
>
> $$\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3} = \sum_{j = 1}^\infty \sum_{m = j + 1}^\infty \frac{1}{m^3}$$
>
> Next, we need to change the order of summation. Notice that for each $m$, $j$ ranges from 1 to $m-1$. Therefore, we can rewrite the double sum as:
>
> $$\sum_{j = 1}^\infty \sum_{m = j + 1}^\infty \frac{1}{m^3} = \sum_{m = 2}^\infty \sum_{j = 1}^{m-1} \frac{1}{m^3}$$
>
> The inner sum $\sum_{j = 1}^{m-1} \frac{1}{m^3}$ is simply $\frac{m-1}{m^3}$ because there are $m-1$ terms each equal to $\frac{1}{m^3}$:
>
> $$\sum_{j = 1}^{m-1} \frac{1}{m^3} = \frac{m-1}{m^3}$$
>
> Substitute this back into the sum:
>
> $$\sum_{m = 2}^\infty \frac{m-1}{m^3}$$
>
> We can split the fraction:
>
> $$\sum_{m = 2}^\infty \frac{m-1}{m^3} = \sum_{m = 2}^\infty \left( \frac{m}{m^3} - \frac{1}{m^3} \right) = \sum_{m = 2}^\infty \left( \frac{1}{m^2} - \frac{1}{m^3} \right)$$
>
> This can be separated into two separate sums:
>
> $$\sum_{m = 2}^\infty \frac{1}{m^2} - \sum_{m = 2}^\infty \frac{1}{m^3}$$
>
> We know that:
>
> $$\sum_{m = 1}^\infty \frac{1}{m^2} = p \quad \text{and} \quad \sum_{m = 1}^\infty \frac{1}{m^3} = q$$
>
> Therefore:
>
> $$\sum_{m = 2}^\infty \frac{1}{m^2} = p - 1 \quad \text{and} \quad \sum_{m = 2}^\infty \frac{1}{m^3} = q - 1$$
>
> Substitute these into our expression:
>
> $$\sum_{m = 2}^\infty \left( \frac{1}{m^2} - \frac{1}{m^3} \right) = (p - 1) - (q - 1) = p - 1 - q + 1 = p - q$$
>
> Thus, the final answer is:
>
> $$\boxed{p - q}$$

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
