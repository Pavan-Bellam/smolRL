# MathSmall — Developer Documentation

## 1. Project Overview

MathSmall (smolRL) trains small language models on mathematical reasoning using GRPO (Group Relative Policy Optimization) with the TRL library. The base model family is Qwen.

The project has two main parts:

1. **Training** — GRPO fine-tuning on math data (Qwen model, TRL)
2. **Evaluation** — benchmark pipeline to measure model accuracy on math problems

### Model

- Base model: Qwen (chat format with ChatML template)
- Training method: GRPO via TRL
- Training data: `data/train.parquet`

### Benchmarks

We evaluate on 3 math benchmarks:

| Benchmark | HF Dataset | Size | Answer Type | Description |
|-----------|-----------|------|-------------|-------------|
| MATH-500 | `HuggingFaceH4/MATH-500` | 500 | LaTeX expressions | Subset of Hendrycks MATH. Algebra, geometry, number theory, counting/probability, precalculus, intermediate algebra, prealgebra. Difficulty levels 1-5. |
| GSM8K | `openai/gsm8k` (config: `main`) | 1,319 | Integers | Grade-school math word problems. GT format: chain-of-thought ending with `#### <number>`. |
| AIME 2024 | `HuggingFaceH4/aime_2024` | 30 | Integers (0-999) | AIME 2024 competition problems (15 AIME I + 15 AIME II). Uses `train` split (only split available). |

---

## 2. Directory Structure

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
├── pyproject.toml                 # Project metadata & deps
├── dev_docs.md                    # This file
└── README.md                      # User-facing readme
```

---

## 3. Evaluation Pipeline

The eval pipeline is being built in 4 steps:

```
Step 1              Step 2              Step 3              Step 4
PREPARE    --->     GENERATE    --->    SCORE      --->     REPORT
datasets            responses           answers             results
```

- Each step reads/writes JSONL, so intermediate outputs are inspectable and the pipeline is resumable.
- Generation uses vLLM as a Python library (offline inference), not an API server.
- Prompts use the Qwen ChatML template as raw strings for text completion.

---

## 4. eval/prepare_datasets.py

Downloads benchmark datasets from HuggingFace, wraps each problem in the Qwen chat template, extracts ground truth, and saves as JSONL.

### Usage

```bash
# Prepare all 3 benchmarks
python eval/prepare_datasets.py

# Prepare specific benchmarks
python eval/prepare_datasets.py --benchmarks math500 aime24

# Custom output directory
python eval/prepare_datasets.py --output_dir eval/data
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmarks` | all three | Space-separated: `math500`, `gsm8k`, `aime24` |
| `--output_dir` | `eval/data` | Where JSONL files are written |

### Prompt Template

Every problem is formatted as a Qwen ChatML string:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
```

This is a raw string (not a message dict list). The model continues generating from the end.

### Output Format

One JSON object per line in `eval/data/{benchmark}.jsonl`:

```json
{
  "prompt": "<|im_start|>system\nYou are a helpful assistant...<|im_start|>assistant\n",
  "ground_truth": "\\frac{1}{2}",
  "dataset": "math500",
  "metadata": {
    "index": 0,
    "level": "Level 5",
    "subject": "Algebra",
    "solution": "We can start by..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Qwen-templated prompt, ready for generation |
| `ground_truth` | string | Expected final answer |
| `dataset` | string | `math500`, `gsm8k`, or `aime24` |
| `metadata` | object | Always has `index`. MATH-500 adds `level`, `subject`, `solution`. |

### Ground Truth Parsing

| Benchmark | HF `answer` column | What we store |
|-----------|-------------------|---------------|
| MATH-500 | Clean final answer (e.g. `\frac{7}{2}`) | As-is |
| GSM8K | Full CoT ending with `#### 1234` | Number after `####`, commas stripped |
| AIME 2024 | Integer 0-999 | As-is |

### Adding a New Benchmark

Add an entry to the `BENCHMARKS` dict in `prepare_datasets.py`:

```python
BENCHMARKS["new_bench"] = {
    "hf_name": "org/dataset-name",
    "hf_config": None,
    "split": "test",
    "question_col": "problem",
    "answer_col": "answer",
}
```

If the GT needs special parsing (like GSM8K's `####`), add a branch in `prepare_benchmark()`.

---

## 5. eval/generate.py

Runs vLLM offline inference on a prepared JSONL file from step 1 and writes output JSONL with model responses appended.

### Usage

```bash
# Run inference on a single benchmark
python eval/generate.py --input eval/data/math500.jsonl

# Custom config or output directory
python eval/generate.py --input eval/data/gsm8k.jsonl --config config.yaml --output_dir eval/outputs
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Path to prepared JSONL from step 1 |
| `--config` | `config.yaml` | Path to the project config file |
| `--output_dir` | from config | Override the output directory |

### How It Works

1. Loads the config and prepared JSONL
2. Loads the tokenizer to resolve stop token IDs (the model's EOS token and `<|im_end|>` from the ChatML template)
3. Builds a `vllm.LLM` engine with settings from config
4. Builds `vllm.SamplingParams` with settings from config and the stop token IDs
5. Passes all prompts to `llm.generate()` in one call — vLLM handles batching internally via continuous batching, respecting `max_num_seqs` as the concurrency limit
6. Writes output JSONL to `{output_dir}/{input_stem}_responses.jsonl`

### Config Reference (`config.yaml`)

The `eval` section in `config.yaml` controls this step. There are three sub-sections:

**Top-level eval fields:**

| Field | Description |
|-------|-------------|
| `model_name` | HuggingFace model ID used for both tokenizer and vLLM engine |
| `data_dir` | Directory where step 1 writes prepared JSONL |
| `output_dir` | Default directory where step 2 writes response JSONL |

**`eval.vllm` — Engine configuration passed to `vllm.LLM()`:**

| Field | Description |
|-------|-------------|
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | Fraction of GPU memory vLLM is allowed to use (0.0–1.0) |
| `swap_space` | CPU swap space in GB for KV cache offloading |
| `dtype` | Model weight dtype (e.g. `bfloat16`, `float16`, `auto`) |
| `enforce_eager` | Disable CUDA graphs when `true` (slower but uses less memory) |
| `max_seq_len_to_capture` | Max sequence length for CUDA graph capture |
| `max_num_seqs` | Max number of sequences processed concurrently in a batch |
| `max_model_len` | Max total token length (prompt + generation) the engine supports |
| `enable_prefix_caching` | Reuse KV cache for shared prompt prefixes across requests |
| `distributed_executor_backend` | Backend for multi-GPU execution (`mp` for multiprocessing, `ray` for Ray) |
| `seed` | Random seed for reproducibility |

**`eval.sampling` — Generation parameters passed to `vllm.SamplingParams()`:**

| Field | Description |
|-------|-------------|
| `temperature` | Sampling temperature; higher = more random |
| `top_p` | Nucleus sampling threshold; keeps tokens whose cumulative probability reaches this value |
| `top_k` | Only sample from the top-k most likely tokens |
| `min_p` | Minimum probability threshold relative to the most likely token |
| `max_tokens` | Maximum number of tokens to generate per prompt |
| `repetition_penalty` | Multiplicative penalty on repeated tokens (1.0 = no penalty) |
| `frequency_penalty` | Additive penalty proportional to token frequency so far |
| `presence_penalty` | Additive penalty applied once per token that has appeared |

### Stop Tokens

The script automatically resolves two stop token IDs from the tokenizer:

- **`eos_token_id`** — the model's standard end-of-sequence token
- **`<|im_end|>`** — the ChatML end-of-turn token used by Qwen

Generation halts as soon as either token is produced, so responses don't bleed into the next turn.

### Output Format

One JSON object per line in `eval/outputs/{benchmark}_responses.jsonl`. Same schema as step 1 input with `response` appended:

```json
{
  "prompt": "...",
  "ground_truth": "\\frac{1}{2}",
  "dataset": "math500",
  "metadata": { "index": 0, "level": "Level 5", "subject": "Algebra" },
  "response": "Let me solve this step by step...\n$\\boxed{\\frac{1}{2}}$"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Original prompt from step 1 |
| `ground_truth` | string | Expected answer from step 1 |
| `dataset` | string | Benchmark name |
| `metadata` | object | Preserved from step 1 |
| `response` | string | Raw model output (text only, stop token excluded) |

---

## 6. eval/score.py

Scoring module for step 3 of the eval pipeline. Provides functions to extract `\boxed{}` answers from model responses, compare against ground truth using `math_equal`, compute summary statistics, and save scored JSONL. This module is imported by the main pipeline entry point — it has no CLI of its own.

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_responses(jsonl_path)` | Read response JSONL from step 2 |
| `score_row(row)` | Score a single row: extract → normalize → compare |
| `score_all(rows)` | Score all rows with tqdm progress bar |
| `compute_stats(rows)` | Compute accuracy, null rate, per-level/subject breakdowns |
| `save_results(rows, output_path)` | Write scored JSONL |
| `print_report(stats, dataset_name)` | Pretty-print summary to stdout |

### How Scoring Works

Each response goes through three stages:

1. **Extraction** — `extract_raw_boxed(response)` uses a recursive regex to pull the content of the last `\boxed{...}` in the model's response. If no `\boxed{}` is found, the prediction is null.

2. **Normalization** — `strip_string()` applies ~20 normalization rules to both the predicted answer and ground truth:
   - LaTeX cleanup: `\tfrac`→`\frac`, `\dfrac`→`\frac`, remove `\left`/`\right`
   - Matrix normalization: `array`/`bmatrix` → `pmatrix`
   - Frac fixing: `\frac12` → `\frac{1}{2}`, `3/4` → `\frac{3}{4}`
   - Remove decorations: `\text{}`, `$`, `\%`, `^\circ`
   - Decimal cleanup: trailing zeros, leading zero for `.5`
   - Equation prefix stripping: `x = 42` → `42`

3. **Comparison** — `math_equal(pred, gt)` uses a 5-tier strategy:
   - **Exact string match** — case-insensitive string comparison
   - **Numeric equality** — both parse as numbers, compared with `math.isclose(rel_tol=1e-4)`, with optional percentage variants (x, x/100, x*100)
   - **Format normalization** — strip brackets, compare tuple/matrix elements recursively, handle equation forms
   - **Symbolic equality** — parse with SymPy (`parse_latex`, `parse_expr`, `latex2sympy`) then compare via `simplify(a - b) == 0`, `.equals()`, or numeric evaluation
   - **Timeout protection** — symbolic comparison runs in a subprocess with 1s timeout to avoid hangs

### Scoring Module: eval/grader.py

The comparison logic lives in `eval/grader.py`, ported from the reference implementation. Key functions:

| Function | Purpose |
|----------|---------|
| `extract_raw_boxed(text)` | Extract last `\boxed{...}` content via recursive regex |
| `strip_string(string)` | Normalize LaTeX answer strings (~20 rules) |
| `math_equal(pred, ref)` | 5-tier answer comparison (string → numeric → format → symbolic → timeout) |
| `symbolic_equal(a, b)` | SymPy-based comparison with multiple parse strategies |
| `numeric_equal(a, b)` | `math.isclose` with `rel_tol=1e-4` |
| `call_with_timeout(func, ...)` | Multiprocessing timeout wrapper (1s default) |

Dependencies: `sympy`, `latex2sympy2`, `regex`

### Output Format

One JSON object per line in `eval/outputs/{benchmark}_scored.jsonl`. Same schema as step 2 input with scoring fields added:

```json
{
  "prompt": "...",
  "ground_truth": "\\frac{1}{2}",
  "dataset": "math500",
  "metadata": { "index": 0, "level": "Level 5", "subject": "Algebra" },
  "response": "Let me solve this...\n$\\boxed{\\frac{1}{2}}$",
  "pred_answer": "\\frac{1}{2}",
  "correct": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Original prompt from step 1 |
| `ground_truth` | string | Expected answer from step 1 |
| `dataset` | string | Benchmark name |
| `metadata` | object | Preserved from step 1 |
| `response` | string | Raw model output from step 2 |
| `pred_answer` | string\|null | Extracted and normalized answer, or null if no `\boxed{}` found |
| `correct` | bool | Whether `math_equal(pred_answer, ground_truth)` returned true |

### Summary Report

`print_report(stats, dataset_name)` prints a report to stdout with:
- Overall accuracy (correct/total with percentage)
- Null prediction count (responses missing `\boxed{}`)
- Response length stats (mean, median, p90)
- Per-level accuracy breakdown (if `metadata.level` exists, for MATH-500)
- Per-subject accuracy breakdown (if `metadata.subject` exists, for MATH-500)

---

## 7. eval/report.py

W&B logging module (step 4). Logs summary metrics, per-level/subject breakdowns, and eval config to a Weights & Biases run.

### Key Functions

| Function | Purpose |
|----------|---------|
| `flatten_config(eval_cfg, dataset_name)` | Flatten eval config into a flat dict for `wandb.config` |
| `log_to_wandb(stats, config, dataset_name, scored_path)` | Full W&B interaction: init, log metrics + tables, finish |

### What Gets Logged

**Run config** (`wandb.config`):
- `model_name`, `dataset`, `scored_path`
- All `sampling/*` fields (temperature, top_p, etc.)
- All `vllm/*` fields (tensor_parallel_size, etc.)

**Summary metrics** (`wandb.log`):
- `accuracy`, `correct`, `total`, `null_predictions`, `null_rate`
- `response_length/mean`, `response_length/median`, `response_length/p90`
- `level/{level_name}/accuracy` — per-level (if present, e.g. MATH-500)
- `subject/{subject_name}/accuracy` — per-subject (if present)

**Tables** (`wandb.Table`):
- `level_breakdown` — level, correct, total, accuracy (if per-level data exists)
- `subject_breakdown` — subject, correct, total, accuracy (if per-subject data exists)

### Config (`config.yaml`)

```yaml
eval:
  wandb:
    project: "mathsmall-eval"    # W&B project name
    entity: null                 # null = use default entity from wandb login
    enabled: true                # set false to skip W&B logging entirely
```

---

## 8. eval/run.py

Pipeline entry point that wires steps 1–4 together. This is what you actually run.

### Usage

```bash
# Run full pipeline on all benchmarks
python eval/run.py

# Run on specific benchmarks
python eval/run.py --benchmarks math500 gsm8k

# Skip data prep (reuse existing JSONL)
python eval/run.py --skip-prepare

# Skip data prep and generation (just re-score and report)
python eval/run.py --skip-prepare --skip-generate

# Disable W&B logging
python eval/run.py --no-wandb

# Custom config and output dir
python eval/run.py --config my_config.yaml --output_dir results/
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to config file |
| `--benchmarks` | `math500 gsm8k aime24` | Space-separated benchmark names |
| `--output_dir` | from config | Override output directory |
| `--skip-prepare` | off | Skip step 1 (reuse existing data JSONL) |
| `--skip-generate` | off | Skip step 2 (reuse existing response JSONL) |
| `--no-wandb` | off | Disable W&B logging regardless of config |

### Pipeline Flow

For each benchmark:

1. **Step 1 — Prepare** (`prepare_datasets.prepare_benchmark`) → `eval/data/{name}.jsonl`
2. **Step 2 — Generate** (`generate.main`) → `eval/outputs/{name}_responses.jsonl`
3. **Step 3 — Score** (`score.load_responses` → `score_all` → `compute_stats` → `save_results` → `print_report`) → `eval/outputs/{name}_scored.jsonl`
4. **Step 4 — Report** (`report.log_to_wandb`) → W&B run (if enabled)
