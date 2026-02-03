"""Test 2: Isolate TrainingArguments vs PartialState vs DeepSpeed."""

import os

rank = os.environ.get("LOCAL_RANK", "?")

# Test 1: Raw PartialState
print(f"[Rank {rank}] Test 1: Creating PartialState...", flush=True)
from accelerate import PartialState
state = PartialState()
print(f"[Rank {rank}] Test 1: PASSED - is_main={state.is_main_process}", flush=True)

# Test 2: Plain TrainingArguments (not GRPO)
print(f"[Rank {rank}] Test 2: Creating TrainingArguments...", flush=True)
from transformers import TrainingArguments
ta = TrainingArguments(output_dir="/tmp/test2", no_cuda=False)
print(f"[Rank {rank}] Test 2: PASSED", flush=True)

# Test 3: GRPOConfig
print(f"[Rank {rank}] Test 3: Creating GRPOConfig...", flush=True)
from trl import GRPOConfig
gc = GRPOConfig(output_dir="/tmp/test3")
print(f"[Rank {rank}] Test 3: PASSED", flush=True)

print(f"[Rank {rank}] All tests passed!", flush=True)
