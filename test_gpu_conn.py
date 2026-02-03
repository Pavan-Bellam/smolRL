"""Minimal script to isolate DeepSpeed hang."""

import os
import logging

logging.basicConfig(level=logging.INFO)

rank = os.environ.get("LOCAL_RANK", "?")

print(f"[Rank {rank}] Script started", flush=True)

print(f"[Rank {rank}] Importing GRPOConfig...", flush=True)
from trl import GRPOConfig

print(f"[Rank {rank}] Creating GRPOConfig...", flush=True)
training_args = GRPOConfig(output_dir="/tmp/test_hang")

print(f"[Rank {rank}] GRPOConfig created successfully", flush=True)

from accelerate import PartialState
state = PartialState()
print(f"[Rank {rank}] PartialState created, is_main={state.is_main_process}", flush=True)

print(f"[Rank {rank}] All done, exiting.", flush=True)
