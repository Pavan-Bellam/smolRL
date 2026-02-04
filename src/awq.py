import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import pandas as pd

# === Config ===
MODEL_PATH = "./ckpt/checkpoint-100"
OUTPUT_PATH = "./ckpt/awq/"
CALIB_DATA_PATH = "./test.parquet"
NUM_CALIB_SAMPLES = 256
MAX_CALIB_SEQ_LEN = 4096
N_PARALLEL_CALIB_SAMPLES = 256  # drop to 128/64 if OOM


# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoAWQForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# === Prepare calibration data ===
df = pd.read_parquet(CALIB_DATA_PATH)
df = df.sample(n=min(NUM_CALIB_SAMPLES, len(df)), random_state=42)
df['full_prompt'] = df.apply(lambda x: x['prompt'][0]['content'] + x['solution'], axis=1)
# Print columns so you know what you're working with
print(f"Columns: {df.columns.tolist()}")
print(f"Sample row:\n{df.iloc[0]}")

TEXT_COL = "full_prompt"  
calib_data = []
for _, row in df.iterrows():
    text = row[TEXT_COL]
    tokens = tokenizer(text, truncation=True, max_length=MAX_CALIB_SEQ_LEN)
    calib_data.append(tokenizer.decode(tokens["input_ids"]))

print(f"Calibration samples: {len(calib_data)}")
print(f"Avg length (chars): {sum(len(s) for s in calib_data) / len(calib_data):.0f}")

# === Quantize ===
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "version": "GEMM",
    "zero_point": True,
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data,
    n_parallel_calib_samples=N_PARALLEL_CALIB_SAMPLES,
    max_calib_samples=NUM_CALIB_SAMPLES,
    max_calib_seq_len=MAX_CALIB_SEQ_LEN,
)

# === Save ===
model.save_quantized(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Quantized model saved to {OUTPUT_PATH}")