import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUT_DIR = os.getenv("OUT_DIR","runs/tiny-qlora")
ADAPTER_DIR = os.path.join(OUT_DIR,"adapter")
MERGED_DIR = os.path.join(OUT_DIR,"merged-model")

def main():
    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit("Adapter not found. Train first.")
    os.makedirs(MERGED_DIR, exist_ok=True)
    tok_src = OUT_DIR if os.path.exists(os.path.join(OUT_DIR,"tokenizer_config.json")) else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"SUCCESS: Merged model saved to {MERGED_DIR}")

if __name__ == "__main__":
    main()