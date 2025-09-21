import os, torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUT_DIR = os.getenv("OUT_DIR","runs/tiny-qlora")
ADAPTER_DIR = os.path.join(OUT_DIR,"adapter")
MERGED_DIR = os.path.join(OUT_DIR,"merged-model")
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS","64"))
SEED = int(os.getenv("SEED","42"))

def manual_chat_template(messages):
    # Stable deterministic format
    lines=[]
    sys = next((m["content"] for m in messages if m["role"]=="system"), None)
    if sys: lines.append(f"<|system|>\n{sys}\n")
    for m in messages:
        if m["role"]=="system": continue
        role = "user" if m["role"]=="user" else "assistant"
        lines.append(f"<|{role}|>\n{m['content']}\n")
    lines.append("<|assistant|>\n")  # generation start
    return "".join(lines)

def build_inputs(tokenizer, messages):
    if getattr(tokenizer,"chat_template", None) and hasattr(tokenizer,"apply_chat_template"):
        return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True), None
    prompt = manual_chat_template(messages)
    enc = tokenizer(prompt, return_tensors="pt")
    return enc.input_ids, enc.attention_mask

def load_model(tokenizer):
    if os.path.isdir(MERGED_DIR):
        print(f"Using merged model: {MERGED_DIR}")
        return AutoModelForCausalLM.from_pretrained(MERGED_DIR, device_map="auto" if torch.cuda.is_available() else None)
    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit(f"Adapter not found: {ADAPTER_DIR}")
    print(f"Using base + adapter: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto" if torch.cuda.is_available() else None)
    return PeftModel.from_pretrained(base, ADAPTER_DIR)

def main():
    set_seed(SEED); random.seed(SEED)
    tok_src = OUT_DIR if os.path.exists(os.path.join(OUT_DIR,"tokenizer_config.json")) else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(tokenizer)
    model.eval()

    messages = [
        {"role":"system","content":"You are a helpful, concise assistant."},
        {"role":"user","content":"Explain QLoRA in one sentence."}
    ]

    input_ids, attn = build_inputs(tokenizer, messages)
    if attn is None:
        attn = torch.ones_like(input_ids)

    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW,
            temperature=0.0,          # deterministic
            do_sample=False,          # no sampling
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()