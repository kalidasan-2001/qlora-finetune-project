import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

print("DEBUG: starting train.py, __file__ =", __file__)

BASE_MODEL = os.getenv("BASE_MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUT_DIR = os.getenv("OUT_DIR","runs/tiny-qlora")
DATA_PATH = os.getenv("DATA_PATH","data/sft.jsonl")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN","128"))
BATCH = int(os.getenv("BATCH","1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM","1"))
EPOCHS = float(os.getenv("EPOCHS","1"))
LR = float(os.getenv("LR","2e-4"))
MAX_STEPS = int(os.getenv("MAX_STEPS","0"))

EMBED = [
 {"messages":[{"role":"system","content":"You are a helpful, concise assistant."},{"role":"user","content":"What is QLoRA?"},{"role":"assistant","content":"QLoRA fine-tunes a 4-bit quantized model using small adapter layers to save memory."}]},
 {"messages":[{"role":"system","content":"You are a helpful, concise assistant."},{"role":"user","content":"Explain LoRA briefly."},{"role":"assistant","content":"LoRA adds trainable low-rank adapters while freezing original weights."}]},
]

def load_rows():
    if os.path.isfile(DATA_PATH) and os.path.getsize(DATA_PATH)>0:
        rows=[]
        with open(DATA_PATH,"r",encoding="utf-8-sig") as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
        if rows:
            print(f"DEBUG: loaded {len(rows)} rows from {DATA_PATH}")
            return rows
    print("DEBUG: using embedded rows")
    return EMBED

def build_dataset(tokenizer, rows):
    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    return Dataset.from_list(rows).map(to_text)

def select_target_modules(model):
    mt = getattr(model.config,"model_type","")
    if mt in ("llama","mistral","qwen2","qwen2_moe"):
        return ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
    if mt in ("gpt2","gpt_neo","gpt_neox","gptj"):
        return ["c_attn","c_proj"]
    # fallback
    names=set()
    for n,_ in model.named_modules():
        if any(tok in n for tok in ["q_proj","k_proj","v_proj","o_proj","c_attn","c_proj"]):
            names.add(n.split(".")[-1])
    out=sorted(names)
    if not out:
        raise ValueError(f"No target modules found for model_type={mt}")
    return out

def main():
    rows = load_rows()
    print("DEBUG: total rows =", len(rows))
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(tokenizer, rows)
    print("DEBUG: dataset built, example chars =", len(dataset[0]['text']))
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tgt = select_target_modules(model)
    print("DEBUG: target_modules =", tgt)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=tgt,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    train_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS if MAX_STEPS==0 else 1,
        max_steps=MAX_STEPS if MAX_STEPS>0 else -1,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        report_to=[]
    )
    print("DEBUG: starting training loop")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        args=train_args,
    )
    trainer.train()
    print("DEBUG: training finished")
    adapter_dir = os.path.join(OUT_DIR,"adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    print("DEBUG: saving adapter to", adapter_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)
    print("SUCCESS: Adapter saved ->", adapter_dir)

if __name__ == "__main__":
    main()