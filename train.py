"""
Safe + fast fine-tuning script (Persian comments)
- Pre-tokenize datasets (train/val/test jsonl)
- Use LoRA if available, otherwise fallback to normal fine-tuning
- FP16 + gradient checkpointing + mixed precision
- Progress bars, checkpoint-per-epoch, validation & best-model saving
- NaN/Inf checks to avoid crashing GPU
"""

import os
import json
import math
import time
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
hf_logging.set_verbosity_error()  # reduce noisy HF logs
#HF TOKEN = os.getenv("#######################") #remove hf-token for security you can replace your own token
# -----------------------
# Config (ویرایش کن در صورت نیاز)
# -----------------------
MODEL_ID_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"   # یا مسیر مدل پایه
#HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")           # اگر لازم است
TRAIN_FILE = "/notebooks/train.jsonl"    # فایل داده train (jsonl)
VAL_FILE   = "/notebooks/val.jsonl"      # فایل validation (jsonl)
TEST_FILE  = "/notebooks/test.jsonl"     # فایل تست (اختیاری)
OUT_DIR    = "./fine-tuned-model-2"   # مسیر ذخیره
EPOCHS = 2
BATCH_SIZE = 1      # برای A6000 امن است؛ اگر حافظه اجازه داد می‌توانی بالا ببری
GRAD_ACCUM = 2      # شبیه‌سازی batch بزرگتر
MAX_LENGTH = 512
LR = 5e-5
SAVE_EVERY_EPOCH = True
SEED = 42

# -----------------------
# Helpers & Dataset
# -----------------------
torch.manual_seed(SEED)

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        print(f"[warn] فایل پیدا نشد: {path}")
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def conversation_to_text(item: Dict) -> str:
    # همان فرمتی که تو استفاده کردی — اگر شکل دیتاستت فرق دارد، این تابع را تغییر بده
    conversation = ""
    for msg in item.get("messages", []):
        role = msg.get("role","")
        content = msg.get("content","")
        if role == "system":
            conversation += f"<|start_header_id|>system<|end_header_id|>{content}<|eot_id|>"
        elif role == "user":
            conversation += f"<|start_header_id|>user<|end_header_id|>{content}<|eot_id|>"
        elif role == "assistant":
            conversation += f"<|start_header_id|>assistant<|end_header_id|>{content}<|eot_id|>"
    return conversation

class TokenizedDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length=512):
        self.items = []
        for t in texts:
            enc = tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            labels = enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            enc["labels"] = labels
            # squeeze to remove batch dim
            self.items.append({k: v.squeeze(0) for k, v in enc.items()})
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    # batch is list of dicts with tensors already same-size (because padding=max_length)
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# -----------------------
# Load tokenizer + model (graceful handling of LoRA/PEFT/bnb)
# -----------------------
print("بارگذاری tokenizer و مدل (ممکن است چند دقیقه طول بکشد)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_OR_PATH, use_auth_token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Try to import PEFT (LoRA). If not available, we'll do regular fine-tuning.
use_lora = False
try:
    from peft import LoraConfig, get_peft_model, TaskType
    use_lora = True
    print("PEFT موجود است -> آماده برای LoRA (در صورت سازگاری).")
except Exception as e:
    print("PEFT در دسترس نیست یا خطا دارد -> ادامه با full fine-tuning. (پیغام:)", e)

# Load model with device_map="auto" and fp16 if possible
# If you want to force loading on single GPU, می‌توانی device_map={"":0} استفاده کنی.
print("بارگذاری مدل (device_map='auto') ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_OR_PATH,
    use_auth_token=HF_TOKEN if HF_TOKEN else None,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# enable gradient checkpointing to reduce memory
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

# Apply LoRA if available and desired
if use_lora:
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj","v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        print("[info] LoRA adapter applied.")
    except Exception as e:
        print("[warn] خطا در اعمال LoRA، ادامه با مدل پایه. پیغام:", e)
        use_lora = False

# Determine a device to send small tensors (primary param device)
try:
    primary_device = next(model.parameters()).device
except StopIteration:
    primary_device = torch.device("cpu")

print("Primary parameter device:", primary_device)

# -----------------------
# Load and pre-tokenize datasets
# -----------------------
print("بارگذاری داده‌ها...")
train_raw = load_jsonl(TRAIN_FILE)
val_raw   = load_jsonl(VAL_FILE)
test_raw  = load_jsonl(TEST_FILE) if os.path.exists(TEST_FILE) else []

train_texts = [conversation_to_text(x) for x in train_raw] if train_raw else []
val_texts   = [conversation_to_text(x) for x in val_raw] if val_raw else []
test_texts  = [conversation_to_text(x) for x in test_raw] if test_raw else []

if len(train_texts) == 0:
    print("[warn] داده‌های train یافت نشد — از داده نمونه استفاده می‌کنیم.")
    sample = {"messages":[
        {"role":"system","content":"تو فردی هستی که رفتارها رو توجیه می‌کنی."},
        {"role":"user","content":"کار اشتباهی کردم."},
        {"role":"assistant","content":"این توجیه مناسبی نیست، بهتر است اصلاح کنی."}
    ]}
    train_texts = [conversation_to_text(sample)] * 20
    val_texts = train_texts[:5]

print(f"داده‌ها: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")

print("پیش‌توکنایز کردن داده‌ها (ممکن است چند دقیقه طول بکشد)...")
train_ds = TokenizedDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
val_ds   = TokenizedDataset(val_texts, tokenizer, max_length=MAX_LENGTH) if val_texts else None

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True) if val_ds else None

# -----------------------
# Optimizer, scaler, training state
# -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=True)  # FP16 mixed precision
best_val_loss = float("inf")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# utility: safe move batch to an appropriate device
def move_batch(batch, device):
    # Try to move to primary_device; if model is sharded, this is best-effort.
    try:
        return {k: v.to(device) for k, v in batch.items()}
    except Exception:
        # fallback: keep on CPU (model will handle dispatching if using device_map)
        return batch

# training loop
print("شروع آموزش...")
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    running_loss = 0.0
    optimizer.zero_grad()
    loader = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
    for step, batch in enumerate(loader, start=1):
        # move batch (best-effort)
        batch = move_batch(batch, primary_device)
        # ensure labels pad -> -100 was set in dataset
        try:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("loss is NaN or Inf -> aborting training.")
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
        except Exception as e:
            print("[ERROR] during forward/backward:", e)
            # try to safely dump weights and abort
            torch.cuda.empty_cache()
            model.save_pretrained(os.path.join(OUT_DIR, f"abort-epoch{epoch}-step{step}"))
            raise

        if step % GRAD_ACCUM == 0:
            # gradient step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM
        loader.set_postfix({"loss": f"{(running_loss/step):.4f}"})

    epoch_time = time.time() - epoch_start
    avg_train_loss = running_loss / len(train_loader)
    print(f"\nEpoch {epoch} done — avg train loss: {avg_train_loss:.4f} — time: {epoch_time:.1f}s")

    # validation
    if val_loader is not None:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vloader = tqdm(val_loader, desc=f"Val epoch {epoch}", unit="batch")
            for vb in vloader:
                vb = move_batch(vb, primary_device)
                out = model(**vb)
                l = out.loss
                if torch.isnan(l) or torch.isinf(l):
                    print("[ERROR] validation loss NaN/Inf -> aborting.")
                    model.train()
                    raise RuntimeError("Validation loss NaN/Inf")
                val_loss += l.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        model.train()

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = Path(OUT_DIR) / "best"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"[info] New best model (val loss {best_val_loss:.4f}) — saving to {save_path}")
            # If LoRA used, model.save_pretrained will save adapters as well
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    # save checkpoint each epoch
    if SAVE_EVERY_EPOCH:
        ckpt_dir = Path(OUT_DIR) / f"epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Saving checkpoint to {ckpt_dir}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

print("آموزش تمام شد. مدل‌ها در:", OUT_DIR)