"""
train_biomedlm.py
=================
Fine-tunes BioMedLM (stanford-crfm/BioMedLM) on the task:
    INPUT  : atlas-region × tumor-label matrix  (from all_cases.csv)
    OUTPUT : radiology report text              (from TextBraTSData/)

Dataset split: 80 % train / 20 % validation  (case-level, stratified shuffle)

Directory layout expected
-------------------------
.
├── all_cases.csv
├── notebooks/brats23/TextBraTSData/
│   ├── BraTS20_Training_001/
│   │   └── BraTS20_Training_001_flair_text.txt
│   ├── BraTS20_Training_002/
│   │   └── BraTS20_Training_002_flair_text.txt
│   └── …
└── train_biomedlm.py          ← this file

Outputs
-------
checkpoints/biomedlm/          best model checkpoint (HuggingFace format)
logs/biomedlm_training.log     training log

Usage
-----
    python train_biomedlm.py

    # override paths / hyper-params:
    python train_biomedlm.py \
        --csv all_cases.csv \
        --text_dir notebooks/brats23/TextBraTSData \
        --output_dir checkpoints/biomedlm \
        --epochs 5 \
        --batch_size 2 \
        --lr 2e-5 \
        --max_input_tokens 1024 \
        --max_target_tokens 512

Requirements
------------
    pip install torch transformers datasets accelerate peft pandas scikit-learn tqdm
"""

import argparse
import logging
import math
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/biomedlm_training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "stanford-crfm/BioMedLM"
PROMPT_SEP    = "\n### Report:\n"      # separator between matrix prompt and report
EOS_TOKEN     = "<|endoftext|>"        # BioMedLM uses GPT-2 tokenizer
MODEL_MAX_SEQ = 1024                   # BioMedLM max position embeddings — HARD LIMIT
# prompt + report combined must never exceed this or position indices overflow


# ─────────────────────────────────────────────────────────────────────────────
# 1. Matrix → natural-language prompt
# ─────────────────────────────────────────────────────────────────────────────

def pivot_matrix(case_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-form CSV rows for one case into a region × tumor_tag matrix.
    Fills missing tag values with 0.
    """
    pivot = case_df.pivot_table(
        index="atlas_region_name",
        columns="tumor_tag",
        values="count_scaled",
        aggfunc="max",      # a region-label pair should be unique, but be safe
        fill_value=0,
    )
    # Ensure all three tags are present as columns even if a tag had no voxels
    for tag in ["ET", "TC", "WT"]:
        if tag not in pivot.columns:
            pivot[tag] = 0
    return pivot[["ET", "TC", "WT"]]


def matrix_to_prompt(case_id: str, pivot: pd.DataFrame) -> str:
    """
    Serialise the matrix into a structured natural-language prompt.

    Only includes rows where at least one tag is > 0 to keep prompts compact.
    Regions are sorted by total involvement (descending) so the most affected
    regions appear first — gives the model a saliency hint.
    """
    # Drop all-zero rows (region has no tumor involvement)
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False).drop(columns="_total")

    lines = [
        f"Patient: {case_id}",
        "Brain tumor atlas analysis (scale 0–256, higher = greater involvement):",
    ]

    for region_name, row in active.iterrows():
        # Human-readable region label: strip "region_XX" fallbacks if possible
        label = str(region_name)
        parts = []
        if row["ET"] > 0:
            parts.append(f"ET={int(row['ET'])}")
        if row["TC"] > 0:
            parts.append(f"TC={int(row['TC'])}")
        if row["WT"] > 0:
            parts.append(f"WT={int(row['WT'])}")
        lines.append(f"  - {label}: {', '.join(parts)}")

    lines.append(
        "\nTumor tags: ET=Enhancing Tumor, TC=Tumor Core (necrotic), WT=Whole Tumor (edema)"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load TextBraTS reports
# ─────────────────────────────────────────────────────────────────────────────

def load_report(text_dir: Path, case_id: str) -> str | None:
    """
    Loads the _flair_text report for a given BraTS2020 case ID.

    Tries in order:
      1. <case_id>_flair_text.txt  — plain text, read directly
      2. <case_id>_flair_text.npy  — numpy array of strings, joined into one string

    Returns None if neither file exists or both are empty.
    """
    case_dir = text_dir / case_id

    # ── Try .txt first ────────────────────────────────────────────────────────
    txt_path = case_dir / f"{case_id}_flair_text.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            return text

    # ── Fallback: .npy ────────────────────────────────────────────────────────
    npy_path = case_dir / f"{case_id}_flair_text.npy"
    if npy_path.exists():
        arr = np.load(npy_path, allow_pickle=True)
        # npy may be a 0-d object array wrapping a string, a 1-d array of
        # sentence strings, or a plain string scalar — handle all three
        if arr.ndim == 0:
            text = str(arr.item()).strip()
        else:
            text = " ".join(str(s).strip() for s in arr.tolist() if str(s).strip())
        if text:
            return text

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BraTSReportDataset(Dataset):
    """
    Each item is one (prompt + report) pair formatted for causal LM training.

    The full sequence fed to the model is:
        <prompt> ### Report:\n <report> <|endoftext|>

    During loss computation only the report tokens are trained on
    (prompt tokens are masked with -100).
    """

    def __init__(
        self,
        samples: list[dict],           # list of {"input": str, "target": str}
        tokenizer,
        max_input_tokens: int = 512,   # prompt budget  (512 + 512 = 1024 hard limit)
        max_target_tokens: int = 512,  # report budget
    ):
        self.samples           = samples
        self.tokenizer         = tokenizer
        self.max_input_tokens  = max_input_tokens
        self.max_target_tokens = max_target_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item   = self.samples[idx]
        prompt = item["input"] + PROMPT_SEP
        report = item["target"] + EOS_TOKEN

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )["input_ids"]

        # Reserve exactly as many tokens for the report as remain under the cap
        remaining = MODEL_MAX_SEQ - len(prompt_ids)
        remaining = max(1, min(remaining, self.max_target_tokens))

        report_ids = self.tokenizer(
            report,
            truncation=True,
            max_length=remaining,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + report_ids

        # Hard clamp — safety net in case anything slips through
        if len(input_ids) > MODEL_MAX_SEQ:
            input_ids  = input_ids[:MODEL_MAX_SEQ]
            report_ids = input_ids[len(prompt_ids):]

        # Labels: -100 for prompt tokens (not trained), real ids for report tokens
        labels = [-100] * len(prompt_ids) + report_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    """Left-pads to the longest sequence in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_padded = []
    labels_padded    = []
    attention_masks  = []

    for item in batch:
        seq_len  = item["input_ids"].size(0)
        pad_len  = max_len - seq_len

        input_ids_padded.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                       item["input_ids"]])
        )
        labels_padded.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long),
                       item["labels"]])
        )
        attention_masks.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long),
                       torch.ones(seq_len, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(input_ids_padded),
        "labels":         torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_masks),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build sample list
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(csv_path: str, text_dir: Path) -> list[dict]:
    """
    For each case in all_cases.csv that also has a TextBraTS report:
      1. Pivot the long-form CSV rows into a region × tag matrix
      2. Serialise the matrix to a natural-language prompt
      3. Load the corresponding report text
    Returns a list of {"case_id", "input", "target"} dicts.
    """
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"  {len(df)} rows  |  {df['case_id'].nunique()} unique cases")

    samples   = []
    skipped   = []

    for case_id, case_df in df.groupby("case_id"):
        report = load_report(text_dir, case_id)
        if report is None:
            skipped.append(case_id)
            continue

        pivot  = pivot_matrix(case_df)
        prompt = matrix_to_prompt(case_id, pivot)

        samples.append({
            "case_id": case_id,
            "input":   prompt,
            "target":  report,
        })

    log.info(f"  Matched pairs  : {len(samples)}")
    log.info(f"  Skipped (no report txt): {len(skipped)}")
    if skipped:
        log.debug(f"  Skipped cases: {skipped[:10]} …")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    text_dir = Path(args.text_dir)
    samples  = build_samples(args.csv, text_dir)

    if len(samples) < 2:
        raise ValueError("Not enough matched (matrix, report) pairs to train.")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )
    log.info(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 tokenizer has no pad token — use eos as pad (standard practice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = BraTSReportDataset(
        train_samples, tokenizer,
        args.max_input_tokens, args.max_target_tokens,
    )
    val_ds = BraTSReportDataset(
        val_samples, tokenizer,
        args.max_input_tokens, args.max_target_tokens,
    )

    _collate = lambda batch: collate_fn(batch, pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Load in BF16 if available (Ampere+), otherwise FP32.
    # Never load in FP16 for training — GradScaler cannot unscale FP16 gradients.
    # BF16 has the same dynamic range as FP32 so it does not need a GradScaler.
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16   # older GPUs without BF16
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    log.info(f"Loading model: {MODEL_NAME}  (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention projections
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    accum_steps   = args.grad_accum
    # scheduler steps once per *effective* batch, not per micro-batch
    total_steps   = (len(train_loader) // accum_steps) * args.epochs
    warmup_steps  = max(1, total_steps // 10)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # GradScaler is only needed for FP16 (not BF16 or FP32)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    best_val_loss = float("inf")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # scale loss by accum_steps so gradients average over the
                # effective batch rather than summing
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps   # undo the /accum_steps for logging
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_ppl      = math.exp(min(avg_train_loss, 20))

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl      = math.exp(min(avg_val_loss, 20))

        log.info(
            f"Epoch {epoch}  "
            f"train_loss={avg_train_loss:.4f}  train_ppl={train_ppl:.2f}  "
            f"val_loss={avg_val_loss:.4f}  val_ppl={val_ppl:.2f}"
        )

        # ── Save best checkpoint ────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            log.info(f"  ✓ New best checkpoint saved → {args.output_dir}")

    log.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Best model saved at: {args.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick inference helper (sanity check after training)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(model, tokenizer, prompt: str, device, max_new_tokens: int = 400):
    """
    Given a matrix prompt string, generate a report using the fine-tuned model.
    Call this after training to sanity-check a single case.
    """
    full_prompt = prompt + PROMPT_SEP
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BioMedLM on BraTS matrix → report")
    # Script lives at:  notebooks/brats23/a2/trainBiomed.py
    # CSV lives at:     notebooks/brats23/a2/atlas_segmentations/segresnet/all_cases.csv
    # TextBraTS at:     notebooks/brats23/TextBraTSData/
    p.add_argument("--csv",
                   default="atlas_segmentations/segresnet/all_cases.csv")
    p.add_argument("--text_dir",
                   default="../TextBraTSData")
    p.add_argument("--output_dir",       default="checkpoints/biomedlm")
    p.add_argument("--epochs",           type=int,   default=15)
    p.add_argument("--batch_size",       type=int,   default=1)
    p.add_argument("--grad_accum",       type=int,   default=4)   # effective batch = 4
    p.add_argument("--lr",               type=float, default=5e-6)
    p.add_argument("--max_input_tokens", type=int,   default=512)   # prompt half of 1024
    p.add_argument("--max_target_tokens",type=int,   default=512)  # report half of 1024
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log.info("=" * 60)
    log.info("BioMedLM fine-tuning — BraTS matrix → radiology report")
    log.info("=" * 60)
    log.info(f"  CSV            : {args.csv}")
    log.info(f"  Text dir       : {args.text_dir}")
    log.info(f"  Output dir     : {args.output_dir}")
    log.info(f"  Epochs         : {args.epochs}")
    log.info(f"  Batch size     : {args.batch_size}  (grad accum={args.grad_accum}, effective={args.batch_size * args.grad_accum})")
    log.info(f"  Learning rate  : {args.lr}")
    log.info(f"  Max input tok  : {args.max_input_tokens}")
    log.info(f"  Max target tok : {args.max_target_tokens}")
    log.info("=" * 60)

    train(args)