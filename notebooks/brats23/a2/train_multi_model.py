"""
train_multi_model.py
====================
Trains multiple candidate models on the BraTS matrix → structured report task
and saves comparable metrics + generation samples for each. Run overnight,
compare in the morning.

Models
------
  1. microsoft/biogpt                  (causal LM, 347M, PubMed-pretrained)
  2. razent/SciFive-base-PMC           (seq2seq T5-base, PMC-pretrained)
  3. razent/SciFive-large-Pubmed       (seq2seq T5-large, PubMed-pretrained)
  4. luqh/ClinicalT5-base              (seq2seq T5-base, clinical notes)
  5. hossboll/clinical-t5              (seq2seq T5, clinical summarization)

Architecture handling
---------------------
  - Seq2seq (T5): input = prompt, target = report; loss on target only.
  - Causal (BioGPT): input = prompt + SEP + report; loss on report tokens only
    (prompt tokens masked with -100).

Usage
-----
    # Train all 5 models with default settings
    python train_multi_model.py

    # Train a specific model
    python train_multi_model.py --model scifive-large

    # Adjust epochs/patience
    python train_multi_model.py --epochs 50 --patience 5

Outputs
-------
  checkpoints_multi/
    biogpt/                  ← LoRA checkpoint
      summary.json           ← best val_loss, epoch, generation samples
    scifive-base/...
    scifive-large/...
    clinicalt5-base/...
    clinicalt5-hoss/...
  logs/
    multi_model_<timestamp>.log
    summary_<timestamp>.json    ← side-by-side comparison
"""

import argparse
import json
import logging
import math
import os
import random
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Model registry ────────────────────────────────────────────────────────────
# kind: "seq2seq" or "causal"
# target_modules: LoRA target module names (differ by architecture)
MODELS = {
    "biogpt": {
        "name":       "microsoft/biogpt",
        "kind":       "causal",
        "target_modules": ["q_proj", "v_proj"],
        "max_input":  768,     # causal needs prompt+target in same window
        "max_target": 200,
    },
    "scifive-base": {
        "name":       "razent/SciFive-base-PMC",
        "kind":       "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":  512,
        "max_target": 200,
    },
    "scifive-large": {
        "name":       "razent/SciFive-large-Pubmed",
        "kind":       "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":  512,
        "max_target": 200,
    },
    "clinicalt5-base": {
        "name":       "luqh/ClinicalT5-base",
        "kind":       "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":  512,
        "max_target": 200,
    },
    "clinicalt5-hoss": {
        "name":       "hossboll/clinical-t5",
        "kind":       "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":  512,
        "max_target": 200,
    },
}

# ── Constants ─────────────────────────────────────────────────────────────────
TASK_PREFIX = (
    "You are a neuroradiology assistant. "
    "Given a brain tumor atlas matrix, write a structured radiology report "
    "using EXACTLY this format (4 lines, each starting with the field name):\n"
    "Lesion area: [lobe(s), hemisphere, signal characteristics]\n"
    "Edema: [location and extent]\n"
    "Necrosis: [location and signal, or 'not observed']\n"
    "Ventricular compression: [description, or 'not observed']\n\n"
    "Atlas matrix:\n"
)

REPORT_FORMAT = (
    "Lesion area: {lesion}\n"
    "Edema: {edema}\n"
    "Necrosis: {necrosis}\n"
    "Ventricular compression: {compression}"
)

# Separator token between prompt and target for causal training
CAUSAL_SEP = "\n\nReport:\n"

# ── Jülich atlas heuristics (unchanged from v4) ──────────────────────────────
_LEFT_RE      = re.compile(r"\b(left|_L_|Left)\b",       re.I)
_RIGHT_RE     = re.compile(r"\b(right|_R_|Right)\b",     re.I)
_BILATERAL_RE = re.compile(r"\b(bilateral|_B_|both)\b",  re.I)

_LOBE_KEYWORDS = {
    "frontal":    re.compile(r"frontal",   re.I),
    "parietal":   re.compile(r"parietal",  re.I),
    "temporal":   re.compile(r"temporal",  re.I),
    "occipital":  re.compile(r"occipital", re.I),
    "insula":     re.compile(r"insul",     re.I),
    "cerebellum": re.compile(r"cerebell",  re.I),
    "brainstem":  re.compile(r"brain.?stem|pons|medulla|midbrain", re.I),
    "basal ganglia": re.compile(r"basal|putamen|caudate|pallidum|thalamus", re.I),
}


def infer_hemisphere(active_regions):
    l = sum(1 for r in active_regions if _LEFT_RE.search(r))
    r_ = sum(1 for r in active_regions if _RIGHT_RE.search(r))
    b = sum(1 for r in active_regions if _BILATERAL_RE.search(r))
    if b > 0 or (l > 0 and r_ > 0):
        return "BILATERAL"
    if l > r_: return "LEFT"
    if r_ > l: return "RIGHT"
    return "UNSPECIFIED"


def infer_primary_lobes(active_regions, pivot):
    scores = {}
    for region_name, row in pivot.iterrows():
        wt = float(row.get("WT", 0))
        if wt <= 0:
            continue
        for lobe, pat in _LOBE_KEYWORDS.items():
            if pat.search(str(region_name)):
                scores[lobe] = scores.get(lobe, 0) + wt
    if not scores:
        return "unspecified"
    top = sorted(scores, key=scores.get, reverse=True)[:2]
    return ", ".join(top)


def pivot_matrix(case_df):
    pv = case_df.pivot_table(
        index="atlas_region_name",
        columns="tumor_tag",
        values="count_scaled",
        aggfunc="max",
        fill_value=0,
    )
    for tag in ["ET", "TC", "WT"]:
        if tag not in pv.columns:
            pv[tag] = 0
    return pv[["ET", "TC", "WT"]]


def matrix_to_prompt(case_id, pivot):
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False).drop(columns="_total")

    region_names = [str(r) for r in active.index]
    hemisphere   = infer_hemisphere(region_names)
    primary      = infer_primary_lobes(region_names, active)

    lines = [
        f"Patient: {case_id}",
        f"Dominant hemisphere: {hemisphere}",
        f"Primary lobe(s): {primary}",
        "Atlas involvement (scale 0-256):",
    ]
    for region_name, row in active.iterrows():
        parts = []
        if row["ET"] > 0: parts.append(f"ET={int(row['ET'])}")
        if row["TC"] > 0: parts.append(f"TC={int(row['TC'])}")
        if row["WT"] > 0: parts.append(f"WT={int(row['WT'])}")
        lines.append(f"  - {region_name}: {', '.join(parts)}")
    lines.append("\nET=Enhancing Tumor  TC=Tumor Core/necrosis  WT=Whole Tumor/edema")
    return TASK_PREFIX + "\n".join(lines)


# ── Report parsing (unchanged) ────────────────────────────────────────────────
_FIELD_PATTERNS = {
    "lesion":      re.compile(r"(?:the\s+)?lesion\s+area\s*(?:is\s+in\s+|[:\-]\s*)(.+?)(?=\nedema|\nnecrosis|\nventricular|$)",  re.I | re.S),
    "edema":       re.compile(r"edema\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nnecrosis|\nventricular|$)",                         re.I | re.S),
    "necrosis":    re.compile(r"necrosis\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nventricular|$)",                         re.I | re.S),
    "compression": re.compile(r"ventricular\s+compression\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nnecrosis|$)",           re.I | re.S),
}


def clean_text(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    fixes = [
        (r"\blobes?ts\b", "lobes"), (r"\bleisure\b", "lesion"), (r"\bleish\b", "lesion"),
        (r"\blobists\b", "lobes"), (r"\blobels\b", "lobes"), (r"\blobestal\b", "lobe"),
        (r"\bparticeal\b", "parietal"), (r"\blession\b", "lesion"),
        (r"\bleocytes\b", "lesion"), (r"\bleon\b", "lesion"),
        (r"\bparaplegic\b", "parahippocampal"), (r"\bparafacial\b", "parahippocampal"),
    ]
    for pat, rep in fixes:
        text = re.sub(pat, rep, text, flags=re.I)
    text = re.sub(r"\b[bcdfghjklmnpqrstvwxyz]{6,}\b", "", text, flags=re.I)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def parse_report(raw):
    text = clean_text(raw)
    fields = {}
    for key, pat in _FIELD_PATTERNS.items():
        m = pat.search(text)
        fields[key] = m.group(1).strip().rstrip(".") if m else None
    if sum(1 for v in fields.values() if v) >= 2:
        return REPORT_FORMAT.format(
            lesion      = fields["lesion"]      or "not observed",
            edema       = fields["edema"]       or "not observed",
            necrosis    = fields["necrosis"]    or "not observed",
            compression = fields["compression"] or "not observed",
        )
    return text


def load_report(text_dir, case_id):
    case_dir = text_dir / case_id
    txt_path = case_dir / f"{case_id}_flair_text.txt"
    if txt_path.exists():
        t = txt_path.read_text(encoding="utf-8").strip()
        if t:
            return parse_report(t)
    npy_path = case_dir / f"{case_id}_flair_text.npy"
    if npy_path.exists():
        import numpy as np
        arr = np.load(npy_path, allow_pickle=True)
        t = str(arr.item()).strip() if arr.ndim == 0 else " ".join(
            str(s).strip() for s in arr.tolist() if str(s).strip()
        )
        if t:
            return parse_report(t)
    return None


# ── Dataset classes ──────────────────────────────────────────────────────────

class Seq2SeqDataset(Dataset):
    """For T5-family models: encoder sees prompt, decoder produces target."""
    def __init__(self, samples, tokenizer, max_input, max_target):
        self.samples = samples
        self.tok = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tok(s["input_base"], truncation=True, max_length=self.max_input, padding=False)
        dec = self.tok(text_target=s["target"], truncation=True, max_length=self.max_target, padding=False)
        labels = [l if l != self.tok.pad_token_id else -100 for l in dec["input_ids"]]
        return {
            "input_ids":      torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


class CausalDataset(Dataset):
    """For BioGPT etc.: concat [prompt + SEP + target], mask prompt in labels."""
    def __init__(self, samples, tokenizer, max_input, max_target):
        self.samples = samples
        self.tok = tokenizer
        self.max_total = max_input  # total budget for prompt + target
        self.max_target = max_target
        # Ensure pad token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt_part = s["input_base"] + CAUSAL_SEP
        target_part = s["target"] + self.tok.eos_token

        # Tokenize separately so we know where prompt ends
        prompt_ids = self.tok(prompt_part, add_special_tokens=False)["input_ids"]
        target_ids = self.tok(target_part, add_special_tokens=False)["input_ids"]

        # Budget: keep full target (truncate left), truncate prompt if needed
        budget = self.max_total
        if len(target_ids) > self.max_target:
            target_ids = target_ids[: self.max_target]
        max_prompt = budget - len(target_ids)
        if len(prompt_ids) > max_prompt:
            # keep tail of prompt so task prefix at start is lost before matrix rows
            prompt_ids = prompt_ids[-max_prompt:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        attention = [1] * len(input_ids)

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


def make_collate(pad_token_id):
    def collate(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        max_lbl = max(item["labels"].size(0) for item in batch)
        ids, att, lbls = [], [], []
        for item in batch:
            ip = max_len - item["input_ids"].size(0)
            lp = max_lbl - item["labels"].size(0)
            ids.append(torch.cat([item["input_ids"], torch.full((ip,), pad_token_id, dtype=torch.long)]))
            att.append(torch.cat([item["attention_mask"], torch.zeros(ip, dtype=torch.long)]))
            lbls.append(torch.cat([item["labels"], torch.full((lp,), -100, dtype=torch.long)]))
        return {
            "input_ids":      torch.stack(ids),
            "attention_mask": torch.stack(att),
            "labels":         torch.stack(lbls),
        }
    return collate


# ── Data building ────────────────────────────────────────────────────────────

_REQUIRED_CSV_COLS = {"case_id", "atlas_region_name", "tumor_tag", "count_scaled"}


def build_samples(csv_path, text_dir, log):
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    log.info(f"  {len(df)} rows | {df['case_id'].nunique()} cases")

    samples, skipped = [], []
    for case_id, case_df in df.groupby("case_id"):
        report = load_report(text_dir, case_id)
        if report is None:
            skipped.append(case_id)
            continue
        pv = pivot_matrix(case_df)
        samples.append({
            "case_id":    case_id,
            "input_base": matrix_to_prompt(case_id, pv),
            "pivot":      pv,
            "target":     report,
        })
    log.info(f"  Matched: {len(samples)} | Skipped: {len(skipped)}")
    return samples


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_seq2seq(model, tokenizer, prompt, device, max_new_tokens, max_input):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            length_penalty=0.8,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def generate_causal(model, tokenizer, prompt, device, max_new_tokens, max_input):
    full_prompt = prompt + CAUSAL_SEP
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_input - max_new_tokens).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Strip the prompt portion, decode only new tokens
    new_tokens = out[0][inputs["input_ids"].size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def preview_generations(model, tokenizer, val_samples, device, kind, max_new, max_input, log, n=3):
    model.eval()
    chosen = random.sample(val_samples, min(n, len(val_samples)))
    log.info("\n" + "=" * 60 + "  PREVIEW")
    previews = []
    for s in chosen:
        try:
            if kind == "seq2seq":
                pred = generate_seq2seq(model, tokenizer, s["input_base"], device, max_new, max_input)
            else:
                pred = generate_causal(model, tokenizer, s["input_base"], device, max_new, max_input)
        except Exception as e:
            pred = f"<generation failed: {e}>"
        log.info(f"\n  Case : {s['case_id']}")
        log.info(f"  PRED : {pred[:400]}")
        log.info(f"  REF  : {s['target'][:400]}")
        previews.append({
            "case_id": s["case_id"],
            "pred":    pred,
            "ref":     s["target"],
        })
    log.info("=" * 60)
    return previews


# ── Training loop per model ───────────────────────────────────────────────────

def train_one_model(model_key, args, samples, log):
    cfg = MODELS[model_key]
    log.info("\n" + "#" * 70)
    log.info(f"# Training model: {model_key}  ({cfg['name']})  kind={cfg['kind']}")
    log.info("#" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, val_samples = train_test_split(
        samples, test_size=0.20, random_state=args.seed, shuffle=True,
    )
    log.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # ── Tokenizer & model ────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True)

    log.info(f"Loading model: {cfg['name']}")
    if cfg["kind"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)
        task_type = TaskType.SEQ_2_SEQ_LM
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)
        task_type = TaskType.CAUSAL_LM
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=task_type,
        r=32,
        lora_alpha=64,
        target_modules=cfg["target_modules"],
        lora_dropout=0.1,
        bias="none",
    )
    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as e:
        log.warning(f"LoRA target {cfg['target_modules']} failed: {e}")
        log.warning("Falling back to auto-detected linear layers.")
        # Let PEFT pick reasonable defaults
        lora_cfg.target_modules = None
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    if cfg["kind"] == "seq2seq":
        DSClass = Seq2SeqDataset
    else:
        DSClass = CausalDataset
    train_ds = DSClass(train_samples, tokenizer, cfg["max_input"], cfg["max_target"])
    val_ds   = DSClass(val_samples,   tokenizer, cfg["max_input"], cfg["max_target"])
    collate  = make_collate(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    accum = args.grad_accum
    total_steps = max(1, (len(train_loader) // accum) * args.epochs)
    warmup = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.output_root) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    best_previews = []
    no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        grad_sum = 0.0
        n_upd = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"[{model_key}] E{epoch}/{args.epochs} train")
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            if (batch["labels"] != -100).sum() == 0:
                continue
            try:
                out = model(**batch)
                loss = out.loss / accum
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                log.warning(f"  OOM on step {step}, skipping batch")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            if step % accum == 0 or step == len(train_loader):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_sum += gn.item()
                n_upd += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}")

        avg_train = train_loss / max(1, len(train_loader))
        avg_gn    = grad_sum / max(1, n_upd)
        tr_ppl    = math.exp(min(avg_train, 20)) if math.isfinite(avg_train) else float("nan")

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{model_key}] E{epoch}/{args.epochs} val  "):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                if torch.isfinite(out.loss):
                    val_loss += out.loss.item()
        avg_val = val_loss / max(1, len(val_loader))
        v_ppl   = math.exp(min(avg_val, 20)) if math.isfinite(avg_val) else float("nan")

        log.info(
            f"[{model_key}] E{epoch:>3}  train={avg_train:.4f} (ppl {tr_ppl:.2f})  "
            f"val={avg_val:.4f} (ppl {v_ppl:.2f})  gn={avg_gn:.3f}"
        )

        previews = preview_generations(
            model, tokenizer, val_samples, device,
            cfg["kind"], cfg["max_target"], cfg["max_input"], log,
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss":   avg_val,
            "train_ppl":  tr_ppl,
            "val_ppl":    v_ppl,
            "grad_norm":  avg_gn,
        })

        if math.isfinite(avg_val) and avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_previews = previews
            no_improve = 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            log.info(f"  ✓ New best ({best_val:.4f}) → {out_dir}")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                log.info("Early stopping.")
                break

    elapsed = time.time() - t0
    summary = {
        "model_key":    model_key,
        "model_name":   cfg["name"],
        "kind":         cfg["kind"],
        "best_val_loss": best_val,
        "best_epoch":   best_epoch,
        "epochs_run":   len(history),
        "elapsed_sec":  elapsed,
        "elapsed_min":  elapsed / 60,
        "history":      history,
        "best_previews": best_previews,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[{model_key}] done in {elapsed/60:.1f} min. Best val={best_val:.4f} at epoch {best_epoch}")

    # Free memory before next model
    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           default="atlas_segmentations/segresnet/all_cases.csv")
    p.add_argument("--text_dir",      default="../TextBraTSData")
    p.add_argument("--output_root",   default="checkpoints_multi")
    p.add_argument("--model",         default="all",
                   help=f"Which model to train. One of: all | {' | '.join(MODELS.keys())}")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--patience",      type=int,   default=6)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"multi_model_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Multi-model BraTS matrix → report training")
    log.info("=" * 70)
    log.info(f"  CSV         : {args.csv}")
    log.info(f"  Text dir    : {args.text_dir}")
    log.info(f"  Output root : {args.output_root}")
    log.info(f"  Epochs      : {args.epochs}  patience={args.patience}")
    log.info(f"  Batch       : {args.batch_size} (accum={args.grad_accum}, eff={args.batch_size*args.grad_accum})")
    log.info(f"  LR          : {args.lr}")
    log.info(f"  Model(s)    : {args.model}")
    log.info("=" * 70)

    samples = build_samples(args.csv, Path(args.text_dir), log)
    if len(samples) < 2:
        raise ValueError("Not enough samples")

    # Which models to train
    if args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        if args.model not in MODELS:
            raise ValueError(f"Unknown model {args.model}. Options: all, {list(MODELS.keys())}")
        model_keys = [args.model]

    all_summaries = []
    for key in model_keys:
        try:
            s = train_one_model(key, args, samples, log)
            all_summaries.append(s)
        except Exception as e:
            log.exception(f"[{key}] training FAILED: {e}")
            all_summaries.append({
                "model_key": key,
                "model_name": MODELS[key]["name"],
                "error": str(e),
            })
            torch.cuda.empty_cache()

    # ── Write comparative summary ────────────────────────────────────────────
    out_path = Path("logs") / f"summary_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"\nComparison summary → {out_path}")

    log.info("\n" + "=" * 70)
    log.info("FINAL RANKING BY BEST VAL LOSS")
    log.info("=" * 70)
    rank = sorted(
        [s for s in all_summaries if "best_val_loss" in s and math.isfinite(s["best_val_loss"])],
        key=lambda x: x["best_val_loss"],
    )
    for i, s in enumerate(rank, 1):
        log.info(
            f"  {i}. {s['model_key']:<20} val_loss={s['best_val_loss']:.4f}  "
            f"epoch {s['best_epoch']}/{s['epochs_run']}  "
            f"({s['elapsed_min']:.1f} min)"
        )
    failed = [s for s in all_summaries if "error" in s]
    if failed:
        log.info("\n  FAILED:")
        for s in failed:
            log.info(f"    {s['model_key']}: {s['error']}")


if __name__ == "__main__":
    main()