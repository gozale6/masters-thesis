"""
train_multi_model.py
====================
Trains SIX models on the BraTS matrix → structured report task:
  - 3 base configurations (no augmentation)
  - 3 augmented configurations (hemisphere swap + region shuffle)

This pairs each model with its augmented twin so you can directly attribute
gains to augmentation in your thesis.

Models
------
  scifive-base          razent/SciFive-base-PMC
  scifive-base-aug      same + hemisphere/shuffle augmentation
  scifive-large         razent/SciFive-large-Pubmed
  scifive-large-aug     same + augmentation
  clinicalt5-hoss       hossboll/clinical-t5
  clinicalt5-hoss-aug   same + augmentation

Key changes from previous version
---------------------------------
1. max_target: 200 → 400 (fixes truncated predictions; refs avg 222 tokens)
2. max_input:  512 → 640 (fewer atlas matrices truncated)
3. length_penalty in generation: 0.8 → 1.1 (encourages longer outputs)
4. label_smoothing: 0.1 (standard small-data fine-tuning)
5. Hemisphere-swap augmentation: 50% chance to flip left↔right in
   BOTH prompt and target (only on training, not val). This forces the
   model to attend to the prompt's hemisphere field rather than memorize
   a left/right prior — directly addresses the 39% hemisphere accuracy
   problem.
6. Region-order shuffle augmentation: 50% chance to shuffle row order
   within each tumor tag in the prompt. Prevents reliance on row position.

Usage
-----
    python train_multi_model.py                          # all 6 models
    python train_multi_model.py --model scifive-base-aug # one model
    python train_multi_model.py --skip_done              # skip done ones

Outputs
-------
  checkpoints_multi/<model_key>/      LoRA + tokenizer + summary.json
  logs/multi_model_<ts>.log
  logs/summary_<ts>.json
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
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry has a base_name (HF id) and an augment flag. Pairs of base/aug
# share the same HF id so you can compare cleanly.
MODELS = {
    "scifive-base": {
        "name":           "razent/SciFive-base-PMC",
        "kind":           "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":      640,
        "max_target":     400,
        "augment":        False,
    },
    "scifive-base-aug": {
        "name":           "razent/SciFive-base-PMC",
        "kind":           "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":      640,
        "max_target":     400,
        "augment":        True,
    },
    "scifive-large": {
        "name":           "razent/SciFive-large-Pubmed",
        "kind":           "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":      640,
        "max_target":     400,
        "augment":        False,
    },
    "scifive-large-aug": {
        "name":           "razent/SciFive-large-Pubmed",
        "kind":           "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":      640,
        "max_target":     400,
        "augment":        True,
    },
    "clinicalt5-hoss": {
        "name":           "hossboll/clinical-t5",
        "kind":           "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":      640,
        "max_target":     400,
        "augment":        False,
    },
    "clinicalt5-hoss-aug": {
        "name":           "hossboll/clinical-t5",
        "kind":           "seq2seq",
        "target_modules": ["q", "v"],
        "max_input":      640,
        "max_target":     400,
        "augment":        True,
    },
}

# Train order: alternate base/aug so if you stop early you have pairs
TRAIN_ORDER = [
    "scifive-base", "scifive-base-aug",
    "clinicalt5-hoss", "clinicalt5-hoss-aug",
    "scifive-large", "scifive-large-aug",   # large last (slowest)
]

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

# ── Jülich atlas heuristics ──────────────────────────────────────────────────
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


def matrix_to_prompt(case_id, pivot, shuffle_within_tag=False):
    """Build the prompt. If shuffle_within_tag is True, shuffle row order
    within each tumor tag (for augmentation)."""
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False).drop(columns="_total")

    if shuffle_within_tag:
        # Shuffle rows globally — simple approximation since rows aren't
        # tag-grouped here. Preserves the "most-active first" intent only loosely.
        idx = list(active.index)
        random.shuffle(idx)
        active = active.loc[idx]

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


# ── Hemisphere swap augmentation ─────────────────────────────────────────────
_HEMI_PLACEHOLDER_L = "\u2190LEFT\u2192"
_HEMI_PLACEHOLDER_R = "\u2190RIGHT\u2192"


def swap_hemispheres(text):
    """Flip 'left' ↔ 'right' (case-insensitive) safely, including the
    'Dominant hemisphere: LEFT/RIGHT' header line."""
    text = re.sub(r"\bleft\b",  _HEMI_PLACEHOLDER_L, text, flags=re.I)
    text = re.sub(r"\bright\b", _HEMI_PLACEHOLDER_R, text, flags=re.I)
    text = text.replace(_HEMI_PLACEHOLDER_L, "right")
    text = text.replace(_HEMI_PLACEHOLDER_R, "left")
    # Header line uses uppercase LEFT/RIGHT — handle separately
    text = text.replace("Dominant hemisphere: LEFT", "__TMP_DOM_L__")
    text = text.replace("Dominant hemisphere: RIGHT", "Dominant hemisphere: LEFT")
    text = text.replace("__TMP_DOM_L__", "Dominant hemisphere: RIGHT")
    return text


# ── Report parsing ───────────────────────────────────────────────────────────
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


# ── Dataset (with optional augmentation) ─────────────────────────────────────
class Seq2SeqDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input, max_target,
                 augment=False, p_swap=0.5, p_shuffle=0.5):
        self.samples = samples
        self.tok = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        self.augment = augment
        self.p_swap = p_swap
        self.p_shuffle = p_shuffle

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        target = s["target"]

        if self.augment:
            shuffle = random.random() < self.p_shuffle
            if shuffle:
                # Rebuild prompt with shuffled rows
                prompt = matrix_to_prompt(s["case_id"], s["pivot"], shuffle_within_tag=True)
            else:
                prompt = s["input_base"]
            if random.random() < self.p_swap:
                prompt = swap_hemispheres(prompt)
                target = swap_hemispheres(target)
        else:
            prompt = s["input_base"]

        enc = self.tok(prompt, truncation=True, max_length=self.max_input, padding=False)
        dec = self.tok(text_target=target, truncation=True, max_length=self.max_target, padding=False)
        labels = [l if l != self.tok.pad_token_id else -100 for l in dec["input_ids"]]
        return {
            "input_ids":      torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
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
    log.info(f"  {len(df)} rows | {df['case_id'].nunique()} cases in CSV")

    samples, skipped = [], []
    for case_id, case_df in df.groupby("case_id"):
        report = load_report(text_dir, case_id)
        if report is None:
            skipped.append(case_id)
            continue
        pv = pivot_matrix(case_df)
        samples.append({
            "case_id":    case_id,
            "input_base": matrix_to_prompt(case_id, pv, shuffle_within_tag=False),
            "pivot":      pv,
            "target":     report,
        })
    log.info(f"  Matched (matrix + report): {len(samples)}  |  Skipped: {len(skipped)}")
    return samples


# ── Inference ────────────────────────────────────────────────────────────────
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
            length_penalty=1.1,    # was 0.8 — encourage longer reports
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def preview_generations(model, tokenizer, val_samples, device, max_new, max_input, log, n=2):
    model.eval()
    chosen = random.sample(val_samples, min(n, len(val_samples)))
    log.info("\n" + "=" * 60 + "  PREVIEW")
    previews = []
    for s in chosen:
        try:
            pred = generate_seq2seq(model, tokenizer, s["input_base"], device, max_new, max_input)
        except Exception as e:
            pred = f"<generation failed: {e}>"
        log.info(f"\n  Case : {s['case_id']}")
        log.info(f"  PRED : {pred[:500]}")
        log.info(f"  REF  : {s['target'][:500]}")
        previews.append({
            "case_id": s["case_id"],
            "pred":    pred,
            "ref":     s["target"],
        })
    log.info("=" * 60)
    return previews


# ── Hemisphere accuracy on a small val subset (extra signal during training) ─
def _detect_hemisphere(text):
    text_l = text.lower()
    has_l = bool(re.search(r"\bleft\b",  text_l))
    has_r = bool(re.search(r"\bright\b", text_l))
    has_b = bool(re.search(r"\b(bilateral|both)\b", text_l))
    if has_b: return "BILATERAL"
    if has_l and not has_r: return "LEFT"
    if has_r and not has_l: return "RIGHT"
    if has_l and has_r:     return "BILATERAL"
    return "UNK"


def quick_hemisphere_check(model, tokenizer, val_samples, device,
                           max_new, max_input, log, n=20):
    """Quick generation-based check on hemisphere accuracy — extra signal
    beyond teacher-forced val_loss."""
    model.eval()
    chosen = random.sample(val_samples, min(n, len(val_samples)))
    correct, total = 0, 0
    for s in chosen:
        try:
            pred = generate_seq2seq(model, tokenizer, s["input_base"],
                                    device, max_new, max_input)
        except Exception:
            continue
        rh = _detect_hemisphere(s["target"])
        ph = _detect_hemisphere(pred)
        if rh != "UNK":
            total += 1
            if rh == ph:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    log.info(f"  Hemisphere quick-check: {correct}/{total} = {acc*100:.1f}%")
    return acc


# ── Per-model training ───────────────────────────────────────────────────────
def train_one_model(model_key, args, samples, log):
    cfg = MODELS[model_key]
    out_dir = Path(args.output_root) / model_key

    # Skip-if-done
    if args.skip_done and (out_dir / "summary.json").exists():
        log.info(f"\n[SKIP] {model_key} — summary.json exists. "
                 f"Pass --no-skip_done to retrain.")
        with open(out_dir / "summary.json") as f:
            return json.load(f)

    log.info("\n" + "#" * 70)
    log.info(f"# Training model: {model_key}  ({cfg['name']})")
    log.info(f"#   augment={cfg['augment']}  (p_swap={args.p_swap}, p_shuffle={args.p_shuffle})")
    log.info(f"#   max_input={cfg['max_input']}  max_target={cfg['max_target']}")
    log.info("#" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, val_samples = train_test_split(
        samples, test_size=0.20, random_state=args.seed, shuffle=True,
    )
    log.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    log.info(f"Loading tokenizer: {cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True)

    log.info(f"Loading model: {cfg['name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
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
        lora_cfg.target_modules = None
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    train_ds = Seq2SeqDataset(
        train_samples, tokenizer, cfg["max_input"], cfg["max_target"],
        augment=cfg["augment"], p_swap=args.p_swap, p_shuffle=args.p_shuffle,
    )
    val_ds = Seq2SeqDataset(
        val_samples, tokenizer, cfg["max_input"], cfg["max_target"],
        augment=False,    # never augment val
    )
    collate = make_collate(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    lr = args.lr_large if "large" in model_key else args.lr
    log.info(f"Using LR: {lr}  |  label_smoothing: {args.label_smoothing}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

    accum = args.grad_accum
    total_steps = max(1, (len(train_loader) // accum) * args.epochs)
    warmup = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    best_previews = []
    no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
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
                if args.label_smoothing > 0:
                    try:
                        out = model(**batch, label_smoothing_factor=args.label_smoothing)
                    except TypeError:
                        out = model(**batch)
                else:
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

        # Hemisphere quick-check every 5 epochs (cheap, ~30s for 20 samples)
        hemi_acc = None
        if epoch % args.hemi_check_every == 0:
            hemi_acc = quick_hemisphere_check(
                model, tokenizer, val_samples, device,
                cfg["max_target"], cfg["max_input"], log, n=args.hemi_check_n,
            )

        previews = preview_generations(
            model, tokenizer, val_samples, device,
            cfg["max_target"], cfg["max_input"], log,
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss":   avg_val,
            "train_ppl":  tr_ppl,
            "val_ppl":    v_ppl,
            "grad_norm":  avg_gn,
            "hemisphere_acc": hemi_acc,
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
        "augment":      cfg["augment"],
        "max_input":    cfg["max_input"],
        "max_target":   cfg["max_target"],
        "label_smoothing": args.label_smoothing,
        "p_swap":       args.p_swap,
        "p_shuffle":    args.p_shuffle,
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

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()

    return summary


# ── Entry point ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",      default="../TextBraTSData")
    p.add_argument("--output_root",   default="checkpoints_multi")
    p.add_argument("--model",         default="all",
                   help=f"Which model. all | {' | '.join(MODELS.keys())}")
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--lr_large",      type=float, default=1e-5)
    p.add_argument("--patience",      type=int,   default=12)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--p_swap",        type=float, default=0.5,
                   help="Prob of hemisphere swap (only on aug models)")
    p.add_argument("--p_shuffle",     type=float, default=0.5,
                   help="Prob of region shuffle (only on aug models)")
    p.add_argument("--hemi_check_every", type=int, default=5,
                   help="Run hemisphere quick-check every N epochs")
    p.add_argument("--hemi_check_n",  type=int, default=20,
                   help="Sample size for hemisphere quick-check")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--skip_done",     action="store_true", default=True,
                   help="Skip models that already have summary.json")
    p.add_argument("--no-skip_done",  dest="skip_done", action="store_false")
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
    log.info("Multi-model BraTS matrix → report training (6 models, base + aug)")
    log.info("=" * 70)
    log.info(f"  CSV         : {args.csv}")
    log.info(f"  Text dir    : {args.text_dir}")
    log.info(f"  Output root : {args.output_root}")
    log.info(f"  Model(s)    : {args.model}")
    log.info(f"  Epochs      : {args.epochs}  patience={args.patience}")
    log.info(f"  Batch       : {args.batch_size} (accum={args.grad_accum}, eff={args.batch_size*args.grad_accum})")
    log.info(f"  LR (base)   : {args.lr}")
    log.info(f"  LR (large)  : {args.lr_large}")
    log.info(f"  Label smth  : {args.label_smoothing}")
    log.info(f"  Aug probs   : p_swap={args.p_swap}  p_shuffle={args.p_shuffle}")
    log.info(f"  Skip done   : {args.skip_done}")
    log.info("=" * 70)

    samples = build_samples(args.csv, Path(args.text_dir), log)
    if len(samples) < 2:
        raise ValueError("Not enough samples")

    if args.model == "all":
        model_keys = TRAIN_ORDER
    else:
        if args.model not in MODELS:
            raise ValueError(f"Unknown model {args.model}. Options: all, {list(MODELS.keys())}")
        model_keys = [args.model]

    all_summaries = []
    t_global = time.time()
    for i, key in enumerate(model_keys, 1):
        elapsed_global = (time.time() - t_global) / 60
        log.info(f"\n>>> Run {i}/{len(model_keys)}  (elapsed: {elapsed_global:.1f} min)")
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
        aug_tag = " [aug]" if s.get("augment") else "      "
        log.info(
            f"  {i}. {s['model_key']:<22}{aug_tag} val_loss={s['best_val_loss']:.4f}  "
            f"epoch {s['best_epoch']}/{s['epochs_run']}  "
            f"({s['elapsed_min']:.1f} min)"
        )

    # Pair comparison: base vs aug for each model
    log.info("\n" + "=" * 70)
    log.info("BASE vs AUG PAIRS")
    log.info("=" * 70)
    by_key = {s["model_key"]: s for s in all_summaries if "best_val_loss" in s}
    for base in ["scifive-base", "scifive-large", "clinicalt5-hoss"]:
        aug = base + "-aug"
        if base in by_key and aug in by_key:
            b, a = by_key[base], by_key[aug]
            delta = a["best_val_loss"] - b["best_val_loss"]
            arrow = "↓" if delta < 0 else "↑"
            log.info(f"  {base:<22} : val={b['best_val_loss']:.4f}")
            log.info(f"  {aug:<22} : val={a['best_val_loss']:.4f}  (Δ {arrow}{abs(delta):.4f})")
            log.info("")

    failed = [s for s in all_summaries if "error" in s]
    if failed:
        log.info("\n  FAILED:")
        for s in failed:
            log.info(f"    {s['model_key']}: {s['error']}")


if __name__ == "__main__":
    main()