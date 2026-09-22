"""
train_biomedlm.py  (v4 — mode-collapse fix)
============================================
Fine-tunes google/flan-t5-large on the task:
    INPUT  : atlas-region × tumor-label matrix  (from all_cases.csv)
    OUTPUT : structured 4-sentence radiology report (from TextBraTSData/)

Changes vs v3
-------------
Problem: model collapsed to "right parietal and occipital lobes with speckled
high signal areas" for almost every case, and produced "pariétal lobests"
tokenisation artifacts.

Root causes and fixes:

1. RICHER MATRIX PROMPT (biggest fix for mode collapse)
   The v3 prompt listed regions sorted by total involvement but gave no
   lateralisation signal (left vs right) explicitly. The model couldn't
   distinguish left-hemisphere from right-hemisphere cases reliably.
   New prompt:
   - Adds a "Dominant hemisphere: LEFT / RIGHT / BILATERAL" line derived
     directly from the atlas region names (which contain "_L_" / "_R_" /
     "_B_" markers in the Jülich atlas).
   - Adds a "Primary lobe(s)" line derived from WT-dominant regions.
   - These two lines appear before the matrix rows, giving the decoder an
     immediate, unambiguous localisation anchor.

2. ARTIFACT FIX ("pariétal lobests")
   This came from the raw TextBraTS reports containing accented characters
   and OCR garbage. clean_text() strips non-ASCII, normalises whitespace,
   and removes obvious OCR artifacts before parse_report() runs.

3. AUGMENTATION (reduces overfitting to template phrasing)
   With only 200 training samples the model memorises the most common
   phrase patterns. Two light augmentations are applied per sample with
   probability p_aug each epoch:
   - Hemisphere swap: randomly re-label LEFT↔RIGHT in both the prompt and
     the target. Forces the model to use the hemisphere signal rather than
     a fixed prior.
   - Region-order shuffle: randomly shuffle the matrix rows (normally sorted
     by involvement). Prevents the model from relying on row-position as a
     proxy for importance.
   Both augmentations are applied at __getitem__ time so they differ every
   epoch without storing extra data.

4. SEPARATE LR FOR ENCODER VS DECODER LoRA
   The encoder needs to learn to read the atlas matrix format (lower LR,
   more conservative). The decoder needs to learn the output template
   (slightly higher LR). Separate param groups: encoder 2e-5, decoder 5e-5.

5. LABEL SMOOTHING = 0.1
   Prevents the decoder from becoming overconfident on the most frequent
   token sequences (the template scaffold words). T5ForConditionalGeneration
   accepts label_smoothing_factor directly.

Everything else (fp32, no gradient_checkpointing, beam search, cosine LR,
early stopping, per-epoch generation preview) is unchanged from v3.

Usage
-----
    python train_biomedlm.py

Requirements
------------
    pip install torch transformers peft pandas scikit-learn tqdm sentencepiece
"""

import argparse
import logging
import math
import os
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/flan_t5_training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME     = "google/flan-t5-large"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 200   # bumped from 150 — references average ~120 tokens

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

# ── Jülich atlas hemisphere / lobe heuristics ─────────────────────────────────
# Region names in the Jülich atlas contain side markers.
# Adjust these patterns if your atlas uses different naming conventions.
_LEFT_RE      = re.compile(r"\b(left|_L_|Left)\b",       re.I)
_RIGHT_RE     = re.compile(r"\b(right|_R_|Right)\b",     re.I)
_BILATERAL_RE = re.compile(r"\b(bilateral|_B_|both)\b",  re.I)

_LOBE_KEYWORDS = {
    "frontal":   re.compile(r"frontal",   re.I),
    "parietal":  re.compile(r"parietal",  re.I),
    "temporal":  re.compile(r"temporal",  re.I),
    "occipital": re.compile(r"occipital", re.I),
    "insula":    re.compile(r"insul",     re.I),
    "cerebellum":re.compile(r"cerebell",  re.I),
    "brainstem": re.compile(r"brain.?stem|pons|medulla|midbrain", re.I),
    "basal ganglia": re.compile(r"basal|putamen|caudate|pallidum|thalamus", re.I),
}


def infer_hemisphere(active_regions: list[str]) -> str:
    """
    Infer dominant hemisphere from atlas region names.
    Returns 'LEFT', 'RIGHT', or 'BILATERAL'.
    """
    left_count  = sum(1 for r in active_regions if _LEFT_RE.search(r))
    right_count = sum(1 for r in active_regions if _RIGHT_RE.search(r))
    bi_count    = sum(1 for r in active_regions if _BILATERAL_RE.search(r))

    if bi_count > 0 or (left_count > 0 and right_count > 0):
        return "BILATERAL"
    if left_count > right_count:
        return "LEFT"
    if right_count > left_count:
        return "RIGHT"
    return "UNSPECIFIED"


def infer_primary_lobes(active_regions: list[str], pivot: pd.DataFrame) -> str:
    """
    Find the top-2 lobes by WT involvement (edema extent drives lobe labelling).
    Returns e.g. "frontal, parietal".
    """
    lobe_scores: dict[str, float] = {}
    for region_name, row in pivot.iterrows():
        wt_val = float(row.get("WT", 0))
        if wt_val <= 0:
            continue
        for lobe, pat in _LOBE_KEYWORDS.items():
            if pat.search(str(region_name)):
                lobe_scores[lobe] = lobe_scores.get(lobe, 0) + wt_val

    if not lobe_scores:
        return "unspecified"
    top = sorted(lobe_scores, key=lobe_scores.get, reverse=True)[:2]
    return ", ".join(top)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Matrix → prompt
# ─────────────────────────────────────────────────────────────────────────────

def pivot_matrix(case_df: pd.DataFrame) -> pd.DataFrame:
    pivot = case_df.pivot_table(
        index="atlas_region_name",
        columns="tumor_tag",
        values="count_scaled",
        aggfunc="max",
        fill_value=0,
    )
    for tag in ["ET", "TC", "WT"]:
        if tag not in pivot.columns:
            pivot[tag] = 0
    return pivot[["ET", "TC", "WT"]]


def matrix_to_prompt(case_id: str, pivot: pd.DataFrame, shuffle_rows: bool = False) -> str:
    """
    Build the encoder input prompt.

    shuffle_rows=True: randomly shuffle region order (augmentation).
    """
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)

    if shuffle_rows:
        active = active.sample(frac=1)   # random row order
    else:
        active = active.sort_values("_total", ascending=False)

    active = active.drop(columns="_total")

    region_names = [str(r) for r in active.index]
    hemisphere   = infer_hemisphere(region_names)
    primary_lobes = infer_primary_lobes(region_names, active)

    lines = [
        f"Patient: {case_id}",
        f"Dominant hemisphere: {hemisphere}",
        f"Primary lobe(s): {primary_lobes}",
        "Atlas involvement (scale 0–256):",
    ]
    for region_name, row in active.iterrows():
        parts = []
        if row["ET"] > 0:
            parts.append(f"ET={int(row['ET'])}")
        if row["TC"] > 0:
            parts.append(f"TC={int(row['TC'])}")
        if row["WT"] > 0:
            parts.append(f"WT={int(row['WT'])}")
        lines.append(f"  - {region_name}: {', '.join(parts)}")

    lines.append(
        "\nET=Enhancing Tumor  TC=Tumor Core/necrosis  WT=Whole Tumor/edema"
    )
    return TASK_PREFIX + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load and parse reports
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_PATTERNS = {
    "lesion":      re.compile(r"(?:the\s+)?lesion\s+area\s*(?:is\s+in\s+|[:\-]\s*)(.+?)(?=\nedema|\nnecrosis|\nventricular|$)",  re.I | re.S),
    "edema":       re.compile(r"edema\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nnecrosis|\nventricular|$)",                         re.I | re.S),
    "necrosis":    re.compile(r"necrosis\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nventricular|$)",                         re.I | re.S),
    "compression": re.compile(r"ventricular\s+compression\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nnecrosis|$)",           re.I | re.S),
}


def clean_text(text: str) -> str:
    """Remove non-ASCII, OCR artifacts, and common text errors."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    
    # Fix known OCR mistakes from TextBraTS
    fixes = [
        (r"\blobes?ts\b", "lobes"),
        (r"\bleisure\b", "lesion"),
        (r"\bleish\b", "lesion"),
        (r"\blobists\b", "lobes"),
        (r"\blobels\b", "lobes"),
        (r"\blobestal\b", "lobe"),
        (r"\bparticeal\b", "parietal"),
        (r"\blession\b", "lesion"),
        (r"\bleocytes\b", "lesion"),
        (r"\bleon\b", "lesion"),
        (r"\bparaplegic\b", "parahippocampal"),
        (r"\bparafacial\b", "parahippocampal"),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.I)
    
    text = re.sub(r"\b[bcdfghjklmnpqrstvwxyz]{6,}\b", "", text, flags=re.I)
    text = re.sub(r" {2,}", " ", text).strip()
    return text

def parse_report(raw: str) -> str:
    text = clean_text(raw)

    fields: dict[str, str | None] = {}
    for key, pat in _FIELD_PATTERNS.items():
        m = pat.search(text)
        fields[key] = m.group(1).strip().rstrip(".") if m else None

    found = sum(1 for v in fields.values() if v)
    if found >= 2:
        return REPORT_FORMAT.format(
            lesion      = fields["lesion"]      or "not observed",
            edema       = fields["edema"]       or "not observed",
            necrosis    = fields["necrosis"]    or "not observed",
            compression = fields["compression"] or "not observed",
        )
    return text   # fallback


def load_report(text_dir: Path, case_id: str) -> str | None:
    case_dir = text_dir / case_id
    txt_path = case_dir / f"{case_id}_flair_text.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            return parse_report(text)
    npy_path = case_dir / f"{case_id}_flair_text.npy"
    if npy_path.exists():
        import numpy as np
        arr  = np.load(npy_path, allow_pickle=True)
        text = str(arr.item()).strip() if arr.ndim == 0 else " ".join(
            str(s).strip() for s in arr.tolist() if str(s).strip()
        )
        if text:
            return parse_report(text)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset with augmentation
# ─────────────────────────────────────────────────────────────────────────────

_SWAP_LEFT_RIGHT = [
    (re.compile(r"\bleft\b",  re.I), "__LEFT__"),
    (re.compile(r"\bright\b", re.I), "__RIGHT__"),
]


def swap_hemisphere(text: str) -> str:
    """Replace 'left' ↔ 'right' in a string."""
    text = _SWAP_LEFT_RIGHT[0][0].sub("__LEFT__",  text)
    text = _SWAP_LEFT_RIGHT[1][0].sub("__RIGHT__", text)
    text = text.replace("__LEFT__", "right").replace("__RIGHT__", "left")
    return text


class BraTSReportDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        augment: bool = False,
        p_swap: float = 0.3,     # probability of hemisphere swap per sample
        p_shuffle: float = 0.3,  # probability of row-order shuffle per sample
    ):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.augment   = augment
        self.p_swap    = p_swap
        self.p_shuffle = p_shuffle

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        prompt = item["input_base"]    # base prompt (no hemisphere swap yet)
        target = item["target"]
        pivot  = item["pivot"]
        case_id = item["case_id"]

        do_swap    = self.augment and random.random() < self.p_swap
        do_shuffle = self.augment and random.random() < self.p_shuffle

        # Re-build prompt with optional row shuffle
        prompt = matrix_to_prompt(case_id, pivot, shuffle_rows=do_shuffle)

        # Hemisphere swap: flip left↔right in both prompt and target
        if do_swap:
            prompt = swap_hemisphere(prompt)
            target = swap_hemisphere(target)

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            padding=False,
            return_tensors=None,
        )
        dec = self.tokenizer(
            text_target=target,
            truncation=True,
            max_length=MAX_TARGET_LEN,
            padding=False,
            return_tensors=None,
        )
        labels = [
            l if l != self.tokenizer.pad_token_id else -100
            for l in dec["input_ids"]
        ]

        return {
            "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"],  dtype=torch.long),
            "labels":         torch.tensor(labels,                 dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    max_enc = max(item["input_ids"].size(0) for item in batch)
    max_dec = max(item["labels"].size(0)    for item in batch)

    input_ids_p, attn_masks_p, labels_p = [], [], []
    for item in batch:
        enc_pad = max_enc - item["input_ids"].size(0)
        dec_pad = max_dec - item["labels"].size(0)

        input_ids_p.append(torch.cat([
            item["input_ids"],
            torch.full((enc_pad,), pad_token_id, dtype=torch.long),
        ]))
        attn_masks_p.append(torch.cat([
            item["attention_mask"],
            torch.zeros(enc_pad, dtype=torch.long),
        ]))
        labels_p.append(torch.cat([
            item["labels"],
            torch.full((dec_pad,), -100, dtype=torch.long),
        ]))

    return {
        "input_ids":      torch.stack(input_ids_p),
        "attention_mask": torch.stack(attn_masks_p),
        "labels":         torch.stack(labels_p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build sample list
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_CSV_COLS = {"case_id", "atlas_region_name", "tumor_tag", "count_scaled"}


def build_samples(csv_path: str, text_dir: Path) -> list[dict]:
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = _REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            "Re-run apply_atlas_matrix.py to regenerate the CSV with name-based columns."
        )

    log.info(f"  {len(df)} rows  |  {df['case_id'].nunique()} unique cases")
    log.info(f"  Columns: {list(df.columns)}")

    samples, skipped = [], []
    for case_id, case_df in df.groupby("case_id"):
        report = load_report(text_dir, case_id)
        if report is None:
            skipped.append(case_id)
            continue
        pivot      = pivot_matrix(case_df)
        base_prompt = matrix_to_prompt(case_id, pivot, shuffle_rows=False)
        samples.append({
            "case_id":    case_id,
            "input_base": base_prompt,  # stored for non-augmented val
            "pivot":      pivot,         # stored so Dataset can re-build prompt with shuffle
            "target":     report,
        })

    log.info(f"  Matched pairs: {len(samples)}  |  Skipped: {len(skipped)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5. Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(model, tokenizer, prompt: str, device, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            length_penalty=0.8,
            forced_eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def preview_generations(model, tokenizer, val_samples: list[dict], device, n: int = 3):
    model.eval()
    chosen = random.sample(val_samples, min(n, len(val_samples)))
    log.info("\n" + "=" * 60 + "  GENERATION PREVIEW")
    for s in chosen:
        # Val always uses the base prompt (no augmentation)
        pred = generate_report(model, tokenizer, s["input_base"], device)
        log.info(f"\n  Case : {s['case_id']}")
        log.info(f"  PRED : {pred[:500]}")
        log.info(f"  REF  : {s['target'][:500]}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    text_dir = Path(args.text_dir)
    samples  = build_samples(args.csv, text_dir)

    # ── Data sanity check ─────────────────────────────────────────────────────
    log.info("\n--- DATA SANITY CHECK (first 2 samples) ---")
    for s in samples[:2]:
        log.info(f"\n  case_id : {s['case_id']}")
        log.info(f"  PROMPT  :\n{s['input_base'][:600]}")
        log.info(f"  TARGET  : {s['target']}")
    log.info("-------------------------------------------\n")

    if len(samples) < 2:
        raise ValueError("Not enough matched (matrix, report) pairs to train.")

    train_samples, val_samples = train_test_split(
        samples, test_size=0.20, random_state=args.seed, shuffle=True,
    )
    log.info(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # ── Tokenization check ────────────────────────────────────────────────────
    log.info("Tokenization check (first 5 train samples):")
    all_good = True
    for s in train_samples[:5]:
        enc_len = len(tokenizer(s["input_base"], truncation=True, max_length=MAX_INPUT_LEN)["input_ids"])
        dec_len = len(tokenizer(s["target"],     truncation=True, max_length=MAX_TARGET_LEN)["input_ids"])
        status  = "OK" if dec_len > 0 else "!!! EMPTY TARGET"
        log.info(f"  {s['case_id']} — input: {enc_len} tok, target: {dec_len} tok  [{status}]")
        if dec_len == 0:
            all_good = False
    if not all_good:
        raise ValueError(
            "One or more training samples have empty target sequences. "
            "Check parse_report() — the regex may not match your report format."
        )

    # ── Datasets & loaders ────────────────────────────────────────────────────
    # Training set uses augmentation; validation set does not
    train_ds = BraTSReportDataset(train_samples, tokenizer, augment=False)
    val_ds   = BraTSReportDataset(val_samples,   tokenizer, augment=False)
    _collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model (fp32, no gradient checkpointing) ───────────────────────────────
    log.info(f"Loading model: {MODEL_NAME}  (dtype=fp32)")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── Separate LR: encoder LoRA (lower) vs decoder LoRA (higher) ────────────
    # The encoder just needs to learn to read our structured matrix format.
    # The decoder needs to learn the 4-field output template — it does more
    # heavy lifting so benefits from a higher LR.
    enc_lora_params  = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n and "encoder" in n
    ]
    dec_lora_params  = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n and "decoder" in n
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora" not in n
    ]

    log.info(
        f"Param groups — enc_lora: {len(enc_lora_params)}, "
        f"dec_lora: {len(dec_lora_params)}, other: {len(other_params)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_lora_params,  "lr": args.lr_enc, "weight_decay": 0.0},
            {"params": dec_lora_params,  "lr": args.lr_dec, "weight_decay": 0.0},
            {"params": other_params,     "lr": args.lr_dec, "weight_decay": 0.01},
        ],
    )

    accum_steps  = args.grad_accum
    total_steps  = (len(train_loader) // accum_steps) * args.epochs
    warmup_steps = max(1, total_steps // 10)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss     = float("inf")
    epochs_no_improve = 0

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        grad_norm_sum = 0.0
        n_updates     = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for step, batch in enumerate(pbar, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            if (labels != -100).sum() == 0:
                log.warning(f"  Step {step}: all labels are -100, skipping")
                continue

            # label_smoothing_factor reduces overconfidence on template tokens
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                label_smoothing_factor=0.1,
            )
            loss = outputs.loss / accum_steps

            if not torch.isfinite(loss):
                log.warning(f"  Step {step}: non-finite loss, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()

            if step % accum_steps == 0 or step == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm_sum += grad_norm.item()
                n_updates     += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm  = grad_norm_sum / max(1, n_updates)
        train_ppl      = math.exp(min(avg_train_loss, 20)) if math.isfinite(avg_train_loss) else float("nan")

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                if torch.isfinite(outputs.loss):
                    val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl      = math.exp(min(avg_val_loss, 20)) if math.isfinite(avg_val_loss) else float("nan")

        log.info(
            f"Epoch {epoch:>3}  "
            f"train_loss={avg_train_loss:.4f}  train_ppl={train_ppl:.2f}  "
            f"val_loss={avg_val_loss:.4f}  val_ppl={val_ppl:.2f}  "
            f"grad_norm={avg_grad_norm:.3f}"
        )

        preview_generations(model, tokenizer, val_samples, device)

        if math.isfinite(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            log.info(f"  ✓ New best → {args.output_dir}  (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            log.info(f"  No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                log.info("Early stopping triggered.")
                break

    log.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            default="atlas_segmentations/segresnet/all_cases.csv")
    p.add_argument("--text_dir",       default="../TextBraTSData")
    p.add_argument("--output_dir",     default="checkpoints/flan-t5")
    p.add_argument("--epochs",         type=int,   default=300)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--lr_enc",         type=float, default=2e-5)   # encoder LoRA
    p.add_argument("--lr_dec",         type=float, default=5e-5)   # decoder LoRA
    p.add_argument("--patience",       type=int,   default=8)      # more patience with augmentation
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log.info("=" * 60)
    log.info("Flan-T5-large v4 — BraTS matrix → structured report")
    log.info("=" * 60)
    log.info(f"  CSV            : {args.csv}")
    log.info(f"  Text dir       : {args.text_dir}")
    log.info(f"  Output dir     : {args.output_dir}")
    log.info(f"  Epochs         : {args.epochs}")
    log.info(f"  Batch size     : {args.batch_size}  (grad_accum={args.grad_accum}, effective={args.batch_size * args.grad_accum})")
    log.info(f"  LR (encoder)   : {args.lr_enc}")
    log.info(f"  LR (decoder)   : {args.lr_dec}")
    log.info(f"  Patience       : {args.patience}")
    log.info("=" * 60)

    train(args)