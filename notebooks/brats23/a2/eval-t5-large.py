"""
evalBiomed.py  (v4 — fixed format compliance + per-field scoring)
==================================================================
Loads the fine-tuned Flan-T5 LoRA checkpoint, runs generation on the
validation split, and reports ROUGE-1/2/L, BLEU, format compliance,
and per-field ROUGE-1.

Fixes vs v3 eval
----------------
- format_compliance was 0% because the regex expected "Lesion area:" as a
  standalone prefix but predictions output "Lesion area: the right..." — the
  value starts with "the". Fixed: regex now anchors on the field label at the
  START of a line, not as part of the value extraction group.
- Per-field ROUGE-1 was N/A for the same reason. Fixed alongside.
- Uses the same richer prompt (hemisphere + primary lobes) as train v4.
- clean_text() applied to references too, so scores aren't penalised for
  accented characters in the reference that the model correctly avoided.

Usage
-----
    conda run -n mastersThesis python notebooks/brats23/a2/evalBiomed.py

Requirements
------------
    pip install torch transformers peft pandas scikit-learn tqdm sentencepiece
    pip install nltk rouge-score
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants (must match train v4) ──────────────────────────────────────────
MODEL_NAME     = "google/flan-t5-large"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 200

TASK_PREFIX = (
    "You are a neuroradiology assistant. "
    "Given a brain tumor atlas matrix, write a structured radiology report "
    "using EXACTLY this format (4 lines, each starting with the field name):\n"
    "Lesion area: [lobe(s), hemisphere, signal characteristics]\n"
    "Edema: [location and extent]\n"
    "Necrosis: [location and signal, or 'not observed']\n"
    "Ventricular compression: [compression description, or 'not observed']\n\n"
    "Atlas matrix:\n"
)

REPORT_FORMAT = (
    "Lesion area: {lesion}\n"
    "Edema: {edema}\n"
    "Necrosis: {necrosis}\n"
    "Ventricular compression: {compression}"
)

# ── Field extraction ──────────────────────────────────────────────────────────
# These patterns extract the VALUE after the label.
# They are anchored to line-start (or after newline) so "Lesion area: the X"
# correctly captures "the X" as the value, not "" as an empty string.
_FIELD_PATTERNS = {
    "lesion": re.compile(
        r"(?:^|\n)\s*(?:the\s+)?lesion\s+area\s*(?:is\s+(?:in\s+)?|[:\-]\s*)(.+?)(?=\nedema|\nnecrosis|\nventricular|\Z)",
        re.I | re.S,
    ),
    "edema": re.compile(
        r"(?:^|\n)\s*edema\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nnecrosis|\nventricular|\Z)",
        re.I | re.S,
    ),
    "necrosis": re.compile(
        r"(?:^|\n)\s*necrosis\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nventricular|\Z)",
        re.I | re.S,
    ),
    "compression": re.compile(
        r"(?:^|\n)\s*ventricular\s+compression\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nnecrosis|\Z)",
        re.I | re.S,
    ),
}

# Format compliance: does the output contain ALL 4 field headers on their own line?
_HEADER_PATTERNS = {
    "lesion":      re.compile(r"(?:^|\n)\s*(?:the\s+)?lesion\s+area\s*[:\-]",      re.I),
    "edema":       re.compile(r"(?:^|\n)\s*edema\s*[:\-]",                          re.I),
    "necrosis":    re.compile(r"(?:^|\n)\s*necrosis\s*[:\-]",                       re.I),
    "compression": re.compile(r"(?:^|\n)\s*ventricular\s+compression\s*[:\-]",      re.I),
}


def extract_field(text: str, field: str) -> str:
    pat = _FIELD_PATTERNS.get(field)
    if pat is None:
        return ""
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def has_all_headers(text: str) -> bool:
    return all(pat.search(text) for pat in _HEADER_PATTERNS.values())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (identical logic to train v4)
# ─────────────────────────────────────────────────────────────────────────────

_LEFT_RE      = re.compile(r"\b(left|_L_|Left)\b",       re.I)
_RIGHT_RE     = re.compile(r"\b(right|_R_|Right)\b",     re.I)
_BILATERAL_RE = re.compile(r"\b(bilateral|_B_|both)\b",  re.I)

_LOBE_KEYWORDS = {
    "frontal":       re.compile(r"frontal",   re.I),
    "parietal":      re.compile(r"parietal",  re.I),
    "temporal":      re.compile(r"temporal",  re.I),
    "occipital":     re.compile(r"occipital", re.I),
    "insula":        re.compile(r"insul",     re.I),
    "cerebellum":    re.compile(r"cerebell",  re.I),
    "brainstem":     re.compile(r"brain.?stem|pons|medulla|midbrain", re.I),
    "basal ganglia": re.compile(r"basal|putamen|caudate|pallidum|thalamus", re.I),
}


def infer_hemisphere(active_regions):
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


def infer_primary_lobes(active_regions, pivot):
    lobe_scores = {}
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


def pivot_matrix(case_df):
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


def matrix_to_prompt(case_id, pivot):
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False).drop(columns="_total")

    region_names  = [str(r) for r in active.index]
    hemisphere    = infer_hemisphere(region_names)
    primary_lobes = infer_primary_lobes(region_names, active)

    lines = [
        f"Patient: {case_id}",
        f"Dominant hemisphere: {hemisphere}",
        f"Primary lobe(s): {primary_lobes}",
        "Atlas involvement (scale 0-256):",
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
    lines.append("\nET=Enhancing Tumor  TC=Tumor Core/necrosis  WT=Whole Tumor/edema")
    return TASK_PREFIX + "\n".join(lines)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\b[bcdfghjklmnpqrstvwxyz]{5,}\b", "", text, flags=re.I)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


_TRAIN_FIELD_PATTERNS = {
    "lesion":      re.compile(r"(?:the\s+)?lesion\s+area\s*(?:is\s+in\s+|[:\-]\s*)(.+?)(?=\nedema|\nnecrosis|\nventricular|$)",  re.I | re.S),
    "edema":       re.compile(r"edema\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nnecrosis|\nventricular|$)",                         re.I | re.S),
    "necrosis":    re.compile(r"necrosis\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nventricular|$)",                         re.I | re.S),
    "compression": re.compile(r"ventricular\s+compression\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nnecrosis|$)",           re.I | re.S),
}


def parse_report(raw: str) -> str:
    text   = clean_text(raw)
    fields = {}
    for key, pat in _TRAIN_FIELD_PATTERNS.items():
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
    return text


def load_report(text_dir, case_id):
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


def build_samples(csv_path, text_dir):
    df = pd.read_csv(csv_path)
    samples, skipped = [], []
    for case_id, case_df in df.groupby("case_id"):
        report = load_report(text_dir, case_id)
        if report is None:
            skipped.append(case_id)
            continue
        pivot  = pivot_matrix(case_df)
        prompt = matrix_to_prompt(case_id, pivot)
        samples.append({"case_id": case_id, "input": prompt, "target": report})
    log.info(f"Matched pairs: {len(samples)}  |  skipped: {len(skipped)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, device, max_new_tokens=200):
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(predictions, references):
    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1, r2, rL   = [], [], []
    bleu_refs    = []
    bleu_hyps    = []
    field_r1     = {f: [] for f in _FIELD_PATTERNS}
    format_count = 0

    for pred, ref in zip(predictions, references):
        # Clean both sides consistently so scores aren't affected by
        # accented chars or OCR garbage in the reference
        pred_clean = clean_text(pred)
        ref_clean  = clean_text(ref)

        s = scorer.score(ref_clean, pred_clean)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

        bleu_refs.append([ref_clean.split()])
        bleu_hyps.append(pred_clean.split())

        if has_all_headers(pred_clean):
            format_count += 1

        for field in _FIELD_PATTERNS:
            pred_field = extract_field(pred_clean, field)
            ref_field  = extract_field(ref_clean,  field)
            if ref_field:
                fs = scorer.score(ref_field, pred_field)
                field_r1[field].append(fs["rouge1"].fmeasure)

    bleu = corpus_bleu(
        bleu_refs, bleu_hyps,
        smoothing_function=SmoothingFunction().method1,
    )
    n = len(predictions)

    return {
        "rouge1":   round(sum(r1) / n, 4),
        "rouge2":   round(sum(r2) / n, 4),
        "rougeL":   round(sum(rL) / n, 4),
        "bleu":     round(bleu, 4),
        "n_samples": n,
        "format_compliance": round(format_count / n, 4),
        "field_rouge1": {
            field: round(sum(scores) / len(scores), 4) if scores else None
            for field, scores in field_r1.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",             default="atlas_segmentations/segresnet/all_cases.csv")
    p.add_argument("--text_dir",        default="../TextBraTSData")
    p.add_argument("--model_dir",       default="checkpoints/flan-t5")
    p.add_argument("--output",          default="eval_results.json")
    p.add_argument("--max_new_tokens",  type=int, default=200)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    text_dir = Path(args.text_dir)
    samples  = build_samples(args.csv, text_dir)
    _, val_samples = train_test_split(
        samples, test_size=0.20, random_state=args.seed, shuffle=True,
    )
    log.info(f"Val samples: {len(val_samples)}")

    model_dir = Path(args.model_dir)
    log.info(f"Loading tokenizer from {model_dir}")
    tokenizer = T5Tokenizer.from_pretrained(model_dir)

    log.info(f"Loading base model: {MODEL_NAME}")
    use_bf16   = False   # fp32 to match training
    base_model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )

    log.info(f"Applying LoRA adapter from {model_dir}")
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()

    # ── Generate ──────────────────────────────────────────────────────────────
    predictions, references, case_ids = [], [], []

    for sample in tqdm(val_samples, desc="Generating"):
        pred = generate(model, tokenizer, sample["input"], device, args.max_new_tokens)
        predictions.append(pred)
        references.append(sample["target"])
        case_ids.append(sample["case_id"])

        if len(predictions) % 10 == 0:
            log.info(f"\n  [{sample['case_id']}]")
            log.info(f"  PRED : {pred}")
            log.info(f"  REF  : {sample['target'][:400]}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(predictions, references)

    log.info("=" * 55)
    log.info(f"  ROUGE-1            : {metrics['rouge1']:.4f}")
    log.info(f"  ROUGE-2            : {metrics['rouge2']:.4f}")
    log.info(f"  ROUGE-L            : {metrics['rougeL']:.4f}")
    log.info(f"  BLEU               : {metrics['bleu']:.4f}")
    log.info(f"  Format compliance  : {metrics['format_compliance']:.2%}")
    log.info(f"  Samples            : {metrics['n_samples']}")
    log.info("  Per-field ROUGE-1:")
    for field, score in metrics["field_rouge1"].items():
        display = f"{score:.4f}" if score is not None else "N/A (no ref field found)"
        log.info(f"    {field:<22}: {display}")
    log.info("=" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_data = {
        "metrics": metrics,
        "predictions": [
            {"case_id": c, "prediction": p, "reference": r}
            for c, p, r in zip(case_ids, predictions, references)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = out_path.parent / "eval_runs" / f"run_{ts}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        shutil.copy(out_path, archive_dir / out_path.name)
        log.info(f"Previous results archived → {archive_dir / out_path.name}")

    model_dir_path = Path(args.model_dir)
    if model_dir_path.exists():
        shutil.copytree(model_dir_path, archive_dir / "model")
        log.info(f"Checkpoint archived → {archive_dir / 'model'}")

    out_path.write_text(json.dumps(output_data, indent=2))
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()