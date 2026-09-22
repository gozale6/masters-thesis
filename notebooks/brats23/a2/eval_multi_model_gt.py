"""
eval_multi_model.py
===================
Evaluate the LoRA checkpoints produced by train_multi_model.py (ground-truth run).
Mirrors the training script: seq2seq-only (3 models), same prompt format,
same train/val split.

Computes ROUGE-1/2/L, BLEU, format compliance, per-field ROUGE-1.
Saves a side-by-side comparison.

Usage
-----
    pip install rouge-score sacrebleu
    python eval_multi_model.py
    python eval_multi_model.py --model scifive-base
"""

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from peft import PeftModel

# ── Model registry (mirror train_multi_model.py) ─────────────────────────────
MODELS = {
    "scifive-base": {
        "name":       "razent/SciFive-base-PMC",
        "max_input":  512,
        "max_target": 200,
    },
    "scifive-large": {
        "name":       "razent/SciFive-large-Pubmed",
        "max_input":  512,
        "max_target": 200,
    },
    "clinicalt5-hoss": {
        "name":       "hossboll/clinical-t5",
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

# ── Atlas prompt helpers ──────────────────────────────────────────────────────
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
        if t: return parse_report(t)
    npy_path = case_dir / f"{case_id}_flair_text.npy"
    if npy_path.exists():
        import numpy as np
        arr = np.load(npy_path, allow_pickle=True)
        t = str(arr.item()).strip() if arr.ndim == 0 else " ".join(
            str(s).strip() for s in arr.tolist() if str(s).strip()
        )
        if t: return parse_report(t)
    return None


def build_samples(csv_path, text_dir, log):
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
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
            "target":     report,
        })
    log.info(f"  Matched: {len(samples)} | Skipped: {len(skipped)}")
    return samples


# ── Generation ────────────────────────────────────────────────────────────────
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(predictions, references, log):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        log.error("rouge-score not installed. Run: pip install rouge-score")
        raise
    try:
        import sacrebleu
    except ImportError:
        log.error("sacrebleu not installed. Run: pip install sacrebleu")
        raise

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score / 100.0

    # Format compliance: all 4 headers present on their own lines
    compliant = 0
    for p in predictions:
        has_all = (
            re.search(r"(?mi)^\s*Lesion area:",             p) and
            re.search(r"(?mi)^\s*Edema:",                   p) and
            re.search(r"(?mi)^\s*Necrosis:",                p) and
            re.search(r"(?mi)^\s*Ventricular compression:", p)
        )
        if has_all:
            compliant += 1
    compliance = compliant / len(predictions) if predictions else 0.0

    # Per-field ROUGE-1
    per_field = {}
    for field_key, pattern in _FIELD_PATTERNS.items():
        field_preds, field_refs = [], []
        for p, r in zip(predictions, references):
            mp = pattern.search(p)
            mr = pattern.search(r)
            fp = mp.group(1).strip() if mp else ""
            fr = mr.group(1).strip() if mr else ""
            if fr:
                field_preds.append(fp)
                field_refs.append(fr)
        if field_refs:
            fs = [scorer.score(fr, fp)["rouge1"].fmeasure
                  for fp, fr in zip(field_preds, field_refs)]
            per_field[field_key] = sum(fs) / len(fs)
        else:
            per_field[field_key] = 0.0

    return {
        "rouge1":      sum(r1) / len(r1) if r1 else 0.0,
        "rouge2":      sum(r2) / len(r2) if r2 else 0.0,
        "rougeL":      sum(rL) / len(rL) if rL else 0.0,
        "bleu":        bleu,
        "format_compliance": compliance,
        "per_field_rouge1":  per_field,
        "n_samples":   len(predictions),
    }


# ── Per-model evaluation ──────────────────────────────────────────────────────
def evaluate_one(model_key, args, samples, log):
    cfg = MODELS[model_key]
    ckpt_dir = Path(args.checkpoint_root) / model_key
    if not ckpt_dir.exists():
        log.warning(f"[{model_key}] checkpoint not found at {ckpt_dir} — skipping")
        return None

    log.info("\n" + "#" * 70)
    log.info(f"# Evaluating: {model_key}  ({cfg['name']})")
    log.info("#" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the exact train/val split used during training
    _, val_samples = train_test_split(
        samples, test_size=0.20, random_state=args.seed, shuffle=True,
    )
    log.info(f"Val samples: {len(val_samples)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True)

    log.info(f"Loading base model: {cfg['name']}")
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)

    log.info(f"Applying LoRA adapter from {ckpt_dir}")
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
    model.to(device)
    model.eval()

    predictions, references, case_ids = [], [], []
    t0 = time.time()
    for s in tqdm(val_samples, desc=f"[{model_key}] generate"):
        try:
            pred = generate_seq2seq(
                model, tokenizer, s["input_base"],
                device, cfg["max_target"], cfg["max_input"],
            )
        except Exception as e:
            log.warning(f"  [{s['case_id']}] generation failed: {e}")
            pred = ""
        predictions.append(pred)
        references.append(s["target"])
        case_ids.append(s["case_id"])
    elapsed = time.time() - t0

    metrics = compute_metrics(predictions, references, log)

    log.info(f"\n[{model_key}] metrics")
    log.info(f"  ROUGE-1            : {metrics['rouge1']:.4f}")
    log.info(f"  ROUGE-2            : {metrics['rouge2']:.4f}")
    log.info(f"  ROUGE-L            : {metrics['rougeL']:.4f}")
    log.info(f"  BLEU               : {metrics['bleu']:.4f}")
    log.info(f"  Format compliance  : {metrics['format_compliance']*100:.1f}%")
    log.info(f"  Samples            : {metrics['n_samples']}")
    log.info(f"  Per-field ROUGE-1:")
    for k, v in metrics["per_field_rouge1"].items():
        log.info(f"    {k:<20} : {v:.4f}")

    out_dir = Path(args.output_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"case_id": cid, "prediction": p, "reference": r}
        for cid, p, r in zip(case_ids, predictions, references)
    ]
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    del model, base
    torch.cuda.empty_cache()

    return {
        "model_key":    model_key,
        "model_name":   cfg["name"],
        "metrics":      metrics,
        "generate_sec": elapsed,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_root", default="checkpoints_multi")
    p.add_argument("--csv",             default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",        default="../TextBraTSData")
    p.add_argument("--output_dir",      default="eval_multi_results")
    p.add_argument("--model",           default="all",
                   help=f"Which model to eval. One of: all | {' | '.join(MODELS.keys())}")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"eval_multi_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Multi-model evaluation (ground truth)")
    log.info("=" * 70)
    log.info(f"  Checkpoint root : {args.checkpoint_root}")
    log.info(f"  CSV             : {args.csv}")
    log.info(f"  Text dir        : {args.text_dir}")
    log.info(f"  Output dir      : {args.output_dir}")
    log.info(f"  Model(s)        : {args.model}")
    log.info("=" * 70)

    samples = build_samples(args.csv, Path(args.text_dir), log)
    if len(samples) < 2:
        raise ValueError("Not enough samples to form val split")

    if args.model == "all":
        model_keys = []
        for key in MODELS:
            if (Path(args.checkpoint_root) / key).exists():
                model_keys.append(key)
        if not model_keys:
            raise ValueError(f"No checkpoints found under {args.checkpoint_root}")
        log.info(f"Found checkpoints: {model_keys}")
    else:
        if args.model not in MODELS:
            raise ValueError(f"Unknown model {args.model}. Options: {list(MODELS.keys())}")
        model_keys = [args.model]

    all_results = []
    for key in model_keys:
        try:
            r = evaluate_one(key, args, samples, log)
            if r is not None:
                all_results.append(r)
        except Exception as e:
            log.exception(f"[{key}] eval FAILED: {e}")
            all_results.append({
                "model_key": key,
                "model_name": MODELS[key]["name"],
                "error": str(e),
            })
            torch.cuda.empty_cache()

    out_path = Path(args.output_dir) / f"comparison_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nComparison saved → {out_path}")

    log.info("\n" + "=" * 70)
    log.info("RANKING BY ROUGE-1")
    log.info("=" * 70)
    log.info(f"  {'model':<20}  {'R-1':>6}  {'R-2':>6}  {'R-L':>6}  {'BLEU':>6}  {'fmt%':>5}")
    log.info(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}")
    rank = sorted(
        [r for r in all_results if "metrics" in r],
        key=lambda x: x["metrics"]["rouge1"],
        reverse=True,
    )
    for r in rank:
        m = r["metrics"]
        log.info(
            f"  {r['model_key']:<20}  "
            f"{m['rouge1']:.4f}  {m['rouge2']:.4f}  {m['rougeL']:.4f}  "
            f"{m['bleu']:.4f}  {m['format_compliance']*100:5.1f}"
        )

    log.info("\n  Per-field ROUGE-1 (lesion / edema / necrosis / compression):")
    for r in rank:
        pf = r["metrics"]["per_field_rouge1"]
        log.info(
            f"  {r['model_key']:<20}  "
            f"{pf.get('lesion', 0):.4f}  {pf.get('edema', 0):.4f}  "
            f"{pf.get('necrosis', 0):.4f}  {pf.get('compression', 0):.4f}"
        )

    failed = [r for r in all_results if "error" in r]
    if failed:
        log.info("\n  FAILED:")
        for r in failed:
            log.info(f"    {r['model_key']}: {r['error']}")


if __name__ == "__main__":
    main()