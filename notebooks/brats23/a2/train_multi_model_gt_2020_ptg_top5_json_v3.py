"""
train_multi_model_v2.py  (v2.1 — anti-collapse revision)
========================================================
Field-wise multi-task trainer. v2 fixed format compliance (0 -> 1.0) but the
eval revealed FIELD COLLAPSE: ~72% of reports had all four fields identical, and
the text was nearly the same across patients ("the <hemi> frontal and parietal
lobes with a mixture of heterogeneous high and low signals..."). The model
learned the single most frequent lesion sentence and emitted it for every field,
because the TASK selector sat at the END of a long prompt and barely influenced
decoding.

v2.1 CHANGES (all target the collapse)
--------------------------------------
1. FRONT-LOAD + BOOKEND the field instruction. T5 was pretrained on leading
   instructions ("summarize:", "question: ... context: ..."), so the field
   question now comes FIRST, and the prompt ENDS with "Answer (<FIELD>):" so the
   field name is the last thing the encoder sees too. The verbose system
   preamble was removed so the instruction isn't drowned out.
       BEFORE:  <long matrix> ... TASK: Report the NECROSIS.
       AFTER:   Question: Describe the NECROSIS ...
                <matrix>
                Answer (NECROSIS):

2. WIDEN LoRA to the feed-forward projections (wi/wo, plus o). r=32 on q,v alone
   may lack the capacity to ROUTE on the instruction; the FFN is where T5 does
   most token-level transformation. Toggle with --no-lora_widen to ablate.

3. FIELD-DISTINCTNESS MONITOR during training. quick_field_distinctness_check
   generates all 4 fields for a few val cases and reports collapse rate +
   distinct ratio every --hemi_check_every epochs, so you can SEE the collapse
   break (or not) as training proceeds — val_loss is teacher-forced and hides it.

>>> IMPORTANT: the eval script must mirror build_field_prompt / matrix_to_context
    EXACTLY or it will feed the checkpoint prompts it never saw. The snippet to
    paste into eval_multi_model_v2.py is identical to the constants + those two
    functions below.

Usage
-----
    python train_multi_model_v2.py --model scifive-base       # test one first
    python train_multi_model_v2.py --no-lora_widen            # ablate change #2
    python train_multi_model_v2.py --audit_only
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
from collections import Counter
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
# target_modules is the BASE set; --lora_widen (default on) adds FF/o modules.
MODELS = {
    "scifive-base":        {"name": "razent/SciFive-base-PMC",     "target_modules": ["q", "v"], "max_input": 640, "max_target": 160, "augment": False},
    "scifive-base-aug":    {"name": "razent/SciFive-base-PMC",     "target_modules": ["q", "v"], "max_input": 640, "max_target": 160, "augment": True},
    "scifive-large":       {"name": "razent/SciFive-large-Pubmed", "target_modules": ["q", "v"], "max_input": 640, "max_target": 160, "augment": False},
    "scifive-large-aug":   {"name": "razent/SciFive-large-Pubmed", "target_modules": ["q", "v"], "max_input": 640, "max_target": 160, "augment": True},
    "clinicalt5-hoss":     {"name": "hossboll/clinical-t5",        "target_modules": ["q", "v"], "max_input": 640, "max_target": 160, "augment": False},
    "clinicalt5-hoss-aug": {"name": "hossboll/clinical-t5",        "target_modules": ["q", "v"], "max_input": 640, "max_target": 160, "augment": True},
}

TRAIN_ORDER = [
    "scifive-base", "scifive-base-aug",
    "clinicalt5-hoss", "clinicalt5-hoss-aug",
    "scifive-large", "scifive-large-aug",
]

# Feed-forward / output projections to add when --lora_widen is on. Covers both
# original T5 (wi, wo) and T5.1.1 gated FFN (wi_0, wi_1). Modules that don't
# exist in a given architecture are simply ignored by PEFT.
_FF_MODULES = ["o", "wi", "wi_0", "wi_1", "wo"]


def resolve_target_modules(cfg, widen):
    mods = list(cfg["target_modules"])
    if widen:
        for m in _FF_MODULES:
            if m not in mods:
                mods.append(m)
    return mods


# ── Field definitions ─────────────────────────────────────────────────────────
FIELDS = ["lesion", "edema", "necrosis", "compression"]

# (1) Front-loaded, clearly DISTINCT questions — each names its field and asks
# something different, so the instruction can actually steer generation.
FIELD_TASK = {
    "lesion":      "Question: In which lobe(s) and hemisphere is the tumor LESION, and what are its signal characteristics?",
    "edema":       "Question: Where is the EDEMA located and how extensive is it?",
    "necrosis":    "Question: Describe the NECROSIS (location and signal). If there is none, answer 'not observed'.",
    "compression": "Question: Describe any VENTRICULAR COMPRESSION (which ventricles, what deformation). If none, answer 'not observed'.",
}

# Trailing answer cue — bookends the field name at the END of the prompt too.
FIELD_LABEL = {
    "lesion":      "LESION AREA",
    "edema":       "EDEMA",
    "necrosis":    "NECROSIS",
    "compression": "VENTRICULAR COMPRESSION",
}

REPORT_FORMAT = (
    "Lesion area: {lesion}\n"
    "Edema: {edema}\n"
    "Necrosis: {necrosis}\n"
    "Ventricular compression: {compression}"
)

# Short matrix header (the long system preamble was removed so the front-loaded
# question dominates the encoder input).
MATRIX_HEADER = "Atlas matrix for one patient:"

# ── Jülich atlas heuristics (unchanged) ───────────────────────────────────────
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
    l = sum(1 for r in active_regions if _LEFT_RE.search(r))
    r_ = sum(1 for r in active_regions if _RIGHT_RE.search(r))
    b = sum(1 for r in active_regions if _BILATERAL_RE.search(r))
    if b > 0 or (l > 0 and r_ > 0):
        return "BILATERAL"
    if l > r_: return "LEFT"
    if r_ > l: return "RIGHT"
    return "UNSPECIFIED"


def infer_primary_lobes(pivot):
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
        index="atlas_region_name", columns="tumor_tag",
        values="count_scaled", aggfunc="max", fill_value=0,
    )
    for tag in ["ET", "TC", "WT"]:
        if tag not in pv.columns:
            pv[tag] = 0
    return pv[["ET", "TC", "WT"]]


# ── Banded region shuffle (FIXED) ─────────────────────────────────────────────
def _ordered_active(pivot, banded_shuffle=False):
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False)
    if banded_shuffle:
        new_index = []
        for _, grp in active.groupby("_total", sort=False):
            idx = list(grp.index)
            random.shuffle(idx)
            new_index.extend(idx)
        active = active.loc[new_index]
    return active.drop(columns="_total")


TOP_N_REGIONS = 5   # Valerio et al. use the top-5 most-affected regions in the report JSON


def matrix_to_context(case_id, pivot, banded_shuffle=False):
    """Build a Valerio-style JSON descriptor instead of the plain-text matrix.

    This is the ONLY function that differs from train_multi_model_v3.py — the
    whole point of this script is the matrix-vs-JSON ablation, so everything else
    (field-wise targets, LoRA, collapse monitor) is held identical.

    Schema mirrors json/SegResNet_sample_*_EN.json from the Valerio repo:
        MRI_Scan.Tumor_Details.Spatial_Distribution = [ {Region, Percentage_of_Tumor, ...}, ... ]
        MRI_Scan.Tumor_Details.Semantic_Segmentation = {Tumor_Core, Peritumoral_Edema, GD_Enhancing_Tumor}

    HONEST ADAPTATION (state this in the thesis): Valerio's percentages come from
    siibra voxel counts. Your CSV only has `count_scaled` per (region, tumor_tag),
    so:
      * Percentage_of_Tumor is derived from the SAME count_scaled values used to
        build the matrix (each region's share of total WT involvement) — so the
        JSON and matrix carry the SAME information. That is what keeps this a
        clean input-representation ablation rather than an information ablation.
      * Percentage_of_Region_Affected (voxels of the region covered) is NOT
        recoverable from your CSV, so it is omitted. The matrix never had it
        either, so neither side gets an unfair advantage.
      * banded_shuffle still applies (reorders equal-salience regions) so the
        augmentation arm behaves the same as the matrix version.
    """
    active = _ordered_active(pivot, banded_shuffle=banded_shuffle)

    # Rank regions by total involvement (WT-weighted, falling back to row sum),
    # exactly as the matrix path orders them, then keep the top-N.
    totals = {}
    for region_name, row in active.iterrows():
        wt = float(row.get("WT", 0))
        tot = wt if wt > 0 else float(row.get("ET", 0) + row.get("TC", 0) + row.get("WT", 0))
        if tot > 0:
            totals[str(region_name)] = tot
    grand = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_REGIONS]

    spatial = [
        {
            "Region": region_name,
            "Percentage_of_Tumor": round(val / grand * 100, 2),
        }
        for region_name, val in ranked
    ]

    descriptor = {
        "MRI_Scan": {
            "Tumor_Details": {
                "Spatial_Distribution": spatial,
                "Semantic_Segmentation": {
                    "Tumor_Core": {"Color": "red"},
                    "Peritumoral_Edema": {"Color": "yellow"},
                    "GD_Enhancing_Tumor": {"Color": "green"},
                },
                "Model_Used": "atlas-segmentation",
            }
        }
    }
    # Compact-but-readable JSON (Valerio used tab indent; we keep 2-space to save tokens).
    return "JSON MRI report data:\n" + json.dumps(descriptor, indent=2)


def build_field_prompt(context, field):
    """(1) Front-load the field question; (3) bookend with a field-named answer
    cue. The matrix sits between."""
    return (
        f"{FIELD_TASK[field]}\n"
        f"Use only the JSON data below. Answer in ONE concise line.\n\n"
        f"{context}\n\n"
        f"Answer ({FIELD_LABEL[field]}):"
    )


# ── Hemisphere swap augmentation (unchanged logic) ────────────────────────────
_HEMI_L = "\u2190LEFT\u2192"
_HEMI_R = "\u2190RIGHT\u2192"


def swap_hemispheres(text):
    text = re.sub(r"\bleft\b",  _HEMI_L, text, flags=re.I)
    text = re.sub(r"\bright\b", _HEMI_R, text, flags=re.I)
    text = text.replace(_HEMI_L, "right").replace(_HEMI_R, "left")
    text = text.replace("Dominant hemisphere: LEFT", "__TMP__")
    text = text.replace("Dominant hemisphere: RIGHT", "Dominant hemisphere: LEFT")
    text = text.replace("__TMP__", "Dominant hemisphere: RIGHT")
    return text


# ── Report parsing -> CANONICAL FIELD DICT ────────────────────────────────────
_FIELD_PATTERNS = {
    "lesion":      re.compile(r"(?:the\s+)?lesion\s+area\s*(?:is\s+in\s+|[:\-]\s*)(.+?)(?=\nedema|\nnecrosis|\nventricular|$)", re.I | re.S),
    "edema":       re.compile(r"edema\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nnecrosis|\nventricular|$)",                       re.I | re.S),
    "necrosis":    re.compile(r"necrosis\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nventricular|$)",                       re.I | re.S),
    "compression": re.compile(r"ventricular\s+compression\s*(?:is\s+|[:\-]\s*)(.+?)(?=\nlesion|\nedema|\nnecrosis|$)",         re.I | re.S),
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
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_fields(raw):
    text = clean_text(raw)
    text_nl = re.sub(r"\s*(edema|necrosis|ventricular)", r"\n\1", text, flags=re.I)
    fields = {}
    for key, pat in _FIELD_PATTERNS.items():
        m = pat.search(text_nl)
        fields[key] = m.group(1).strip().rstrip(".") if m else None
    n_matched = sum(1 for v in fields.values() if v)
    return fields, n_matched


def load_report_fields(text_dir, case_id):
    case_dir = text_dir / case_id
    txt_path = case_dir / f"{case_id}_flair_text.txt"
    raw = None
    if txt_path.exists():
        t = txt_path.read_text(encoding="utf-8").strip()
        if t:
            raw = t
    if raw is None:
        npy_path = case_dir / f"{case_id}_flair_text.npy"
        if npy_path.exists():
            import numpy as np
            arr = np.load(npy_path, allow_pickle=True)
            raw = (str(arr.item()).strip() if arr.ndim == 0
                   else " ".join(str(s).strip() for s in arr.tolist() if str(s).strip()))
    if not raw:
        return None, 0
    return parse_fields(raw)


def report_hemisphere(lesion_value):
    if not lesion_value:
        return "UNK"
    t = lesion_value.lower()
    has_l = bool(re.search(r"\bleft\b", t))
    has_r = bool(re.search(r"\bright\b", t))
    if re.search(r"\b(bilateral|both)\b", t): return "BILATERAL"
    if has_l and has_r: return "BILATERAL"
    if has_l: return "LEFT"
    if has_r: return "RIGHT"
    return "UNK"


# ── Data building + DIAGNOSTICS ───────────────────────────────────────────────
_REQUIRED_CSV_COLS = {"case_id", "atlas_region_name", "tumor_tag", "count_scaled"}


def build_cases(csv_path, text_dir, log, min_fields=2):
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    log.info(f"  {len(df)} rows | {df['case_id'].nunique()} cases in CSV")

    cases = []
    n_no_report = 0
    match_hist = Counter()
    hemi_agree = hemi_total = 0
    # NEW: measure GT per-field redundancy. If the GT fields themselves overlap a
    # lot, the task is partly ill-posed and some collapse is expected.
    gt_distinct_ratios = []

    for case_id, case_df in df.groupby("case_id"):
        fields, n_matched = load_report_fields(text_dir, case_id)
        if fields is None:
            n_no_report += 1
            continue
        match_hist[n_matched] += 1
        if n_matched < min_fields:
            continue

        pv = pivot_matrix(case_df)
        context = matrix_to_context(case_id, pv, banded_shuffle=False)

        prompt_hemi = infer_hemisphere([str(r) for r in pv.index])
        gt_hemi = report_hemisphere(fields.get("lesion"))
        if gt_hemi != "UNK" and prompt_hemi != "UNSPECIFIED":
            hemi_total += 1
            if prompt_hemi == gt_hemi:
                hemi_agree += 1

        canon = {k: (fields.get(k) or "not observed") for k in FIELDS}
        vals = [canon[k].strip().lower() for k in FIELDS]
        gt_distinct_ratios.append(len(set(vals)) / len(FIELDS))

        cases.append({"case_id": case_id, "pivot": pv, "context": context, "fields": canon})

    total_with_report = sum(match_hist.values())
    log.info("\n" + "=" * 70)
    log.info("DATA DIAGNOSTICS (read these before trusting any training run)")
    log.info("=" * 70)
    log.info(f"  Cases with no GT report file        : {n_no_report}")
    log.info(f"  Cases with a GT report              : {total_with_report}")
    log.info(f"  Parsed-field histogram (0..4 fields):")
    for k in range(5):
        c = match_hist.get(k, 0)
        pct = 100 * c / total_with_report if total_with_report else 0
        log.info(f"      {k} field(s): {c:>4}  ({pct:4.1f}%)")
    log.info(f"  Cases KEPT (>= {min_fields} fields)            : {len(cases)}")
    if total_with_report:
        full = match_hist.get(4, 0)
        log.info(f"  Cases parsing ALL 4 fields cleanly  : {full} ({100*full/total_with_report:.1f}%)")
        if match_hist.get(0, 0) + match_hist.get(1, 0) > 0.3 * total_with_report:
            log.warning("  >>> Over 30% of reports parse <2 fields. Fix GT extraction first.")
    agree_pct = 100 * hemi_agree / hemi_total if hemi_total else 0
    log.info(f"  Hemisphere agreement (prompt vs GT) : {hemi_agree}/{hemi_total} = {agree_pct:.1f}%")
    if hemi_total and agree_pct < 70:
        log.warning("  >>> Prompt vs GT hemisphere disagree a lot. Keep hemisphere-swap OFF.")
    gt_dr = sum(gt_distinct_ratios) / len(gt_distinct_ratios) if gt_distinct_ratios else 0
    log.info(f"  GT field distinct ratio (target)    : {gt_dr:.2f}  (1.0 = all 4 GT fields differ)")
    if gt_dr < 0.9:
        log.warning("  >>> GT fields themselves overlap; some prediction collapse is "
                    "baked into the data, not just the model.")
    log.info("=" * 70 + "\n")
    return cases


def expand_to_examples(cases):
    examples = []
    for c in cases:
        for field in FIELDS:
            examples.append({
                "case_id": c["case_id"], "pivot": c["pivot"],
                "field": field, "context": c["context"], "target": c["fields"][field],
            })
    return examples


# ── Dataset ───────────────────────────────────────────────────────────────────
class FieldDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input, max_target,
                 augment=False, p_swap=0.5, p_shuffle=0.0):
        self.examples = examples
        self.tok = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        self.augment = augment
        self.p_swap = p_swap
        self.p_shuffle = p_shuffle

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        target = ex["target"]

        if self.augment and self.p_shuffle > 0 and random.random() < self.p_shuffle:
            context = matrix_to_context(ex["case_id"], ex["pivot"], banded_shuffle=True)
        else:
            context = ex["context"]

        prompt = build_field_prompt(context, ex["field"])

        if self.augment and random.random() < self.p_swap:
            prompt = swap_hemispheres(prompt)
            if ex["field"] in ("lesion", "edema"):
                target = swap_hemispheres(target)

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
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(att), "labels": torch.stack(lbls)}
    return collate


# ── Inference: per-field generate + assemble ──────────────────────────────────
def generate_field(model, tokenizer, prompt, device, max_new_tokens, max_input):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True,
            no_repeat_ngram_size=3, repetition_penalty=1.3, length_penalty=1.0,
        )
    return clean_text(tokenizer.decode(out[0], skip_special_tokens=True).strip())


def predict_report(model, tokenizer, case, device, max_new_tokens, max_input):
    values = {}
    for field in FIELDS:
        prompt = build_field_prompt(case["context"], field)
        val = generate_field(model, tokenizer, prompt, device, max_new_tokens, max_input)
        values[field] = val or "not observed"
    return REPORT_FORMAT.format(**values), values


def ground_truth_report(case):
    return REPORT_FORMAT.format(**case["fields"])


def preview_generations(model, tokenizer, val_cases, device, max_new, max_input, log, n=2):
    model.eval()
    chosen = random.sample(val_cases, min(n, len(val_cases)))
    log.info("\n" + "=" * 60 + "  PREVIEW")
    previews = []
    for c in chosen:
        try:
            pred, _ = predict_report(model, tokenizer, c, device, max_new, max_input)
        except Exception as e:
            pred = f"<generation failed: {e}>"
        ref = ground_truth_report(c)
        log.info(f"\n  Case : {c['case_id']}")
        log.info(f"  PRED :\n{pred}")
        log.info(f"  REF  :\n{ref}")
        previews.append({"case_id": c["case_id"], "pred": pred, "ref": ref})
    log.info("=" * 60)
    return previews


def quick_hemisphere_check(model, tokenizer, val_cases, device, max_new, max_input, log, n=20):
    model.eval()
    chosen = random.sample(val_cases, min(n, len(val_cases)))
    correct = total = 0
    for c in chosen:
        prompt = build_field_prompt(c["context"], "lesion")
        try:
            pred_lesion = generate_field(model, tokenizer, prompt, device, max_new, max_input)
        except Exception:
            continue
        gt = report_hemisphere(c["fields"]["lesion"])
        ph = report_hemisphere(pred_lesion)
        if gt != "UNK":
            total += 1
            if gt == ph:
                correct += 1
    acc = correct / total if total else 0.0
    log.info(f"  Hemisphere quick-check (lesion field): {correct}/{total} = {acc*100:.1f}%")
    return acc


def quick_field_distinctness_check(model, tokenizer, val_cases, device, max_new, max_input, log, n=10):
    """(3) The collapse monitor. Generates all 4 fields for n cases and reports
    how often they're identical. distinct_ratio: 0.25 = all identical (full
    collapse), 1.0 = all four differ. WATCH THIS climb across epochs."""
    model.eval()
    chosen = random.sample(val_cases, min(n, len(val_cases)))
    collapsed = 0
    ratios = []
    for c in chosen:
        vals = []
        for field in FIELDS:
            prompt = build_field_prompt(c["context"], field)
            try:
                vals.append(generate_field(model, tokenizer, prompt, device, max_new, max_input).strip().lower())
            except Exception:
                vals.append("")
        uniq = len(set(vals))
        ratios.append(uniq / len(FIELDS))
        if uniq == 1:
            collapsed += 1
    cr = collapsed / len(chosen) if chosen else 0.0
    dr = sum(ratios) / len(ratios) if ratios else 0.0
    log.info(f"  Field-distinctness check: collapse={cr*100:.0f}%  distinct_ratio={dr:.2f}  "
             f"(0.25=full collapse, 1.0=all differ)")
    return {"collapse_rate": cr, "distinct_ratio": dr}


# ── Per-model training ────────────────────────────────────────────────────────
def train_one_model(model_key, args, cases, log):
    cfg = MODELS[model_key]
    out_dir = Path(args.output_root) / model_key

    if args.skip_done and (out_dir / "summary.json").exists():
        log.info(f"\n[SKIP] {model_key} — summary.json exists. Pass --no-skip_done to retrain.")
        with open(out_dir / "summary.json") as f:
            return json.load(f)

    target_modules = resolve_target_modules(cfg, args.lora_widen)
    log.info("\n" + "#" * 70)
    log.info(f"# Training model: {model_key}  ({cfg['name']})")
    log.info(f"#   augment={cfg['augment']}  p_swap={args.p_swap}  p_shuffle={args.p_shuffle}")
    log.info(f"#   LoRA r={args.lora_r}  targets={target_modules}  (widen={args.lora_widen})")
    log.info(f"#   max_input={cfg['max_input']}  max_target={cfg['max_target']}")
    log.info("#" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cases, val_cases = train_test_split(cases, test_size=0.20, random_state=args.seed, shuffle=True)
    train_examples = expand_to_examples(train_cases)
    val_examples   = expand_to_examples(val_cases)
    log.info(f"Cases  -> train {len(train_cases)} | val {len(val_cases)}")
    log.info(f"Examples (x4 fields) -> train {len(train_examples)} | val {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=target_modules, lora_dropout=0.1, bias="none",
    )
    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as e:
        log.warning(f"LoRA targets {target_modules} failed: {e}; auto-detecting.")
        lora_cfg.target_modules = None
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    train_ds = FieldDataset(train_examples, tokenizer, cfg["max_input"], cfg["max_target"],
                            augment=cfg["augment"], p_swap=args.p_swap, p_shuffle=args.p_shuffle)
    val_ds   = FieldDataset(val_examples, tokenizer, cfg["max_input"], cfg["max_target"], augment=False)
    collate = make_collate(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"))

    lr = args.lr_large if "large" in model_key else args.lr
    log.info(f"Using LR: {lr}  |  label_smoothing: {args.label_smoothing}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    accum = args.grad_accum
    total_steps = max(1, (len(train_loader) // accum) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, best_previews, no_improve, history = float("inf"), 0, [], 0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = grad_sum = 0.0
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
                    optimizer.zero_grad(); continue
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                log.warning(f"  OOM on step {step}, skipping batch")
                optimizer.zero_grad(); torch.cuda.empty_cache(); continue

            if step % accum == 0 or step == len(train_loader):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_sum += gn.item(); n_upd += 1
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

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

        log.info(f"[{model_key}] E{epoch:>3}  train={avg_train:.4f} (ppl {tr_ppl:.2f})  "
                 f"val={avg_val:.4f} (ppl {v_ppl:.2f})  gn={avg_gn:.3f}")

        hemi_acc = None
        distinct = None
        if epoch % args.hemi_check_every == 0:
            hemi_acc = quick_hemisphere_check(model, tokenizer, val_cases, device,
                                              cfg["max_target"], cfg["max_input"], log, n=args.hemi_check_n)
            distinct = quick_field_distinctness_check(model, tokenizer, val_cases, device,
                                                      cfg["max_target"], cfg["max_input"], log,
                                                      n=args.distinct_check_n)
        previews = preview_generations(model, tokenizer, val_cases, device,
                                       cfg["max_target"], cfg["max_input"], log)

        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
                        "train_ppl": tr_ppl, "val_ppl": v_ppl, "grad_norm": avg_gn,
                        "hemisphere_acc": hemi_acc,
                        "collapse_rate": (distinct or {}).get("collapse_rate"),
                        "distinct_ratio": (distinct or {}).get("distinct_ratio")})

        if math.isfinite(avg_val) and avg_val < best_val:
            best_val, best_epoch, best_previews, no_improve = avg_val, epoch, previews, 0
            model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
            log.info(f"  ✓ New best ({best_val:.4f}) → {out_dir}")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                log.info("Early stopping."); break

    elapsed = time.time() - t0
    summary = {
        "model_key": model_key, "model_name": cfg["name"], "augment": cfg["augment"],
        "lora_r": args.lora_r, "lora_targets": target_modules, "lora_widen": args.lora_widen,
        "max_input": cfg["max_input"], "max_target": cfg["max_target"],
        "label_smoothing": args.label_smoothing, "p_swap": args.p_swap, "p_shuffle": args.p_shuffle,
        "best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": len(history),
        "elapsed_sec": elapsed, "elapsed_min": elapsed / 60,
        "history": history, "best_previews": best_previews,
        "mode": "field_wise_multitask_v2.1",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[{model_key}] done in {elapsed/60:.1f} min. Best val={best_val:.4f} at epoch {best_epoch}")

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",    default="../TextBraTSData")
    p.add_argument("--output_root", default="checkpoints_multi_v3_json")
    p.add_argument("--model",       default="all", help=f"all | {' | '.join(MODELS.keys())}")
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--lr_large",      type=float, default=1e-5)
    p.add_argument("--patience",      type=int,   default=12)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--min_fields",    type=int,   default=2)
    p.add_argument("--lora_r",        type=int,   default=32)
    p.add_argument("--lora_widen",    action="store_true", default=True,
                   help="Add FFN/o projections to LoRA targets (change #2). On by default.")
    p.add_argument("--no-lora_widen", dest="lora_widen", action="store_false")
    p.add_argument("--p_swap",        type=float, default=0.5)
    p.add_argument("--p_shuffle",     type=float, default=0.0)
    p.add_argument("--hemi_check_every", type=int, default=5)
    p.add_argument("--hemi_check_n",  type=int, default=20)
    p.add_argument("--distinct_check_n", type=int, default=10,
                   help="Cases for the field-collapse monitor (change #3)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--audit_only",    action="store_true")
    p.add_argument("--skip_done",     action="store_true", default=True)
    p.add_argument("--no-skip_done",  dest="skip_done", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"multi_model_v3_json_{ts}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()], force=True)
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Field-wise multi-task BraTS matrix → report training (v3-JSON, Valerio-style input — matrix-vs-JSON ablation)")
    log.info("=" * 70)
    for k in ["csv", "text_dir", "output_root", "model", "epochs", "patience",
              "batch_size", "grad_accum", "lr", "lr_large", "label_smoothing",
              "min_fields", "lora_r", "lora_widen", "p_swap", "p_shuffle", "skip_done"]:
        log.info(f"  {k:<14}: {getattr(args, k)}")
    log.info("=" * 70)

    cases = build_cases(args.csv, Path(args.text_dir), log, min_fields=args.min_fields)
    if args.audit_only:
        log.info("audit_only set — diagnostics printed above. Exiting before training.")
        return
    if len(cases) < 2:
        raise ValueError("Not enough usable cases after filtering. Check diagnostics above.")

    model_keys = TRAIN_ORDER if args.model == "all" else [args.model]
    if args.model != "all" and args.model not in MODELS:
        raise ValueError(f"Unknown model {args.model}. Options: all, {list(MODELS.keys())}")

    all_summaries, t_global = [], time.time()
    for i, key in enumerate(model_keys, 1):
        log.info(f"\n>>> Run {i}/{len(model_keys)}  (elapsed: {(time.time()-t_global)/60:.1f} min)")
        try:
            all_summaries.append(train_one_model(key, args, cases, log))
        except Exception as e:
            log.exception(f"[{key}] training FAILED: {e}")
            all_summaries.append({"model_key": key, "model_name": MODELS[key]["name"], "error": str(e)})
            torch.cuda.empty_cache()

    out_path = Path("logs") / f"summary_v3_json_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"\nComparison summary → {out_path}")

    log.info("\n" + "=" * 70)
    log.info("FINAL RANKING BY BEST VAL LOSS")
    log.info("=" * 70)
    rank = sorted([s for s in all_summaries if "best_val_loss" in s and math.isfinite(s["best_val_loss"])],
                  key=lambda x: x["best_val_loss"])
    for i, s in enumerate(rank, 1):
        tag = " [aug]" if s.get("augment") else "      "
        log.info(f"  {i}. {s['model_key']:<22}{tag} val_loss={s['best_val_loss']:.4f}  "
                 f"epoch {s['best_epoch']}/{s['epochs_run']}  ({s['elapsed_min']:.1f} min)")

    failed = [s for s in all_summaries if "error" in s]
    if failed:
        log.info("\n  FAILED:")
        for s in failed:
            log.info(f"    {s['model_key']}: {s['error']}")


if __name__ == "__main__":
    main()