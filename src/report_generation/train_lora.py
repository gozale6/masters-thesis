"""
train_multi_model_gt_2020_ptg_v4.py  (MATRIX arm, v4 — anti *cross-patient* collapse)
==========================================================================
v2.1 fixed WITHIN-REPORT field collapse (the 4 fields of one report now differ:
distinct_field_ratio=1.0, field_collapse_rate=0.0). But the eval predictions
revealed a SECOND collapse the v2.1 monitor cannot see:

    CROSS-PATIENT TEMPLATE COLLAPSE
    -------------------------------
    The model learned ~2 templates (one per hemisphere) and emits a near-
    identical report for every patient, only flipping left/right and a couple of
    lobe words. e.g. the lesion field for cases _166 / _016 / _313 is byte-for-
    byte identical. Aggregate metrics (ROUGE ~0.65, BERTScore ~0.92) hide this
    because the GT reports are stylistically homogeneous, so a per-hemisphere
    template scores well. lobe_jaccard ~0.64 is mostly "guess frontal+parietal".
    Bilateral GT (_120, _040) and temporal/occipital GT (_016, _220, _154) get
    flattened to the template.

v4 CHANGES (all target cross-patient collapse / weak input conditioning)
--------------------------------------------------------------------------
1. USE THE RICH CSV. apply_atlas_matrix_gt.py now emits pct_of_tumor,
   pct_of_region and is_top5. Salience uses pct_of_tumor (real tumor share); the
   matrix now also shows region occupancy and marks the top-5 rows.

2. EXPLICIT LOBE ROLLUP (--no-lobe_rollup to ablate). The OLD `Primary lobe(s):`
   line used _LOBE_KEYWORDS, which only matched the literal words frontal/
   parietal/... — Julich CYTOARCHITECTONIC names (Area 45 (IFG), Area hOc1
   (V1)...) almost never contain them, so that line was usually "unspecified".
   region_to_lobe() now maps the gyral/area abbreviations the names DO carry, and
   the context gets a proper per-lobe involvement summary + per-region [lobe]
   tags. The map is a HEURISTIC seed — build_cases() prints coverage + the
   unmatched names so you can audit/extend _ATLAS_LOBE_PATTERNS.
   >>> Mirror this (and 3-6) into the JSON arm or the matrix-vs-JSON ablation is
       contaminated. See train_multi_model_gt_2020_ptg_v4_json.py.

3. CROSS-PATIENT MONITOR. quick_cross_patient_check() generates one field across
   N DIFFERENT patients and counts unique outputs. THIS is the number that
   exposes template collapse (the v2.1 within-report check said 1.0 while the
   model was collapsed). Watch it climb.

4. CONTENT-BASED CHECKPOINT SELECTION (--select_metric content, default).
   Teacher-forced val_loss rewards the template-memorizer, so it was selecting
   the wrong epoch. We now select `best` on 0.5*lobe_jaccard + 0.5*hemisphere_acc
   over a small val subset. val_loss is still logged. --select_metric val_loss
   reverts.

5. HEMISPHERE-SWAP AUG TURNED DOWN. The swap *hurt* hemisphere accuracy
   (0.905 -> 0.797) in every aug pair while barely moving content. Default p_swap
   lowered 0.50 -> 0.15. Pass --p_swap 0.5 to reproduce the old runs.

6. GT DIVERSITY DIAGNOSTICS. build_cases() reports GT lobe-set distribution, GT
   hemisphere distribution and cross-patient GT lesion distinctness, so you know
   the achievable ceiling before trusting any score.

FINE-TUNE-MODE AXIS (12-run sweep)
--------------------------------------------------------------------------
Each of the 6 models is now trained TWICE — once with LoRA and once with FULL
fine-tuning — so you can see what LoRA is costing/saving vs. tuning all weights.
Controlled by --ft_mode {lora|full|both}, default 'both' → 12 runs. Full-FT uses
its own (much lower) learning rates (--lr_full / --lr_large_full) and saves a
FULL model (no adapter), so:
  * output_root grows a lot (full checkpoints, not tiny adapters);
  * full-FT of the *large* model is memory-hungry — on a 16GB card it may OOM at
    batch_size 2. If so, run --model with the smaller ones, lower the batch, or
    enable gradient checkpointing (see the ft_mode=='full' branch).
  * >>> the EVAL loader must branch: full-FT dirs have NO adapter_config.json,
        so load them with AutoModelForSeq2SeqLM directly, not PeftModel.

>>> The eval script must mirror build_field_prompt / matrix_to_context EXACTLY.
    Re-copy the constants + those two functions into eval_multi_model_gt_2020_ptg_v4.py.

Usage
-----
    python train_multi_model_gt_2020_ptg_v4.py                             # 12 runs (lora+full x 6)
    python train_multi_model_gt_2020_ptg_v4.py --ft_mode lora              # original 6 (LoRA only)
    python train_multi_model_gt_2020_ptg_v4.py --ft_mode full              # 6 full fine-tunes
    python train_multi_model_gt_2020_ptg_v4.py --model scifive-base        # 2 runs (lora+full) for one
    python train_multi_model_gt_2020_ptg_v4.py --no-lobe_rollup            # ablate change #2
    python train_multi_model_gt_2020_ptg_v4.py --select_metric val_loss    # old selection
    python train_multi_model_gt_2020_ptg_v4.py --audit_only
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
from collections import Counter, defaultdict
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

# Each model in TRAIN_ORDER is trained once per mode -> 12 runs when ft_mode=both.
FT_MODES = ["lora", "full"]

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

FIELD_TASK = {
    "lesion":      "Question: In which lobe(s) and hemisphere is the tumor LESION, and what are its signal characteristics?",
    "edema":       "Question: Where is the EDEMA located and how extensive is it?",
    "necrosis":    "Question: Describe the NECROSIS (location and signal). If there is none, answer 'not observed'.",
    "compression": "Question: Describe any VENTRICULAR COMPRESSION (which ventricles, what deformation). If none, answer 'not observed'.",
}

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

MATRIX_HEADER = "Atlas matrix for one patient:"

# ── Hemisphere heuristics (unchanged) ─────────────────────────────────────────
_LEFT_RE      = re.compile(r"\b(left|_L_|Left)\b",       re.I)
_RIGHT_RE     = re.compile(r"\b(right|_R_|Right)\b",     re.I)
_BILATERAL_RE = re.compile(r"\b(bilateral|_B_|both)\b",  re.I)


def infer_hemisphere(active_regions):
    l = sum(1 for r in active_regions if _LEFT_RE.search(r))
    r_ = sum(1 for r in active_regions if _RIGHT_RE.search(r))
    b = sum(1 for r in active_regions if _BILATERAL_RE.search(r))
    if b > 0 or (l > 0 and r_ > 0):
        return "BILATERAL"
    if l > r_: return "LEFT"
    if r_ > l: return "RIGHT"
    return "UNSPECIFIED"


# ── (change #2) ATLAS region-name -> LOBE mapping ─────────────────────────────
# Replaces the old thin _LOBE_KEYWORDS / infer_primary_lobes. HEURISTIC SEED:
# Julich-Brain v3 names are cytoarchitectonic (e.g. "Area 45 (IFG)",
# "Area hOc1 (V1, 17, CalcS)"), so we match the gyral/area abbreviations they
# carry, not the literal lobe words (which rarely appear). build_cases() prints
# coverage + the unmatched names — AUDIT THEM and extend this table. First match
# wins. Ambiguous picks (fusiform -> temporal; precentral -> frontal) flagged so
# you can move them.
_ATLAS_LOBE_PATTERNS = {
    "frontal":     re.compile(r"frontal|\bIFG\b|\bMFG\b|\bSFG\b|\bFp\d|frontal ?pole|\bArea ?4[45]\b|broca|\bPreCG\b|precentral|\bSMA\b|pre-?SMA|frontal opercul|\bFO\d", re.I),
    "parietal":    re.compile(r"parietal|\bSPL\b|\bIPL\b|\bIPS\b|\bPostCG\b|postcentral|\bSMG\b|\bAG\b|\bPCu\b|precuneus|\bPF[a-z]{0,2}\b|\bPG[ap]\b|\bArea ?[57][A-Za-z]?\b", re.I),
    "temporal":    re.compile(r"temporal|\bSTG\b|\bMTG\b|\bITG\b|\bFusG\b|fusiform|\bFG[1-4]\b|\bTE\b|\bTE\d|heschl|planum|\bPhG\b|parahippocamp|hippocamp|\bCA[1-3]\b|\bDG\b|entorhinal|\bEC\b|amygdal", re.I),
    "occipital":   re.compile(r"occipital|\bhOc\d|\bV[1-5]\b|\bV3[AB]?\b|\bV4\b|calcarine|calcs|cuneus|lingual|\bLOC\b|\bhMT\b|\bMST\b", re.I),
    "insula":      re.compile(r"insula|insular|\bIg\d|\bId\d|\bIa\b", re.I),
    "cerebellum":  re.compile(r"cerebell|dentate|vermis|nodul|fastigial|interposed|flocc", re.I),
    "subcortical": re.compile(r"putamen|caudate|pallidum|accumbens|\bGP[ei]?\b|striatum|thalam|\bSTN\b|substantia ?nigra|red ?nucleus|mammillary", re.I),
    "limbic":      re.compile(r"cingul|\bACC\b|\bMCC\b|\bPCC\b|septal|fornix|\bBA2[34]\b", re.I),
}


def region_to_lobe(name):
    s = str(name)
    for lobe, pat in _ATLAS_LOBE_PATTERNS.items():
        if pat.search(s):
            return lobe
    return "other"


def lobe_coverage_report(region_names, log):
    counts = Counter()
    unmatched = []
    for nm in sorted(set(str(n) for n in region_names)):
        lobe = region_to_lobe(nm)
        counts[lobe] += 1
        if lobe == "other":
            unmatched.append(nm)
    total = sum(counts.values()) or 1
    matched = total - counts.get("other", 0)
    log.info("  ATLAS region -> lobe coverage (change #2):")
    log.info(f"      matched {matched}/{total} = {100*matched/total:.1f}% of distinct region names")
    for lobe in ["frontal", "parietal", "temporal", "occipital", "insula",
                 "cerebellum", "subcortical", "limbic", "other"]:
        if counts.get(lobe):
            log.info(f"        {lobe:<12}: {counts[lobe]}")
    if unmatched:
        log.warning("  >>> UNMATCHED region names (extend _ATLAS_LOBE_PATTERNS). "
                    f"showing {min(25, len(unmatched))}/{len(unmatched)}:")
        for nm in unmatched[:25]:
            log.warning(f"        {nm}")
        if 100 * matched / total < 75:
            log.warning("  >>> Lobe coverage < 75%. The lobe rollup will be weak until you "
                        "fix the map; the model cannot read lobes it can't see. This was very "
                        "likely the root cause of the per-hemisphere template.")


# ── (change #1) Pivot now carries pct_of_region + is_top5 when present ─────────
def pivot_matrix(case_df):
    has_pct = "pct_of_tumor" in case_df.columns
    val_col = "pct_of_tumor" if has_pct else "count_scaled"
    pv = case_df.pivot_table(
        index="atlas_region_name", columns="tumor_tag",
        values=val_col, aggfunc="max", fill_value=0,
    )
    for tag in ["ET", "TC", "WT"]:
        if tag not in pv.columns:
            pv[tag] = 0
    pv = pv[["ET", "TC", "WT"]].copy()

    if "pct_of_region" in case_df.columns:
        pr = case_df.groupby("atlas_region_name")["pct_of_region"].max()
        pv["pct_region"] = pr.reindex(pv.index).fillna(0.0)
    else:
        pv["pct_region"] = 0.0

    if "is_top5" in case_df.columns:
        t5 = case_df.groupby("atlas_region_name")["is_top5"].any()
        pv["is_top5"] = t5.reindex(pv.index).fillna(False).astype(bool)
    else:
        pv["is_top5"] = False
    return pv


_SALIENCE_COLS = ["ET", "TC", "WT"]


def _ordered_active(pivot, banded_shuffle=False, full_shuffle=False):
    sal = pivot[_SALIENCE_COLS]
    active = pivot[(sal > 0).any(axis=1)].copy()
    active["_total"] = active[_SALIENCE_COLS].sum(axis=1)
    active = active.sort_values("_total", ascending=False)
    if full_shuffle:
        # Shuffle ALL active regions (ignores salience order). Stronger than
        # banded — useful precisely because pct_of_tumor floats rarely tie, so
        # banded_shuffle barely permutes anything.
        idx = list(active.index)
        random.shuffle(idx)
        active = active.loc[idx]
    elif banded_shuffle:
        new_index = []
        for _, grp in active.groupby("_total", sort=False):
            idx = list(grp.index)
            random.shuffle(idx)
            new_index.extend(idx)
        active = active.loc[new_index]
    return active.drop(columns="_total")


def _region_salience(row):
    wt = float(row.get("WT", 0))
    return wt if wt > 0 else float(row.get("ET", 0) + row.get("TC", 0) + row.get("WT", 0))


def summarize_lobe_involvement(active):
    """Aggregate per-region tumor share into a lobe breakdown (change #2)."""
    buckets = defaultdict(float)
    for region_name, row in active.iterrows():
        buckets[region_to_lobe(region_name)] += _region_salience(row)
    grand = sum(buckets.values()) or 1.0
    ordered = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    return [(lobe, round(val / grand * 100, 1)) for lobe, val in ordered if val > 0]


def matrix_to_context(case_id, pivot, banded_shuffle=False, full_shuffle=False, lobe_rollup=True):
    """Plain-text matrix body (no instruction/field — those live in
    build_field_prompt). v4: real lobe rollup, per-region [lobe] tags, region
    occupancy, top-5 mark, pct_of_tumor salience."""
    active = _ordered_active(pivot, banded_shuffle=banded_shuffle, full_shuffle=full_shuffle)
    region_names = [str(r) for r in active.index]
    hemisphere   = infer_hemisphere(region_names)

    lines = [
        MATRIX_HEADER,
        f"Patient: {case_id}",
        f"Dominant hemisphere: {hemisphere}",
    ]
    if lobe_rollup:
        involve = summarize_lobe_involvement(active)
        lobe_str = ", ".join(f"{lobe} {pct:.0f}%" for lobe, pct in involve) if involve else "unspecified"
        lines.append(f"Lobe involvement: {lobe_str}")
    lines.append("Atlas involvement (values = % of that tumor compartment; * = top-5 region):")
    for region_name, row in active.iterrows():
        parts = []
        if row["ET"] > 0: parts.append(f"ET={row['ET']:.0f}")
        if row["TC"] > 0: parts.append(f"TC={row['TC']:.0f}")
        if row["WT"] > 0: parts.append(f"WT={row['WT']:.0f}")
        pr = float(row.get("pct_region", 0) or 0)
        extra = f", region%={pr:.0f}" if pr > 0 else ""
        star = "*" if bool(row.get("is_top5", False)) else " "
        lobe_tag = f" [{region_to_lobe(region_name)}]" if lobe_rollup else ""
        lines.append(f" {star}- {region_name}{lobe_tag}: {', '.join(parts)}{extra}")
    lines.append("ET=Enhancing Tumor  TC=Tumor Core/necrosis  WT=Whole Tumor/edema")
    return "\n".join(lines)


def build_field_prompt(context, field):
    return (
        f"{FIELD_TASK[field]}\n"
        f"Use only the atlas matrix below. Answer in ONE concise line.\n\n"
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


# ── Report parsing -> CANONICAL FIELD DICT (unchanged) ────────────────────────
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


# ── (change #4/#6) LOBE EXTRACTION FROM REPORT TEXT ───────────────────────────
# Distinct from region_to_lobe: reports already speak in lobe words, so match
# them directly. Used for content-based selection AND GT diversity diagnostics.
_REPORT_LOBE_WORDS = {
    "frontal":    re.compile(r"frontal", re.I),
    "parietal":   re.compile(r"parietal", re.I),
    "temporal":   re.compile(r"temporal", re.I),
    "occipital":  re.compile(r"occipital", re.I),
    "insula":     re.compile(r"insula|insular", re.I),
    "cerebellum": re.compile(r"cerebell", re.I),
    "brainstem":  re.compile(r"brain.?stem|pons|medulla|midbrain", re.I),
}


def lobes_in_text(t):
    return frozenset(l for l, p in _REPORT_LOBE_WORDS.items() if p.search(t or ""))


def lobe_jaccard_text(pred, ref):
    a, b = lobes_in_text(pred), lobes_in_text(ref)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Data building + DIAGNOSTICS ───────────────────────────────────────────────
_REQUIRED_CSV_COLS = {"case_id", "atlas_region_name", "tumor_tag"}  # value cols detected below


def build_cases(csv_path, text_dir, log, min_fields=2, lobe_rollup=True):
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if "pct_of_tumor" not in df.columns and "count_scaled" not in df.columns:
        raise ValueError("CSV needs at least one of: pct_of_tumor, count_scaled")
    if "is_top5" in df.columns:  # csv.DictWriter wrote bools as 'True'/'False' strings
        df["is_top5"] = df["is_top5"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    rich = "pct_of_tumor" in df.columns
    log.info(f"  {len(df)} rows | {df['case_id'].nunique()} cases in CSV | "
             f"rich_cols={'yes' if rich else 'no (legacy count_scaled)'}")

    cases = []
    n_no_report = 0
    match_hist = Counter()
    hemi_agree = hemi_total = 0
    gt_distinct_ratios = []
    gt_lobe_sets = Counter()
    gt_hemi = Counter()
    gt_lesion_strings = []
    all_region_names = set()

    for case_id, case_df in df.groupby("case_id"):
        fields, n_matched = load_report_fields(text_dir, case_id)
        if fields is None:
            n_no_report += 1
            continue
        match_hist[n_matched] += 1
        if n_matched < min_fields:
            continue

        pv = pivot_matrix(case_df)
        all_region_names.update(str(r) for r in pv.index)
        context = matrix_to_context(case_id, pv, banded_shuffle=False, lobe_rollup=lobe_rollup)

        prompt_hemi = infer_hemisphere([str(r) for r in pv.index])
        ghemi = report_hemisphere(fields.get("lesion"))
        if ghemi != "UNK" and prompt_hemi != "UNSPECIFIED":
            hemi_total += 1
            if prompt_hemi == ghemi:
                hemi_agree += 1

        canon = {k: (fields.get(k) or "not observed") for k in FIELDS}
        vals = [canon[k].strip().lower() for k in FIELDS]
        gt_distinct_ratios.append(len(set(vals)) / len(FIELDS))

        gt_lobe_sets[lobes_in_text(canon["lesion"])] += 1
        gt_hemi[ghemi] += 1
        gt_lesion_strings.append(canon["lesion"].strip().lower())

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

    # ── change #6: GT diversity / ceiling ──────────────────────────────────────
    if cases:
        n = len(cases)
        uniq_lesions = len(set(gt_lesion_strings))
        log.info("\n  --- GT DIVERSITY (the ceiling your model can reach) ---")
        log.info(f"  Cross-patient GT lesion distinctness: {uniq_lesions}/{n} = {uniq_lesions/n:.2f} unique")
        log.info("  GT hemisphere distribution:")
        for h, c in gt_hemi.most_common():
            log.info(f"        {h:<11}: {c:>4} ({100*c/n:.1f}%)")
        log.info("  GT lesion lobe-set distribution (top 8):")
        for lobeset, c in gt_lobe_sets.most_common(8):
            label = "+".join(sorted(lobeset)) if lobeset else "(none)"
            log.info(f"        {label:<28}: {c:>4} ({100*c/n:.1f}%)")
        top_share = gt_lobe_sets.most_common(1)[0][1] / n if gt_lobe_sets else 0
        if top_share > 0.5:
            log.warning(f"  >>> The single most common lobe-set covers {100*top_share:.0f}% of cases. "
                        "A per-hemisphere template will already score high — interpret "
                        "ROUGE/BERTScore with that in mind, and lean on the cross-patient monitor.")

    # ── change #2: lobe coverage over the atlas names actually used ────────────
    if lobe_rollup and all_region_names:
        log.info("")
        lobe_coverage_report(all_region_names, log)
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
                 augment=False, p_swap=0.15, p_shuffle=0.0, lobe_rollup=True,
                 shuffle_mode="banded"):
        self.examples = examples
        self.tok = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        self.augment = augment
        self.p_swap = p_swap
        self.p_shuffle = p_shuffle
        self.lobe_rollup = lobe_rollup
        self.shuffle_mode = shuffle_mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        target = ex["target"]

        if self.augment and self.p_shuffle > 0 and random.random() < self.p_shuffle:
            context = matrix_to_context(
                ex["case_id"], ex["pivot"],
                banded_shuffle=(self.shuffle_mode == "banded"),
                full_shuffle=(self.shuffle_mode == "full"),
                lobe_rollup=self.lobe_rollup)
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
    """v2.1 WITHIN-report monitor: are the 4 fields of ONE report different?"""
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
    log.info(f"  Within-report distinctness: collapse={cr*100:.0f}%  distinct_ratio={dr:.2f}  "
             f"(0.25=full collapse, 1.0=all differ)")
    return {"collapse_rate": cr, "distinct_ratio": dr}


def quick_cross_patient_check(model, tokenizer, val_cases, device, max_new, max_input, log,
                              fields=("lesion", "necrosis"), n=20):
    """(change #3) Exposes TEMPLATE collapse: generate ONE field across N
    DIFFERENT patients and count unique outputs. Low ratio = same thing to
    everyone. The within-report check above is blind to this."""
    model.eval()
    chosen = random.sample(val_cases, min(n, len(val_cases)))
    out = {}
    for field in fields:
        outs = []
        for c in chosen:
            prompt = build_field_prompt(c["context"], field)
            try:
                outs.append(generate_field(model, tokenizer, prompt, device, max_new, max_input).strip().lower())
            except Exception:
                outs.append("")
        nonempty = [o for o in outs if o]
        ratio = len(set(nonempty)) / len(nonempty) if nonempty else 0.0
        out[field] = ratio
        log.info(f"  Cross-patient[{field}]: {len(set(nonempty))}/{len(nonempty)} unique = {ratio:.2f}  "
                 f"(1.0=all differ, low=TEMPLATE COLLAPSE)")
    return out


def eval_content_score(model, tokenizer, val_cases, device, max_new, max_input, log, n=24):
    """(change #4) Generation-based selection signal. lesion-only to stay cheap:
    0.5*lobe_jaccard(text) + 0.5*hemisphere_acc. Higher is better."""
    model.eval()
    chosen = random.sample(val_cases, min(n, len(val_cases)))
    jaccs = []
    hemi_hit = hemi_tot = 0
    for c in chosen:
        prompt = build_field_prompt(c["context"], "lesion")
        try:
            pred = generate_field(model, tokenizer, prompt, device, max_new, max_input)
        except Exception:
            continue
        ref = c["fields"]["lesion"]
        jaccs.append(lobe_jaccard_text(pred, ref))
        g = report_hemisphere(ref)
        if g != "UNK":
            hemi_tot += 1
            if g == report_hemisphere(pred):
                hemi_hit += 1
    lobe = sum(jaccs) / len(jaccs) if jaccs else 0.0
    hemi = hemi_hit / hemi_tot if hemi_tot else 0.0
    score = 0.5 * lobe + 0.5 * hemi
    log.info(f"  Content score (lesion): lobe_jacc={lobe:.3f}  hemi={hemi:.3f}  -> {score:.3f}")
    return {"score": score, "lobe_jaccard": lobe, "hemi_acc": hemi}


# ── Per-model training ────────────────────────────────────────────────────────
def train_one_model(model_key, ft_mode, args, cases, log):
    cfg = MODELS[model_key]
    run_key = f"{model_key}-{ft_mode}"
    out_dir = Path(args.output_root) / run_key

    if args.skip_done and (out_dir / "summary.json").exists():
        log.info(f"\n[SKIP] {run_key} — summary.json exists. Pass --no-skip_done to retrain.")
        with open(out_dir / "summary.json") as f:
            return json.load(f)

    target_modules = resolve_target_modules(cfg, args.lora_widen)
    log.info("\n" + "#" * 70)
    log.info(f"# Training model: {model_key}  ({cfg['name']})  [ft_mode={ft_mode}]")
    log.info(f"#   augment={cfg['augment']}  aug_preset={args.aug_preset}  "
             f"p_swap={args.p_swap}  p_shuffle={args.p_shuffle}  shuffle_mode={args.shuffle_mode}")
    if ft_mode == "lora":
        log.info(f"#   LoRA r={args.lora_r}  targets={target_modules}  (widen={args.lora_widen})")
    else:
        log.info(f"#   FULL fine-tuning (all weights trainable, no LoRA)")
    log.info(f"#   lobe_rollup={args.lobe_rollup}  select_metric={args.select_metric}")
    log.info(f"#   max_input={cfg['max_input']}  max_target={cfg['max_target']}")
    log.info("#" * 70)
    if cfg["augment"] and abs(args.p_swap - 0.15) > 1e-9:
        log.warning(f"#   NOTE p_swap={args.p_swap}. v4 default is 0.15 (hemisphere swap hurt "
                    "hemisphere accuracy in v2.1). Prior runs used 0.5.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cases, val_cases = train_test_split(cases, test_size=0.20, random_state=args.seed, shuffle=True)
    train_examples = expand_to_examples(train_cases)
    val_examples   = expand_to_examples(val_cases)
    log.info(f"Cases  -> train {len(train_cases)} | val {len(val_cases)}")
    log.info(f"Examples (x4 fields) -> train {len(train_examples)} | val {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)

    if ft_mode == "lora":
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
    else:  # full fine-tuning — every weight trainable, no adapters
        for p in model.parameters():
            p.requires_grad_(True)
        # If the *large* model OOMs on a 16GB card, uncomment the next two lines
        # (turns off the KV cache during teacher-forced training; generation in
        # the monitors re-enables it per call).
        # model.gradient_checkpointing_enable()
        # model.config.use_cache = False
        n_all = sum(p.numel() for p in model.parameters())
        n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"FULL FT trainable params: {n_tr:,} / {n_all:,} ({100*n_tr/n_all:.2f}%)")
    model.to(device)

    train_ds = FieldDataset(train_examples, tokenizer, cfg["max_input"], cfg["max_target"],
                            augment=cfg["augment"], p_swap=args.p_swap, p_shuffle=args.p_shuffle,
                            lobe_rollup=args.lobe_rollup, shuffle_mode=args.shuffle_mode)
    val_ds   = FieldDataset(val_examples, tokenizer, cfg["max_input"], cfg["max_target"],
                            augment=False, lobe_rollup=args.lobe_rollup)
    collate = make_collate(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"))

    if ft_mode == "full":
        lr = args.lr_large_full if "large" in model_key else args.lr_full
    else:
        lr = args.lr_large if "large" in model_key else args.lr
    log.info(f"Using LR: {lr}  (ft_mode={ft_mode})  |  label_smoothing: {args.label_smoothing}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    accum = args.grad_accum
    total_steps = max(1, (len(train_loader) // accum) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    # (change #4) selection on a "higher is better" value: content score, or
    # -val_loss when --select_metric val_loss (or content disabled).
    best_sel = -float("inf")
    best_val, best_epoch, best_previews, best_content, no_improve, history = float("inf"), 0, [], None, 0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = grad_sum = 0.0
        n_upd = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{run_key}] E{epoch}/{args.epochs} train")
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
            for batch in tqdm(val_loader, desc=f"[{run_key}] E{epoch}/{args.epochs} val  "):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                if torch.isfinite(out.loss):
                    val_loss += out.loss.item()
        avg_val = val_loss / max(1, len(val_loader))
        v_ppl   = math.exp(min(avg_val, 20)) if math.isfinite(avg_val) else float("nan")

        log.info(f"[{run_key}] E{epoch:>3}  train={avg_train:.4f} (ppl {tr_ppl:.2f})  "
                 f"val={avg_val:.4f} (ppl {v_ppl:.2f})  gn={avg_gn:.3f}")

        # ── Generation-based monitors ─────────────────────────────────────────
        content = None
        if args.content_check_n > 0:
            content = eval_content_score(model, tokenizer, val_cases, device,
                                         cfg["max_target"], cfg["max_input"], log, n=args.content_check_n)

        hemi_acc = distinct = cross = None
        if epoch % args.hemi_check_every == 0:
            hemi_acc = quick_hemisphere_check(model, tokenizer, val_cases, device,
                                              cfg["max_target"], cfg["max_input"], log, n=args.hemi_check_n)
            distinct = quick_field_distinctness_check(model, tokenizer, val_cases, device,
                                                      cfg["max_target"], cfg["max_input"], log,
                                                      n=args.distinct_check_n)
            cross = quick_cross_patient_check(model, tokenizer, val_cases, device,
                                              cfg["max_target"], cfg["max_input"], log,
                                              n=args.cross_patient_n)
        previews = preview_generations(model, tokenizer, val_cases, device,
                                       cfg["max_target"], cfg["max_input"], log)

        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
                        "train_ppl": tr_ppl, "val_ppl": v_ppl, "grad_norm": avg_gn,
                        "content_score": (content or {}).get("score"),
                        "content_lobe_jaccard": (content or {}).get("lobe_jaccard"),
                        "content_hemi_acc": (content or {}).get("hemi_acc"),
                        "hemisphere_acc": hemi_acc,
                        "collapse_rate": (distinct or {}).get("collapse_rate"),
                        "distinct_ratio": (distinct or {}).get("distinct_ratio"),
                        "cross_patient": cross})

        # ── (change #4) checkpoint selection ──────────────────────────────────
        use_content = (args.select_metric == "content" and content is not None)
        sel_value = content["score"] if use_content else -avg_val
        improved = sel_value > best_sel and math.isfinite(sel_value)
        if improved:
            best_sel = sel_value
            best_val, best_epoch, best_previews, no_improve = avg_val, epoch, previews, 0
            best_content = content
            model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
            tag = f"content={content['score']:.3f}" if use_content else f"val={avg_val:.4f}"
            log.info(f"  ✓ New best ({tag}) → {out_dir}")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                log.info("Early stopping."); break

    elapsed = time.time() - t0
    summary = {
        "model_key": model_key, "ft_mode": ft_mode, "run_key": run_key,
        "model_name": cfg["name"], "augment": cfg["augment"],
        "lora_r": (args.lora_r if ft_mode == "lora" else None),
        "lora_targets": (target_modules if ft_mode == "lora" else None),
        "lora_widen": args.lora_widen,
        "lobe_rollup": args.lobe_rollup, "select_metric": args.select_metric,
        "learning_rate": lr,
        "max_input": cfg["max_input"], "max_target": cfg["max_target"],
        "label_smoothing": args.label_smoothing, "p_swap": args.p_swap, "p_shuffle": args.p_shuffle,
        "best_val_loss": best_val, "best_content": best_content, "best_epoch": best_epoch,
        "epochs_run": len(history), "elapsed_sec": elapsed, "elapsed_min": elapsed / 60,
        "history": history, "best_previews": best_previews,
        "mode": "field_wise_multitask_matrix_v4",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[{run_key}] done in {elapsed/60:.1f} min. Best val={best_val:.4f} at epoch {best_epoch}"
             + (f" (content={best_content['score']:.3f})" if best_content else ""))

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
# ── (augmentation sweep) presets: (p_swap, p_shuffle) ─────────────────────────
# Only the 3 aug models in TRAIN_ORDER consume these; the no-aug models ignore
# them, so the aug-vs-no-aug structure of the 6-model suite is preserved. Sweep
# the whole suite with one flag, e.g. --aug_preset shuffle_only.
_AUG_PRESETS = {
    "tuned":        (0.15, 0.50),   # default: swap DOWN (it hurt hemi acc), shuffle ON
    "legacy":       (0.50, 0.00),   # reproduce the old v3 aug runs
    "swap_only":    (0.50, 0.00),
    "shuffle_only": (0.00, 0.50),
    "both_heavy":   (0.50, 0.50),
    "light":        (0.15, 0.15),
    "off":          (0.00, 0.00),
}


def resolve_aug(args):
    """Preset sets p_swap/p_shuffle; explicit --p_swap/--p_shuffle override it."""
    ps, psh = _AUG_PRESETS[args.aug_preset]
    if args.p_swap is not None:
        ps = args.p_swap
    if args.p_shuffle is not None:
        psh = args.p_shuffle
    args.p_swap, args.p_shuffle = ps, psh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",    default="../TextBraTSData")
    p.add_argument("--output_root", default="checkpoints_multi_v4")
    p.add_argument("--model",       default="all", help=f"all | {' | '.join(MODELS.keys())}")
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--lr_large",      type=float, default=1e-5)
    # full fine-tuning LRs (much lower than LoRA — all weights move)
    p.add_argument("--lr_full",       type=float, default=1e-5,
                   help="LR for FULL fine-tuning (non-large models).")
    p.add_argument("--lr_large_full", type=float, default=5e-6,
                   help="LR for FULL fine-tuning of *large* models.")
    p.add_argument("--patience",      type=int,   default=12)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--min_fields",    type=int,   default=2)
    p.add_argument("--lora_r",        type=int,   default=32)
    p.add_argument("--lora_widen",    action="store_true", default=True)
    p.add_argument("--no-lora_widen", dest="lora_widen", action="store_false")
    # fine-tune-mode axis: each model runs once per selected mode (both -> 12 runs)
    p.add_argument("--ft_mode", choices=["lora", "full", "both"], default="both",
                   help="Per-model fine-tune mode. 'both' runs each model twice "
                        "(LoRA + full fine-tune) -> 12 runs total.")
    # (change #2)
    p.add_argument("--lobe_rollup",    action="store_true", default=True,
                   help="Inject explicit area->lobe rollup into the matrix context.")
    p.add_argument("--no-lobe_rollup", dest="lobe_rollup", action="store_false")
    # (change #5 / augmentation sweep) preset sets these; flags override.
    p.add_argument("--aug_preset", choices=list(_AUG_PRESETS.keys()), default="tuned",
                   help="(p_swap,p_shuffle) preset for the aug models. Sweep the whole "
                        "6-model suite with one flag. tuned=(0.15,0.5).")
    p.add_argument("--shuffle_mode", choices=["banded", "full"], default="banded",
                   help="Region-shuffle aug. 'banded' permutes equal-salience regions only "
                        "(weak with float pct_of_tumor); 'full' shuffles all active regions.")
    p.add_argument("--p_swap",        type=float, default=None,
                   help="Override preset hemisphere-swap prob.")
    p.add_argument("--p_shuffle",     type=float, default=None,
                   help="Override preset region-shuffle prob.")
    # (change #4)
    p.add_argument("--select_metric", choices=["content", "val_loss"], default="content",
                   help="Checkpoint selection signal. 'content' avoids the teacher-forced trap.")
    p.add_argument("--content_check_n", type=int, default=24,
                   help="Val cases for the content score each epoch. 0 disables (-> val_loss).")
    # monitors
    p.add_argument("--hemi_check_every", type=int, default=5)
    p.add_argument("--hemi_check_n",  type=int, default=20)
    p.add_argument("--distinct_check_n", type=int, default=10)
    p.add_argument("--cross_patient_n", type=int, default=20,
                   help="Cases for the cross-patient TEMPLATE-collapse monitor (change #3)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--audit_only",    action="store_true")
    p.add_argument("--skip_done",     action="store_true", default=True)
    p.add_argument("--no-skip_done",  dest="skip_done", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    resolve_aug(args)   # preset -> p_swap/p_shuffle (flags override)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"multi_model_v4_{ts}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()], force=True)
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Field-wise multi-task BraTS matrix -> report training (v4, anti cross-patient collapse)")
    log.info("=" * 70)
    for k in ["csv", "text_dir", "output_root", "model", "ft_mode", "epochs", "patience",
              "batch_size", "grad_accum", "lr", "lr_large", "lr_full", "lr_large_full",
              "label_smoothing", "min_fields", "lora_r", "lora_widen", "lobe_rollup",
              "select_metric", "content_check_n", "aug_preset", "shuffle_mode",
              "p_swap", "p_shuffle", "skip_done"]:
        log.info(f"  {k:<16}: {getattr(args, k)}")
    log.info("=" * 70)

    cases = build_cases(args.csv, Path(args.text_dir), log,
                        min_fields=args.min_fields, lobe_rollup=args.lobe_rollup)
    if args.audit_only:
        log.info("audit_only set — diagnostics printed above. Exiting before training.")
        return
    if len(cases) < 2:
        raise ValueError("Not enough usable cases after filtering. Check diagnostics above.")

    model_keys = TRAIN_ORDER if args.model == "all" else [args.model]
    if args.model != "all" and args.model not in MODELS:
        raise ValueError(f"Unknown model {args.model}. Options: all, {list(MODELS.keys())}")

    modes = FT_MODES if args.ft_mode == "both" else [args.ft_mode]
    # interleave lora+full per model so partial runs still give a comparison
    runs = [(key, mode) for key in model_keys for mode in modes]

    all_summaries, t_global = [], time.time()
    for i, (key, mode) in enumerate(runs, 1):
        log.info(f"\n>>> Run {i}/{len(runs)}  [{key}-{mode}]  (elapsed: {(time.time()-t_global)/60:.1f} min)")
        try:
            all_summaries.append(train_one_model(key, mode, args, cases, log))
        except Exception as e:
            log.exception(f"[{key}-{mode}] training FAILED: {e}")
            all_summaries.append({"model_key": key, "ft_mode": mode,
                                  "run_key": f"{key}-{mode}",
                                  "model_name": MODELS[key]["name"], "error": str(e)})
            torch.cuda.empty_cache()

    out_path = Path("logs") / f"summary_v4_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"\nComparison summary → {out_path}")

    log.info("\n" + "=" * 70)
    log.info("FINAL RANKING BY BEST VAL LOSS")
    log.info("=" * 70)
    rank = sorted([s for s in all_summaries if "best_val_loss" in s and math.isfinite(s["best_val_loss"])],
                  key=lambda x: x["best_val_loss"])
    for i, s in enumerate(rank, 1):
        tag  = " [aug]" if s.get("augment") else "      "
        mode = s.get("ft_mode", "lora")
        cs   = s.get("best_content") or {}
        cstr = f"  content={cs['score']:.3f}" if cs.get("score") is not None else ""
        log.info(f"  {i}. {s['model_key']:<22}{tag} [{mode:<4}] val_loss={s['best_val_loss']:.4f}{cstr}  "
                 f"epoch {s['best_epoch']}/{s['epochs_run']}  ({s['elapsed_min']:.1f} min)")

    failed = [s for s in all_summaries if "error" in s]
    if failed:
        log.info("\n  FAILED:")
        for s in failed:
            log.info(f"    {s.get('run_key', s['model_key'])}: {s['error']}")


if __name__ == "__main__":
    main()