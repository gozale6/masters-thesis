"""
eval_multi_model_v2.py
======================
Evaluation for the FIELD-WISE multi-task checkpoints produced by
train_multi_model_v2.py.

WHAT'S DIFFERENT FROM v1 EVAL (and why)
---------------------------------------
1. GENERATION matches training. v2 trains one example per field; a report is
   produced by querying the model FOUR times (one per field) and ASSEMBLING the
   4-line skeleton. This eval does exactly that via predict_report(). It does
   NOT do a single full-report generate — that would not match how the model
   was trained and would score garbage.

2. PROMPT/PARSE helpers are copied verbatim from train_multi_model_v2.py so the
   eval prompts are byte-identical to training prompts. >>> IF YOU CHANGE THE
   PROMPT IN TRAINING (e.g. front-load the TASK line), MAKE THE SAME CHANGE HERE
   or the checkpoint will receive prompts it never saw. <<<

3. Val split matches training. We rebuild cases the SAME way (same --min_fields)
   and split on CASES with the same seed, so the val set is identical.

4. Hemisphere accuracy is computed on the LESION FIELD only (the v1 metric
   scanned the whole report and spuriously flagged BILATERAL). Added a
   lesion-field Lobe Jaccard for the same reason.

5. NEW field-collapse diagnostic. The first v2 model emitted the SAME text for
   all four fields (it ignored the task selector). field_collapse_rate measures
   how often all four predicted fields are identical, and distinct_field_ratio
   reports the average number of unique field values / 4. Watch these: a high
   collapse rate means the model isn't conditioning on the field instruction,
   regardless of how good ROUGE looks.

Usage
-----
    python eval_multi_model_v2.py
    python eval_multi_model_v2.py --model scifive-base
    python eval_multi_model_v2.py --models scifive-base clinicalt5-hoss
"""

import argparse
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ── Model registry (mirror train_multi_model_v2.py) ──────────────────────────
MODELS = {
    "scifive-base":        {"name": "razent/SciFive-base-PMC",     "max_input": 640, "max_target": 160, "augment": False},
    "scifive-base-aug":    {"name": "razent/SciFive-base-PMC",     "max_input": 640, "max_target": 160, "augment": True},
    "scifive-large":       {"name": "razent/SciFive-large-Pubmed", "max_input": 640, "max_target": 160, "augment": False},
    "scifive-large-aug":   {"name": "razent/SciFive-large-Pubmed", "max_input": 640, "max_target": 160, "augment": True},
    "clinicalt5-hoss":     {"name": "hossboll/clinical-t5",        "max_input": 640, "max_target": 160, "augment": False},
    "clinicalt5-hoss-aug": {"name": "hossboll/clinical-t5",        "max_input": 640, "max_target": 160, "augment": True},
}

DISPLAY_ORDER = [
    "scifive-base", "scifive-base-aug",
    "clinicalt5-hoss", "clinicalt5-hoss-aug",
    "scifive-large", "scifive-large-aug",
]
PAIRS = [
    ("scifive-base", "scifive-base-aug"),
    ("clinicalt5-hoss", "clinicalt5-hoss-aug"),
    ("scifive-large", "scifive-large-aug"),
]

# ── Field definitions (mirror training) ──────────────────────────────────────
FIELDS = ["lesion", "edema", "necrosis", "compression"]

FIELD_TASK = {
    "lesion":      "TASK: Report the LESION AREA (lobe(s), hemisphere, signal characteristics).",
    "edema":       "TASK: Report the EDEMA (location and extent).",
    "necrosis":    "TASK: Report the NECROSIS (location and signal, or 'not observed').",
    "compression": "TASK: Report VENTRICULAR COMPRESSION (description, or 'not observed').",
}

REPORT_FORMAT = (
    "Lesion area: {lesion}\n"
    "Edema: {edema}\n"
    "Necrosis: {necrosis}\n"
    "Ventricular compression: {compression}"
)

CONTEXT_PREFIX = (
    "You are a neuroradiology assistant. Below is a brain tumor atlas matrix for "
    "one patient. Answer ONLY the requested field in one concise line.\n\n"
    "Atlas matrix:\n"
)

# ── Atlas helpers (mirror training) ──────────────────────────────────────────
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


def _ordered_active(pivot):
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False)
    return active.drop(columns="_total")


def matrix_to_context(case_id, pivot):
    active = _ordered_active(pivot)
    region_names = [str(r) for r in active.index]
    hemisphere   = infer_hemisphere(region_names)
    primary      = infer_primary_lobes(active)
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
    return CONTEXT_PREFIX + "\n".join(lines)


def build_field_prompt(context, field):
    return f"{context}\n\n{FIELD_TASK[field]}"


# ── Report parsing (mirror training) ─────────────────────────────────────────
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
    if re.search(r"\b(bilateral|both)\b", t): return "BILATERAL"
    has_l = bool(re.search(r"\bleft\b", t))
    has_r = bool(re.search(r"\bright\b", t))
    if has_l and has_r: return "BILATERAL"
    if has_l: return "LEFT"
    if has_r: return "RIGHT"
    return "UNK"


# ── Build cases (mirror training, with same filtering) ───────────────────────
_REQUIRED_CSV_COLS = {"case_id", "atlas_region_name", "tumor_tag", "count_scaled"}


def build_cases(csv_path, text_dir, log, min_fields=2):
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    cases = []
    for case_id, case_df in df.groupby("case_id"):
        fields, n_matched = load_report_fields(text_dir, case_id)
        if fields is None or n_matched < min_fields:
            continue
        pv = pivot_matrix(case_df)
        canon = {k: (fields.get(k) or "not observed") for k in FIELDS}
        cases.append({
            "case_id": case_id,
            "pivot":   pv,
            "context": matrix_to_context(case_id, pv),
            "fields":  canon,
        })
    log.info(f"  Usable cases (>= {min_fields} fields): {len(cases)}")
    return cases


# ── Generation: per-field + assemble (mirror training) ───────────────────────
def generate_field(model, tokenizer, prompt, device, max_new_tokens, max_input):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_input).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            length_penalty=1.0,
        )
    return clean_text(tokenizer.decode(out[0], skip_special_tokens=True).strip())


def predict_report(model, tokenizer, case, device, max_new_tokens, max_input):
    """Query once per field, assemble the 4-line report. Returns (report, fields)."""
    values = {}
    for field in FIELDS:
        prompt = build_field_prompt(case["context"], field)
        val = generate_field(model, tokenizer, prompt, device, max_new_tokens, max_input)
        values[field] = val or "not observed"
    return REPORT_FORMAT.format(**values), values


def ground_truth_report(case):
    return REPORT_FORMAT.format(**case["fields"])


# ── Metric helpers ───────────────────────────────────────────────────────────
def _safe_tokenize(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


# ── Lexical / n-gram metrics ─────────────────────────────────────────────────
def compute_rouge(predictions, references, log):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        log.warning("rouge-score not installed — skipping ROUGE")
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1) / len(r1) if r1 else 0.0,
        "rouge2": sum(r2) / len(r2) if r2 else 0.0,
        "rougeL": sum(rL) / len(rL) if rL else 0.0,
    }


def compute_bleu_n(predictions, references, log):
    out = {}
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sm = SmoothingFunction().method1
        refs  = [[_safe_tokenize(r)] for r in references]
        preds = [_safe_tokenize(p) for p in predictions]
        for n in (1, 2, 3, 4):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                out[f"bleu{n}"] = corpus_bleu(refs, preds, weights=weights, smoothing_function=sm)
            except Exception as e:
                log.warning(f"NLTK BLEU-{n} failed: {e}")
    except ImportError:
        log.warning("nltk not installed — skipping BLEU-1..4")
    return out


def compute_meteor(predictions, references, log):
    try:
        import nltk
        for res in ("corpora/wordnet", "corpora/omw-1.4"):
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(res.split("/")[-1], quiet=True)
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        log.warning("nltk not installed — skipping METEOR")
        return {}
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            scores.append(meteor_score([_safe_tokenize(ref)], _safe_tokenize(pred)))
        except Exception:
            pass
    return {"meteor": sum(scores) / len(scores) if scores else 0.0}


def compute_cider(predictions, references, log):
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        log.warning("pycocoevalcap not installed — skipping CIDEr")
        return {}
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    try:
        score, _ = Cider().compute_score(gts, res)
        return {"cider": float(score)}
    except Exception as e:
        log.warning(f"CIDEr failed: {e}")
        return {}


def compute_chrf(predictions, references, log):
    try:
        import sacrebleu
    except ImportError:
        return {}
    try:
        chrf  = sacrebleu.corpus_chrf(predictions, [references], word_order=0).score / 100.0
        chrf2 = sacrebleu.corpus_chrf(predictions, [references], word_order=2).score / 100.0
        return {"chrf": chrf, "chrf++": chrf2}
    except Exception as e:
        log.warning(f"chrF failed: {e}")
        return {}


# ── Semantic similarity ──────────────────────────────────────────────────────
def compute_bertscore(predictions, references, log, device):
    try:
        from bert_score import score as bertscore_fn
    except ImportError:
        log.warning("bert-score not installed — skipping BERTScore")
        return {}
    try:
        P, R, F1 = bertscore_fn(predictions, references, lang="en", device=device,
                                verbose=False, rescale_with_baseline=False)
        return {"bertscore_p": float(P.mean()), "bertscore_r": float(R.mean()),
                "bertscore_f1": float(F1.mean())}
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")
        return {}


# ── Intrinsic text quality ───────────────────────────────────────────────────
def compute_lexical_diversity(predictions, log):
    ttrs, maas = [], []
    for p in predictions:
        toks = _safe_tokenize(p)
        n_tokens, n_types = len(toks), len(set(toks))
        if n_tokens == 0:
            continue
        ttrs.append(n_types / n_tokens)
        if n_tokens > 1 and n_types > 0:
            log_n, log_t = math.log(n_tokens), math.log(n_types)
            if log_n ** 2 > 0:
                maas.append((log_n - log_t) / (log_n ** 2))
    return {"ttr": sum(ttrs) / len(ttrs) if ttrs else 0.0,
            "maas": sum(maas) / len(maas) if maas else 0.0}


def compute_readability(predictions, log):
    try:
        import textstat
    except ImportError:
        log.warning("textstat not installed — skipping FRES")
        return {}
    scores = []
    for p in predictions:
        try:
            scores.append(textstat.flesch_reading_ease(p))
        except Exception:
            pass
    return {"fres": sum(scores) / len(scores) if scores else 0.0}


def compute_coherence_and_coverage(predictions, prompts, log, device):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        log.warning("sentence-transformers / sklearn not installed — skipping CohS/ECS/TCS")
        return {}
    log.info("  Loading sentence-transformer (all-MiniLM-L6-v2) …")
    try:
        st = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    except Exception as e:
        log.warning(f"Could not load sentence-transformer: {e}")
        return {}

    coh_scores, ecs_scores, tcs_scores = [], [], []

    def split_sents(text):
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [s.strip() for s in parts if s.strip()]

    STOPWORDS = set("""a an the and or but if then else for of in on at to with by from is are was were be been being
                       has have had do does did this that these those it its they them their what which who whom""".split())

    for pred, prompt in zip(predictions, prompts):
        sents = split_sents(pred)
        if len(sents) >= 2:
            try:
                embs = st.encode(sents, convert_to_numpy=True, show_progress_bar=False)
                sims = [float(cosine_similarity(embs[i].reshape(1, -1), embs[i + 1].reshape(1, -1))[0, 0])
                        for i in range(len(sents) - 1)]
                coh_scores.append(sum(sims) / len(sims))
            except Exception:
                pass
        prompt_sents = split_sents(prompt)
        if pred.strip() and prompt_sents:
            try:
                pred_emb = st.encode([pred], convert_to_numpy=True, show_progress_bar=False)
                ref_embs = st.encode(prompt_sents, convert_to_numpy=True, show_progress_bar=False)
                ecs_scores.append(float(cosine_similarity(pred_emb, ref_embs)[0].mean()))
            except Exception:
                pass
        pt = set(t for t in _safe_tokenize(pred)   if t not in STOPWORDS)
        rt = set(t for t in _safe_tokenize(prompt) if t not in STOPWORDS)
        if pt and rt:
            tcs_scores.append(len(pt & rt) / len(pt | rt) if (pt | rt) else 0.0)

    return {"cohs": sum(coh_scores) / len(coh_scores) if coh_scores else 0.0,
            "ecs":  sum(ecs_scores) / len(ecs_scores) if ecs_scores else 0.0,
            "tcs":  sum(tcs_scores) / len(tcs_scores) if tcs_scores else 0.0}


# ── Task-specific (operate on stored field dicts — no fragile re-parsing) ─────
def compute_per_field_rouge1(pred_fields_list, ref_fields_list, log):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    out = {}
    for field in FIELDS:
        scores = []
        for pf, rf in zip(pred_fields_list, ref_fields_list):
            ref_val = rf.get(field, "")
            if ref_val and ref_val.lower() != "not observed":
                scores.append(scorer.score(ref_val, pf.get(field, "") or "")["rouge1"].fmeasure)
        out[f"per_field_{field}"] = sum(scores) / len(scores) if scores else 0.0
    return out


def compute_format_compliance(predictions):
    if not predictions:
        return {"format_compliance": 0.0}
    n_ok = 0
    for p in predictions:
        if (re.search(r"(?mi)^\s*Lesion area:", p) and re.search(r"(?mi)^\s*Edema:", p)
                and re.search(r"(?mi)^\s*Necrosis:", p)
                and re.search(r"(?mi)^\s*Ventricular compression:", p)):
            n_ok += 1
    return {"format_compliance": n_ok / len(predictions)}


def compute_hemisphere_accuracy(pred_fields_list, ref_fields_list):
    """Lesion-field only (fixes v1 whole-report scan bug)."""
    correct = total = 0
    for pf, rf in zip(pred_fields_list, ref_fields_list):
        rh = report_hemisphere(rf.get("lesion"))
        ph = report_hemisphere(pf.get("lesion"))
        if rh != "UNK":
            total += 1
            if rh == ph:
                correct += 1
    return {"hemisphere_accuracy": correct / total if total else 0.0}


def _detect_lobes(text):
    return {lobe for lobe, pat in _LOBE_KEYWORDS.items() if text and pat.search(text)}


def compute_lobe_jaccard(predictions, references):
    """Whole-report lobe overlap (kept for continuity with v1 columns)."""
    scores = []
    for p, r in zip(predictions, references):
        pset, rset = _detect_lobes(p), _detect_lobes(r)
        if rset:
            scores.append(len(pset & rset) / len(pset | rset) if (pset | rset) else 0.0)
    return {"lobe_jaccard": sum(scores) / len(scores) if scores else 0.0}


def compute_lobe_jaccard_lesion(pred_fields_list, ref_fields_list):
    """Lobe overlap on the LESION field only — the localization signal that
    matters. (You flagged a case where hemisphere was right but lobe wrong.)"""
    scores = []
    for pf, rf in zip(pred_fields_list, ref_fields_list):
        pset, rset = _detect_lobes(pf.get("lesion")), _detect_lobes(rf.get("lesion"))
        if rset:
            scores.append(len(pset & rset) / len(pset | rset) if (pset | rset) else 0.0)
    return {"lobe_jaccard_lesion": sum(scores) / len(scores) if scores else 0.0}


def compute_field_collapse(pred_fields_list):
    """NEW. Detects the failure mode where the model emits the same text for all
    four fields (ignores the task selector).
      field_collapse_rate  = fraction of cases with all 4 fields identical
      distinct_field_ratio = avg (# unique field values) / 4   (1.0 = all differ)
    """
    if not pred_fields_list:
        return {"field_collapse_rate": 0.0, "distinct_field_ratio": 0.0}
    collapsed, distinct_ratios = 0, []
    for pf in pred_fields_list:
        vals = [(pf.get(f) or "").strip().lower() for f in FIELDS]
        uniq = len(set(vals))
        distinct_ratios.append(uniq / len(FIELDS))
        if uniq == 1:
            collapsed += 1
    return {"field_collapse_rate": collapsed / len(pred_fields_list),
            "distinct_field_ratio": sum(distinct_ratios) / len(distinct_ratios)}


# ── Length / fluency ─────────────────────────────────────────────────────────
def compute_length_stats(predictions, references):
    if not predictions:
        return {}
    pred_lens = [len(_safe_tokenize(p)) for p in predictions]
    ref_lens  = [len(_safe_tokenize(r)) for r in references]
    avg_pred = sum(pred_lens) / len(pred_lens)
    avg_ref  = sum(ref_lens) / len(ref_lens) if ref_lens else 0.0
    return {"avg_pred_len": avg_pred, "avg_ref_len": avg_ref,
            "len_ratio": avg_pred / avg_ref if avg_ref > 0 else 0.0,
            "empty_rate": sum(1 for p in predictions if not p.strip()) / len(predictions)}


def compute_repetition_rate(predictions, n=3):
    rates = []
    for p in predictions:
        toks = _safe_tokenize(p)
        if len(toks) < n + 1:
            continue
        ngs = _ngrams(toks, n)
        if not ngs:
            continue
        counts = Counter(ngs)
        n_repeat = sum(c for c in counts.values() if c > 1) - len([c for c in counts.values() if c > 1])
        rates.append(n_repeat / len(ngs))
    return {"repetition_3gram": sum(rates) / len(rates) if rates else 0.0}


# ── Master metrics fn ────────────────────────────────────────────────────────
def compute_all_metrics(predictions, references, prompts,
                        pred_fields_list, ref_fields_list, log, device):
    out = {"n_samples": len(predictions)}
    log.info("  → ROUGE");                 out.update(compute_rouge(predictions, references, log))
    log.info("  → BLEU 1-4");              out.update(compute_bleu_n(predictions, references, log))
    log.info("  → METEOR");                out.update(compute_meteor(predictions, references, log))
    log.info("  → CIDEr");                 out.update(compute_cider(predictions, references, log))
    log.info("  → chrF / chrF++");         out.update(compute_chrf(predictions, references, log))
    log.info("  → BERTScore");             out.update(compute_bertscore(predictions, references, log, device))
    log.info("  → Lexical diversity");     out.update(compute_lexical_diversity(predictions, log))
    log.info("  → Readability (FRES)");    out.update(compute_readability(predictions, log))
    log.info("  → Coherence + Coverage");  out.update(compute_coherence_and_coverage(predictions, prompts, log, device))
    log.info("  → Per-field ROUGE-1");     out.update(compute_per_field_rouge1(pred_fields_list, ref_fields_list, log))
    log.info("  → Format compliance");     out.update(compute_format_compliance(predictions))
    log.info("  → Hemisphere accuracy");   out.update(compute_hemisphere_accuracy(pred_fields_list, ref_fields_list))
    log.info("  → Lobe Jaccard");          out.update(compute_lobe_jaccard(predictions, references))
    log.info("  → Lobe Jaccard (lesion)"); out.update(compute_lobe_jaccard_lesion(pred_fields_list, ref_fields_list))
    log.info("  → Field collapse");        out.update(compute_field_collapse(pred_fields_list))
    log.info("  → Length stats");          out.update(compute_length_stats(predictions, references))
    log.info("  → Repetition rate");       out.update(compute_repetition_rate(predictions))
    return out


# ── Per-model evaluation ─────────────────────────────────────────────────────
def evaluate_one(model_key, args, cases, log):
    cfg = MODELS[model_key]
    ckpt_dir = Path(args.checkpoint_root) / model_key
    if not ckpt_dir.exists():
        log.warning(f"[{model_key}] checkpoint not found at {ckpt_dir} — skipping")
        return None

    log.info("\n" + "#" * 70)
    log.info(f"# Evaluating: {model_key}  ({cfg['name']})  augment={cfg['augment']}")
    log.info("#" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"

    # MATCH training split: split on CASES, same seed/test_size.
    _, val_cases = train_test_split(cases, test_size=0.20, random_state=args.seed, shuffle=True)
    log.info(f"Val cases: {len(val_cases)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True)

    log.info(f"Loading base model: {cfg['name']}")
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)
    log.info(f"Applying LoRA from {ckpt_dir}")
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
    model.to(device)
    model.eval()

    predictions, references, prompts = [], [], []
    pred_fields_list, ref_fields_list, case_ids = [], [], []
    t0 = time.time()
    for c in tqdm(val_cases, desc=f"[{model_key}] generate"):
        try:
            pred, pfields = predict_report(model, tokenizer, c, device,
                                           cfg["max_target"], cfg["max_input"])
        except Exception as e:
            log.warning(f"  [{c['case_id']}] gen failed: {e}")
            pred, pfields = "", {f: "" for f in FIELDS}
        predictions.append(pred)
        pred_fields_list.append(pfields)
        references.append(ground_truth_report(c))
        ref_fields_list.append(c["fields"])
        prompts.append(c["context"])     # coverage metrics compare to the matrix context
        case_ids.append(c["case_id"])
    gen_time = time.time() - t0
    log.info(f"  Generated {len(predictions)} reports in {gen_time:.1f}s "
             f"({gen_time/max(1,len(predictions)):.2f}s/sample — note: 4 generations/report)")

    log.info("\nComputing metrics …")
    metrics = compute_all_metrics(predictions, references, prompts,
                                  pred_fields_list, ref_fields_list, log, device_str)

    log.info(f"\n[{model_key}] METRICS")
    _log_metrics(metrics, log)

    out_dir = Path(args.output_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"case_id": cid, "prediction": p, "reference": r,
             "pred_fields": pf, "ref_fields": rf}
            for cid, p, r, pf, rf in zip(case_ids, predictions, references,
                                         pred_fields_list, ref_fields_list)]
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    del model, base
    torch.cuda.empty_cache()
    return {"model_key": model_key, "model_name": cfg["name"], "augment": cfg["augment"],
            "metrics": metrics, "generate_sec": gen_time}


def _log_metrics(m, log):
    def _fmt(k, w=4):
        v = m.get(k)
        if v is None: return "  —"
        return f"{v:.{w}f}" if isinstance(v, float) else str(v)
    log.info("  ── n-gram overlap ──")
    log.info(f"    ROUGE-1 / 2 / L  : {_fmt('rouge1')} / {_fmt('rouge2')} / {_fmt('rougeL')}")
    log.info(f"    BLEU 1 / 2 / 3 / 4 : {_fmt('bleu1')} / {_fmt('bleu2')} / {_fmt('bleu3')} / {_fmt('bleu4')}")
    log.info(f"    METEOR           : {_fmt('meteor')}")
    log.info(f"    CIDEr            : {_fmt('cider')}")
    log.info(f"    chrF / chrF++    : {_fmt('chrf')} / {_fmt('chrf++')}")
    log.info("  ── semantic similarity ──")
    log.info(f"    BERTScore P/R/F1 : {_fmt('bertscore_p')} / {_fmt('bertscore_r')} / {_fmt('bertscore_f1')}")
    log.info("  ── intrinsic text quality ──")
    log.info(f"    TTR / Maas'      : {_fmt('ttr')} / {_fmt('maas')}")
    log.info(f"    FRES             : {_fmt('fres', 2)}")
    log.info(f"    CohS / ECS / TCS : {_fmt('cohs')} / {_fmt('ecs')} / {_fmt('tcs')}")
    log.info("  ── task-specific ──")
    log.info(f"    Per-field ROUGE-1: lesion {_fmt('per_field_lesion')}  edema {_fmt('per_field_edema')}  "
             f"necrosis {_fmt('per_field_necrosis')}  compression {_fmt('per_field_compression')}")
    log.info(f"    Format compliance: {m.get('format_compliance', 0)*100:.1f}%")
    log.info(f"    Hemisphere acc   : {m.get('hemisphere_accuracy', 0)*100:.1f}%  (lesion field)")
    log.info(f"    Lobe Jaccard     : {_fmt('lobe_jaccard')}  | lesion-only {_fmt('lobe_jaccard_lesion')}")
    log.info("  ── field conditioning (NEW) ──")
    log.info(f"    Field collapse   : {m.get('field_collapse_rate', 0)*100:.1f}%  "
             f"(all 4 fields identical — HIGH IS BAD)")
    log.info(f"    Distinct ratio   : {_fmt('distinct_field_ratio')}  (1.0 = all fields differ)")
    log.info("  ── length / fluency ──")
    log.info(f"    Avg pred / ref   : {_fmt('avg_pred_len', 1)} / {_fmt('avg_ref_len', 1)}")
    log.info(f"    Length ratio     : {_fmt('len_ratio', 2)}")
    log.info(f"    Empty rate       : {m.get('empty_rate', 0)*100:.1f}%")
    log.info(f"    Repetition (3-g) : {_fmt('repetition_3gram')}")
    log.info(f"    n_samples        : {m.get('n_samples')}")


# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison_table(all_results, log):
    by_key = {r["model_key"]: r for r in all_results if "metrics" in r}
    ordered = [k for k in DISPLAY_ORDER if k in by_key]
    if not ordered:
        return
    log.info("\n" + "=" * 110)
    log.info("MODEL COMPARISON  (base/aug pairs grouped)")
    log.info("=" * 110)
    header = f"  {'metric':<22} " + " ".join(f"{k[:18]:>18}" for k in ordered)
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    def _row(label, key, fmt="{:.4f}"):
        cells = []
        for k in ordered:
            v = by_key[k]["metrics"].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "  —")
        return f"  {label:<22} " + " ".join(f"{c:>18}" for c in cells)

    log.info("  ── n-gram overlap ──")
    for label, key in [("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"),
                       ("BLEU-1", "bleu1"), ("BLEU-2", "bleu2"), ("BLEU-3", "bleu3"), ("BLEU-4", "bleu4"),
                       ("METEOR", "meteor"), ("CIDEr", "cider"), ("chrF", "chrf"), ("chrF++", "chrf++")]:
        log.info(_row(label, key))
    log.info("  ── semantic ──")
    for label, key in [("BERTScore-P", "bertscore_p"), ("BERTScore-R", "bertscore_r"), ("BERTScore-F1", "bertscore_f1")]:
        log.info(_row(label, key))
    log.info("  ── intrinsic text quality ──")
    for label, key in [("TTR", "ttr"), ("Maas'", "maas"), ("FRES", "fres"),
                       ("CohS", "cohs"), ("ECS", "ecs"), ("TCS", "tcs")]:
        log.info(_row(label, key, "{:.4f}" if key != "fres" else "{:.2f}"))
    log.info("  ── task-specific ──")
    for label, key in [("PerField lesion", "per_field_lesion"), ("PerField edema", "per_field_edema"),
                       ("PerField necrosis", "per_field_necrosis"), ("PerField compression", "per_field_compression"),
                       ("Format compliance", "format_compliance"), ("Hemisphere acc", "hemisphere_accuracy"),
                       ("Lobe Jaccard", "lobe_jaccard"), ("Lobe Jaccard lesion", "lobe_jaccard_lesion")]:
        log.info(_row(label, key))
    log.info("  ── field conditioning ──")
    for label, key in [("Field collapse rate", "field_collapse_rate"), ("Distinct field ratio", "distinct_field_ratio")]:
        log.info(_row(label, key))
    log.info("  ── length / fluency ──")
    for label, key, fmt in [("Avg pred len", "avg_pred_len", "{:.1f}"), ("Avg ref len", "avg_ref_len", "{:.1f}"),
                            ("Length ratio", "len_ratio", "{:.2f}"), ("Empty rate", "empty_rate", "{:.4f}"),
                            ("Repetition 3-gr", "repetition_3gram", "{:.4f}")]:
        log.info(_row(label, key, fmt))


# ── Base vs Aug pair comparison ──────────────────────────────────────────────
def print_pair_comparison(all_results, log):
    by_key = {r["model_key"]: r for r in all_results if "metrics" in r}
    log.info("\n" + "=" * 110)
    log.info("BASE vs AUG PAIR COMPARISON")
    log.info("  Δ = aug - base   (positive = aug improved that metric;")
    log.info("                    for maas/empty_rate/repetition/field_collapse, lower is better)")
    log.info("=" * 110)

    METRICS_LOWER_IS_BETTER = {"maas", "empty_rate", "repetition_3gram", "field_collapse_rate"}
    GROUPS = [
        ("n-gram overlap", [("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"),
                            ("BLEU-1", "bleu1"), ("BLEU-2", "bleu2"), ("BLEU-3", "bleu3"), ("BLEU-4", "bleu4"),
                            ("METEOR", "meteor"), ("CIDEr", "cider"), ("chrF", "chrf"), ("chrF++", "chrf++")]),
        ("semantic", [("BERTScore-P", "bertscore_p"), ("BERTScore-R", "bertscore_r"), ("BERTScore-F1", "bertscore_f1")]),
        ("intrinsic", [("TTR", "ttr"), ("Maas'", "maas"), ("FRES", "fres"),
                       ("CohS", "cohs"), ("ECS", "ecs"), ("TCS", "tcs")]),
        ("task-specific", [("PerField lesion", "per_field_lesion"), ("PerField edema", "per_field_edema"),
                           ("PerField necrosis", "per_field_necrosis"), ("PerField compression", "per_field_compression"),
                           ("Format compliance", "format_compliance"), ("Hemisphere acc", "hemisphere_accuracy"),
                           ("Lobe Jaccard", "lobe_jaccard"), ("Lobe Jaccard lesion", "lobe_jaccard_lesion")]),
        ("field conditioning", [("Field collapse", "field_collapse_rate"), ("Distinct ratio", "distinct_field_ratio")]),
        ("length / fluency", [("Avg pred len", "avg_pred_len"), ("Length ratio", "len_ratio"),
                              ("Empty rate", "empty_rate"), ("Repetition 3-gr", "repetition_3gram")]),
    ]

    for base_key, aug_key in PAIRS:
        if base_key not in by_key or aug_key not in by_key:
            log.info(f"\n  [skip] {base_key} ↔ {aug_key} — one or both missing")
            continue
        log.info(f"\n  ── Pair: {base_key}  ↔  {aug_key} ──")
        log.info(f"  {'metric':<22} {'base':>10} {'aug':>10} {'Δ (aug-base)':>14} {'verdict':>10}")
        log.info(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")
        base_m, aug_m = by_key[base_key]["metrics"], by_key[aug_key]["metrics"]
        wins = losses = ties = 0
        for group_name, items in GROUPS:
            log.info(f"  ── {group_name} ──")
            for label, key in items:
                bv, av = base_m.get(key), aug_m.get(key)
                if not isinstance(bv, (int, float)) or not isinstance(av, (int, float)):
                    log.info(f"  {label:<22} {'—':>10} {'—':>10} {'—':>14} {'':>10}")
                    continue
                delta = av - bv
                if abs(delta) < 1e-6:
                    verdict = "tie"; ties += 1
                elif key in METRICS_LOWER_IS_BETTER:
                    verdict = "aug ✓" if delta < 0 else "base"
                    wins += delta < 0; losses += delta >= 0
                else:
                    verdict = "aug ✓" if delta > 0 else "base"
                    wins += delta > 0; losses += delta <= 0
                bv_s = f"{bv:.4f}" if abs(bv) < 100 else f"{bv:.1f}"
                av_s = f"{av:.4f}" if abs(av) < 100 else f"{av:.1f}"
                d_s  = f"{delta:+.4f}" if abs(delta) < 100 else f"{delta:+.1f}"
                log.info(f"  {label:<22} {bv_s:>10} {av_s:>10} {d_s:>14} {verdict:>10}")
        log.info(f"\n    >>> {aug_key} wins: {int(wins)}  losses: {int(losses)}  ties: {ties}")


# ── Entry point ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_root", default="checkpoints_multi_v2")
    p.add_argument("--csv",             default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",        default="../TextBraTSData")
    p.add_argument("--output_dir",      default="eval_multi_results_v2")
    p.add_argument("--model",           default="all",
                   help=f"all | {' | '.join(MODELS.keys())}")
    p.add_argument("--models",          nargs="+", default=None, help="Subset; overrides --model")
    p.add_argument("--min_fields",      type=int, default=2,
                   help="MUST match the value used in training so the val split is identical")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"eval_multi_v2_{ts}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        force=True)
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Field-wise multi-model evaluation (v2)")
    log.info("=" * 70)
    for k in ["checkpoint_root", "csv", "text_dir", "output_dir", "min_fields", "seed"]:
        log.info(f"  {k:<16}: {getattr(args, k)}")
    log.info("=" * 70)

    cases = build_cases(args.csv, Path(args.text_dir), log, min_fields=args.min_fields)
    if len(cases) < 2:
        raise ValueError("Not enough usable cases to form val split")

    if args.models:
        for m in args.models:
            if m not in MODELS:
                raise ValueError(f"Unknown model {m}. Options: {list(MODELS.keys())}")
        model_keys = args.models
    elif args.model == "all":
        model_keys = [k for k in DISPLAY_ORDER if (Path(args.checkpoint_root) / k).exists()]
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
            r = evaluate_one(key, args, cases, log)
            if r is not None:
                all_results.append(r)
        except Exception as e:
            log.exception(f"[{key}] eval FAILED: {e}")
            all_results.append({"model_key": key, "error": str(e)})
            torch.cuda.empty_cache()

    out_path = Path(args.output_dir) / f"comparison_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nComparison saved → {out_path}")

    print_comparison_table(all_results, log)
    print_pair_comparison(all_results, log)


if __name__ == "__main__":
    main()