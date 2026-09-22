"""
eval_multi_model.py
===================
Comprehensive evaluation for the SIX LoRA checkpoints produced by
train_multi_model.py (3 base + 3 augmented variants).

Auto-detects all checkpoints under <checkpoint_root>/. Produces:
  - Per-model metrics
  - Side-by-side ranking
  - BASE vs AUG pair comparison (base vs base-aug deltas)

Metrics (graceful skip if a library is missing)
-----------------------------------------------
Lexical / n-gram overlap:
  ROUGE-1, ROUGE-2, ROUGE-L                 (rouge-score)
  BLEU-1, BLEU-2, BLEU-3, BLEU-4 (corpus)   (sacrebleu, nltk)
  METEOR                                    (nltk)
  CIDEr                                     (pycocoevalcap)
  chrF, chrF++                              (sacrebleu)

Semantic similarity:
  BERTScore P/R/F1                          (bert-score)

Intrinsic text quality (matches Valerio et al. 2025):
  TTR, Maas', FRES, CohS, ECS, TCS

Task-specific:
  Per-field ROUGE-1 (lesion/edema/necrosis/compression)
  Format compliance, Hemisphere accuracy, Lobe Jaccard

Length / fluency:
  Avg pred / ref length, length ratio
  Empty rate, repetition rate

Install (recommended)
---------------------
    pip install rouge-score sacrebleu nltk textstat \\
                bert-score sentence-transformers \\
                git+https://github.com/salaniz/pycocoevalcap.git

Usage
-----
    python eval_multi_model.py
    python eval_multi_model.py --model scifive-base-aug
    python eval_multi_model.py --models scifive-base scifive-base-aug   # subset
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

# ── Model registry (mirror train_multi_model.py) ─────────────────────────────
MODELS = {
    "scifive-base": {
        "name":       "razent/SciFive-base-PMC",
        "max_input":  640,
        "max_target": 400,
        "augment":    False,
    },
    "scifive-base-aug": {
        "name":       "razent/SciFive-base-PMC",
        "max_input":  640,
        "max_target": 400,
        "augment":    True,
    },
    "scifive-large": {
        "name":       "razent/SciFive-large-Pubmed",
        "max_input":  640,
        "max_target": 400,
        "augment":    False,
    },
    "scifive-large-aug": {
        "name":       "razent/SciFive-large-Pubmed",
        "max_input":  640,
        "max_target": 400,
        "augment":    True,
    },
    "clinicalt5-hoss": {
        "name":       "hossboll/clinical-t5",
        "max_input":  640,
        "max_target": 400,
        "augment":    False,
    },
    "clinicalt5-hoss-aug": {
        "name":       "hossboll/clinical-t5",
        "max_input":  640,
        "max_target": 400,
        "augment":    True,
    },
}

# Display order: base/aug pairs grouped together
DISPLAY_ORDER = [
    "scifive-base", "scifive-base-aug",
    "clinicalt5-hoss", "clinicalt5-hoss-aug",
    "scifive-large", "scifive-large-aug",
]

# Pairs for base-vs-aug comparison
PAIRS = [
    ("scifive-base", "scifive-base-aug"),
    ("clinicalt5-hoss", "clinicalt5-hoss-aug"),
    ("scifive-large", "scifive-large-aug"),
]

# ── Constants (mirror training) ──────────────────────────────────────────────
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

# ── Atlas prompt helpers (mirror training) ───────────────────────────────────
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


# ── Report parsing (mirror training) ─────────────────────────────────────────
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


# ── Generation ───────────────────────────────────────────────────────────────
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
            length_penalty=1.1,   # match training-time generation
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


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
                score = corpus_bleu(refs, preds, weights=weights, smoothing_function=sm)
                out[f"bleu{n}"] = score
            except Exception as e:
                log.warning(f"NLTK BLEU-{n} failed: {e}")
    except ImportError:
        log.warning("nltk not installed — skipping BLEU-1..4")
    return out


def compute_meteor(predictions, references, log):
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            log.info("Downloading NLTK wordnet …")
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        log.warning("nltk not installed — skipping METEOR")
        return {}
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            s = meteor_score([_safe_tokenize(ref)], _safe_tokenize(pred))
            scores.append(s)
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
        P, R, F1 = bertscore_fn(
            predictions, references,
            lang="en",
            device=device,
            verbose=False,
            rescale_with_baseline=False,
        )
        return {
            "bertscore_p":  float(P.mean()),
            "bertscore_r":  float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")
        return {}


# ── Intrinsic text quality (Valerio parity) ──────────────────────────────────
def compute_lexical_diversity(predictions, log):
    ttrs, maas = [], []
    for p in predictions:
        toks = _safe_tokenize(p)
        n_tokens = len(toks)
        n_types  = len(set(toks))
        if n_tokens == 0:
            continue
        ttr = n_types / n_tokens
        ttrs.append(ttr)
        if n_tokens > 1 and n_types > 0:
            log_n = math.log(n_tokens)
            log_t = math.log(n_types)
            denom = log_n ** 2
            if denom > 0:
                maas.append((log_n - log_t) / denom)
    return {
        "ttr":  sum(ttrs) / len(ttrs) if ttrs else 0.0,
        "maas": sum(maas) / len(maas) if maas else 0.0,
    }


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
        import numpy as np
    except ImportError:
        log.warning("sentence-transformers / sklearn not installed — skipping CohS/ECS")
        return {}

    log.info("  Loading sentence-transformer (all-MiniLM-L6-v2) …")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
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
                embs = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
                sims = []
                for i in range(len(sents) - 1):
                    a = embs[i].reshape(1, -1)
                    b = embs[i + 1].reshape(1, -1)
                    sims.append(float(cosine_similarity(a, b)[0, 0]))
                coh_scores.append(sum(sims) / len(sims))
            except Exception:
                pass

        prompt_sents = split_sents(prompt)
        if pred.strip() and prompt_sents:
            try:
                pred_emb = model.encode([pred], convert_to_numpy=True, show_progress_bar=False)
                ref_embs = model.encode(prompt_sents, convert_to_numpy=True, show_progress_bar=False)
                sims = cosine_similarity(pred_emb, ref_embs)[0]
                ecs_scores.append(float(sims.mean()))
            except Exception:
                pass

        pt = set(t for t in _safe_tokenize(pred)   if t not in STOPWORDS)
        rt = set(t for t in _safe_tokenize(prompt) if t not in STOPWORDS)
        if pt and rt:
            inter = pt & rt
            union = pt | rt
            tcs_scores.append(len(inter) / len(union) if union else 0.0)

    return {
        "cohs": sum(coh_scores) / len(coh_scores) if coh_scores else 0.0,
        "ecs":  sum(ecs_scores) / len(ecs_scores) if ecs_scores else 0.0,
        "tcs":  sum(tcs_scores) / len(tcs_scores) if tcs_scores else 0.0,
    }


# ── Task-specific ────────────────────────────────────────────────────────────
def _extract_field(text, pattern):
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def compute_per_field_rouge1(predictions, references, log):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    out = {}
    for field_key, pat in _FIELD_PATTERNS.items():
        scores = []
        for p, r in zip(predictions, references):
            fp = _extract_field(p, pat)
            fr = _extract_field(r, pat)
            if fr:
                scores.append(scorer.score(fr, fp)["rouge1"].fmeasure)
        out[f"per_field_{field_key}"] = sum(scores) / len(scores) if scores else 0.0
    return out


def compute_format_compliance(predictions):
    if not predictions:
        return {"format_compliance": 0.0}
    n_ok = 0
    for p in predictions:
        ok = (
            re.search(r"(?mi)^\s*Lesion area:",             p) and
            re.search(r"(?mi)^\s*Edema:",                   p) and
            re.search(r"(?mi)^\s*Necrosis:",                p) and
            re.search(r"(?mi)^\s*Ventricular compression:", p)
        )
        if ok:
            n_ok += 1
    return {"format_compliance": n_ok / len(predictions)}


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


def compute_hemisphere_accuracy(predictions, references):
    correct, total = 0, 0
    for p, r in zip(predictions, references):
        rh = _detect_hemisphere(r)
        ph = _detect_hemisphere(p)
        if rh != "UNK":
            total += 1
            if rh == ph:
                correct += 1
    return {"hemisphere_accuracy": correct / total if total > 0 else 0.0}


def _detect_lobes(text):
    out = set()
    for lobe, pat in _LOBE_KEYWORDS.items():
        if pat.search(text):
            out.add(lobe)
    return out


def compute_lobe_jaccard(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        pset = _detect_lobes(p)
        rset = _detect_lobes(r)
        if rset:
            inter = len(pset & rset)
            union = len(pset | rset)
            scores.append(inter / union if union > 0 else 0.0)
    return {"lobe_jaccard": sum(scores) / len(scores) if scores else 0.0}


# ── Length / fluency ─────────────────────────────────────────────────────────
def compute_length_stats(predictions, references):
    if not predictions:
        return {}
    pred_lens = [len(_safe_tokenize(p)) for p in predictions]
    ref_lens  = [len(_safe_tokenize(r)) for r in references]
    avg_pred = sum(pred_lens) / len(pred_lens)
    avg_ref  = sum(ref_lens)  / len(ref_lens) if ref_lens else 0.0
    n_empty  = sum(1 for p in predictions if not p.strip())
    return {
        "avg_pred_len":  avg_pred,
        "avg_ref_len":   avg_ref,
        "len_ratio":     avg_pred / avg_ref if avg_ref > 0 else 0.0,
        "empty_rate":    n_empty / len(predictions),
    }


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
        n_total = len(ngs)
        n_repeat = sum(c for c in counts.values() if c > 1) - len([c for c in counts.values() if c > 1])
        rates.append(n_repeat / n_total)
    return {"repetition_3gram": sum(rates) / len(rates) if rates else 0.0}


# ── Master metrics fn ────────────────────────────────────────────────────────
def compute_all_metrics(predictions, references, prompts, log, device):
    out = {"n_samples": len(predictions)}
    log.info("  → ROUGE")
    out.update(compute_rouge(predictions, references, log))
    log.info("  → BLEU 1-4")
    out.update(compute_bleu_n(predictions, references, log))
    log.info("  → METEOR")
    out.update(compute_meteor(predictions, references, log))
    log.info("  → CIDEr")
    out.update(compute_cider(predictions, references, log))
    log.info("  → chrF / chrF++")
    out.update(compute_chrf(predictions, references, log))
    log.info("  → BERTScore")
    out.update(compute_bertscore(predictions, references, log, device))
    log.info("  → Lexical diversity (TTR, Maas')")
    out.update(compute_lexical_diversity(predictions, log))
    log.info("  → Readability (FRES)")
    out.update(compute_readability(predictions, log))
    log.info("  → Coherence + Coverage (CohS, ECS, TCS)")
    out.update(compute_coherence_and_coverage(predictions, prompts, log, device))
    log.info("  → Per-field ROUGE-1")
    out.update(compute_per_field_rouge1(predictions, references, log))
    log.info("  → Format compliance")
    out.update(compute_format_compliance(predictions))
    log.info("  → Hemisphere accuracy")
    out.update(compute_hemisphere_accuracy(predictions, references))
    log.info("  → Lobe Jaccard")
    out.update(compute_lobe_jaccard(predictions, references))
    log.info("  → Length stats")
    out.update(compute_length_stats(predictions, references))
    log.info("  → Repetition rate")
    out.update(compute_repetition_rate(predictions))
    return out


# ── Per-model evaluation ─────────────────────────────────────────────────────
def evaluate_one(model_key, args, samples, log):
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
    log.info(f"Applying LoRA from {ckpt_dir}")
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
    model.to(device)
    model.eval()

    predictions, references, prompts, case_ids = [], [], [], []
    t0 = time.time()
    for s in tqdm(val_samples, desc=f"[{model_key}] generate"):
        try:
            pred = generate_seq2seq(
                model, tokenizer, s["input_base"],
                device, cfg["max_target"], cfg["max_input"],
            )
        except Exception as e:
            log.warning(f"  [{s['case_id']}] gen failed: {e}")
            pred = ""
        predictions.append(pred)
        references.append(s["target"])
        prompts.append(s["input_base"])
        case_ids.append(s["case_id"])
    gen_time = time.time() - t0
    log.info(f"  Generated {len(predictions)} reports in {gen_time:.1f}s "
             f"({gen_time/len(predictions):.2f}s/sample)")

    log.info("\nComputing metrics …")
    metrics = compute_all_metrics(predictions, references, prompts, log, device_str)

    log.info(f"\n[{model_key}] METRICS")
    _log_metrics(metrics, log)

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
        "augment":      cfg["augment"],
        "metrics":      metrics,
        "generate_sec": gen_time,
    }


def _log_metrics(m, log):
    def _fmt(k, w=4):
        v = m.get(k)
        if v is None: return "  —"
        if isinstance(v, float): return f"{v:.{w}f}"
        return str(v)

    log.info("  ── n-gram overlap ──")
    log.info(f"    ROUGE-1 / 2 / L  : {_fmt('rouge1')} / {_fmt('rouge2')} / {_fmt('rougeL')}")
    log.info(f"    BLEU 1 / 2 / 3 / 4 : "
             f"{_fmt('bleu1')} / {_fmt('bleu2')} / {_fmt('bleu3')} / {_fmt('bleu4')}")
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
    log.info(f"    Per-field ROUGE-1:")
    log.info(f"      lesion         : {_fmt('per_field_lesion')}")
    log.info(f"      edema          : {_fmt('per_field_edema')}")
    log.info(f"      necrosis       : {_fmt('per_field_necrosis')}")
    log.info(f"      compression    : {_fmt('per_field_compression')}")
    log.info(f"    Format compliance: {m.get('format_compliance', 0)*100:.1f}%")
    log.info(f"    Hemisphere acc   : {m.get('hemisphere_accuracy', 0)*100:.1f}%")
    log.info(f"    Lobe Jaccard     : {_fmt('lobe_jaccard')}")
    log.info("  ── length / fluency ──")
    log.info(f"    Avg pred / ref   : {_fmt('avg_pred_len', 1)} / {_fmt('avg_ref_len', 1)}")
    log.info(f"    Length ratio     : {_fmt('len_ratio', 2)}")
    log.info(f"    Empty rate       : {m.get('empty_rate', 0)*100:.1f}%")
    log.info(f"    Repetition (3-g) : {_fmt('repetition_3gram')}")
    log.info(f"    n_samples        : {m.get('n_samples')}")


# ── Big comparison table (all 6 models) ──────────────────────────────────────
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
    for label, key in [
        ("ROUGE-1",       "rouge1"),
        ("ROUGE-2",       "rouge2"),
        ("ROUGE-L",       "rougeL"),
        ("BLEU-1",        "bleu1"),
        ("BLEU-2",        "bleu2"),
        ("BLEU-3",        "bleu3"),
        ("BLEU-4",        "bleu4"),
        ("METEOR",        "meteor"),
        ("CIDEr",         "cider"),
        ("chrF",          "chrf"),
        ("chrF++",        "chrf++"),
    ]:
        log.info(_row(label, key))

    log.info("  ── semantic ──")
    for label, key in [
        ("BERTScore-P",  "bertscore_p"),
        ("BERTScore-R",  "bertscore_r"),
        ("BERTScore-F1", "bertscore_f1"),
    ]:
        log.info(_row(label, key))

    log.info("  ── intrinsic text quality ──")
    for label, key in [
        ("TTR",           "ttr"),
        ("Maas'",         "maas"),
        ("FRES",          "fres"),
        ("CohS",          "cohs"),
        ("ECS",           "ecs"),
        ("TCS",           "tcs"),
    ]:
        fmt = "{:.4f}" if key != "fres" else "{:.2f}"
        log.info(_row(label, key, fmt))

    log.info("  ── task-specific ──")
    for label, key in [
        ("PerField lesion",      "per_field_lesion"),
        ("PerField edema",       "per_field_edema"),
        ("PerField necrosis",    "per_field_necrosis"),
        ("PerField compression", "per_field_compression"),
        ("Format compliance",    "format_compliance"),
        ("Hemisphere acc",       "hemisphere_accuracy"),
        ("Lobe Jaccard",         "lobe_jaccard"),
    ]:
        log.info(_row(label, key))

    log.info("  ── length / fluency ──")
    for label, key, fmt in [
        ("Avg pred len",    "avg_pred_len",     "{:.1f}"),
        ("Avg ref len",     "avg_ref_len",      "{:.1f}"),
        ("Length ratio",    "len_ratio",        "{:.2f}"),
        ("Empty rate",      "empty_rate",       "{:.4f}"),
        ("Repetition 3-gr", "repetition_3gram", "{:.4f}"),
    ]:
        log.info(_row(label, key, fmt))


# ── Base vs Aug pair comparison ──────────────────────────────────────────────
def print_pair_comparison(all_results, log):
    by_key = {r["model_key"]: r for r in all_results if "metrics" in r}

    log.info("\n" + "=" * 110)
    log.info("BASE vs AUG PAIR COMPARISON")
    log.info("  Δ = aug - base   (positive = aug improved that metric;")
    log.info("                    for empty_rate / repetition_3gram, lower is better)")
    log.info("=" * 110)

    METRICS_HIGHER_IS_BETTER = {
        "rouge1", "rouge2", "rougeL",
        "bleu1", "bleu2", "bleu3", "bleu4",
        "meteor", "cider", "chrf", "chrf++",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "ttr", "fres", "cohs", "ecs", "tcs",
        "per_field_lesion", "per_field_edema", "per_field_necrosis", "per_field_compression",
        "format_compliance", "hemisphere_accuracy", "lobe_jaccard",
    }
    METRICS_LOWER_IS_BETTER = {"maas", "empty_rate", "repetition_3gram"}

    GROUPS = [
        ("n-gram overlap", [
            ("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"),
            ("BLEU-1", "bleu1"), ("BLEU-2", "bleu2"), ("BLEU-3", "bleu3"), ("BLEU-4", "bleu4"),
            ("METEOR", "meteor"), ("CIDEr", "cider"),
            ("chrF", "chrf"), ("chrF++", "chrf++"),
        ]),
        ("semantic", [
            ("BERTScore-P", "bertscore_p"),
            ("BERTScore-R", "bertscore_r"),
            ("BERTScore-F1", "bertscore_f1"),
        ]),
        ("intrinsic", [
            ("TTR", "ttr"), ("Maas'", "maas"), ("FRES", "fres"),
            ("CohS", "cohs"), ("ECS", "ecs"), ("TCS", "tcs"),
        ]),
        ("task-specific", [
            ("PerField lesion", "per_field_lesion"),
            ("PerField edema", "per_field_edema"),
            ("PerField necrosis", "per_field_necrosis"),
            ("PerField compression", "per_field_compression"),
            ("Format compliance", "format_compliance"),
            ("Hemisphere acc", "hemisphere_accuracy"),
            ("Lobe Jaccard", "lobe_jaccard"),
        ]),
        ("length / fluency", [
            ("Avg pred len", "avg_pred_len"),
            ("Length ratio", "len_ratio"),
            ("Empty rate", "empty_rate"),
            ("Repetition 3-gr", "repetition_3gram"),
        ]),
    ]

    for base_key, aug_key in PAIRS:
        if base_key not in by_key or aug_key not in by_key:
            log.info(f"\n  [skip] {base_key} ↔ {aug_key} — one or both missing")
            continue

        log.info(f"\n  ── Pair: {base_key}  ↔  {aug_key} ──")
        log.info(f"  {'metric':<22} {'base':>10} {'aug':>10} {'Δ (aug-base)':>14} {'verdict':>10}")
        log.info(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")

        base_m = by_key[base_key]["metrics"]
        aug_m  = by_key[aug_key]["metrics"]

        # Track wins/losses for a summary line
        wins, losses, ties = 0, 0, 0

        for group_name, items in GROUPS:
            log.info(f"  ── {group_name} ──")
            for label, key in items:
                bv = base_m.get(key)
                av = aug_m.get(key)
                if not isinstance(bv, (int, float)) or not isinstance(av, (int, float)):
                    log.info(f"  {label:<22} {'—':>10} {'—':>10} {'—':>14} {'':>10}")
                    continue
                delta = av - bv
                # Verdict
                if abs(delta) < 1e-6:
                    verdict = "tie"
                    ties += 1
                elif key in METRICS_LOWER_IS_BETTER:
                    if delta < 0:
                        verdict = "aug ✓"
                        wins += 1
                    else:
                        verdict = "base"
                        losses += 1
                else:
                    if delta > 0:
                        verdict = "aug ✓"
                        wins += 1
                    else:
                        verdict = "base"
                        losses += 1
                bv_s = f"{bv:.4f}" if abs(bv) < 100 else f"{bv:.1f}"
                av_s = f"{av:.4f}" if abs(av) < 100 else f"{av:.1f}"
                d_s  = f"{delta:+.4f}" if abs(delta) < 100 else f"{delta:+.1f}"
                log.info(f"  {label:<22} {bv_s:>10} {av_s:>10} {d_s:>14} {verdict:>10}")

        log.info(f"\n    >>> {aug_key} wins: {wins}  losses: {losses}  ties: {ties}")


# ── Entry point ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_root", default="checkpoints_multi")
    p.add_argument("--csv",             default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",        default="../TextBraTSData")
    p.add_argument("--output_dir",      default="eval_multi_results")
    p.add_argument("--model",           default="all",
                   help=f"Single model to eval. all | {' | '.join(MODELS.keys())}")
    p.add_argument("--models",          nargs="+", default=None,
                   help="Subset of models. Overrides --model.")
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
    log.info("Multi-model evaluation — comprehensive metrics + base/aug pairs")
    log.info("=" * 70)
    log.info(f"  Checkpoint root : {args.checkpoint_root}")
    log.info(f"  CSV             : {args.csv}")
    log.info(f"  Text dir        : {args.text_dir}")
    log.info(f"  Output dir      : {args.output_dir}")
    log.info("=" * 70)

    samples = build_samples(args.csv, Path(args.text_dir), log)
    if len(samples) < 2:
        raise ValueError("Not enough samples to form val split")

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
            r = evaluate_one(key, args, samples, log)
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