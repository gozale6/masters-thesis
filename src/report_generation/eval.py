"""
eval_multi_model_gt_2020_ptg_v4.py
==================================
Evaluation for the v4 MATRIX trainer (train_multi_model_gt_2020_ptg_v4.py).

PROMPT PARITY (the thing that silently breaks evals):
    Instead of re-copying the constants / matrix_to_context / build_field_prompt,
    this script IMPORTS them from the training module. A checkpoint is therefore
    ALWAYS fed the exact prompt format it was trained on. If you fork the JSON
    arm, point TRAIN_MODULE at it — nothing else changes.

What it does
    1. build_cases() (imported) -> identical cases to training.
    2. Reproduces the SAME 80/20 val split (same seed / test_size / shuffle), so
       you evaluate on the split each checkpoint was selected on.
    3. Loads each LoRA checkpoint (base model + adapter from out_dir).
    4. Generates the 4 fields per case with predict_report() (imported) and
       assembles the report exactly as training does.
    5. Computes metrics (incl. the v4 cross-patient template-collapse ratio) and
       writes predictions_<model>.json in the same schema you've been using.

Custom metrics
    cohs / ecs / tcs from your v3 results were bespoke — I don't have their
    definitions, so they are NOT guessed. Register them in CUSTOM_METRIC_HOOKS
    below (each a fn(pred_report, ref_report) -> float) and they'll be averaged
    in automatically. ttr / maas / fres are standard and implemented.

Heavy metric libs (rouge_score, sacrebleu, nltk, bert_score, pycocoevalcap,
textstat) are imported defensively — missing ones are skipped with a log line,
so the script still runs. Use --metrics_lite to skip the slow ones
(bertscore, cider) for quick iteration.

Usage
-----
    python eval_multi_model_gt_2020_ptg_v4.py                       # all models
    python eval_multi_model_gt_2020_ptg_v4.py --model scifive-base
    python eval_multi_model_gt_2020_ptg_v4.py --metrics_lite
"""

import argparse
import importlib
import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.model_selection import train_test_split

# ── Import the EXACT training pieces (prompt parity) ──────────────────────────
TRAIN_MODULE = "train_multi_model_gt_2020_ptg_v4"   # JSON arm: ..._v4_json
tm = importlib.import_module(TRAIN_MODULE)

FIELDS            = tm.FIELDS
REPORT_FORMAT     = tm.REPORT_FORMAT
MODELS            = tm.MODELS
TRAIN_ORDER       = tm.TRAIN_ORDER
build_cases       = tm.build_cases
predict_report    = tm.predict_report
generate_field    = tm.generate_field
build_field_prompt = tm.build_field_prompt
clean_text        = tm.clean_text
parse_fields      = tm.parse_fields
report_hemisphere = tm.report_hemisphere
lobes_in_text     = tm.lobes_in_text
lobe_jaccard_text = tm.lobe_jaccard_text
ground_truth_report = tm.ground_truth_report

# ── Register bespoke metrics here: name -> fn(pred_report, ref_report) -> float ─
# e.g. CUSTOM_METRIC_HOOKS["cohs"] = my_coherence_fn
CUSTOM_METRIC_HOOKS = {}

# ── Defensive metric-lib imports ──────────────────────────────────────────────
_LIBS = {}
try:
    from rouge_score import rouge_scorer
    _LIBS["rouge"] = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
except Exception:
    _LIBS["rouge"] = None
try:
    import sacrebleu
    _LIBS["sacrebleu"] = sacrebleu
except Exception:
    _LIBS["sacrebleu"] = None
try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    _LIBS["nltk"] = (corpus_bleu, SmoothingFunction, meteor_score)
except Exception:
    _LIBS["nltk"] = None
try:
    from bert_score import score as _bertscore
    _LIBS["bertscore"] = _bertscore
except Exception:
    _LIBS["bertscore"] = None
try:
    from pycocoevalcap.cider.cider import Cider
    _LIBS["cider"] = Cider
except Exception:
    _LIBS["cider"] = None
try:
    import textstat
    _LIBS["textstat"] = textstat
except Exception:
    _LIBS["textstat"] = None


def _avail(log):
    have = [k for k, v in _LIBS.items() if v is not None]
    miss = [k for k, v in _LIBS.items() if v is None]
    log.info(f"  metric libs available: {have}")
    if miss:
        log.warning(f"  metric libs MISSING (those metrics skipped): {miss}")


# ── Metric helpers ────────────────────────────────────────────────────────────
def _tok(s):
    return (s or "").split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repetition_3gram(text):
    g = _ngrams(_tok(text), 3)
    if not g:
        return 0.0
    return 1.0 - (len(set(g)) / len(g))


def ttr(all_tokens):
    return (len(set(all_tokens)) / len(all_tokens)) if all_tokens else 0.0


def maas(all_tokens):
    N, V = len(all_tokens), len(set(all_tokens))
    if N <= 1 or V <= 0:
        return 0.0
    return (math.log(N) - math.log(V)) / (math.log(N) ** 2)


def jaccard_words(a, b):
    sa, sb = set(_tok(a.lower())), set(_tok(b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rougeL_f(pred, ref):
    if _LIBS["rouge"] is None:
        return None
    return _LIBS["rouge"].score(ref or "", pred or "")["rougeL"].fmeasure


def compute_metrics(records, log, lite=False):
    """records: list of dicts with prediction, reference, pred_fields, ref_fields."""
    preds = [r["prediction"] for r in records]
    refs  = [r["reference"] for r in records]
    n = len(records)
    M = {"n_samples": n}

    # ----- ROUGE (corpus avg of per-pair F1) -----
    if _LIBS["rouge"] is not None:
        r1 = r2 = rl = 0.0
        for p, rf in zip(preds, refs):
            sc = _LIBS["rouge"].score(rf, p)
            r1 += sc["rouge1"].fmeasure; r2 += sc["rouge2"].fmeasure; rl += sc["rougeL"].fmeasure
        M["rouge1"], M["rouge2"], M["rougeL"] = r1 / n, r2 / n, rl / n

    # ----- BLEU 1-4 (corpus, nltk) + METEOR -----
    if _LIBS["nltk"] is not None:
        corpus_bleu, SmoothingFunction, meteor_score = _LIBS["nltk"]
        sm = SmoothingFunction().method1
        hyp_tok = [_tok(p) for p in preds]
        ref_tok = [[_tok(rf)] for rf in refs]
        for k, name in [(1, "bleu1"), (2, "bleu2"), (3, "bleu3"), (4, "bleu4")]:
            w = tuple([1.0 / k] * k + [0.0] * (4 - k))
            try:
                M[name] = corpus_bleu(ref_tok, hyp_tok, weights=w, smoothing_function=sm)
            except Exception:
                M[name] = 0.0
        try:
            M["meteor"] = sum(meteor_score([_tok(rf)], _tok(p))
                              for p, rf in zip(preds, refs)) / n
        except Exception as e:
            log.warning(f"  meteor failed (nltk wordnet data?): {e}")

    # ----- chrF / chrF++ -----
    if _LIBS["sacrebleu"] is not None:
        sb = _LIBS["sacrebleu"]
        try:
            M["chrf"]   = sb.corpus_chrf(preds, [refs], word_order=0).score / 100.0
            M["chrf++"] = sb.corpus_chrf(preds, [refs], word_order=2).score / 100.0
        except Exception as e:
            log.warning(f"  chrf failed: {e}")

    # ----- CIDEr -----
    if _LIBS["cider"] is not None and not lite:
        try:
            gts = {i: [refs[i]] for i in range(n)}
            res = {i: [preds[i]] for i in range(n)}
            M["cider"], _ = _LIBS["cider"]().compute_score(gts, res)
        except Exception as e:
            log.warning(f"  cider failed: {e}")

    # ----- BERTScore -----
    if _LIBS["bertscore"] is not None and not lite:
        try:
            P, R, F = _LIBS["bertscore"](preds, refs, lang="en", verbose=False, rescale_with_baseline=False)
            M["bertscore_p"]  = P.mean().item()
            M["bertscore_r"]  = R.mean().item()
            M["bertscore_f1"] = F.mean().item()
        except Exception as e:
            log.warning(f"  bertscore failed: {e}")

    # ----- diversity / readability -----
    all_pred_tokens = [t for p in preds for t in _tok(p.lower())]
    M["ttr"]  = ttr(all_pred_tokens)
    M["maas"] = maas(all_pred_tokens)
    if _LIBS["textstat"] is not None:
        try:
            M["fres"] = _LIBS["textstat"].flesch_reading_ease(" ".join(preds))
        except Exception:
            pass

    # ----- custom hooks (cohs / ecs / tcs / ...) -----
    for name, fn in CUSTOM_METRIC_HOOKS.items():
        try:
            M[name] = sum(fn(p, rf) for p, rf in zip(preds, refs)) / n
        except Exception as e:
            log.warning(f"  custom metric '{name}' failed: {e}")

    # ----- per-field rougeL -----
    if _LIBS["rouge"] is not None:
        for f in FIELDS:
            vals = [rougeL_f(r["pred_fields"][f], r["ref_fields"][f]) for r in records]
            M[f"per_field_{f}"] = sum(vals) / n

    # ----- structural / clinical -----
    # format compliance: re-parse the assembled report, require all 4 fields
    compliant = sum(1 for r in records if parse_fields(r["prediction"])[1] == 4)
    M["format_compliance"] = compliant / n

    # hemisphere accuracy on the lesion field (ref != UNK)
    hh = ht = 0
    for r in records:
        g = report_hemisphere(r["ref_fields"]["lesion"])
        if g != "UNK":
            ht += 1
            if g == report_hemisphere(r["pred_fields"]["lesion"]):
                hh += 1
    M["hemisphere_accuracy"] = hh / ht if ht else 0.0

    # lobe jaccard (full report) and lesion-only
    M["lobe_jaccard"] = sum(lobe_jaccard_text(r["prediction"], r["reference"]) for r in records) / n
    M["lobe_jaccard_lesion"] = sum(
        lobe_jaccard_text(r["pred_fields"]["lesion"], r["ref_fields"]["lesion"]) for r in records) / n

    # within-report field collapse (on the PREDICTIONS)
    collapsed = 0
    distinct_ratios = []
    for r in records:
        vals = [r["pred_fields"][f].strip().lower() for f in FIELDS]
        u = len(set(vals))
        distinct_ratios.append(u / len(FIELDS))
        if u == 1:
            collapsed += 1
    M["field_collapse_rate"] = collapsed / n
    M["distinct_field_ratio"] = sum(distinct_ratios) / n

    # v4: CROSS-PATIENT template collapse — unique field strings across patients
    for f in ("lesion", "necrosis"):
        strings = [r["pred_fields"][f].strip().lower() for r in records]
        strings = [s for s in strings if s]
        M[f"cross_patient_{f}"] = (len(set(strings)) / len(strings)) if strings else 0.0

    # lengths / empty / repetition
    pred_len = [len(_tok(p)) for p in preds]
    ref_len  = [len(_tok(rf)) for rf in refs]
    M["avg_pred_len"] = sum(pred_len) / n
    M["avg_ref_len"]  = sum(ref_len) / n
    M["len_ratio"]    = (M["avg_pred_len"] / M["avg_ref_len"]) if M["avg_ref_len"] else 0.0
    empties = sum(1 for r in records for f in FIELDS if not r["pred_fields"][f].strip())
    M["empty_rate"] = empties / (n * len(FIELDS))
    M["repetition_3gram"] = sum(repetition_3gram(p) for p in preds) / n
    return M


# ── Checkpoint loading + per-model evaluation ─────────────────────────────────
def load_checkpoint(model_key, out_dir, log):
    """Load base model + LoRA adapter. Reads base name + dims from summary.json,
    falling back to the MODELS registry. Returns (model, tokenizer, cfg_like)."""
    summary_path = out_dir / "summary.json"
    base_name = MODELS[model_key]["name"]
    max_input = MODELS[model_key]["max_input"]
    max_target = MODELS[model_key]["max_target"]
    augment = MODELS[model_key]["augment"]
    lobe_rollup = True
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        base_name   = s.get("model_name", base_name)
        max_input   = s.get("max_input", max_input)
        max_target  = s.get("max_target", max_target)
        augment     = s.get("augment", augment)
        lobe_rollup = s.get("lobe_rollup", lobe_rollup)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(out_dir))
    model.to(device).eval()
    return model, tokenizer, {"max_input": max_input, "max_target": max_target,
                              "augment": augment, "lobe_rollup": lobe_rollup,
                              "model_name": base_name, "device": device}


def evaluate_one(model_key, args, cases, log):
    out_dir = Path(args.output_root) / model_key
    if not (out_dir / "adapter_config.json").exists() and not (out_dir / "adapter_model.safetensors").exists():
        log.warning(f"[{model_key}] no adapter found in {out_dir} — skipping.")
        return {"model_key": model_key, "error": "no checkpoint"}

    log.info("\n" + "#" * 70)
    log.info(f"# Evaluating {model_key}  ({out_dir})")
    log.info("#" * 70)

    model, tokenizer, cfg = load_checkpoint(model_key, out_dir, log)
    if cfg["lobe_rollup"] != args.lobe_rollup:
        log.warning(f"[{model_key}] checkpoint trained with lobe_rollup={cfg['lobe_rollup']} but eval "
                    f"built contexts with lobe_rollup={args.lobe_rollup}. Re-run eval with "
                    f"--{'lobe_rollup' if cfg['lobe_rollup'] else 'no-lobe_rollup'} for prompt parity.")

    # SAME val split each checkpoint was selected on
    _, val_cases = train_test_split(cases, test_size=0.20, random_state=args.seed, shuffle=True)
    log.info(f"  val cases: {len(val_cases)}")

    device = cfg["device"]
    t0 = time.time()
    records = []
    for c in val_cases:
        try:
            pred_report, pred_fields = predict_report(
                model, tokenizer, c, device, cfg["max_target"], cfg["max_input"])
        except Exception as e:
            log.warning(f"  [{c['case_id']}] generation failed: {e}")
            pred_fields = {f: "not observed" for f in FIELDS}
            pred_report = REPORT_FORMAT.format(**pred_fields)
        records.append({
            "case_id": c["case_id"],
            "prediction": pred_report,
            "reference": ground_truth_report(c),
            "pred_fields": pred_fields,
            "ref_fields": c["fields"],
        })
    gen_sec = time.time() - t0

    metrics = compute_metrics(records, log, lite=args.metrics_lite)
    log.info(f"  rougeL={metrics.get('rougeL'):.4f}  bertscore_f1={metrics.get('bertscore_f1', float('nan'))}  "
             f"hemi={metrics.get('hemisphere_accuracy'):.3f}  "
             f"lobe_jacc={metrics.get('lobe_jaccard'):.3f}  "
             f"cross_patient_lesion={metrics.get('cross_patient_lesion'):.2f}")

    # predictions JSON (your existing schema)
    pred_path = Path(args.results_dir) / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"  predictions -> {pred_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"model_key": model_key, "model_name": cfg["model_name"],
            "augment": cfg["augment"], "metrics": metrics, "generate_sec": gen_sec}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default="atlas_segmentations/ground_truth/all_cases.csv")
    p.add_argument("--text_dir",    default="../TextBraTSData")
    p.add_argument("--output_root", default="checkpoints_multi_v4",
                   help="Where train_multi_model_gt_2020_ptg_v4.py wrote checkpoints.")
    p.add_argument("--results_dir", default="eval_results_v4")
    p.add_argument("--model",       default="all", help=f"all | {' | '.join(MODELS.keys())}")
    p.add_argument("--min_fields",  type=int, default=2)
    p.add_argument("--seed",        type=int, default=42,
                   help="MUST match the training seed to reproduce the val split.")
    # context must be built the SAME way the checkpoints were trained
    p.add_argument("--lobe_rollup",    action="store_true", default=True)
    p.add_argument("--no-lobe_rollup", dest="lobe_rollup", action="store_false")
    p.add_argument("--metrics_lite", action="store_true",
                   help="Skip the slow metrics (bertscore, cider).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(Path("logs") / f"eval_v4_{ts}.log"),
                                  logging.StreamHandler()], force=True)
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("EVAL — v4 matrix trainer (prompts imported from training for parity)")
    log.info("=" * 70)
    _avail(log)

    # Build cases ONCE — identical to training (same lobe_rollup), shared by all models.
    cases = build_cases(args.csv, Path(args.text_dir), log,
                        min_fields=args.min_fields, lobe_rollup=args.lobe_rollup)
    if len(cases) < 2:
        raise ValueError("Not enough usable cases. Check diagnostics above.")

    model_keys = TRAIN_ORDER if args.model == "all" else [args.model]
    if args.model != "all" and args.model not in MODELS:
        raise ValueError(f"Unknown model {args.model}. Options: all, {list(MODELS.keys())}")

    results = []
    for i, key in enumerate(model_keys, 1):
        log.info(f"\n>>> Eval {i}/{len(model_keys)}")
        try:
            results.append(evaluate_one(key, args, cases, log))
        except Exception as e:
            log.exception(f"[{key}] eval FAILED: {e}")
            results.append({"model_key": key, "error": str(e)})

    out_path = Path(args.results_dir) / f"eval_summary_v4_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSummary -> {out_path}")

    # ranking
    log.info("\n" + "=" * 70)
    log.info("RANKING BY rougeL  (+ the v4 cross-patient ratio — watch that it's NOT ~0)")
    log.info("=" * 70)
    rank = sorted([r for r in results if "metrics" in r],
                  key=lambda r: r["metrics"].get("rougeL", 0), reverse=True)
    for i, r in enumerate(rank, 1):
        m = r["metrics"]
        tag = " [aug]" if r.get("augment") else "      "
        log.info(f"  {i}. {r['model_key']:<22}{tag} rougeL={m.get('rougeL', 0):.4f}  "
                 f"bert_f1={m.get('bertscore_f1', float('nan'))}  "
                 f"hemi={m.get('hemisphere_accuracy', 0):.3f}  "
                 f"lobe_jacc={m.get('lobe_jaccard', 0):.3f}  "
                 f"x-patient(les)={m.get('cross_patient_lesion', 0):.2f}")
    failed = [r for r in results if "error" in r]
    if failed:
        log.info("\n  FAILED/SKIPPED:")
        for r in failed:
            log.info(f"    {r['model_key']}: {r['error']}")


if __name__ == "__main__":
    main()