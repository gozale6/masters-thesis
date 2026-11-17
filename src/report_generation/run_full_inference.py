#!/usr/bin/env python
"""Full validation-set inference pass over every checkpoint from this
session's two new sweeps (placement, selection): instead of train.py's own
eval_content_score() on a random 24-case subset of val (used only to pick the
best epoch during training), this generates all 4 fields for EVERY val case
and recomputes the same content_score formula (0.5*lesion_lobe_jaccard +
0.5*hemisphere_acc) on the full split for a reliable final number, plus
per-field lobe_jaccard (edema/necrosis/compression too, not just lesion) and
every predicted report alongside its ground truth, for qualitative review.

"Prompt parity" matters here (see CLAUDE.md): a checkpoint must be evaluated
with the exact context builder it was trained with. Each checkpoint's own
variant_config.json records that (n_terms, shape, style, placement,
selection, max_regions), so this script re-activates it via
fuzzy_mf.activate_variant() before rebuilding cases and generating anything --
mirrors fuzzy_mf.run_sweep()'s own activate/build/restore ordering, and
recovers the exact same train/val split training used (build_cases() +
train_test_split() with the same seed).

Resumable: skips any (variant, model) pair whose output JSON already exists.

Usage:
    python src/report_generation/run_full_inference.py --dry-run
    python src/report_generation/run_full_inference.py
    python src/report_generation/run_full_inference.py --sweeps placement
    python src/report_generation/run_full_inference.py --only trapezoidal_2term_hybrid_percentile:clinicalt5-hoss

Inference only (no backward pass) -- much cheaper than training, but ~130
checkpoints x the full val split x 4 fields each still adds up. Start it
under nohup/tmux/screen:

    nohup python src/report_generation/run_full_inference.py > full_inference_stdout.log 2>&1 &
"""
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuzzy_mf as fm

t5v4 = fm.t5v4

OUT_ROOT = fm.PROJECT_ROOT / "results" / "report_generation" / "full_inference"
COMPARISON_CSV = fm.PROJECT_ROOT / "results" / "report_generation" / "full_inference_comparison.csv"

SWEEP_ROOTS = {
    "placement": fm.PLACEMENT_SWEEP_ROOT,
    "selection": fm.SELECTION_SWEEP_ROOT,
    "original": fm.SWEEP_ROOT,  # the 96-run shape x term-count x style sweep (same variant_config.json layout)
}

# Crisp baseline + the 2 hand-tuned fuzzy runs: flat <dir>/<model_key>/ layout,
# no variant_config.json (predate fuzzy_mf.py's sweep machinery) -- cfg comes
# from fuzzy_mf._PRE_SWEEP_RUNS instead of a file.
BASELINE_RUNS = fm._PRE_SWEEP_RUNS
ALL_SWEEP_NAMES = list(SWEEP_ROOTS) + ["baseline"]


def iter_checkpoints(sweeps):
    """Yields (sweep_name, variant_name, model_key, variant_cfg, ckpt_dir) for
    every trained (variant, model) pair under the chosen sweeps."""
    for sweep_name in sweeps:
        if sweep_name == "baseline":
            for run in BASELINE_RUNS:
                cfg = {"n_terms": run["n_terms"], "shape": run["shape"], "style": run["style"]}
                for model_dir in sorted(Path(run["dir"]).glob("*")):
                    if model_dir.name not in t5v4.MODELS:  # multi_v4/ also has -full/-lora split dirs
                        continue
                    if not (model_dir / "summary.json").exists():
                        continue
                    if not (model_dir / "adapter_config.json").exists():
                        continue
                    yield sweep_name, run["name"], model_dir.name, cfg, model_dir
            continue

        root = SWEEP_ROOTS[sweep_name]
        for variant_dir in sorted(Path(root).glob("*")):
            cfg_path = variant_dir / "variant_config.json"
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            for model_dir in sorted(variant_dir.glob("*")):
                if model_dir.name not in t5v4.MODELS:
                    continue
                if not (model_dir / "summary.json").exists():
                    continue
                if not (model_dir / "adapter_config.json").exists():
                    continue
                yield sweep_name, variant_dir.name, model_dir.name, cfg, model_dir


def load_finetuned(model_key, out_dir, device):
    cfg = t5v4.MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"], torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(out_dir))
    model.to(device).eval()
    return model, tokenizer


def evaluate_checkpoint(model, tokenizer, val_cases, device, max_target, max_input):
    """One generation pass over every val case (all 4 fields each). Returns
    (metrics_dict, predictions_list). content_score mirrors train.py's
    eval_content_score() formula exactly, just computed over every val case
    instead of a random 24-case sample."""
    predictions = []
    jaccs = {f: [] for f in t5v4.FIELDS}
    hemi_hit = hemi_tot = 0
    for c in val_cases:
        pred, values = t5v4.predict_report(model, tokenizer, c, device, max_target, max_input)
        ref = t5v4.ground_truth_report(c)
        for field in t5v4.FIELDS:
            jaccs[field].append(t5v4.lobe_jaccard_text(values[field], c["fields"][field]))
        g = t5v4.report_hemisphere(c["fields"]["lesion"])
        if g != "UNK":
            hemi_tot += 1
            if g == t5v4.report_hemisphere(values["lesion"]):
                hemi_hit += 1
        predictions.append({"case_id": c["case_id"], "pred": pred, "pred_fields": values,
                             "ref": ref, "ref_fields": c["fields"]})

    lobe_jaccard_by_field = {f: (sum(v) / len(v) if v else 0.0) for f, v in jaccs.items()}
    hemi_acc = hemi_hit / hemi_tot if hemi_tot else 0.0
    lesion_lobe_jaccard = lobe_jaccard_by_field["lesion"]
    content_score = 0.5 * lesion_lobe_jaccard + 0.5 * hemi_acc
    metrics = {
        "full_content_score": content_score,
        "full_lobe_jaccard_lesion": lesion_lobe_jaccard,
        "full_hemi_acc": hemi_acc,
        "full_lobe_jaccard_by_field": lobe_jaccard_by_field,
        "n_val_cases": len(val_cases),
    }
    return metrics, predictions


def run_one(sweep_name, variant_name, model_key, cfg, ckpt_dir, log, device):
    out_dir = ckpt_dir

    placement = cfg.get("placement", "uniform")
    selection = cfg.get("selection", "fixed")
    max_regions = cfg.get("max_regions") or 8
    if cfg["style"] == "crisp":
        fm.restore_crisp()  # the plain train.py matrix_to_context -- activate_variant() only knows hybrid/fuzzy_only
    else:
        fm.activate_variant(cfg["n_terms"], cfg["shape"], cfg["style"],
                             placement=placement, selection=selection, max_regions=max_regions)
    try:
        args = fm.make_args(str(ckpt_dir.parent))
        cases = t5v4.build_cases(args.csv, Path(args.text_dir), log,
                                  min_fields=args.min_fields, lobe_rollup=args.lobe_rollup)
        _, val_cases = t5v4.train_test_split(cases, test_size=0.20, random_state=args.seed, shuffle=True)

        model_cfg = t5v4.MODELS[model_key]
        model, tokenizer = load_finetuned(model_key, out_dir, device)

        t0 = time.time()
        metrics, predictions = evaluate_checkpoint(model, tokenizer, val_cases, device,
                                                    model_cfg["max_target"], model_cfg["max_input"])
        elapsed_min = (time.time() - t0) / 60

        result = {
            "sweep": sweep_name, "variant": variant_name, "model": model_key,
            "n_terms": cfg["n_terms"], "shape": cfg["shape"], "style": cfg["style"],
            "placement": placement, "selection": selection, "max_regions": cfg.get("max_regions"),
            "augment": model_cfg["augment"], "inference_elapsed_min": elapsed_min,
            **metrics, "predictions": predictions,
        }
        log.info(f"  -> full_content={metrics['full_content_score']:.3f}  "
                 f"lobe_jaccard(lesion)={metrics['full_lobe_jaccard_lesion']:.3f}  "
                 f"hemi_acc={metrics['full_hemi_acc']:.3f}  "
                 f"n={metrics['n_val_cases']}  ({elapsed_min:.1f} min)")

        del model
        torch.cuda.empty_cache()
        return result
    finally:
        fm.restore_crisp()


def append_comparison_row(result):
    row = {k: v for k, v in result.items() if k not in ("predictions", "full_lobe_jaccard_by_field")}
    for field, val in result["full_lobe_jaccard_by_field"].items():
        row[f"full_lobe_jaccard_{field}"] = val
    df_row = pd.DataFrame([row])
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_row.to_csv(COMPARISON_CSV, mode="a", header=not COMPARISON_CSV.exists(), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweeps", nargs="+", choices=ALL_SWEEP_NAMES, default=["placement", "selection"],
                         help="Which sweeps to run inference over (default: placement + selection; "
                              "pass 'original' and/or 'baseline' for a like-for-like comparison against "
                              "the pre-session crisp/MF-design-sweep numbers).")
    parser.add_argument("--only", nargs="+", metavar="VARIANT:MODEL",
                         help="Run only these variant:model pairs instead of the full queue.")
    parser.add_argument("--dry-run", action="store_true", help="List queued checkpoints and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    log = logging.getLogger("full_inference")

    queue = list(iter_checkpoints(args.sweeps))
    if args.only:
        wanted = set(args.only)
        queue = [q for q in queue if f"{q[1]}:{q[2]}" in wanted]
        missing = wanted - {f"{q[1]}:{q[2]}" for q in queue}
        if missing:
            parser.error(f"Unknown/untrained variant:model pair(s): {sorted(missing)}")

    print(f"{len(queue)} checkpoint(s) queued for full-val-set inference.")
    print(f"Output          -> {OUT_ROOT}/<sweep>/<variant>/<model_key>.json")
    print(f"Comparison CSV  -> {COMPARISON_CSV}")

    if args.dry_run:
        for sweep_name, variant_name, model_key, _, _ in queue:
            done = (OUT_ROOT / sweep_name / variant_name / f"{model_key}.json").exists()
            print(f"  {'[done]' if done else '[ pending ]':<12} {sweep_name}/{variant_name}/{model_key}")
        print("\n--dry-run: not running inference.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (sweep_name, variant_name, model_key, cfg, ckpt_dir) in enumerate(queue, 1):
        out_path = OUT_ROOT / sweep_name / variant_name / f"{model_key}.json"
        if out_path.exists():
            log.info(f"[{i}/{len(queue)}] [SKIP] {sweep_name}/{variant_name}/{model_key}: already done.")
            continue
        log.info(f"[{i}/{len(queue)}] {sweep_name}/{variant_name}/{model_key}")
        try:
            result = run_one(sweep_name, variant_name, model_key, cfg, ckpt_dir, log, device)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            append_comparison_row(result)
        except Exception as e:
            log.error(f"[{i}/{len(queue)}] {sweep_name}/{variant_name}/{model_key}: FAILED -- {e}")
            log.error(traceback.format_exc())

    log.info("Inference sweep finished.")


if __name__ == "__main__":
    main()
