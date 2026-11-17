#!/usr/bin/env python
"""Terminal entry point for the data-driven ("percentile") MF-placement sweep:
the same 18 (shape, term-count, style) combinations as the original MF-design
sweep, but with each term set fit to the real empirical distribution of
tumor-share / region-occupancy values (see fuzzy_mf.make_term_set_percentile())
instead of tiling a fixed universe evenly. Since shape and term count already
turned out not to matter (run_fuzzy_sweep.py's results), this isolates a
different question: does *where* the terms sit matter, given real data is
skewed rather than uniform?

Trains only REPRESENTATIVE_MODEL by default (not the full 6-model suite) --
a first pass to see if placement moves the needle at all before paying for
every architecture. Pass --models to widen it.

Logic lives in fuzzy_mf.py; this script is just the thing you actually run.
See CLAUDE.md ("MF-design sweep") for the background.

Usage:
    python src/report_generation/run_placement_sweep.py              # run the full queue (1 model x 18 variants)
    python src/report_generation/run_placement_sweep.py --dry-run     # just list the queue, no training
    python src/report_generation/run_placement_sweep.py --only trapezoidal_2term_hybrid_percentile
    python src/report_generation/run_placement_sweep.py --models clinicalt5-hoss scifive-large

Meant to run for a long time unattended -- start it under nohup/tmux/screen:

    nohup python src/report_generation/run_placement_sweep.py > placement_sweep_stdout.log 2>&1 &
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuzzy_mf as fm


def activate(variant):
    fm.activate_variant(variant["n_terms"], variant["shape"], variant["style"], placement=variant["placement"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the queued variants and exit -- no training.")
    parser.add_argument("--only", nargs="+", metavar="VARIANT_NAME",
                         help="Run only these variant names instead of the full queue.")
    parser.add_argument("--models", nargs="+", metavar="MODEL_KEY", default=None,
                         help=f"Model keys to train per variant (default: [{fm.REPRESENTATIVE_MODEL}]). "
                              f"Valid: {list(fm.t5v4.MODELS)}")
    args = parser.parse_args()

    models = fm.resolve_models(args.models if args.models else [fm.REPRESENTATIVE_MODEL])

    variants = fm.PLACEMENT_VARIANTS
    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            parser.error(f"Unknown variant name(s): {sorted(missing)}. "
                         f"Valid names: {[v['name'] for v in fm.PLACEMENT_VARIANTS]}")

    total_runs = len(variants) * len(models)
    print(f"{len(variants)} variant(s) queued x {len(models)} model(s) = {total_runs} training run(s).")
    print(f"Models: {models}")
    print(f"Checkpoints -> {fm.PLACEMENT_SWEEP_ROOT}/<variant_name>/<model_key>/")
    print(f"Sweep logs  -> {fm.PLACEMENT_SWEEP_LOGS_DIR}/")
    for v in variants:
        done = fm.variant_done(fm.PLACEMENT_SWEEP_ROOT / v["name"], models=models)
        print(f"  {'[done]' if done else '[ pending ]':<12} {v['name']}")

    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    print()
    fm.run_sweep(variants, sweep_root=fm.PLACEMENT_SWEEP_ROOT, sweep_logs_dir=fm.PLACEMENT_SWEEP_LOGS_DIR,
                 models=models, activate_fn=activate)


if __name__ == "__main__":
    main()
