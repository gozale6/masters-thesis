#!/usr/bin/env python
"""Terminal entry point for the fuzzy region-SELECTION sweep: instead of using
fuzzy logic to describe each matrix row (the MF-design sweep) or to place its
membership functions (the placement sweep), this uses it to decide which rows
reach the LLM at all -- replacing both the crisp baseline's silent tokenizer
truncation and the fuzzy variants' fixed is_top5-then-salience cap
(fuzzy_mf._cap_regions()) with ranking by the same fuzzy severity score used
to describe each region (fuzzy_mf._select_regions_fuzzy()): relevance and
severity are the same signal, so the regions that survive are the ones the
fuzzy system itself judges clinically worst, not just biggest.

Grid = K (how many regions survive) x design (shape, term-count, style for
the severity score that ranks them). Default design is
trapezoidal_2term_hybrid, the best performer from the original MF-design
sweep (content_score 0.802) -- pass --design to test others too.

Trains only REPRESENTATIVE_MODEL by default. Pass --models to widen it.

Logic lives in fuzzy_mf.py; this script is just the thing you actually run.
See CLAUDE.md ("MF-design sweep") for the background.

Usage:
    python src/report_generation/run_selection_sweep.py                      # K in [4,6,8,12], default design
    python src/report_generation/run_selection_sweep.py --dry-run
    python src/report_generation/run_selection_sweep.py --k 2 4 6 8 10 12 16
    python src/report_generation/run_selection_sweep.py --design trapezoidal:2:hybrid gaussian:3:fuzzy_only
    python src/report_generation/run_selection_sweep.py --only trapezoidal_2term_hybrid_fuzzysel_k4

Meant to run for a long time unattended -- start it under nohup/tmux/screen:

    nohup python src/report_generation/run_selection_sweep.py > selection_sweep_stdout.log 2>&1 &
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuzzy_mf as fm


def activate(variant):
    fm.activate_variant(variant["n_terms"], variant["shape"], variant["style"],
                         placement=variant.get("placement", "uniform"),
                         selection=variant["selection"], max_regions=variant["max_regions"])


def parse_design(spec):
    try:
        shape, n_terms, style = spec.split(":")
        return {"shape": shape, "n_terms": int(n_terms), "style": style}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Bad --design {spec!r}; expected SHAPE:N_TERMS:STYLE, e.g. trapezoidal:2:hybrid")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the queued variants and exit -- no training.")
    parser.add_argument("--only", nargs="+", metavar="VARIANT_NAME",
                         help="Run only these variant names instead of the full queue.")
    parser.add_argument("--k", nargs="+", type=int, metavar="K", default=None,
                         help=f"Region counts to sweep (default: {fm.K_GRID}).")
    parser.add_argument("--design", nargs="+", type=parse_design, metavar="SHAPE:N_TERMS:STYLE", default=None,
                         help="MF design(s) for the ranking severity score "
                              f"(default: {fm.DEFAULT_SELECTION_DESIGNS}). "
                              f"shape in {fm.SHAPES}, n_terms in {fm.TERM_COUNTS}, style in {fm.STYLES}.")
    parser.add_argument("--all-designs", action="store_true",
                         help="Use all 18 (shape, n_terms, style) combinations instead of --design "
                              "(same design grid as the original MF-design sweep).")
    parser.add_argument("--models", nargs="+", metavar="MODEL_KEY", default=None,
                         help=f"Model keys to train per variant (default: [{fm.REPRESENTATIVE_MODEL}]). "
                              f"Pass 'all' for the full 6-model TRAIN_ORDER. Valid: {list(fm.t5v4.MODELS)}")
    args = parser.parse_args()

    if args.models == ["all"]:
        models = fm.resolve_models(None)
    else:
        models = fm.resolve_models(args.models if args.models else [fm.REPRESENTATIVE_MODEL])

    designs = fm.ALL_DESIGNS if args.all_designs else args.design
    variants = fm.build_selection_variants(designs=designs, k_grid=args.k)

    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            parser.error(f"Unknown variant name(s): {sorted(missing)}. "
                         f"Valid names: {[v['name'] for v in fm.build_selection_variants(designs=designs, k_grid=args.k)]}")

    total_runs = len(variants) * len(models)
    print(f"{len(variants)} variant(s) queued x {len(models)} model(s) = {total_runs} training run(s).")
    print(f"Models: {models}")
    print(f"Checkpoints -> {fm.SELECTION_SWEEP_ROOT}/<variant_name>/<model_key>/")
    print(f"Sweep logs  -> {fm.SELECTION_SWEEP_LOGS_DIR}/")
    for v in variants:
        done = fm.variant_done(fm.SELECTION_SWEEP_ROOT / v["name"], models=models)
        print(f"  {'[done]' if done else '[ pending ]':<12} {v['name']}")

    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    print()
    fm.run_sweep(variants, sweep_root=fm.SELECTION_SWEEP_ROOT, sweep_logs_dir=fm.SELECTION_SWEEP_LOGS_DIR,
                 models=models, activate_fn=activate)


if __name__ == "__main__":
    main()
