#!/usr/bin/env python
"""Terminal entry point for the fuzzy membership-function design sweep: 16
variants (shape family x term count x prompt style) x the full 6-model suite =
96 training runs, queued sequentially.

Logic lives in fuzzy_mf.py (shared with notebooks/fuzzy_report_generation.ipynb
section 9); this script is just the thing you actually run. See CLAUDE.md
("MF-design sweep") for the background.

Usage:
    python src/report_generation/run_fuzzy_sweep.py              # run the full queue
    python src/report_generation/run_fuzzy_sweep.py --dry-run     # just list the queue, no training
    python src/report_generation/run_fuzzy_sweep.py --only trapezoidal_2term_hybrid gaussian_4term_fuzzy_only

Meant to run for a long time unattended -- start it under nohup/tmux/screen so
it survives your terminal closing, e.g.:

    nohup python src/report_generation/run_fuzzy_sweep.py > sweep_stdout.log 2>&1 &
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuzzy_mf as fm


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the queued variants and exit -- no training.")
    parser.add_argument("--only", nargs="+", metavar="VARIANT_NAME",
                         help="Run only these variant names instead of the full queue.")
    args = parser.parse_args()

    variants = fm.VARIANTS
    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            parser.error(f"Unknown variant name(s): {sorted(missing)}. "
                         f"Valid names: {[v['name'] for v in fm.VARIANTS]}")

    total_runs = len(variants) * len(fm.t5v4.TRAIN_ORDER)
    print(f"{len(variants)} variant(s) queued x {len(fm.t5v4.TRAIN_ORDER)} models "
          f"= {total_runs} training run(s).")
    print(f"Checkpoints -> {fm.SWEEP_ROOT}/<variant_name>/<model_key>/")
    print(f"Sweep logs  -> {fm.SWEEP_LOGS_DIR}/")
    for v in variants:
        done = fm.variant_done(fm.SWEEP_ROOT / v["name"])
        print(f"  {'[done]' if done else '[ pending ]':<12} {v['name']}")

    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    print()
    fm.run_sweep(variants)


if __name__ == "__main__":
    main()
