#!/usr/bin/env python
"""Compare every trained fuzzy-prompt configuration -- the crisp baseline, the
2 original hand-tuned 4-term trapezoidal fuzzy runs, all 16 MF-design sweep
variants, and (opt-in) the newer placement- and selection-sweep variants --
in one table, ranked by content_score (= 0.5*lobe_jaccard + 0.5*hemi_acc, the
same selection metric train_one_model() uses). Read-only: just reads each
checkpoint's summary.json, trains nothing.

Usage:
    python src/report_generation/compare_fuzzy_results.py
    python src/report_generation/compare_fuzzy_results.py --top 20
    python src/report_generation/compare_fuzzy_results.py --by-variant   # one row per variant (averaged across its models)
    python src/report_generation/compare_fuzzy_results.py --include-placement --include-selection
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import fuzzy_mf as fm


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=30, help="Rows to print (default: 30).")
    parser.add_argument("--by-variant", action="store_true",
                         help="Collapse to one row per variant, averaged across its models.")
    parser.add_argument("--include-placement", action="store_true",
                         help="Also pull in the data-driven MF-placement sweep (multi_v4_fuzzy_placement_sweep/).")
    parser.add_argument("--include-selection", action="store_true",
                         help="Also pull in the fuzzy region-selection sweep (multi_v4_fuzzy_selection_sweep/).")
    parser.add_argument("--csv", type=str, default=None,
                         help="Also write the full table to this CSV path.")
    args = parser.parse_args()

    df = fm.collect_all_results()
    extra_roots = []
    if args.include_placement:
        extra_roots.append(fm.PLACEMENT_SWEEP_ROOT)
    if args.include_selection:
        extra_roots.append(fm.SELECTION_SWEEP_ROOT)
    if extra_roots:
        df = pd.concat([df, fm.collect_many(extra_roots)], ignore_index=True)

    if df.empty:
        print("No summary.json files found yet -- nothing trained, or paths have moved.")
        return

    print(f"{len(df)} model-run rows across {df['variant'].nunique()} variants.\n")

    group_cols = ["variant", "n_terms", "shape", "style", "placement", "selection", "max_regions"]
    display_cols = ["variant", "model", "shape", "n_terms", "style", "placement", "selection", "max_regions",
                     "augment", "content_score", "lobe_jaccard", "hemi_acc", "best_val_loss", "best_epoch",
                     "elapsed_min"]

    if args.by_variant:
        g = (df.groupby(group_cols, as_index=False, dropna=False)
               .agg(content_score=("content_score", "mean"),
                    lobe_jaccard=("lobe_jaccard", "mean"),
                    hemi_acc=("hemi_acc", "mean"),
                    best_val_loss=("best_val_loss", "mean"),
                    n_models=("model", "count")))
        g = g.sort_values("content_score", ascending=False)
        print(g.head(args.top).to_string(index=False))
        out = g
    else:
        out = df.sort_values("content_score", ascending=False)
        print(out[display_cols].head(args.top).to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nFull table ({len(df)} rows) written to {args.csv}")


if __name__ == "__main__":
    main()
