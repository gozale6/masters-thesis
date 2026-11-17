"""Generalized fuzzy membership-function toolkit for the report-generation prompt
layer: shape family (trapezoidal/triangular/gaussian) x term count (2/3/4) x
prompt style (hybrid/fuzzy_only), generalizing the single hand-tuned 4-term
trapezoidal design in notebooks/fuzzy_report_generation.ipynb (sections 2-6),
which did not beat the crisp multi_v4/ baseline.

Two further axes build on that MF-design sweep, both still going through
matrix_to_fuzzy_context_variant() / activate_variant() below:

- PLACEMENT ("uniform" vs "percentile", see compute_empirical_distributions() /
  make_term_set_percentile()): instead of tiling each term evenly across a
  fixed universe, fit term centers/widths to the actual empirical distribution
  of tumor-share and region-occupancy values in the training CSV.
- SELECTION ("fixed" vs "fuzzy", see _select_regions_fuzzy()): which matrix
  rows reach the LLM at all. "fixed" is the original is_top5-then-salience
  cap (_cap_regions); "fuzzy" ranks by the same fuzzy severity score used to
  describe each region, so relevance and severity are the same signal.

Single source of truth for this logic -- both
notebooks/fuzzy_report_generation.ipynb (section 9) and run_fuzzy_sweep.py /
run_placement_sweep.py / run_selection_sweep.py (the terminal entry points)
import from here rather than each keeping their own copy.

See CLAUDE.md ("MF-design sweep") for the overall picture.
"""
import argparse
import itertools
import json
import logging
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import train as t5v4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "results" / "segmentations" / "atlas" / "ground_truth" / "all_cases.csv"
TEXT_DIR = PROJECT_ROOT / "notebooks" / "brats23" / "TextBraTSData"

SWEEP_ROOT = PROJECT_ROOT / "checkpoints" / "report_generation" / "multi_v4_fuzzy_sweep"
SWEEP_LOGS_DIR = PROJECT_ROOT / "logs" / "fuzzy_sweep"

PLACEMENT_SWEEP_ROOT = PROJECT_ROOT / "checkpoints" / "report_generation" / "multi_v4_fuzzy_placement_sweep"
PLACEMENT_SWEEP_LOGS_DIR = PROJECT_ROOT / "logs" / "fuzzy_placement_sweep"

SELECTION_SWEEP_ROOT = PROJECT_ROOT / "checkpoints" / "report_generation" / "multi_v4_fuzzy_selection_sweep"
SELECTION_SWEEP_LOGS_DIR = PROJECT_ROOT / "logs" / "fuzzy_selection_sweep"

# Strongest base architecture per the MF-design sweep results (clinicalt5-hoss
# and scifive-large beat scifive-base; clinicalt5-hoss picked as the single
# representative model for the cheaper 1-model grid searches below).
REPRESENTATIVE_MODEL = "clinicalt5-hoss"

_ORIGINAL_MATRIX_TO_CONTEXT = t5v4.matrix_to_context  # captured before anything monkeypatches it


# ── Membership-function shapes ─────────────────────────────────────────────────
def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def trimf(x, a, b, c):
    """Triangular membership function: 0 at a and c, 1 at the single peak b."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def gaussmf(x, mean, sigma):
    """Gaussian membership function (no hard corners, never hits exactly 0)."""
    if sigma <= 0:
        return 1.0 if x == mean else 0.0
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2)


_SHAPE_FNS = {
    "trapezoidal": lambda x, p: trapmf(x, *p),
    "triangular": lambda x, p: trimf(x, *p),
    "gaussian": lambda x, p: gaussmf(x, *p),
}


def fuzzify_generic(x, term_set, shape):
    """term_set: {label: params}. shape: 'trapezoidal' | 'triangular' | 'gaussian'."""
    fn = _SHAPE_FNS[shape]
    return {label: fn(x, params) for label, params in term_set.items()}


def dominant_label(memberships):
    """Pick the highest-membership term. Returns (term, degree)."""
    if not memberships or max(memberships.values()) <= 0:
        return "UNSPECIFIED", 0.0
    term = max(memberships, key=memberships.get)
    return term, memberships[term]


# Ordinal label schemes, keyed by n_terms. Shared by every severity-like
# quantity (tumor share, region occupancy, severity output); lobe dominance and
# hemisphere confidence get their own schemes since they're semantically
# different (dominance vs. a signed left/right split).
SEVERITY_LABELS = {
    2: ["MILD", "SEVERE"],
    3: ["MILD", "MODERATE", "SEVERE"],
    4: ["MINIMAL", "MILD", "MODERATE", "SEVERE"],
}
LOBE_LABELS = {
    2: ["MINOR", "DOMINANT"],
    3: ["MINOR", "MAJOR", "DOMINANT"],
    4: ["TRACE", "MINOR", "MAJOR", "DOMINANT"],
}
HEMI_LABELS = {
    2: ["RIGHT", "LEFT"],
    3: ["RIGHT", "MIXED", "LEFT"],
    4: ["STRONG_RIGHT", "MILD_RIGHT", "MILD_LEFT", "STRONG_LEFT"],
}


def make_term_set(n_terms, shape, universe, labels, overlap=0.5):
    """Build an n-term set of membership functions evenly tiled across `universe`
    = (lo, hi) for the given `shape` family. `overlap` (0-1) controls how far each
    term's ramp extends past its own tile -- 0.5 (the default) reproduces the
    same "generous overlap, rarely a hard 0/1 member of one term" style the
    original hand-tuned PCT_TERMS/LOBE_TERMS/etc. used in the notebook, just
    generated instead of tuned by hand for every (n_terms, shape) combination.
    Returns {label: params}, params shaped per `shape`: a 4-tuple for trapezoidal
    (a, b, c, d), a 3-tuple for triangular (a, b, c), or (mean, sigma) for gaussian.
    """
    lo, hi = universe
    width = (hi - lo) / n_terms
    centers = [lo + width * (i + 0.5) for i in range(n_terms)]
    half_span = width * (0.5 + overlap)

    term_set = {}
    for label, center in zip(labels, centers):
        if shape == "gaussian":
            term_set[label] = (center, half_span / 2.0)
            continue
        a, d = center - half_span, center + half_span
        if shape == "triangular":
            term_set[label] = (a, center, d)
        else:  # trapezoidal: flat plateau spanning this term's own tile
            b, c = center - width / 2.0, center + width / 2.0
            term_set[label] = (a, b, c, d)
    return term_set


# ── Data-driven ("percentile") MF placement ────────────────────────────────────
def compute_empirical_distributions(csv_path=CSV_PATH):
    """One-time pass over the raw feature-matrix CSV (independent of GT-report
    availability -- just the atlas matrix itself, ~368 cases, a few seconds) to
    get the actual distribution of tumor_share (log10, the same transform
    fuzzy_severity_variant fuzzifies) and region occupancy, for data-driven MF
    placement instead of evenly tiling a fixed universe."""
    df = pd.read_csv(csv_path)
    shares, occs = [], []
    for _, case_df in df.groupby("case_id"):
        pivot = t5v4.pivot_matrix(case_df)
        active = t5v4._ordered_active(pivot)
        for _, row in active.iterrows():
            shares.append(t5v4._region_salience(row))
            occs.append(float(row.get("pct_region", 0) or 0))
    return {
        "tumor_share_log": np.log10(np.clip(np.asarray(shares, dtype=float), 0.1, None)),
        "region_occ": np.asarray(occs, dtype=float),
    }


_EMPIRICAL_DIST = None  # lazy singleton -- computed once per process


def get_empirical_distributions(csv_path=CSV_PATH):
    global _EMPIRICAL_DIST
    if _EMPIRICAL_DIST is None:
        _EMPIRICAL_DIST = compute_empirical_distributions(csv_path)
    return _EMPIRICAL_DIST


def make_term_set_percentile(n_terms, shape, data, labels, overlap=0.5, lo_pct=1.0, hi_pct=99.0):
    """Like make_term_set(), but term centers/widths are fit to the empirical
    distribution of `data` (an array of observed values) via equal-frequency
    quantile bins, instead of tiling a fixed universe evenly -- e.g. if real
    tumor-share values are heavily right-skewed, uniform tiling wastes most
    terms on a near-empty tail while percentile bins put one term per
    actually-occupied region of the distribution. Clipped to the
    (lo_pct, hi_pct) percentiles first to stay robust to outliers. Falls back
    to make_term_set() over the data's own min/max if `data` is too small to
    bin meaningfully.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < n_terms * 5:
        lo, hi = (float(data.min()), float(data.max())) if data.size else (0.0, 1.0)
        return make_term_set(n_terms, shape, (lo, hi), labels, overlap=overlap)

    lo, hi = np.percentile(data, [lo_pct, hi_pct])
    clipped = np.clip(data, lo, hi)
    edges = np.percentile(clipped, np.linspace(0, 100, n_terms + 1))
    for i in range(1, len(edges)):  # guard against ties collapsing a bin on skewed data
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    term_set = {}
    for i, label in enumerate(labels):
        a_edge, d_edge = edges[i], edges[i + 1]
        width = d_edge - a_edge
        half_span = width * (0.5 + overlap)
        in_bin = clipped[(clipped >= a_edge) & (clipped <= d_edge)]
        center = float(np.median(in_bin)) if in_bin.size else (a_edge + d_edge) / 2.0
        a, d = center - half_span, center + half_span
        if shape == "gaussian":
            term_set[label] = (center, half_span / 2.0)
        elif shape == "triangular":
            term_set[label] = (a, center, d)
        else:  # trapezoidal: flat plateau is the bin's own quantile range
            term_set[label] = (a, a_edge, d_edge, d)
    return term_set


def make_term_set_for(n_terms, shape, universe, labels, overlap=0.5, placement="uniform", data=None):
    """Dispatcher: placement='uniform' -> make_term_set() (unchanged default
    behavior); placement='percentile' -> make_term_set_percentile() fit to
    `data`."""
    if placement == "percentile":
        if data is None:
            raise ValueError("placement='percentile' requires `data` (empirical values to fit quantiles to).")
        return make_term_set_percentile(n_terms, shape, data, labels, overlap=overlap)
    return make_term_set(n_terms, shape, universe, labels, overlap=overlap)


# ── Generic Mamdani inference ───────────────────────────────────────────────────
def make_rule_table(n_terms):
    """(tumor_share_term_idx, region_occ_term_idx) -> output_term_idx. Generalizes
    the notebook's hand-written 4x4 RULES table as
    output_index = round((2*i + j) / 3), clipped to [0, n_terms-1] -- extent
    (tumor share) counted twice, concentration (region occupancy) once, matching
    the qualitative bias of the original hand-tuned table (not a bit-for-bit
    reproduction -- that table isn't derivable from one closed-form rule either).
    """
    return {
        (i, j): max(0, min(n_terms - 1, round((2 * i + j) / 3)))
        for i in range(n_terms)
        for j in range(n_terms)
    }


_OUTPUT_UNIVERSE_VARIANT = np.linspace(0, 100, 201)


def fuzzy_severity_variant(tumor_share, region_occ, n_terms, shape, placement="uniform", dist=None):
    """Generalization of the notebook's fuzzy_severity(), parametrized by term
    count and shape family instead of the fixed 4-term trapezoidal design.
    placement='percentile' fits the two input term sets (tumor share, region
    occupancy) to `dist` (from get_empirical_distributions()) instead of
    tiling their universes evenly; the output/burden term sets stay uniform
    either way (0-100 is a synthetic target scale, not an observed one)."""
    labels = SEVERITY_LABELS[n_terms]
    share_data = dist["tumor_share_log"] if dist is not None else None
    occ_data = dist["region_occ"] if dist is not None else None
    # same log10(share%) range as SHARE_TERMS_LOG when placement='uniform'
    share_terms = make_term_set_for(n_terms, shape, (-2.0, 2.6), labels, placement=placement, data=share_data)
    occ_terms = make_term_set_for(n_terms, shape, (0, 100), labels, placement=placement, data=occ_data)
    rules = make_rule_table(n_terms)

    tumor_mf = fuzzify_generic(math.log10(max(tumor_share, 0.1)), share_terms, shape)
    region_mf = fuzzify_generic(region_occ, occ_terms, shape)

    out_strength = {label: 0.0 for label in labels}
    for i, t_label in enumerate(labels):
        for j, r_label in enumerate(labels):
            strength = min(tumor_mf[t_label], region_mf[r_label])
            out_label = labels[rules[(i, j)]]
            if strength > out_strength[out_label]:
                out_strength[out_label] = strength

    out_terms = make_term_set(n_terms, shape, (0, 100), labels)
    agg = np.zeros_like(_OUTPUT_UNIVERSE_VARIANT)
    for label, strength in out_strength.items():
        if strength <= 0:
            continue
        curve = np.array([_SHAPE_FNS[shape](x, out_terms[label]) for x in _OUTPUT_UNIVERSE_VARIANT])
        agg = np.maximum(agg, np.minimum(strength, curve))

    crisp = float((_OUTPUT_UNIVERSE_VARIANT * agg).sum() / agg.sum()) if agg.sum() > 0 else 0.0
    label = max(out_strength, key=out_strength.get)
    return label, crisp, out_strength


def fuzzy_hemisphere_confidence_variant(region_share, n_terms, shape):
    """Generalization of the notebook's fuzzy_hemisphere_confidence()."""
    left = sum(s for n, s in region_share.items() if t5v4._LEFT_RE.search(n))
    right = sum(s for n, s in region_share.items() if t5v4._RIGHT_RE.search(n))
    total = left + right
    if total <= 0:
        return "UNSPECIFIED", 0.0
    pct_left = 100.0 * left / total
    terms = make_term_set(n_terms, shape, (0, 100), HEMI_LABELS[n_terms])
    return dominant_label(fuzzify_generic(pct_left, terms, shape))


def _cap_regions(active, max_regions):
    """Keep at most max_regions rows, prioritizing is_top5-flagged rows first,
    then filling with the next-highest-salience rows. Same as the notebook's
    section-4 helper (token-budget cap -- see section 4.5 there)."""
    if max_regions is None or len(active) <= max_regions:
        return active
    is_top5 = active["is_top5"].astype(bool) if "is_top5" in active.columns else pd.Series(False, index=active.index)
    top5_idx = list(active.index[is_top5])[:max_regions]
    remaining = max_regions - len(top5_idx)
    fill_idx = [i for i in active.index if i not in set(top5_idx)][: max(remaining, 0)]
    keep = set(top5_idx) | set(fill_idx)
    return active.loc[[i for i in active.index if i in keep]]


def _select_regions_fuzzy(active_full, region_sev, max_regions, prioritize_top5=True):
    """Alternative to _cap_regions(): instead of ranking the "fill" rows by raw
    salience (ET+TC+WT), rank by the fuzzy severity score already computed for
    every region (region_sev[name] = (label, score)) -- i.e. relevance and
    severity are the same signal, so the regions that get through to the LLM
    are the ones the fuzzy system itself judges clinically worst, not just
    biggest. is_top5-flagged rows are still prioritized first (same
    tie-breaking convention as _cap_regions), then filled by descending
    severity score."""
    if max_regions is None or len(active_full) <= max_regions:
        return active_full
    order = sorted(active_full.index, key=lambda r: region_sev[str(r)][1], reverse=True)
    if prioritize_top5 and "is_top5" in active_full.columns:
        is_top5 = active_full["is_top5"].astype(bool)
        top5_idx = [r for r in order if is_top5.get(r, False)][:max_regions]
        remaining = max_regions - len(top5_idx)
        fill_idx = [r for r in order if r not in set(top5_idx)][: max(remaining, 0)]
        keep = set(top5_idx) | set(fill_idx)
    else:
        keep = set(order[:max_regions])
    return active_full.loc[[i for i in active_full.index if i in keep]]


def matrix_to_fuzzy_context_variant(case_id, pivot, n_terms, shape, style="hybrid",
                                     banded_shuffle=False, full_shuffle=False,
                                     lobe_rollup=True, max_regions=8,
                                     placement="uniform", selection="fixed"):
    """Drop-in context builder parametrized by (n_terms, shape, style). Mirrors
    the notebook's matrix_to_fuzzy_context() line-for-line, swapping in the
    generic inference/label functions everywhere the original used the fixed
    4-term trapezoidal ones.

    placement: 'uniform' (default, tiles each term's universe evenly) |
        'percentile' (fits term sets to the real tumor-share/occupancy
        distributions, see make_term_set_percentile()).
    selection: 'fixed' (default, _cap_regions(): is_top5 then raw salience) |
        'fuzzy' (_select_regions_fuzzy(): is_top5 then fuzzy severity score --
        the fuzzy system decides which rows reach the LLM, not just how they
        read)."""
    if style not in ("hybrid", "fuzzy_only"):
        raise ValueError(f"Unknown style {style!r}; use 'hybrid' | 'fuzzy_only'")
    if selection not in ("fixed", "fuzzy"):
        raise ValueError(f"Unknown selection {selection!r}; use 'fixed' | 'fuzzy'")

    active_full = t5v4._ordered_active(pivot, banded_shuffle=banded_shuffle, full_shuffle=full_shuffle)
    region_names_full = [str(r) for r in active_full.index]
    hemisphere = t5v4.infer_hemisphere(region_names_full)

    dist = get_empirical_distributions() if placement == "percentile" else None
    region_share, region_sev = {}, {}
    for region_name, row in active_full.iterrows():
        share = t5v4._region_salience(row)
        occ = float(row.get("pct_region", 0) or 0)
        label, score, _ = fuzzy_severity_variant(share, occ, n_terms, shape, placement=placement, dist=dist)
        region_share[str(region_name)] = share
        region_sev[str(region_name)] = (label, score)

    hemi_label, hemi_deg = fuzzy_hemisphere_confidence_variant(region_share, n_terms, shape)

    total_share = sum(region_share.values())
    burden_score = (sum(region_sev[n][1] * s for n, s in region_share.items()) / total_share
                    if total_share > 0 else 0.0)
    severity_labels = SEVERITY_LABELS[n_terms]
    burden_terms = make_term_set(n_terms, shape, (0, 100), severity_labels)
    burden_label, _ = dominant_label(fuzzify_generic(burden_score, burden_terms, shape))

    lines = [
        t5v4.MATRIX_HEADER,
        f"Patient: {case_id}",
        f"Dominant hemisphere: {hemisphere} (fuzzy side-confidence: {hemi_label}, {hemi_deg:.2f})",
        f"Overall fuzzy tumor burden: {burden_label} ({burden_score:.0f}/100)",
    ]

    if lobe_rollup:
        involve = t5v4.summarize_lobe_involvement(active_full)
        lobe_terms = make_term_set(n_terms, shape, (0, 100), LOBE_LABELS[n_terms])
        if involve:
            parts = []
            for lobe, pct in involve:
                dlabel, _ = dominant_label(fuzzify_generic(pct, lobe_terms, shape))
                parts.append(f"{lobe} {dlabel}" if style == "fuzzy_only" else f"{lobe} {dlabel}({pct:.0f}%)")
            lines.append(f"Lobe involvement (fuzzy): {', '.join(parts)}")
        else:
            lines.append("Lobe involvement (fuzzy): unspecified")

    active = _select_regions_fuzzy(active_full, region_sev, max_regions) if selection == "fuzzy" \
        else _cap_regions(active_full, max_regions)
    sel_desc = "fuzzy severity score" if selection == "fuzzy" else "salience"
    lines.append(f"Atlas involvement (fuzzy severity = extent x local occupancy; "
                 f"* = top-5 region; top {len(active)}/{len(active_full)} regions by {sel_desc}):")

    tag_share_terms = make_term_set(n_terms, shape, (-2.0, 2.6), severity_labels) if style == "fuzzy_only" else None
    for region_name, row in active.iterrows():
        rn = str(region_name)
        sev_label, sev_score = region_sev[rn]
        star = "*" if bool(row.get("is_top5", False)) else " "
        lobe_tag = f" [{t5v4.region_to_lobe(region_name)}]" if lobe_rollup else ""

        if style == "fuzzy_only":
            tag_parts = []
            for tag in ("ET", "TC", "WT"):
                if row[tag] > 0:
                    tlabel, _ = dominant_label(fuzzify_generic(math.log10(max(row[tag], 0.1)), tag_share_terms, shape))
                    tag_parts.append(f"{tag}={tlabel}")
            lines.append(f" {star}- {region_name}{lobe_tag}: {', '.join(tag_parts)}, severity={sev_label}")
        else:  # hybrid
            parts = []
            if row["ET"] > 0: parts.append(f"ET={row['ET']:.0f}")
            if row["TC"] > 0: parts.append(f"TC={row['TC']:.0f}")
            if row["WT"] > 0: parts.append(f"WT={row['WT']:.0f}")
            pr = float(row.get("pct_region", 0) or 0)
            extra = f", region%={pr:.0f}" if pr > 0 else ""
            lines.append(f" {star}- {region_name}{lobe_tag}: {', '.join(parts)}{extra}, "
                         f"severity={sev_label}({sev_score:.0f})")

    lines.append("ET=Enhancing Tumor  TC=Tumor Core/necrosis  WT=Whole Tumor/edema")
    return "\n".join(lines)


# ── The variant grid ────────────────────────────────────────────────────────────
SHAPES = ["trapezoidal", "triangular", "gaussian"]
TERM_COUNTS = [2, 3, 4]
STYLES = ["hybrid", "fuzzy_only"]

# (trapezoidal, 4-term, *) == the original hand-tuned design, already trained as
# multi_v4_fuzzy_hybrid / multi_v4_fuzzy_fuzzy_only -- excluded so the sweep never
# retrains what's already done.
ALREADY_TRAINED = {("trapezoidal", 4, "hybrid"), ("trapezoidal", 4, "fuzzy_only")}


def build_variants():
    return [
        {"name": f"{shape}_{n}term_{style}", "n_terms": n, "shape": shape, "style": style}
        for shape, n, style in itertools.product(SHAPES, TERM_COUNTS, STYLES)
        if (shape, n, style) not in ALREADY_TRAINED
    ]


VARIANTS = build_variants()


# ── Placement grid: same 18 (shape, n_terms, style) combos, all untrained with
# percentile placement (the original 16-variant sweep + the 2 hand-tuned runs
# all used placement='uniform') ─────────────────────────────────────────────
PLACEMENTS = ["uniform", "percentile"]


def build_placement_variants():
    return [
        {"name": f"{shape}_{n}term_{style}_percentile", "n_terms": n, "shape": shape,
         "style": style, "placement": "percentile"}
        for shape, n, style in itertools.product(SHAPES, TERM_COUNTS, STYLES)
    ]


PLACEMENT_VARIANTS = build_placement_variants()


# ── Selection grid: how many regions (K) reach the LLM, picked by fuzzy
# severity instead of the fixed is_top5-then-salience rule. Default MF design
# is the best performer from the original sweep (trapezoidal_2term_hybrid,
# content_score 0.802) -- pass `designs` to test selection on other designs too.
K_GRID = [4, 6, 8, 12]
DEFAULT_SELECTION_DESIGNS = [{"shape": "trapezoidal", "n_terms": 2, "style": "hybrid"}]

# All 18 (shape, n_terms, style) combinations -- for running fuzzy selection at
# the same scale/grid as the original MF-design sweep (run_selection_sweep.py
# --all-designs), instead of just the single best-known design.
ALL_DESIGNS = [
    {"shape": shape, "n_terms": n, "style": style}
    for shape, n, style in itertools.product(SHAPES, TERM_COUNTS, STYLES)
]


def build_selection_variants(designs=None, k_grid=None):
    designs = DEFAULT_SELECTION_DESIGNS if designs is None else designs
    k_grid = K_GRID if k_grid is None else k_grid
    variants = []
    for d in designs:
        for k in k_grid:
            variants.append({
                "name": f"{d['shape']}_{d['n_terms']}term_{d['style']}_fuzzysel_k{k}",
                "n_terms": d["n_terms"], "shape": d["shape"], "style": d["style"],
                "selection": "fuzzy", "max_regions": k, "placement": d.get("placement", "uniform"),
            })
    return variants


SELECTION_VARIANTS = build_selection_variants()


def resolve_models(names):
    """Validate a list of model-key strings against MODELS; None/empty ->
    the full TRAIN_ORDER (used by the original 6-model sweep)."""
    if not names:
        return list(t5v4.TRAIN_ORDER)
    unknown = [n for n in names if n not in t5v4.MODELS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Valid: {list(t5v4.MODELS)}")
    return list(names)


# ── Wiring into the v4 training loop (mirrors notebook section 6) ──────────────
def activate_variant(n_terms, shape, style, placement="uniform", selection="fixed", max_regions=8):
    def _fn(case_id, pivot, banded_shuffle=False, full_shuffle=False, lobe_rollup=True):
        return matrix_to_fuzzy_context_variant(case_id, pivot, n_terms=n_terms, shape=shape, style=style,
                                                banded_shuffle=banded_shuffle, full_shuffle=full_shuffle,
                                                lobe_rollup=lobe_rollup, placement=placement,
                                                selection=selection, max_regions=max_regions)
    t5v4.matrix_to_context = _fn


def restore_crisp():
    t5v4.matrix_to_context = _ORIGINAL_MATRIX_TO_CONTEXT


def make_args(output_root, csv=str(CSV_PATH), text_dir=str(TEXT_DIR), **overrides):
    """Mirrors train.py's parse_args() defaults exactly (see src/report_generation/train.py)."""
    defaults = dict(
        csv=csv, text_dir=text_dir, output_root=output_root, model="all",
        epochs=300, batch_size=2, grad_accum=4, lr=3e-5, lr_large=1e-5, patience=12,
        label_smoothing=0.1, min_fields=2, lora_r=32, lora_widen=True,
        lobe_rollup=True, aug_preset="tuned", shuffle_mode="banded",
        p_swap=None, p_shuffle=None, select_metric="content", content_check_n=24,
        hemi_check_every=5, hemi_check_n=20, distinct_check_n=10, cross_patient_n=20,
        seed=42, audit_only=False, skip_done=True,
    )
    defaults.update(overrides)
    args = argparse.Namespace(**defaults)
    t5v4.resolve_aug(args)  # aug_preset -> concrete p_swap / p_shuffle
    return args


def variant_done(output_root, models=None):
    """A variant counts as complete once every model in `models` (default: the
    full TRAIN_ORDER) has a summary.json."""
    models = models or t5v4.TRAIN_ORDER
    return all((Path(output_root) / key / "summary.json").exists() for key in models)


# ── The sweep runner ─────────────────────────────────────────────────────────
def run_sweep(variants, sweep_root=SWEEP_ROOT, sweep_logs_dir=SWEEP_LOGS_DIR, models=None, activate_fn=None):
    """Trains every variant sequentially. Catches exceptions per variant (logs,
    moves on) and skips any variant that already has summary.json for every
    model in `models`, so this is safe to interrupt and resume by just calling
    it again.

    models: which model keys to train per variant (default: the full
        TRAIN_ORDER, i.e. the original 6-model MF-design sweep). Pass a
        shorter list (e.g. [REPRESENTATIVE_MODEL]) for a cheaper 1-model grid
        search.
    activate_fn: callable(variant_dict) -> None that wires the variant into
        t5v4.matrix_to_context (default: the original 3-key
        (n_terms, shape, style) activate_variant() call -- pass a lambda that
        also forwards placement/selection/max_regions for the newer grids)."""
    models = resolve_models(models)
    activate_fn = activate_fn or (lambda v: activate_variant(v["n_terms"], v["shape"], v["style"]))

    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_logs_dir.mkdir(parents=True, exist_ok=True)
    for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    sweep_log_path = sweep_logs_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                         handlers=[logging.FileHandler(sweep_log_path), logging.StreamHandler()], force=True)
    log = logging.getLogger("fuzzy_sweep")
    log.info(f"Sweep starting: {len(variants)} variants queued x {len(models)} model(s) -> {sweep_root}")
    print(f"Watch live with: tail -f {sweep_log_path}")

    results = []
    for vi, variant in enumerate(variants, 1):
        name = variant["name"]
        output_root = sweep_root / name

        if variant_done(output_root, models=models):
            log.info(f"[{vi}/{len(variants)}] {name}: already complete, skipping.")
            results.append({"variant": name, "status": "skipped_already_done"})
            continue

        extra = {k: v for k, v in variant.items() if k not in ("name", "n_terms", "shape", "style")}
        log.info(f"[{vi}/{len(variants)}] {name}: starting (n_terms={variant['n_terms']}, "
                 f"shape={variant['shape']}, style={variant['style']}{', ' + str(extra) if extra else ''})")
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "variant_config.json").write_text(json.dumps(variant, indent=2))

        activate_fn(variant)
        try:
            args = make_args(str(output_root))
            cases = t5v4.build_cases(args.csv, Path(args.text_dir), log,
                                      min_fields=args.min_fields, lobe_rollup=args.lobe_rollup)
            for mi, key in enumerate(models, 1):
                log.info(f"  [{name}] model {mi}/{len(models)}: {key}")
                t5v4.train_one_model(key, args, cases, log)
            log.info(f"[{vi}/{len(variants)}] {name}: DONE")
            results.append({"variant": name, "status": "done"})
        except Exception as e:
            log.error(f"[{vi}/{len(variants)}] {name}: FAILED -- {e}")
            log.error(traceback.format_exc())
            results.append({"variant": name, "status": "failed", "error": str(e)})
        finally:
            restore_crisp()

    log.info("Sweep finished.")
    for r in results:
        log.info(f"  {r['variant']}: {r['status']}")
    return results


# ── Comparing results afterward (read-only, no training) ───────────────────────
def collect_sweep_results(sweep_root=SWEEP_ROOT):
    """Reads every finished variant's summary.json (written by
    train_one_model(): best_val_loss, best_content={score,lobe_jaccard,hemi_acc},
    best_epoch, augment, elapsed_min, ...) into one flat DataFrame. Works for
    any sweep root regardless of how many models it trained per variant --
    rows just come out for whichever model_key summaries actually exist."""
    rows = []
    for variant_dir in sorted(Path(sweep_root).glob("*")):
        cfg_path = variant_dir / "variant_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        for model_key in t5v4.TRAIN_ORDER:
            summary_path = variant_dir / model_key / "summary.json"
            if not summary_path.exists():
                continue
            s = json.loads(summary_path.read_text())
            content = s.get("best_content") or {}
            rows.append({
                "variant": cfg["name"], "n_terms": cfg["n_terms"], "shape": cfg["shape"],
                "style": cfg["style"], "placement": cfg.get("placement", "uniform"),
                "selection": cfg.get("selection", "fixed"), "max_regions": cfg.get("max_regions"),
                "model": model_key, "augment": s.get("augment"),
                "best_val_loss": s.get("best_val_loss"), "best_epoch": s.get("best_epoch"),
                "content_score": content.get("score"), "lobe_jaccard": content.get("lobe_jaccard"),
                "hemi_acc": content.get("hemi_acc"), "epochs_run": s.get("epochs_run"),
                "elapsed_min": s.get("elapsed_min"),
            })
    return pd.DataFrame(rows)


def collect_many(sweep_roots):
    """Concatenate collect_sweep_results() over several sweep roots (e.g. the
    original MF-design sweep, the data-driven placement sweep, and the fuzzy
    region-selection sweep) into one comparable DataFrame."""
    frames = [collect_sweep_results(root) for root in sweep_roots]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Pre-sweep checkpoints that live outside multi_v4_fuzzy_sweep/ -- the crisp
# baseline and the two hand-tuned 4-term trapezoidal fuzzy runs that the sweep
# deliberately excluded (see ALREADY_TRAINED). Labeled with the same
# (n_terms, shape, style) scheme as the sweep variants so they drop straight
# into the same comparison table.
_PRE_SWEEP_RUNS = [
    {"name": "trapezoidal_4term_crisp", "n_terms": 4, "shape": "trapezoidal", "style": "crisp",
     "dir": PROJECT_ROOT / "checkpoints" / "report_generation" / "multi_v4"},
    {"name": "trapezoidal_4term_hybrid", "n_terms": 4, "shape": "trapezoidal", "style": "hybrid",
     "dir": PROJECT_ROOT / "checkpoints" / "report_generation" / "multi_v4_fuzzy_hybrid"},
    {"name": "trapezoidal_4term_fuzzy_only", "n_terms": 4, "shape": "trapezoidal", "style": "fuzzy_only",
     "dir": PROJECT_ROOT / "checkpoints" / "report_generation" / "multi_v4_fuzzy_fuzzy_only"},
]


def collect_all_results(sweep_root=SWEEP_ROOT):
    """collect_sweep_results() plus the pre-sweep crisp baseline and the two
    hand-tuned 4-term trapezoidal fuzzy runs (_PRE_SWEEP_RUNS), so the full
    18-combination grid -- not just the 16 sweep variants -- is comparable in
    one table. Read-only, no training; safe to call any time."""
    rows = []
    for run in _PRE_SWEEP_RUNS:
        run_dir = run["dir"]
        for model_key in t5v4.TRAIN_ORDER:
            summary_path = run_dir / model_key / "summary.json"
            if not summary_path.exists():
                continue
            s = json.loads(summary_path.read_text())
            content = s.get("best_content") or {}
            rows.append({
                "variant": run["name"], "n_terms": run["n_terms"], "shape": run["shape"],
                "style": run["style"], "placement": "uniform", "selection": "fixed", "max_regions": None,
                "model": model_key, "augment": s.get("augment"),
                "best_val_loss": s.get("best_val_loss"), "best_epoch": s.get("best_epoch"),
                "content_score": content.get("score"), "lobe_jaccard": content.get("lobe_jaccard"),
                "hemi_acc": content.get("hemi_acc"), "epochs_run": s.get("epochs_run"),
                "elapsed_min": s.get("elapsed_min"),
            })
    pre_sweep_df = pd.DataFrame(rows)
    return pd.concat([pre_sweep_df, collect_sweep_results(sweep_root)], ignore_index=True)
