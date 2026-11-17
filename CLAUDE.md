# CLAUDE.md

Operational context for working in this repo. See [README.md](README.md) for the
full workflow/branching conventions (experiment folders, `config.yaml`/`notes.md`,
git-lfs, branch naming) — this file only covers what README doesn't.

## Pipeline

1. **Segmentation** (`src/segmentation/{unet,segresnet,swinunetr}/`) — train on
   BraTS20/BraTS23 (`brats20/`, `brats23/`, local disk only, not tracked by git).
2. **Atlas application** (`src/report_generation/apply_matrix_gt_2020_ptg_top5.py`
   is the current/last version) — maps a case's segmentation onto the Julich-Brain
   atlas (via `siibra`) and writes a **pathological feature matrix** CSV per case:
   `case_id, atlas_region_name, tumor_tag (ET/TC/WT), voxel_count, count_scaled,
   pct_of_tumor, pct_of_region, is_top5`. The ground-truth version lives at
   `results/segmentations/atlas/ground_truth/all_cases.csv` (368 cases); there are
   also `segresnet/`, `swinunetr/`, `unet/` variants under the same
   `results/segmentations/atlas/` built from each segmentation model's own output
   instead of ground truth.
3. **Report generation** (`src/report_generation/train.py`, the v4 "field-wise
   multi-task" trainer — **this is the current best/last version**) — turns each
   case's feature matrix into a text prompt (`matrix_to_context()` +
   `build_field_prompt()`: hemisphere anchor line, lobe-involvement rollup, ranked
   region rows) and LoRA-fine-tunes T5-family models to answer 4 fields (lesion,
   edema, necrosis, ventricular compression) that get assembled into a
   4-line report. Ground-truth radiology reports live under
   `notebooks/brats23/TextBraTSData/<case_id>/<case_id>_flair_text.txt`.

## The "6 versions"

`train.py`'s `TRAIN_ORDER` / `MODELS` trains **6 models** = 3 base models
(`scifive-base`, `scifive-large`, `clinicalt5-hoss`) x 2 (with/without data
augmentation, i.e. `-aug` suffix = hemisphere-swap + region-shuffle augmentation).
Trained checkpoints + `summary.json` (per-epoch val loss, content score, lobe
Jaccard, hemisphere accuracy) live at
`checkpoints/report_generation/multi_v4/<model_key>/`.

## Fuzzy-logic experiment

`notebooks/fuzzy_report_generation.ipynb` inserts a fuzzy-logic layer (trapezoidal
membership functions + a small Mamdani inference system) between the crisp
pathological feature matrix and the LLM prompt, to test whether linguistic
severity/dominance labels (MINIMAL/MILD/MODERATE/SEVERE, lobe dominance,
hemisphere confidence) help the T5 models produce better reports than raw
percentages. It does **not** fork `train.py` — it builds a fuzzy replacement for
`matrix_to_context()` and monkeypatches it onto the imported `train` module at
runtime, so the entire proven v4 pipeline (data loading, LoRA setup, the 6-model
loop, early stopping, mode-collapse monitors) runs completely unchanged; only the
prompt's number-vs-language framing differs. Fuzzy checkpoints land in
`checkpoints/report_generation/multi_v4_fuzzy_<style>/` (styles: `crisp` /
`hybrid` / `fuzzy_only`), never overwriting the original `multi_v4/` baseline.

Training in that notebook is gated behind a `RUN_TRAINING = False` flag — it is
meant to be run by hand, not automatically.

**MF-design sweep:** since the single hand-tuned 4-term trapezoidal design
didn't beat the crisp baseline, `src/report_generation/fuzzy_mf.py` generalizes
the membership-function design itself — shape family (`trapezoidal` /
`triangular` / `gaussian`) x term count (2/3/4) x style (`hybrid`/`fuzzy_only`) —
into a 16-variant queue (18 combinations minus the 2 already trained), each
training the full 6-model suite (96 training runs total). `fuzzy_mf.py` is the
single source of truth for this logic; both
`notebooks/fuzzy_report_generation.ipynb` (section 9, exploratory/educational)
and `src/report_generation/run_fuzzy_sweep.py` (the actual entry point) import
from it rather than keeping separate copies. Checkpoints land in
`checkpoints/report_generation/multi_v4_fuzzy_sweep/<variant_name>/<model_key>/`,
each with its own `variant_config.json`. **Run it from the terminal**, not a
notebook flag — `python src/report_generation/run_fuzzy_sweep.py` (`--dry-run`
to preview the queue, `--only NAME [NAME ...]` to run a subset); long-running,
so start it under `nohup`/`tmux`/`screen`. Resumable — re-running skips any
variant whose 6 models already have `summary.json`. `fuzzy_mf.collect_sweep_results()`
reads every finished variant's `summary.json` into one comparison table.

Result: none of the 18 (shape x term-count x style) combinations decisively beat
the crisp baseline (best +0.007, well within same-variant cross-model noise of up
to 0.167). What did matter: augmentation (consistently negative) and base
architecture (`clinicalt5-hoss`/`scifive-large` > `scifive-base`) — not MF shape
or term count. Best performer: `trapezoidal_2term_hybrid` (content_score 0.802).

**Placement sweep (`run_placement_sweep.py`):** since shape/term-count didn't
matter, this tests a different axis — *where* each term sits. The original sweep
tiles every term set evenly across a fixed universe
(`fuzzy_mf.make_term_set()`); `fuzzy_mf.make_term_set_percentile()` instead fits
term centers/widths to the real empirical distribution of tumor-share and
region-occupancy values in the training CSV
(`fuzzy_mf.compute_empirical_distributions()`), via equal-frequency quantile
bins. Same 18 (shape, term-count, style) combinations, all re-run with
`placement="percentile"`. Trains only `REPRESENTATIVE_MODEL`
(`clinicalt5-hoss`, the strongest base architecture) by default instead of the
full 6-model suite, per the "test with 1 model first" lesson from the MF-design
sweep — pass `--models` to widen it. Checkpoints land in
`checkpoints/report_generation/multi_v4_fuzzy_placement_sweep/<variant_name>/<model_key>/`.

**Selection sweep (`run_selection_sweep.py`):** a different use of fuzzy logic
entirely — not describing a region's values, but deciding *which regions reach
the LLM at all*. The crisp baseline passes the full matrix and lets the
tokenizer silently truncate at the 640-token budget (most cases exceed it); the
fuzzy MF-design/placement sweeps cap at a fixed top-8 via
`fuzzy_mf._cap_regions()` (is_top5 rows first, then fill by raw salience
ET+TC+WT). `fuzzy_mf._select_regions_fuzzy()` instead fills by the same fuzzy
severity score already computed to *describe* each region — relevance and
severity are the same signal, so the rows that survive are the ones the fuzzy
system itself judges clinically worst, not just biggest. Grid = K (region count,
default `[4, 6, 8, 12]`) x MF design for the ranking score (default
`trapezoidal_2term_hybrid`, the MF-design sweep's best performer; pass
`--design SHAPE:N_TERMS:STYLE` to test others). Also 1-model-by-default.
Checkpoints land in
`checkpoints/report_generation/multi_v4_fuzzy_selection_sweep/<variant_name>/<model_key>/`.

Both new sweeps reuse `fuzzy_mf.run_sweep()` / `variant_done()` (now generalized
with `models=`/`activate_fn=` params instead of hardcoding the 6-model
`TRAIN_ORDER` and the 3-key `activate_variant()` call) and
`fuzzy_mf.collect_sweep_results()` unchanged. `compare_fuzzy_results.py
--include-placement --include-selection` merges all three sweeps (plus the
crisp baseline and the 2 hand-tuned runs) into one table.

**1-model results (2026-10-25/26):** both new axes beat the single-model crisp
baseline (0.821) but neither beat the original sweep's best uniform/fixed
design for that same model (`gaussian_3term_hybrid`, 0.861) — percentile
placement's best was `triangular_3term_hybrid_percentile` (0.851), fuzzy
selection's best was K=6 (0.844, non-monotonic: K=4/8/12 all scored lower,
suggesting a real sweet spot around 6 regions rather than "more is better").
Empirically best single model across the original 96-run sweep (mean
content_score over all 16 variants): `scifive-large` (0.8147), narrowly ahead
of `clinicalt5-hoss` (0.8111) — close enough that either is a reasonable
"representative model" pick for future 1-model passes.

**Full-scale selection sweep:** `run_selection_sweep.py --all-designs --k 6
--models all` reruns the fuzzy-selection idea at the same scale as the
original MF-design sweep — all 18 (shape, n_terms, style) designs x the full
6-model `TRAIN_ORDER`, K fixed at 6 (the best value from the 1-model pass) —
108 runs, ~same cost as the original 96-run sweep (~85-100 GPU-hours, resumes
and skips the one (model, variant) pair already trained during the 1-model
pass). Completed 2026-10-27 (survived one mid-sweep power cut; resumed clean).

**Full-validation-set inference (`run_full_inference.py`):** every
`train_one_model()` checkpoint only ever got checked against a random
24-case subset of val (`eval_content_score`'s `n=24`, used just to pick the
best epoch) — this script re-generates all 4 fields for every val case
(same train/val split, recovered via the same `build_cases()` +
`train_test_split(seed=42)` call train_one_model() made) and recomputes the
identical content_score formula on the full ~74-case split, for a reliable
final number instead of a noisy sample (confirmed noisy in practice: one
checkpoint scored 0.840 on training's 24-case sample vs. 0.750 on the full
74-case split). Also reports per-field lobe_jaccard (edema/necrosis/
compression, not just lesion) and saves every predicted report next to its
ground truth. Covers the two new sweeps (placement, selection) only, not the
original 96-run sweep or crisp baseline. Resumable (skips any
(variant, model) pair whose output JSON already exists). Output ->
`results/report_generation/full_inference/<sweep>/<variant>/<model_key>.json`
(full predictions) + `results/report_generation/full_inference_comparison.csv`
(metrics only, one row per checkpoint, appended as each finishes).

Default `--sweeps placement selection` only (the two new tracks). Pass
`--sweeps original baseline` to also cover the original 96-run MF-design
sweep (`fm.SWEEP_ROOT`, same `variant_config.json` layout, no script changes
needed) and the crisp baseline + 2 hand-tuned fuzzy runs (`fm._PRE_SWEEP_RUNS`
— flat `<dir>/<model_key>/` layout predating the sweep machinery, so
`iter_checkpoints()` special-cases it; `style="crisp"` skips
`activate_variant()` entirely via `fm.restore_crisp()`, since
`matrix_to_fuzzy_context_variant()` only knows `hybrid`/`fuzzy_only`). Needed
for a true apples-to-apples comparison against the pre-session deck's numbers
(those are all training-time 24-case estimates, not full-val). Also filters
`multi_v4/`'s `-full`/`-lora` subdirectories (an older checkpoint-saving
split, not valid `MODELS` keys) out of the baseline listing.

**Full-val results so far confirm training-time numbers run high across the
board**, not just for the new sweeps: `trapezoidal_4term_crisp/clinicalt5-hoss`
scores 0.767 on the full val set vs. 0.821 at training time — same ~0.05-0.09
gap seen everywhere else. Any comparison against the pre-session deck's
0.795/0.802/0.861 numbers needs the `original`/`baseline` full-val numbers,
not a direct read against those training-time figures.

## Known gotchas

- `src/report_generation/eval.py` imports a stale pre-restructure module name
  (`train_multi_model_gt_2020_ptg_v4`) that no longer exists after the "clean
  thesis layout" commit renamed it to `train.py` — it will not run as-is. Not
  fixed as part of the fuzzy work; the fuzzy notebook has its own
  `quick_compare_checkpoint()` helper that reimplements the same
  load-checkpoint-and-generate pattern correctly against `train.py`.
- "Prompt parity": whatever built a checkpoint's training prompts must also build
  its eval prompts, in the same process. For fuzzy checkpoints this means calling
  `activate_context(style)` with the checkpoint's own training style before
  generating anything with it.
- The dedicated conda env is `mastersThesis`
  (`/home/lexie/miniconda3/envs/mastersThesis`, jupyter kernel `mastersthesis`) —
  it has `torch`/`transformers`/`peft`/`pandas`/etc.; the base env does not.
- `entregaRNA/` and `entregaRNA.zip` are a separate delivery snapshot (older
  6-model checkpoints without the lora/full split multi_v4 has), excluded from git
  via `.gitignore` for size — don't confuse them with the current `checkpoints/`.

## Collaboration rules

- Never commit, push, or merge without the user explicitly asking in that moment
  — this applies even mid-task, not just at the end.
- Never run model training/fine-tuning; only prepare the code/notebook and tell
  the user to run it themselves.
