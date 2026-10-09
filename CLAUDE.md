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
