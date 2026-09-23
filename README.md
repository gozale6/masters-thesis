# Generación de Reportes Radiológicos de RM Cerebral mediante una Matriz de Características Patológicas Basada en Atlas, un Estudio con TextBraTS

Master's thesis project — Universidad Autónoma de Chihuahua

---

## Project Overview

Two-phase pipeline for automated brain MRI radiology report generation:

1. **Segmentation (Phase 1)** — Train deep learning models (U-Net, SegResNet, SwinUNETR) on BraTS data to segment tumor regions
2. **Report Generation (Phase 2)** — Fine-tune clinical LLMs (SciFive, ClinicalT5) using an atlas-based pathological feature matrix derived from segmentation outputs

---

## Repository Structure

```
masters-thesis/
├── src/
│   ├── segmentation/         # Phase 1: segmentation model code
│   │   ├── unet/
│   │   ├── segresnet/
│   │   └── swinunetr/
│   ├── report_generation/    # Phase 2: LLM fine-tuning code
│   └── utils/
├── experiments/              # One folder per experiment — config + logs + notes
│   ├── 001_segmentation/
│   └── 002_report_generation/
├── checkpoints/              # Saved model weights (tracked via Git LFS)
│   ├── segmentation/
│   └── report_generation/
├── results/                  # Evaluation outputs and predictions
│   ├── segmentations/
│   └── report_generation/
├── data/
│   ├── splits/               # Train/val JSON split files
│   ├── manifest.csv
│   └── BraTS2023_2017_GLI_Mapping.xlsx
├── notebooks/                # Exploratory analysis only (not production code)
└── references/               # Papers and external resources
```

> **Datasets** (`brats20/`, `brats23/`) live on local disk and are not tracked by git.
> See [data/README.md](data/README.md) for download instructions.

---

## How to Work on This Project

### Daily workflow

```bash
# 1. Make sure you're up to date
git pull origin main

# 2. Start a new experiment or feature on its own branch
git checkout -b experiment/003-scifive-large-full-finetune

# 3. Work, edit code, run experiments...

# 4. Commit frequently with meaningful messages
git add src/report_generation/train.py experiments/003_scifive_large_full/
git commit -m "Try full fine-tune on scifive-large: adjust lr to 1e-4"

# 5. When the experiment is done, push and merge to main
git push origin experiment/003-scifive-large-full-finetune
# Then open a pull request or merge directly:
git checkout main
git merge experiment/003-scifive-large-full-finetune
git push origin main
```

---

### Branch naming convention

| Prefix | When to use | Example |
|---|---|---|
| `experiment/NNN-description` | New model run or hyperparameter search | `experiment/004-clinicalt5-aug-lora` |
| `feature/description` | New code capability (new model, new metric) | `feature/add-rouge-score` |
| `fix/description` | Bug fix | `fix/inference-batch-size` |

---

### Adding a new experiment

Every experiment gets its own numbered folder under `experiments/`:

```
experiments/
└── 003_scifive_large_full/
    ├── config.yaml     ← hyperparameters and settings used
    ├── results.json    ← metrics (BLEU, ROUGE, BERTScore, etc.)
    └── notes.md        ← what you tried, what worked, what to try next
```

**config.yaml example:**
```yaml
model: scifive-large-pubmed
strategy: full-finetune
dataset: brats2023
learning_rate: 1e-4
batch_size: 4
epochs: 10
augmentation: false
data_split: data/splits/train_list.json
```

**notes.md example:**
```markdown
## Hypothesis
Full fine-tuning on scifive-large should outperform LoRA on this domain.

## Result
BLEU: 0.31 — slight improvement over LoRA (0.28). Training time 3x longer.

## Next steps
- Try with augmentation
- Compare BERTScore on 2020 vs 2023 split
```

---

### What goes where

| Thing | Where |
|---|---|
| Reusable, cleaned-up code | `src/` |
| One-off experiment scripts | `experiments/NNN/` |
| Saved model weights | `checkpoints/` (LFS tracked) |
| Evaluation outputs / predictions | `results/` |
| Exploratory / analysis notebooks | `notebooks/` |
| Train/val data splits | `data/splits/` |

### Rules

- **Never name files `_v2`, `_v3`** — use git commits and branches instead
- **Never commit directly to `main`** for experiments — use a branch
- **Always fill in `config.yaml` and `notes.md`** before closing an experiment branch
- **Large files** (model weights, datasets) are tracked via Git LFS — commit them normally, git handles the rest
- **`src/` always holds the current best version** of code — old versions live in git history

---

## Setup

```bash
# Clone the repo
git clone https://github.com/gozale6/masters-thesis.git
cd masters-thesis

# Install git-lfs (required to download checkpoints)
sudo apt-get install git-lfs
git lfs pull

# Download datasets (BraTS 2020 / 2023) and place at:
#   /path/to/brats20/
#   /path/to/brats23/
# Update data paths in config files accordingly
```
