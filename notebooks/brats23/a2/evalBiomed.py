"""
evalBiomed.py
=============
Loads the fine-tuned BioMedLM LoRA checkpoint, runs generation on the
validation split (reproduced with the same seed as training), and reports
ROUGE-1/2/L and corpus BLEU.

Usage
-----
    conda run -n mastersThesis python notebooks/brats23/a2/evalBiomed.py

    # override paths:
    conda run -n mastersThesis python notebooks/brats23/a2/evalBiomed.py \
        --csv        atlas_segmentations/segresnet/all_cases.csv \
        --text_dir   ../TextBraTSData \
        --model_dir  checkpoints/biomedlm \
        --output     eval_results.json \
        --max_new_tokens 400
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ── copy the three helpers from trainBiomed (keep identical logic) ────────────
PROMPT_SEP    = "\n### Report:\n"
EOS_TOKEN     = "<|endoftext|>"
MODEL_MAX_SEQ = 1024

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def pivot_matrix(case_df):
    pivot = case_df.pivot_table(
        index="atlas_region_name",
        columns="tumor_tag",
        values="count_scaled",
        aggfunc="max",
        fill_value=0,
    )
    for tag in ["ET", "TC", "WT"]:
        if tag not in pivot.columns:
            pivot[tag] = 0
    return pivot[["ET", "TC", "WT"]]


def matrix_to_prompt(case_id, pivot):
    active = pivot[(pivot > 0).any(axis=1)].copy()
    active["_total"] = active.sum(axis=1)
    active = active.sort_values("_total", ascending=False).drop(columns="_total")
    lines = [
        f"Patient: {case_id}",
        "Brain tumor atlas analysis (scale 0–256, higher = greater involvement):",
    ]
    for region_name, row in active.iterrows():
        parts = []
        if row["ET"] > 0:
            parts.append(f"ET={int(row['ET'])}")
        if row["TC"] > 0:
            parts.append(f"TC={int(row['TC'])}")
        if row["WT"] > 0:
            parts.append(f"WT={int(row['WT'])}")
        lines.append(f"  - {region_name}: {', '.join(parts)}")
    lines.append(
        "\nTumor tags: ET=Enhancing Tumor, TC=Tumor Core (necrotic), WT=Whole Tumor (edema)"
    )
    return "\n".join(lines)


def load_report(text_dir, case_id):
    case_dir = text_dir / case_id
    txt_path = case_dir / f"{case_id}_flair_text.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    npy_path = case_dir / f"{case_id}_flair_text.npy"
    if npy_path.exists():
        import numpy as np
        arr = np.load(npy_path, allow_pickle=True)
        text = str(arr.item()).strip() if arr.ndim == 0 else " ".join(
            str(s).strip() for s in arr.tolist() if str(s).strip()
        )
        if text:
            return text
    return None


def build_samples(csv_path, text_dir):
    df = pd.read_csv(csv_path)
    samples, skipped = [], []
    for case_id, case_df in df.groupby("case_id"):
        report = load_report(text_dir, case_id)
        if report is None:
            skipped.append(case_id)
            continue
        pivot  = pivot_matrix(case_df)
        prompt = matrix_to_prompt(case_id, pivot)
        samples.append({"case_id": case_id, "input": prompt, "target": report})
    log.info(f"Matched pairs: {len(samples)}  |  skipped: {len(skipped)}")
    return samples


# ── generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, device, max_new_tokens):
    full_prompt = prompt + PROMPT_SEP
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_MAX_SEQ - max_new_tokens,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_list, r2_list, rL_list = [], [], []
    bleu_refs, bleu_hyps = [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rL_list.append(scores["rougeL"].fmeasure)

        bleu_refs.append([ref.split()])
        bleu_hyps.append(pred.split())

    bleu = corpus_bleu(
        bleu_refs, bleu_hyps,
        smoothing_function=SmoothingFunction().method1,
    )

    return {
        "rouge1": round(sum(r1_list) / len(r1_list), 4),
        "rouge2": round(sum(r2_list) / len(r2_list), 4),
        "rougeL": round(sum(rL_list) / len(rL_list), 4),
        "bleu":   round(bleu, 4),
        "n_samples": len(predictions),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            default="atlas_segmentations/segresnet/all_cases.csv")
    p.add_argument("--text_dir",       default="../TextBraTSData")
    p.add_argument("--model_dir",      default="checkpoints/biomedlm")
    p.add_argument("--output",         default="eval_results.json")
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── reproduce exact val split ─────────────────────────────────────────────
    text_dir = Path(args.text_dir)
    samples  = build_samples(args.csv, text_dir)
    _, val_samples = train_test_split(
        samples, test_size=0.20, random_state=args.seed, shuffle=True
    )
    log.info(f"Val samples: {len(val_samples)}")

    # ── load base model + LoRA adapter ────────────────────────────────────────
    model_dir = Path(args.model_dir)
    log.info(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_id = "stanford-crfm/BioMedLM"
    log.info(f"Loading base model: {base_model_id}")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32
    # device_map="auto" loads directly onto GPU layer by layer —
    # avoids the CPU→GPU copy that doubles peak memory usage
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=dtype, device_map="auto"
    )

    log.info(f"Applying LoRA adapter from {model_dir}")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    # ── generate predictions ──────────────────────────────────────────────────
    predictions, references, case_ids = [], [], []

    for sample in tqdm(val_samples, desc="Generating"):
        pred = generate(model, tokenizer, sample["input"], device, args.max_new_tokens)
        predictions.append(pred)
        references.append(sample["target"])
        case_ids.append(sample["case_id"])

    # ── compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(predictions, references)

    log.info("=" * 50)
    log.info(f"  ROUGE-1 : {metrics['rouge1']:.4f}")
    log.info(f"  ROUGE-2 : {metrics['rouge2']:.4f}")
    log.info(f"  ROUGE-L : {metrics['rougeL']:.4f}")
    log.info(f"  BLEU    : {metrics['bleu']:.4f}")
    log.info(f"  Samples : {metrics['n_samples']}")
    log.info("=" * 50)

    # ── save results ──────────────────────────────────────────────────────────
    output = {
        "metrics": metrics,
        "predictions": [
            {"case_id": c, "prediction": p, "reference": r}
            for c, p, r in zip(case_ids, predictions, references)
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Archive results + model checkpoint together under a timestamped folder
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = out_path.parent / "eval_runs" / f"run_{ts}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Archive eval JSON
    if out_path.exists():
        shutil.copy(out_path, archive_dir / out_path.name)
        log.info(f"Previous results archived → {archive_dir / out_path.name}")

    # Archive model checkpoint
    model_archive = archive_dir / "model"
    shutil.copytree(model_dir, model_archive)
    log.info(f"Model checkpoint archived → {model_archive}")

    out_path.write_text(json.dumps(output, indent=2))
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
