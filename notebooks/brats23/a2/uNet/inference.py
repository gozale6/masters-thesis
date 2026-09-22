# inference_unet.py
import os
import json
import argparse
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='monai.transforms')

# ============================================================================
# Configuration — must match training
# ============================================================================

class Config:
    # Data
    data_json = "../val_list.json"

    # Model — must match training exactly
    roi_size      = (128, 128, 128)
    in_channels   = 4
    out_channels  = 3

    # UNet architecture
    spatial_dims   = 3
    channels       = (16, 32, 64, 128, 256)
    strides        = (2, 2, 2, 2)
    num_res_units  = 2
    kernel_size    = 3
    up_kernel_size = 3
    dropout        = 0.0

    # Inference
    sw_batch_size = 2
    overlap       = 0.5

    # Output
    save_preds  = True
    output_dir  = "./predictions_unet"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# ============================================================================
# Label conversion (identical to training)
# ============================================================================

class ConvertBRaTSLabelsd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label  = d[key]
            result = torch.zeros((3, *label.shape[1:]), dtype=label.dtype, device=label.device)
            result[0] = (label[0] == 3)
            result[1] = torch.logical_or(label[0] == 1, label[0] == 3)
            result[2] = torch.logical_or(
                torch.logical_or(label[0] == 1, label[0] == 2),
                label[0] == 3
            )
            d[key] = result.float()
        return d

def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ConvertBRaTSLabelsd(keys=["label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

# ============================================================================
# Model
# ============================================================================

def create_model():
    return UNet(
        spatial_dims   = config.spatial_dims,
        in_channels    = config.in_channels,
        out_channels   = config.out_channels,
        channels       = config.channels,
        strides        = config.strides,
        kernel_size    = config.kernel_size,
        up_kernel_size = config.up_kernel_size,
        num_res_units  = config.num_res_units,
        act            = "PRELU",
        norm           = "INSTANCE",
        dropout        = config.dropout,
        bias           = True,
        adn_ordering   = "NDA",
    )

# ============================================================================
# Inference
# ============================================================================

def run_inference(checkpoint_path, data_json, save_preds):
    print("=" * 70)
    print("UNet Inference — BraTS 2023")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Data JSON  : {data_json}")
    print(f"Device     : {config.device}")
    print(f"Save preds : {save_preds}")
    print("=" * 70)

    # Load checkpoint
    checkpoint    = torch.load(checkpoint_path, map_location=config.device)
    trained_epoch = checkpoint.get("epoch", "?")
    print(f"\nLoaded checkpoint from epoch {trained_epoch}")
    if "dice_scores" in checkpoint:
        ds = checkpoint["dice_scores"]
        print(f"  Validation Dice at save — ET: {ds['et']:.4f}  TC: {ds['tc']:.4f}  "
              f"WT: {ds['wt']:.4f}  Avg: {ds['avg']:.4f}")

    # Model
    model = create_model().to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Data
    with open(data_json) as f:
        data_files = json.load(f)
    print(f"Samples to evaluate: {len(data_files)}\n")

    dataset = Dataset(data=data_files, transform=get_transforms())
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=True)

    if save_preds:
        os.makedirs(config.output_dir, exist_ok=True)

    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd_metric   = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", percentile=95)
    per_case    = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Inference")):
            inputs = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            outputs = sliding_window_inference(
                inputs        = inputs,
                roi_size      = config.roi_size,
                sw_batch_size = config.sw_batch_size,
                predictor     = model,
                overlap       = config.overlap,
            )

            outputs_prob = torch.sigmoid(outputs)
            outputs_bin  = (outputs_prob > 0.5).float()

            # Per-case Dice
            case_dice_m = DiceMetric(include_background=True, reduction="mean_batch")
            case_dice_m(y_pred=outputs_bin, y=labels)
            case_dice = case_dice_m.aggregate().cpu().numpy()
            case_dice_m.reset()

            # Per-case HD95
            case_hd_m = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", percentile=95)
            case_hd_m(y_pred=outputs_bin, y=labels)
            case_hd = case_hd_m.aggregate().cpu().numpy()
            case_hd_m.reset()

            per_case.append({
                "idx":     idx,
                "et":      float(case_dice[0]),
                "tc":      float(case_dice[1]),
                "wt":      float(case_dice[2]),
                "avg":     float(case_dice.mean()),
                "hd95_et": float(case_hd[0]),
                "hd95_tc": float(case_hd[1]),
                "hd95_wt": float(case_hd[2]),
                "hd95_avg":float(case_hd.mean()),
            })

            # Accumulate for global metrics
            dice_metric(y_pred=outputs_bin, y=labels)
            hd_metric(y_pred=outputs_bin, y=labels)

            if save_preds:
                pred_np   = outputs_bin[0].cpu().numpy().astype(np.uint8)
                label_map = np.zeros(pred_np.shape[1:], dtype=np.uint8)
                label_map[pred_np[2] == 1] = 2
                label_map[pred_np[1] == 1] = 1
                label_map[pred_np[0] == 1] = 3
                img = nib.Nifti1Image(label_map, affine=np.eye(4))
                nib.save(img, os.path.join(config.output_dir, f"pred_{idx:04d}.nii.gz"))

            torch.cuda.empty_cache()

    # Aggregate metrics
    mean_dice = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()
    mean_hd   = hd_metric.aggregate().cpu().numpy()
    hd_metric.reset()

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  {'':25}  {'Dice':>6}  {'HD95':>8}")
    print(f"  {'-'*43}")
    print(f"  {'ET (Enhancing Tumor)':25}  {mean_dice[0]:>6.4f}  {mean_hd[0]:>8.2f}")
    print(f"  {'TC (Tumor Core)':25}  {mean_dice[1]:>6.4f}  {mean_hd[1]:>8.2f}")
    print(f"  {'WT (Whole Tumor)':25}  {mean_dice[2]:>6.4f}  {mean_hd[2]:>8.2f}")
    print(f"  {'-'*43}")
    print(f"  {'Average':25}  {mean_dice.mean():>6.4f}  {mean_hd.mean():>8.2f}")
    print("=" * 70)

    print("\nPer-case breakdown:")
    print(f"  {'Idx':>4}  {'ET':>6}  {'TC':>6}  {'WT':>6}  {'Avg':>6}  "
          f"{'HD95_ET':>8}  {'HD95_TC':>8}  {'HD95_WT':>8}  {'HD95_Avg':>9}")
    print("  " + "-" * 80)
    for c in per_case:
        print(f"  {c['idx']:>4}  {c['et']:>6.4f}  {c['tc']:>6.4f}  {c['wt']:>6.4f}  {c['avg']:>6.4f}  "
              f"{c['hd95_et']:>8.2f}  {c['hd95_tc']:>8.2f}  {c['hd95_wt']:>8.2f}  {c['hd95_avg']:>9.2f}")

    results = {
        "checkpoint": checkpoint_path,
        "epoch":      trained_epoch,
        "aggregate": {
            "dice_et":  float(mean_dice[0]),
            "dice_tc":  float(mean_dice[1]),
            "dice_wt":  float(mean_dice[2]),
            "dice_avg": float(mean_dice.mean()),
            "hd95_et":  float(mean_hd[0]),
            "hd95_tc":  float(mean_hd[1]),
            "hd95_wt":  float(mean_hd[2]),
            "hd95_avg": float(mean_hd.mean()),
        },
        "per_case": per_case,
    }
    out_json = os.path.join(
        os.path.dirname(checkpoint_path), "inference_results.json"
    )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_json}")

    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet inference on BraTS 2023")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints_unet/best_model.pth",
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default=config.data_json,
        help="JSON file listing {'image': [...], 'label': '...'} dicts",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        default=config.save_preds,
        help="Save predicted NIfTI segmentation masks",
    )
    args = parser.parse_args()

    run_inference(
        checkpoint_path = args.checkpoint,
        data_json       = args.data_json,
        save_preds      = args.save_preds,
    )
