# run_val_inference.py
# Runs SegResNet inference on the BraTS 2023 Challenge Validation Data.
# No ground-truth labels are available — predictions are saved as NIfTI files.
# Run: python run_val_inference.py

import os
import json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.transforms")

# ============================================================================
# Paths — edit these
# ============================================================================

VAL_DATA_ROOT   = "/home/lexie/Desktop/mastersThesis/brats23/BraTS2023-Challenge-ValidationData"
CHECKPOINT_PATH = "./checkpoints_segresnet/best_model.pth"
OUTPUT_DIR      = "./predictions_segresnet_val"

# ============================================================================
# Model config — must match training exactly
# ============================================================================

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ROI_SIZE      = (128, 128, 128)
IN_CHANNELS   = 4
OUT_CHANNELS  = 3
SPATIAL_DIMS  = 3
INIT_FILTERS  = 16
BLOCKS_DOWN   = (1, 2, 2, 4)
BLOCKS_UP     = (1, 1, 1)
DROPOUT_PROB  = 0.2
SW_BATCH_SIZE = 2
OVERLAP       = 0.5

# ============================================================================
# Build datalist from the validation folder (images only, no labels)
# ============================================================================

def build_datalist(data_root):
    data_root = Path(data_root)
    datalist  = []
    missing   = []

    for patient_dir in sorted(data_root.glob("BraTS-GLI-*")):
        patient_id = patient_dir.name
        t1n = list(patient_dir.glob("*-t1n.nii.gz"))
        t1c = list(patient_dir.glob("*-t1c.nii.gz"))
        t2w = list(patient_dir.glob("*-t2w.nii.gz"))
        t2f = list(patient_dir.glob("*-t2f.nii.gz"))

        if t1n and t1c and t2w and t2f:
            datalist.append({
                "image":      [str(t1n[0]), str(t1c[0]), str(t2w[0]), str(t2f[0])],
                "patient_id": patient_id,
            })
        else:
            missing.append(patient_id)

    if missing:
        print(f"Warning: incomplete cases skipped: {missing}")

    return datalist

# ============================================================================
# Transforms (image only — no label)
# ============================================================================

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("SegResNet — Inference on BraTS 2023 Challenge Validation Data")
    print("=" * 70)
    print(f"Data root  : {VAL_DATA_ROOT}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Device     : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Load checkpoint
    checkpoint    = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    trained_epoch = checkpoint.get("epoch", "?")
    print(f"\nLoaded checkpoint from epoch {trained_epoch}")
    if "dice_scores" in checkpoint:
        ds = checkpoint["dice_scores"]
        print(f"  Train-val Dice at save  "
              f"ET:{ds['et']:.4f}  TC:{ds['tc']:.4f}  "
              f"WT:{ds['wt']:.4f}  Avg:{ds['avg']:.4f}")

    # Build model
    model = SegResNet(
        spatial_dims   = SPATIAL_DIMS,
        in_channels    = IN_CHANNELS,
        out_channels   = OUT_CHANNELS,
        init_filters   = INIT_FILTERS,
        blocks_down    = BLOCKS_DOWN,
        blocks_up      = BLOCKS_UP,
        dropout_prob   = DROPOUT_PROB,
        norm           = ("GROUP", {"num_groups": 8}),
        act            = ("RELU", {"inplace": True}),
        use_conv_final = True,
        upsample_mode  = "nontrainable",
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build datalist
    datalist = build_datalist(VAL_DATA_ROOT)
    print(f"Cases found: {len(datalist)}\n")

    val_loader = DataLoader(
        Dataset(data=datalist, transform=val_transforms),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Inference
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            inputs     = batch["image"].to(DEVICE)
            patient_id = batch["patient_id"][0]

            outputs = sliding_window_inference(
                inputs        = inputs,
                roi_size      = ROI_SIZE,
                sw_batch_size = SW_BATCH_SIZE,
                predictor     = model,
                overlap       = OVERLAP,
            )

            pred_bin = (torch.sigmoid(outputs) > 0.5).float()
            pred_np  = pred_bin[0].cpu().numpy().astype(np.uint8)

            # Reconstruct integer label map from binary channels
            label_map = np.zeros(pred_np.shape[1:], dtype=np.uint8)
            label_map[pred_np[2] == 1] = 2   # WT -> edema (2)
            label_map[pred_np[1] == 1] = 1   # TC -> NCR/NET (1)
            label_map[pred_np[0] == 1] = 3   # ET (3)

            out_path = os.path.join(OUTPUT_DIR, f"{patient_id}-seg.nii.gz")
            nib.save(nib.Nifti1Image(label_map, affine=np.eye(4)), out_path)

            torch.cuda.empty_cache()

    print(f"\nDone. {len(datalist)} predictions saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
