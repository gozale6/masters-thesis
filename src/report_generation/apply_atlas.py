"""
apply_atlas.py
==============
Copies raw predictions (with corrected patient affines) and produces
atlas-annotated NIfTI files in MNI152 space using the Julich-Brain
Cytoarchitectonic Atlas via siibra.

Key fix from reference postprocessing code
------------------------------------------
BraTS predictions must be rotated by 180° in the axial plane (np.rot90 k=2)
before assigning the MNI152 affine.  Without this, the prediction and the
atlas live in different coordinate systems → all-black output.

Output layout
-------------
raw_segmentations/
    unet/          pred_0000.nii.gz …  (BraTS labels 0/1/2/3, patient affine)
    segresnet/
    swinunetr/

atlas_segmentations/
    unet/          pred_0000.nii.gz …  (int16, MNI152 space)
    segresnet/       Julich-Brain region ID at cortical/subcortical tumor voxels
    swinunetr/       Original tumor label 1/2/3 where atlas has no coverage (WM)
                     0 everywhere outside the tumor mask

atlas_segmentations/{model}/regions/
    pred_0000.json …  top-5 Julich-Brain regions per case (% of tumor / % of region)
"""

import os
import json
import warnings

import numpy as np
import nibabel as nib
import siibra
from nilearn import datasets, image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_ASSIGN_POINTS = 5_000   # subsample for siibra assignment (statistics only)
TOP_REGIONS       = 5       # regions to report in JSON

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
VAL_JSON  = os.path.join(BASE, "val_list.json")

PRED_DIRS = {
    "unet":      os.path.join(BASE, "uNet",      "predictions_unet"),
    "segresnet": os.path.join(BASE, "segResNet", "predictions_segresnet"),
    "swinunetr": os.path.join(BASE, "swinUNETR", "predictions_swinunetr"),
}

RAW_OUT   = os.path.join(BASE, "raw_segmentations")
ATLAS_OUT = os.path.join(BASE, "atlas_segmentations")

# ── Val list ──────────────────────────────────────────────────────────────────
with open(VAL_JSON) as f:
    val_list = json.load(f)

print(f"Samples : {len(val_list)}")

for model in PRED_DIRS:
    os.makedirs(os.path.join(RAW_OUT,   model), exist_ok=True)
    os.makedirs(os.path.join(ATLAS_OUT, model), exist_ok=True)
    os.makedirs(os.path.join(ATLAS_OUT, model, "regions"), exist_ok=True)

# ── MNI152 template (91×109×91, 2 mm) ────────────────────────────────────────
print("\nLoading MNI152 template …")
mni_template = datasets.load_mni152_template()

# ── Julich-Brain atlas via siibra ─────────────────────────────────────────────
print("Loading Julich-Brain atlas via siibra …")
atlas        = siibra.atlases['human']
julichbrain  = atlas.get_parcellation('julich 3')
julich_pmaps = siibra.get_map(
    parcellation=julichbrain,
    space=siibra.spaces.get('mni152'),
    maptype=siibra.MapType.LABELLED
)

# Fetch atlas NIfTI and resample once to MNI template grid
print("Resampling Julich-Brain atlas → MNI template grid …")
atlas_nii  = julich_pmaps.fetch()                              # (193,229,193) 1 mm
atlas_mni  = image.resample_to_img(atlas_nii, mni_template, interpolation='nearest')
atlas_data = atlas_mni.get_fdata().astype(np.int32)            # (91,109,91)
print(f"Atlas shape: {atlas_data.shape}  unique labels: {len(np.unique(atlas_data))}")


# ── Helper: region statistics via siibra PointSet.assign ─────────────────────
def get_affected_areas(pmaps, voxel_coords, mni_affine, top=5):
    """
    Return top brain regions and their percentages using siibra assignment.
    voxel_coords: tuple of arrays (i, j, k) — output of np.where on MNI grid.
    Subsamples to MAX_ASSIGN_POINTS for speed.
    """
    total = len(voxel_coords[0])
    if total == 0:
        return {}

    # Subsample
    rng = np.random.default_rng(0)
    if total > MAX_ASSIGN_POINTS:
        sel = rng.choice(total, MAX_ASSIGN_POINTS, replace=False)
        coords = tuple(arr[sel] for arr in voxel_coords)
    else:
        coords = voxel_coords

    points = siibra.PointCloud(
        tuple(zip(*coords)),
        space='mni152',
        sigma_mm=5
    ).transform(mni_affine, space='mni152')

    assignments = pmaps.assign(points)
    top_r       = assignments['region'].value_counts()[:top]

    results = {}
    for region_name, count in top_r.items():
        try:
            p_map = pmaps.fetch(region=region_name)
            n_vox = int(np.count_nonzero(p_map.get_fdata()))
        except Exception:
            n_vox = 1
        results[str(region_name)] = {
            "Percentage_of_Tumor":           round(count / len(assignments) * 100, 2),
            "Percentage_of_Region_Affected": round(count / max(n_vox, 1) * 100, 2),
        }
    return results


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nProcessing {len(val_list)} cases × {len(PRED_DIRS)} models …\n")

for idx, sample in enumerate(tqdm(val_list, desc="Cases")):

    pred_name   = f"pred_{idx:04d}.nii.gz"
    orig_affine = nib.load(sample["image"][0]).affine

    for model, pred_dir in PRED_DIRS.items():

        pred_path = os.path.join(pred_dir, pred_name)
        if not os.path.exists(pred_path):
            tqdm.write(f"[WARN] missing: {pred_path}")
            continue

        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

        # ── 1. Raw: corrected patient affine ──────────────────────────────
        nib.save(
            nib.Nifti1Image(pred_data, affine=orig_affine),
            os.path.join(RAW_OUT, model, pred_name)
        )

        # ── 2. Align to MNI152 via rot90 (reference code approach) ────────
        # A 180° rotation in the axial plane maps BraTS voxel orientation
        # onto the MNI152 grid, after which the MNI affine applies correctly.
        pred_rot = np.rot90(pred_data, k=2)

        pred_mni = image.resample_to_img(
            nib.Nifti1Image(pred_rot.astype(np.float32), affine=mni_template.affine),
            mni_template,
            interpolation='nearest',
        )
        pred_mni_data = np.round(pred_mni.get_fdata()).astype(np.uint8)  # (91,109,91)
        tumor_mask    = pred_mni_data > 0

        # ── 3. Atlas-labeled NIfTI ────────────────────────────────────────
        # Tumor voxels   → Julich-Brain region ID (1-1157)
        # WM/unassigned  → original tumor label 1/2/3 (atlas has 0 there)
        # Background     → 0
        atlas_seg = np.zeros(mni_template.shape[:3], dtype=np.int16)
        atlas_seg[tumor_mask] = atlas_data[tumor_mask].astype(np.int16)

        wm_mask = tumor_mask & (atlas_seg == 0)
        atlas_seg[wm_mask] = pred_mni_data[wm_mask].astype(np.int16)

        nib.save(
            nib.Nifti1Image(atlas_seg, affine=mni_template.affine),
            os.path.join(ATLAS_OUT, model, pred_name)
        )

        # ── 4. Region statistics JSON (siibra PointSet assign) ────────────
        voxel_coords = np.where(tumor_mask)
        try:
            regions = get_affected_areas(
                julich_pmaps, voxel_coords, mni_template.affine, top=TOP_REGIONS
            )
        except Exception as e:
            tqdm.write(f"[WARN] region assign failed for {model}/{pred_name}: {e}")
            regions = {}

        json_path = os.path.join(ATLAS_OUT, model, "regions",
                                 pred_name.replace(".nii.gz", ".json"))
        with open(json_path, "w") as jf:
            json.dump({"top_regions": regions}, jf, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nDone.")
print(f"  Raw   segmentations → {RAW_OUT}")
print(f"  Atlas segmentations → {ATLAS_OUT}")
print(
    "\n  Atlas NIfTI encoding (int16, MNI152 2mm space):"
    "\n    1–1157  Julich-Brain Cytoarchitectonic region ID"
    "\n    1/2/3   Original tumor label (white matter, no atlas coverage)"
    "\n    0       Background"
)
