"""
apply_atlas_matrix_2020.py
==========================
Variant of apply_atlas_matrix.py for SegResNet only.

Key differences from the original:
  - Single model: segresnet
  - Uses val_list_2020.json (BraTS2023 IDs with patient_id field)
  - Maps each BraTS2023 case ID → BraTS2020 ID via BraTS2023_2017_GLI_Mapping.xlsx
  - case_id in output = BraTS2020 ID (e.g. BraTS20_Training_174)
    so it can be joined directly with TextBraTS report texts
  - Cases with no BraTS2020 mapping are skipped with a warning

Output layout
-------------
atlas_segmentations/
    segresnet/
        matrices/
            BraTS20_Training_174.csv   …  per-case matrix (BraTS2020 ID as filename)
        all_cases.csv                  …  all cases concatenated

CSV columns
-----------
  case_id            BraTS2020 ID  (e.g. BraTS20_Training_174)
  brats23_id         Original BraTS2023 ID  (e.g. BraTS-GLI-00100-000)
  pred_index         Zero-padded index in val_list  (e.g. pred_0012) — for traceability
  atlas_region_id    Julich-Brain integer ID (0 = WM / unassigned)
  atlas_region_name  Human-readable Julich region name
  brats_label        1 (NCR) | 2 (ED) | 3 (ET)
  tumor_tag          TC | WT | ET
  voxel_count        Raw voxel count in that region × label cell
  count_scaled       0–256 (256 = busiest region × label pair in this case)

BraTS label → tumor tag
-----------------------
  1 (NCR/NET) → TC   Necrotic Core
  2 (ED)      → WT   Peritumoral Edema
  3 (ET)      → ET   Enhancing Tumor
"""

import os
import csv
import json
import warnings

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── BraTS label → tumor tag ───────────────────────────────────────────────────
LABEL_TAG = {
    1: "TC",
    2: "WT",
    3: "ET",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
VAL_JSON     = os.path.join(BASE, "val_list_2020.json")
MAPPING_XLSX = os.path.join(BASE, "BraTS2023_2017_GLI_Mapping.xlsx")

PRED_DIR = os.path.join(BASE, "segResNet", "predictions_segresnet")
ATLAS_OUT = os.path.join(BASE, "atlas_segmentations", "segresnet")

os.makedirs(os.path.join(ATLAS_OUT, "matrices"), exist_ok=True)

# ── Load val list ─────────────────────────────────────────────────────────────
with open(VAL_JSON) as f:
    val_list = json.load(f)

print(f"Val list entries : {len(val_list)}")

# ── Build BraTS2023 → BraTS2020 mapping ───────────────────────────────────────
# Columns: BraTS2023 (col 0), BraTS2021 (col 1), BraTS2020 (col 2)
print("Loading BraTS2023 → BraTS2020 mapping …")
df_map = pd.read_excel(MAPPING_XLSX, header=0, usecols=[0, 2])
df_map.columns = ["brats23_id", "brats20_id"]
df_map = df_map[df_map["brats20_id"].notna() & (df_map["brats20_id"] != "N/A")]
brats23_to_brats20: dict[str, str] = dict(
    zip(df_map["brats23_id"].str.strip(), df_map["brats20_id"].str.strip())
)
print(f"  {len(brats23_to_brats20)} cases with a BraTS2020 mapping")

# ── MNI152 template ───────────────────────────────────────────────────────────
print("\nLoading MNI152 template …")
mni_template = datasets.load_mni152_template()

# ── Julich-Brain atlas via siibra ─────────────────────────────────────────────
print("Loading Julich-Brain atlas via siibra …")
import siibra

atlas        = siibra.atlases["human"]
julichbrain  = atlas.get_parcellation("julich 3")
julich_pmaps = siibra.get_map(
    parcellation=julichbrain,
    space=siibra.spaces.get("mni152"),
    maptype=siibra.MapType.LABELLED,
)

# Region ID → name lookup
print("Building region ID → name lookup …")
region_id_to_name: dict[int, str] = {}
try:
    for region in julichbrain.regiontree:
        rid = region.index
        if rid is not None:
            region_id_to_name[int(rid)] = region.name
except Exception:
    pass

# Resample atlas once
print("Resampling Julich-Brain atlas → MNI template grid …")
atlas_nii  = julich_pmaps.fetch()
atlas_mni  = image.resample_to_img(atlas_nii, mni_template, interpolation="nearest")
atlas_data = atlas_mni.get_fdata().astype(np.int32)
print(f"Atlas shape: {atlas_data.shape}  unique labels: {len(np.unique(atlas_data))}")

# ── CSV header ────────────────────────────────────────────────────────────────
CSV_HEADER = [
    "case_id",           # BraTS2020 ID — use this to join with TextBraTS
    "brats23_id",        # original BraTS2023 ID — for traceability
    "pred_index",        # pred_XXXX — position in val_list
    "atlas_region_id",
    "atlas_region_name",
    "brats_label",
    "tumor_tag",
    "voxel_count",
    "count_scaled",      # 0–256
]


def build_matrix_rows(
    case_id: str,
    brats23_id: str,
    pred_index: str,
    pred_mni_data: np.ndarray,
) -> list[dict]:
    """
    Return one row per (atlas_region, brats_label) pair with non-zero overlap.
    count_scaled: 0–256, where 256 = most-occupied pair in this case.
    """
    rows = []

    for brats_label, tumor_tag in LABEL_TAG.items():
        label_mask = pred_mni_data == brats_label
        if not np.any(label_mask):
            continue

        region_ids_at_label = atlas_data[label_mask]
        unique_ids, counts   = np.unique(region_ids_at_label, return_counts=True)

        for region_id, count in zip(unique_ids, counts):
            region_id   = int(region_id)
            region_name = region_id_to_name.get(
                region_id,
                "WM/unassigned" if region_id == 0 else f"region_{region_id}",
            )
            rows.append({
                "case_id":           case_id,
                "brats23_id":        brats23_id,
                "pred_index":        pred_index,
                "atlas_region_id":   region_id,
                "atlas_region_name": region_name,
                "brats_label":       brats_label,
                "tumor_tag":         tumor_tag,
                "voxel_count":       int(count),
                "count_scaled":      None,
            })

    # Normalise per case
    if rows:
        max_count = max(r["voxel_count"] for r in rows)
        for r in rows:
            r["count_scaled"] = int(round(r["voxel_count"] / max_count * 256))

    return rows


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nProcessing {len(val_list)} cases (segresnet only) …\n")

all_rows     = []
skipped_no_map   = []
skipped_no_pred  = []

for idx, sample in enumerate(tqdm(val_list, desc="Cases")):

    # BraTS2023 ID comes from the patient_id field in val_list_2020.json
    brats23_id = sample.get("patient_id", "")
    pred_index = f"pred_{idx:04d}"
    pred_name  = f"{pred_index}.nii.gz"

    # Resolve BraTS2020 ID
    brats20_id = brats23_to_brats20.get(brats23_id)
    if brats20_id is None:
        tqdm.write(f"[SKIP] no BraTS2020 mapping for {brats23_id} ({pred_index})")
        skipped_no_map.append(brats23_id)
        continue

    pred_path = os.path.join(PRED_DIR, pred_name)
    if not os.path.exists(pred_path):
        tqdm.write(f"[WARN] prediction file missing: {pred_path}")
        skipped_no_pred.append(pred_index)
        continue

    # Load and reorient prediction to MNI152
    pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)
    pred_rot  = np.rot90(pred_data, k=2)
    pred_mni  = image.resample_to_img(
        nib.Nifti1Image(pred_rot.astype(np.float32), affine=mni_template.affine),
        mni_template,
        interpolation="nearest",
    )
    pred_mni_data = np.round(pred_mni.get_fdata()).astype(np.uint8)

    # Build rows
    rows = build_matrix_rows(brats20_id, brats23_id, pred_index, pred_mni_data)

    # Per-case CSV named by BraTS2020 ID
    csv_path = os.path.join(ATLAS_OUT, "matrices", f"{brats20_id}.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    all_rows.extend(rows)

# ── Aggregated all_cases.csv ──────────────────────────────────────────────────
agg_path = os.path.join(ATLAS_OUT, "all_cases.csv")
with open(agg_path, "w", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(all_rows)

# ── Summary ───────────────────────────────────────────────────────────────────
processed = len(val_list) - len(skipped_no_map) - len(skipped_no_pred)
print(f"\nDone.")
print(f"  Processed          : {processed} cases")
print(f"  Skipped (no map)   : {len(skipped_no_map)} cases — no BraTS2020 ID in mapping file")
print(f"  Skipped (no pred)  : {len(skipped_no_pred)} cases — prediction .nii.gz not found")
print(f"  Total rows written : {len(all_rows)}")
print(f"  Aggregated CSV     : {agg_path}")
print(
    "\n  CSV columns:"
    "\n    case_id            BraTS2020 ID  → join key with TextBraTS"
    "\n    brats23_id         BraTS2023 ID  → traceability"
    "\n    pred_index         pred_XXXX     → traceability"
    "\n    atlas_region_id    Julich-Brain integer ID"
    "\n    atlas_region_name  human-readable region name"
    "\n    brats_label        1 (NCR) | 2 (ED) | 3 (ET)"
    "\n    tumor_tag          TC | WT | ET"
    "\n    voxel_count        raw voxel count"
    "\n    count_scaled       0–256 (256 = busiest pair in this case)"
)