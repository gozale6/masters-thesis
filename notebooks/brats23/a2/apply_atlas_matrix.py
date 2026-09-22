"""
apply_atlas_matrix.py
=====================
Produces CSV matrices per case.

Each CSV row:
    case_id | atlas_region_name | tumor_tag | voxel_count | count_scaled

case_id is the BraTS2020 folder name (e.g. BraTS20_Training_001), mapped
from the BraTS2023 patient_id in val_list.json via the provided Excel mapping
file. Cases without a BraTS2020 counterpart are skipped.

BraTS label → tumor tag mapping
  1 (NCR/NET) → TC
  2 (ED)      → WT
  3 (ET)      → ET
  0           → background (skipped)

count_scaled: per case, max voxel_count across (region, label) pairs → 256.
"""

import os
import csv
import json
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
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
BASE     = os.path.dirname(os.path.abspath(__file__))
VAL_JSON = os.path.join(BASE, "val_list.json")
MAPPING_XLSX = os.path.join(BASE, "BraTS2023_2017_GLI_Mapping.xlsx")

PRED_DIRS = {
    "unet":      os.path.join(BASE, "uNet",      "predictions_unet"),
    "segresnet": os.path.join(BASE, "segResNet", "predictions_segresnet"),
    "swinunetr": os.path.join(BASE, "swinUNETR", "predictions_swinunetr"),
}

ATLAS_OUT = os.path.join(BASE, "atlas_segmentations")

# ── Val list ──────────────────────────────────────────────────────────────────
with open(VAL_JSON) as f:
    val_list = json.load(f)

print(f"Samples : {len(val_list)}")

# ── BraTS23 → BraTS20 mapping ────────────────────────────────────────────────
print(f"Loading BraTS23 → BraTS20 mapping: {MAPPING_XLSX}")
mapping_df = pd.read_excel(MAPPING_XLSX)
brats23_to_brats20 = dict(
    zip(mapping_df["BraTS2023"], mapping_df["BraTS2020"])
)
# Drop NaN values (cases without a BraTS20 counterpart)
brats23_to_brats20 = {
    k: v for k, v in brats23_to_brats20.items()
    if isinstance(v, str) and v.strip()
}
print(f"  BraTS23 → BraTS20 pairs available: {len(brats23_to_brats20)}")

# Count how many val_list entries have a BraTS20 counterpart
mapped_count = sum(
    1 for s in val_list
    if isinstance(s, dict) and s.get("patient_id") in brats23_to_brats20
)
print(f"  Val cases with BraTS20 mapping: {mapped_count} / {len(val_list)}")

for model in PRED_DIRS:
    os.makedirs(os.path.join(ATLAS_OUT, model, "matrices"), exist_ok=True)

# ── MNI152 template ───────────────────────────────────────────────────────────
print("\nLoading MNI152 template …")
mni_template = datasets.load_mni152_template()

# ── Julich-Brain atlas via siibra ─────────────────────────────────────────────
print("Loading Julich-Brain atlas via siibra …")
import siibra

atlas       = siibra.atlases['human']
julichbrain = atlas.get_parcellation('julich 3')
julich_pmaps = siibra.get_map(
    parcellation=julichbrain,
    space=siibra.spaces.get('mni152'),
    maptype=siibra.MapType.LABELLED
)

# ── Build per-fragment (label, fragment) → region name lookup ─────────────────
# The Julich-Brain labelled map uses per-fragment labels (left vs. right
# hemisphere). Fetching without a fragment merges them and re-labels 1..N,
# which breaks label → region lookup. So we:
#   1) build (label, fragment) → region_name from the parcellation
#   2) fetch each fragment separately (labels match siibra's index)
#   3) compose into one volume with disjoint global-id ranges per fragment
print("Building per-fragment (label, fragment) → region name lookup …")

fragments: list[str] = []
for attr in ("fragments", "_fragments"):
    f = getattr(julich_pmaps, attr, None)
    if f:
        try:
            fragments = list(f.keys()) if hasattr(f, "keys") else list(f)
            if fragments:
                break
        except Exception:
            pass
if not fragments:
    fragments = ["left hemisphere", "right hemisphere"]
print(f"  Fragments: {fragments}")

all_regions = []
seen = set()
try:
    for r in julichbrain:
        if id(r) not in seen:
            seen.add(id(r))
            all_regions.append(r)
except Exception:
    pass
print(f"  Regions in parcellation: {len(all_regions)}")

frag_label_to_name: dict[tuple[str, int], str] = {}
resolved = 0
for region in all_regions:
    try:
        idxs = julich_pmaps.get_index(region)
    except Exception:
        continue
    if idxs is None:
        continue
    if not isinstance(idxs, (list, tuple)):
        idxs = [idxs]
    for idx in idxs:
        label = getattr(idx, "label", None)
        frag  = getattr(idx, "fragment", None)
        if label is None or frag is None:
            continue
        try:
            label = int(label)
        except Exception:
            continue
        if label == 0:
            continue
        frag_label_to_name[(frag, label)] = region.name
        resolved += 1
print(f"  (fragment, label) pairs resolved: {resolved}")

# ── Fetch each fragment and compose ──────────────────────────────────────────
print("\nFetching each hemisphere fragment and composing …")
composed_data = None
global_id_to_name: dict[int, str] = {0: "WM/unassigned"}

for frag_i, fragment_name in enumerate(fragments):
    print(f"  Fetching fragment: {fragment_name} …")
    try:
        frag_nii = julich_pmaps.fetch(fragment=fragment_name)
    except Exception as e:
        print(f"    [WARN] fetch failed: {e}")
        continue

    frag_mni = image.resample_to_img(frag_nii, mni_template, interpolation='nearest')
    frag_data = frag_mni.get_fdata().astype(np.int32)

    offset = frag_i * 1000
    local_labels = np.unique(frag_data)
    local_labels = local_labels[local_labels != 0]
    print(f"    local labels: {len(local_labels)}  range: {local_labels.min()}–{local_labels.max()}")

    hits = 0
    for local_label in local_labels:
        global_id = int(local_label) + offset
        name = frag_label_to_name.get((fragment_name, int(local_label)))
        if name is None:
            name = f"region_{int(local_label)}_{fragment_name.replace(' ', '_')}"
        else:
            hits += 1
        global_id_to_name[global_id] = name
    print(f"    named via map lookup: {hits}/{len(local_labels)}")

    frag_global = np.where(frag_data != 0, frag_data + offset, 0).astype(np.int32)

    if composed_data is None:
        composed_data = frag_global
    else:
        bg = composed_data == 0
        composed_data = np.where(bg, frag_global, composed_data)

if composed_data is None:
    raise RuntimeError("Could not fetch any Julich-Brain fragments.")

atlas_data = composed_data
print(f"Composed atlas shape: {atlas_data.shape}  unique ids: {len(np.unique(atlas_data))}")

unique_in_volume = sorted(int(l) for l in np.unique(atlas_data))
missing = [l for l in unique_in_volume if l != 0 and l not in global_id_to_name]
if missing:
    print(f"[WARN] {len(missing)} global ids in volume have no name; first few: {missing[:10]}")
else:
    named = sum(1 for l in unique_in_volume if l != 0)
    print(f"All {named} non-background global ids have resolved names.")

CSV_HEADER = [
    "case_id",
    "atlas_region_name",
    "tumor_tag",
    "voxel_count",
    "count_scaled",
]


def build_matrix_rows(case_id: str, pred_mni_data: np.ndarray) -> list[dict]:
    """Build per-case rows (region × tumor_tag) with count_scaled 0–256."""
    rows = []

    for brats_label, tumor_tag in LABEL_TAG.items():
        label_mask = pred_mni_data == brats_label
        if not np.any(label_mask):
            continue

        ids_at_label = atlas_data[label_mask]
        unique_ids, counts = np.unique(ids_at_label, return_counts=True)

        for gid, count in zip(unique_ids, counts):
            gid = int(gid)
            region_name = global_id_to_name.get(
                gid, "WM/unassigned" if gid == 0 else f"region_{gid}"
            )
            rows.append({
                "case_id":           case_id,
                "atlas_region_name": region_name,
                "tumor_tag":         tumor_tag,
                "voxel_count":       int(count),
                "count_scaled":      None,
            })

    if rows:
        max_count = max(r["voxel_count"] for r in rows)
        for r in rows:
            r["count_scaled"] = int(round(r["voxel_count"] / max_count * 256))

    return rows


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nProcessing {len(val_list)} cases × {len(PRED_DIRS)} models …\n")

all_rows: dict[str, list[dict]] = {m: [] for m in PRED_DIRS}

skipped_no_mapping = 0
skipped_missing_pred = 0

for idx, sample in enumerate(tqdm(val_list, desc="Cases")):

    # Prediction file on disk is still pred_XXXX (produced by segmentation pipeline)
    pred_name = f"pred_{idx:04d}.nii.gz"

    # The CSV case_id must be the BraTS20 folder name so training can find reports
    brats23_id = sample.get("patient_id") if isinstance(sample, dict) else None
    brats20_id = brats23_to_brats20.get(brats23_id) if brats23_id else None

    if not brats20_id:
        skipped_no_mapping += 1
        continue

    case_id = brats20_id

    for model, pred_dir in PRED_DIRS.items():

        pred_path = os.path.join(pred_dir, pred_name)
        if not os.path.exists(pred_path):
            tqdm.write(f"[WARN] missing: {pred_path}")
            skipped_missing_pred += 1
            continue

        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

        # Rotate 180° in axial plane → MNI152 orientation
        pred_rot = np.rot90(pred_data, k=2)
        pred_mni = image.resample_to_img(
            nib.Nifti1Image(pred_rot.astype(np.float32), affine=mni_template.affine),
            mni_template,
            interpolation='nearest',
        )
        pred_mni_data = np.round(pred_mni.get_fdata()).astype(np.uint8)

        rows = build_matrix_rows(case_id, pred_mni_data)

        # Per-case CSV uses the BraTS20 id as filename (easier to cross-ref with TextBraTSData)
        csv_path = os.path.join(ATLAS_OUT, model, "matrices", f"{case_id}.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(rows)

        all_rows[model].extend(rows)

# ── Write aggregated all_cases.csv per model ──────────────────────────────────
for model, rows in all_rows.items():
    agg_path = os.path.join(ATLAS_OUT, model, "all_cases.csv")
    with open(agg_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    unique_cases = len({r["case_id"] for r in rows})
    print(f"  {model}: {len(rows)} rows  |  {unique_cases} cases  → {agg_path}")

print(f"\nSkipped (no BraTS20 mapping): {skipped_no_mapping}")
if skipped_missing_pred:
    print(f"Skipped (prediction file missing): {skipped_missing_pred}")

print("\nDone.")
print(
    "\n  CSV columns:"
    "\n    case_id            BraTS20_Training_XXX (matches TextBraTSData folder)"
    "\n    atlas_region_name  Julich-Brain region name (WM/unassigned = background)"
    "\n    tumor_tag          TC | WT | ET"
    "\n    voxel_count        raw voxel count in that region×label cell"
    "\n    count_scaled       0–256 (256 = busiest region×label pair in that case)"
)