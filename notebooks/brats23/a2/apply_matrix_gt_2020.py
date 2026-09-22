"""
apply_atlas_matrix_gt.py
========================
Produces CSV atlas matrices per case using BraTS2020 GROUND-TRUTH
segmentation labels (not model predictions).

This gives coverage of all BraTS20 cases with text reports (~369 cases)
instead of only the ~75 that had BraTS23 predictions with a BraTS20 mapping.

Each CSV row:
    case_id | atlas_region_name | tumor_tag | voxel_count | count_scaled

case_id is the BraTS2020 folder name (e.g. BraTS20_Training_001). Only cases
that have BOTH a _seg.nii(.gz) file AND a text report in TextBraTSData/ are
included — the text model can't learn from cases without a report.

BraTS2020 label → tumor tag mapping
  Note: BraTS2020 uses labels {1, 2, 4}; we also accept {1, 2, 3} in case some
  files were re-labelled. Either way:
    1 (NCR/NET)  → TC   Necrotic Core
    2 (ED)       → WT   Peritumoral Edema
    3 or 4 (ET)  → ET   Enhancing Tumor
    0            → background (skipped)

count_scaled: per case, max voxel_count across (region, label) pairs → 256.
"""

import os
import csv
import glob
import warnings

import numpy as np
import nibabel as nib
from nilearn import datasets, image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── BraTS label → tumor tag ───────────────────────────────────────────────────
# Accepts both BraTS2020 (labels 1/2/4) and BraTS2023 (labels 1/2/3).
LABEL_TAG = {
    1: "TC",   # NCR/NET
    2: "WT",   # Edema
    3: "ET",   # (BraTS23)
    4: "ET",   # (BraTS20)
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# Ground-truth segmentation source (BraTS20 training data)
BRATS20_ROOT = "/home/lexie/Desktop/mastersThesis/brats20/BraTS2020_TrainingData"

# Text reports (used to filter: only keep cases with a matching report)
TEXT_DIR = os.path.join(BASE, "..", "TextBraTSData")

# Output: a new folder for GT matrices, separate from prediction-based ones
ATLAS_OUT = os.path.join(BASE, "atlas_segmentations", "ground_truth")
os.makedirs(os.path.join(ATLAS_OUT, "matrices"), exist_ok=True)

# ── Discover cases ────────────────────────────────────────────────────────────
# A case is usable only if BOTH the seg file and the text report exist.
print(f"Scanning BraTS20 root: {BRATS20_ROOT}")
all_brats20_dirs = sorted(
    d for d in os.listdir(BRATS20_ROOT)
    if d.startswith("BraTS20_") and os.path.isdir(os.path.join(BRATS20_ROOT, d))
)
print(f"  BraTS20 folders found: {len(all_brats20_dirs)}")

def find_seg_file(case_dir: str) -> str | None:
    """Locate the _seg.nii or _seg.nii.gz file inside a case folder."""
    for pat in ("*_seg.nii.gz", "*_seg.nii"):
        hits = glob.glob(os.path.join(case_dir, pat))
        if hits:
            return hits[0]
    return None


def has_text_report(case_id: str) -> bool:
    case_dir = os.path.join(TEXT_DIR, case_id)
    if not os.path.isdir(case_dir):
        return False
    txt = os.path.join(case_dir, f"{case_id}_flair_text.txt")
    npy = os.path.join(case_dir, f"{case_id}_flair_text.npy")
    return os.path.exists(txt) or os.path.exists(npy)


usable_cases: list[tuple[str, str]] = []   # (case_id, seg_path)
skipped_no_seg = 0
skipped_no_report = 0
for case_id in all_brats20_dirs:
    case_dir = os.path.join(BRATS20_ROOT, case_id)
    seg_path = find_seg_file(case_dir)
    if seg_path is None:
        skipped_no_seg += 1
        continue
    if not has_text_report(case_id):
        skipped_no_report += 1
        continue
    usable_cases.append((case_id, seg_path))

print(f"  Usable cases (seg + report): {len(usable_cases)}")
print(f"  Skipped (no seg file)       : {skipped_no_seg}")
print(f"  Skipped (no text report)    : {skipped_no_report}")

if not usable_cases:
    raise RuntimeError("No usable cases found. Check BRATS20_ROOT and TEXT_DIR paths.")

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

# ── Build per-fragment (label, fragment) → region name lookup ────────────────
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
print(f"  (fragment, label) pairs resolved: {len(frag_label_to_name)}")

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

# ── Row builder ───────────────────────────────────────────────────────────────
CSV_HEADER = [
    "case_id",
    "atlas_region_name",
    "tumor_tag",
    "voxel_count",
    "count_scaled",
]


def build_matrix_rows(case_id: str, seg_mni_data: np.ndarray) -> list[dict]:
    """Build per-case rows (region × tumor_tag) with count_scaled 0–256."""
    rows = []

    # Merge ET variants (3 and 4 both map to ET) so we don't emit duplicate rows
    # with the same tumor_tag. Group by tumor_tag across all labels.
    tag_masks: dict[str, np.ndarray] = {}
    for brats_label, tumor_tag in LABEL_TAG.items():
        m = seg_mni_data == brats_label
        if m.any():
            tag_masks[tumor_tag] = tag_masks.get(tumor_tag, np.zeros_like(m)) | m

    for tumor_tag, mask in tag_masks.items():
        ids_at_label = atlas_data[mask]
        if ids_at_label.size == 0:
            continue
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
print(f"\nProcessing {len(usable_cases)} cases (ground truth) …\n")

all_rows: list[dict] = []
unexpected_labels_reported = set()
skipped_bad_seg = 0

for case_id, seg_path in tqdm(usable_cases, desc="Cases"):
    try:
        seg_data = nib.load(seg_path).get_fdata().astype(np.int32)
    except Exception as e:
        tqdm.write(f"[WARN] {case_id}: could not load seg ({e})")
        skipped_bad_seg += 1
        continue

    # Sanity: report unexpected labels once
    present = set(int(x) for x in np.unique(seg_data) if int(x) != 0)
    unexpected = present - set(LABEL_TAG.keys())
    if unexpected and not unexpected.issubset(unexpected_labels_reported):
        tqdm.write(f"[INFO] {case_id}: unexpected labels {unexpected} (will be ignored)")
        unexpected_labels_reported |= unexpected

    # Rotate 180° in axial plane → MNI152 orientation (same convention as before)
    seg_rot = np.rot90(seg_data, k=2)
    seg_mni = image.resample_to_img(
        nib.Nifti1Image(seg_rot.astype(np.float32), affine=mni_template.affine),
        mni_template,
        interpolation='nearest',
    )
    seg_mni_data = np.round(seg_mni.get_fdata()).astype(np.int32)

    rows = build_matrix_rows(case_id, seg_mni_data)

    # Per-case CSV
    csv_path = os.path.join(ATLAS_OUT, "matrices", f"{case_id}.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    all_rows.extend(rows)

# ── Aggregated CSV ────────────────────────────────────────────────────────────
agg_path = os.path.join(ATLAS_OUT, "all_cases.csv")
with open(agg_path, "w", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=CSV_HEADER)
    writer.writeheader()
    writer.writerows(all_rows)

unique_cases = len({r["case_id"] for r in all_rows})
print(f"\nGround truth: {len(all_rows)} rows  |  {unique_cases} cases  → {agg_path}")
if skipped_bad_seg:
    print(f"Skipped (load failed): {skipped_bad_seg}")

print("\nDone.")
print(
    "\n  Output layout:"
    f"\n    {ATLAS_OUT}/"
    f"\n      matrices/<case_id>.csv    per-case atlas matrices"
    f"\n      all_cases.csv             combined (point train script at this)"
    "\n  CSV columns:"
    "\n    case_id            BraTS20_Training_XXX (matches TextBraTSData folder)"
    "\n    atlas_region_name  Julich-Brain region name (WM/unassigned = background)"
    "\n    tumor_tag          TC | WT | ET"
    "\n    voxel_count        raw voxel count in that region×label cell"
    "\n    count_scaled       0–256 (256 = busiest region×label pair in that case)"
)