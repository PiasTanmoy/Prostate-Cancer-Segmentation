
import sys
from pathlib import Path

import nibabel as nib          # pip install nibabel
import numpy as np
import pandas as pd


PIXEL_VALUES = list(range(6))   # 0,1,2,3,4,5

mask_dir = "input/picai_labels/csPCa_lesion_delineations/human_expert/resampled"
mask_dir = Path(mask_dir)
if not mask_dir.exists():
    raise FileNotFoundError(f"{mask_dir} does not exist")

rows = []
for mask_path in sorted(mask_dir.glob("*.nii.gz")):
    vol = nib.load(str(mask_path)).get_fdata()
    # cast to int in case the mask is float‐encoded
    counts = np.bincount(vol.astype(int).ravel(), minlength=6)[:6]

    rows.append(
        {"mask": mask_path.stem, **{v: int(counts[v]) for v in PIXEL_VALUES}}
    )
    print(f"Processed {mask_path.stem} with counts: {counts}")

df = pd.DataFrame(rows, columns=["mask"] + PIXEL_VALUES)
df.to_csv("mask_pixel_frequencies_all.csv", index=False)
print("✅ Saved frequencies to mask_pixel_frequencies_all.csv")
