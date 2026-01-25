#!/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib


def get_nii_spatial_info(nii_path: str):

    img = nib.load(nii_path)
    affine = img.affine

    # ---- Extract voxel spacing from affine ----
    # Spacing is the norm of each axis vector
    spacing_x = np.linalg.norm(affine[0:3, 0])
    spacing_y = np.linalg.norm(affine[0:3, 1])
    spacing_z = np.linalg.norm(affine[0:3, 2])

    # ---- Extract orientation cosines ----
    row_cosines = affine[0:3, 0] / spacing_x if spacing_x != 0 else np.array([1, 0, 0])
    col_cosines = affine[0:3, 1] / spacing_y if spacing_y != 0 else np.array([0, 1, 0])
    slice_cosines = affine[0:3, 2] / spacing_z if spacing_z != 0 else np.array([0, 0, 1])

    # ---- Image Position Patient (origin) ----
    origin = affine[0:3, 3]

    # ---- Report in DICOM-style format ----
    info = {
        "PixelSpacing": [float(spacing_y), float(spacing_x)],   # row, column
        "SliceThickness": float(spacing_z),
        "ImageOrientationPatient": [
            float(row_cosines[0]), float(row_cosines[1]), float(row_cosines[2]),
            float(col_cosines[0]), float(col_cosines[1]), float(col_cosines[2])
        ],
        "ImagePositionPatient": [
            float(origin[0]), float(origin[1]), float(origin[2])
        ],
        "VolumeShape": img.shape
    }

    return info


# ---------------- Example usage ----------------
if __name__ == "__main__":
    nii_file = "G:/mry1/TOM500/data preprocess/mask2/29.nii"
    spatial_info = get_nii_spatial_info(nii_file)

    print("=== NII 文件详细信息===")
    for k, v in spatial_info.items():
        print(f"{k}: {v}")
