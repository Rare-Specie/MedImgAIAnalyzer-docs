#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def nii_to_npz(nii_path: str, output_root: str = None):
    output_root = output_root if output_root else "."
    ensure_dir(output_root)

    name = os.path.splitext(os.path.basename(nii_path))[0]
    output_path = os.path.join(output_root, f"{name}.npz")

    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # voxel spacing
    spacing = (
        np.linalg.norm(affine[:3, 0]),
        np.linalg.norm(affine[:3, 1]),
        np.linalg.norm(affine[:3, 2])
    )

    np.savez_compressed(
        output_path,
        image=data,
        affine=affine,
        spacing=np.array(spacing),
        source_type="nii",
        source_name=os.path.basename(nii_path)
    )
    print(f"成功转换为NPZ文件！\n"
          f"原始NII文件路径：{nii_path} \n"
          f"输出NPZ文件路径：{output_path}")

nii_to_npz("G:/mry1/TOM500/data preprocess/mask1/1.nii",
    output_root="G:/mry1/TOM500/data preprocess/npzoutput")