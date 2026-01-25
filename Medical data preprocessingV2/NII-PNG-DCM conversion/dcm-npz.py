#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pydicom


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def dcm_folder_to_npz(dcm_folder: str, output_root: str = None):
    output_root = output_root if output_root else "."
    ensure_dir(output_root)

    folder_name = os.path.basename(dcm_folder.rstrip("/"))
    output_path = os.path.join(output_root, f"{folder_name}.npz")

    files = [
        os.path.join(dcm_folder, f)
        for f in os.listdir(dcm_folder)
        if f.lower().endswith(".dcm")
    ]

    if not files:
        raise ValueError("No DICOM files found")

    datasets = [pydicom.dcmread(f) for f in files]
    datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))

    volume = np.stack([ds.pixel_array for ds in datasets], axis=-1).astype(np.int16)

    ds0 = datasets[0]
    spacing = [
        float(ds0.PixelSpacing[0]),
        float(ds0.PixelSpacing[1]),
        float(getattr(ds0, "SliceThickness", 1.0))
    ]

    affine = np.diag([spacing[1], spacing[0], spacing[2], 1.0])

    np.savez_compressed(
        output_path,
        image=volume,
        affine=affine,
        spacing=np.array(spacing),
        source_type="dcm",
        source_name=folder_name
    )

    print(f"成功转换为NPZ文件！\n"
          f"原始DCM文件路径：{dcm_folder} \n"
          f"输出NPZ文件路径：{output_path}")

#下面为主函数，根据实际路径修改！
dcm_folder_to_npz("G:/mry1/TOM500/data preprocess/dicom/26_dcm",
   output_root="G:/mry1/TOM500/data preprocess/npzoutput")
