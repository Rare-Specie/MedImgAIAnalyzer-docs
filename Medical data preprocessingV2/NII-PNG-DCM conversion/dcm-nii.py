#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pydicom
import nibabel as nib

def ensure_dir(path: str):#创建文件夹
    if not os.path.exists(path):
        os.makedirs(path)

def load_original_nii_geometry(original_nii_path: str):
    #从原始NII文件中加载信息
    nii_img = nib.load(original_nii_path)
    affine = nii_img.affine
    return affine

def load_dicom_series_pixels(dcm_folder: str):
    files = [
        os.path.join(dcm_folder, f)
        for f in os.listdir(dcm_folder)
        if f.lower().endswith(".dcm")
    ]

    if not files:
        raise ValueError(f"没有在{dcm_folder}找到DICOM文件")

    datasets = [pydicom.dcmread(f) for f in files]


    datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))


    slices = [ds.pixel_array.astype(np.int16) for ds in datasets]
    volume = np.stack(slices, axis=-1)

    return volume


def convert_dcm_to_nii_with_original_geometry(
    dcm_folder: str,
    original_nii_path: str,
    output_root: str = None
):

 #26_dcm → nii_dcm26.nii

    folder_name = os.path.basename(dcm_folder.rstrip("/"))

    if not folder_name.endswith("_dcm"):
        raise ValueError("必须以'_dcm'为结尾")

    series_id = folder_name.replace("_dcm", "")

    output_root = output_root if output_root else "."
    ensure_dir(output_root)

    #命名规则！
    output_name = f"nii_dcm{series_id}.nii"
    output_path = os.path.join(output_root, output_name)

    affine = load_original_nii_geometry(original_nii_path)

    volume = load_dicom_series_pixels(dcm_folder)

    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, output_path)

    print(f"DCM成功转换为NII!\n"
          f"输入文件地址：{dcm_folder}\n"
          f"输出文件地址：{output_path}\n")
    print(f"原始图像信息: {original_nii_path}")
    print(f"图像尺寸大小: {volume.shape}")


def batch_convert_dcm_to_nii(
    dcm_root: str,
    original_nii_root: str,
    output_root: str = None
):

    folders = [
        os.path.join(dcm_root, f)
        for f in os.listdir(dcm_root)
        if f.endswith("_dcm") and os.path.isdir(os.path.join(dcm_root, f))
    ]

    for folder in folders:
        folder_name = os.path.basename(folder)
        series_id = folder_name.replace("_dcm", "")

        #与原来NII文件相对应
        original_nii_path = os.path.join(original_nii_root, f"{series_id}.nii")

        if not os.path.exists(original_nii_path):
            print(f"[警告] 丢失原来NII文件{original_nii_path}")
            continue

        convert_dcm_to_nii_with_original_geometry(
            dcm_folder=folder,
            original_nii_path=original_nii_path,
            output_root=output_root
        )

    print("转换成功!")
if __name__ == "__main__":
    #下面需要根据自己的路径去修改！
    #当前是单一的DCM文件转NII文件
    convert_dcm_to_nii_with_original_geometry(
        dcm_folder="G:/mry1/TOM500/data preprocess/dicom/1_dcm",
        original_nii_path="G:/mry1/TOM500/data preprocess/mask1/1.nii",
        output_root="G:/mry1/TOM500/data preprocess/niioutput/"
    )
    #下面是多个DCM文件转NII文件
    #batch_convert_dcm_to_nii(
    #dcm_root="G:/mry1/TOM500/data preprocess/dicom",
    #output_root="G:/mry1/TOM500/data preprocess/niioutput"
    #)
    pass
