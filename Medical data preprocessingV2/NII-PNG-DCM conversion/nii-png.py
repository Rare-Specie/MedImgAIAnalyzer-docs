#!/user/bin/env python3
# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

def nii_to_png_single(nii_file_path, out_root_dir, slice_axis=2):
    if not os.path.exists(nii_file_path):
        raise FileNotFoundError(f"NII文件不存在：{nii_file_path}")
    if not (nii_file_path.endswith('.nii') or nii_file_path.endswith('.nii.gz')):
        raise ValueError("输入文件必须是.nii或.nii.gz格式")

    nii_filename = os.path.basename(nii_file_path)
    base_name = os.path.splitext(nii_filename)[0]
    if base_name.endswith('.nii'):  # 处理.nii.gz双重后缀（如"1.nii.gz"→"1"）
        base_name = os.path.splitext(base_name)[0]


    out_dir = os.path.join(out_root_dir, f"{base_name}_png")
    os.makedirs(out_dir, exist_ok=True)


    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()
    slice_num = nii_data.shape[slice_axis]  # 切片数量


    for i in tqdm(range(slice_num), desc=f"转换 {nii_filename}"):
        if slice_axis == 0:
            slice_data = nii_data[i, :, :]
        elif slice_axis == 1:
            slice_data = nii_data[:, i, :]
        else:
            slice_data = nii_data[:, :, i]

        # 归一化到0-255（适配PNG显示）
        data_min = np.min(slice_data)
        data_max = np.max(slice_data)
        if data_max > data_min:
            slice_data = (slice_data - data_min) / (data_max - data_min) * 255
        slice_data = slice_data.astype(np.uint8)

        # 按要求命名PNG：png{base_name}_{index:03d}.png（如"png1_000.png"）
        png_filename = f"png_nii{i + 1:03d}.png"
        png_save_path = os.path.join(out_dir, png_filename)
        cv2.imwrite(png_save_path, slice_data)

    print(f"输入NII：{nii_file_path}")
    print(f"输出PNG文件夹：{out_dir}")
    print(f"PNG文件：png_nii001.png ~ png_nii{slice_num:03d}.png（共{slice_num}张）")


def nii_to_png_batch(nifti_dir, out_root_dir, slice_axis=2):
    # 筛选有效NII文件
    nifti_files = sorted(glob(os.path.join(nifti_dir, "*.nii")) + glob(os.path.join(nifti_dir, "*.nii.gz")))
    if len(nifti_files) == 0:
        raise ValueError(f"NII文件夹中无有效文件：{nifti_dir}")

    # 批量处理每个NII文件
    for idx, nii_file in enumerate(nifti_files):
        print(f"\n处理第 {idx + 1}/{len(nifti_files)} 个文件")
        try:
            nii_to_png_single(nii_file, out_root_dir, slice_axis)
        except Exception as e:
            print(f" 处理失败 {os.path.basename(nii_file)}：{str(e)}")

    # 输出批量处理结果
    print(f"输入NII文件夹：{nifti_dir}")
    print(f"输出PNG根目录：{out_root_dir}")


#下面的路径需要自己去调整！！！
if __name__ == "__main__":
    convert_mode=("single")#single单个NII，batch一系列NII
    OUT_ROOT_DIR="G:/mry1/TOM500/data preprocess/png"#输出路径
    SLICE_AXIS=2

    if convert_mode=="single":
       input_nii = "G:/mry1/TOM500/data preprocess/mask2/39.nii"  # 输入.nii文件路径
       nii_to_png_single(input_nii,OUT_ROOT_DIR,SLICE_AXIS)

    elif convert_mode=="batch":
        input_nii_dir="G:/mry1/TOM500/data preprocess/mask2"#输入.nii路径
        nii_to_png_batch(input_nii_dir,OUT_ROOT_DIR,SLICE_AXIS)