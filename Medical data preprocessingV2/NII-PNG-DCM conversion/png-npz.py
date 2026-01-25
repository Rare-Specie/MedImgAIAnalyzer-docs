#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import cv2
import pydicom
from glob import glob
from typing import Dict,Optional

def read_single_file(file_path: str) -> np.ndarray:
    # 读取NII文件（.nii/.nii.gz）
    if file_path.endswith(('.nii', '.nii.gz')):
        img = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(img)  # 格式：(z,y,x)（3D）

    # 读取DICOM文件（单个.dcm）
    elif file_path.lower().endswith('.dcm'):
        dicom_data = pydicom.dcmread(file_path)
        return dicom_data.pixel_array  # 格式：(y,x)（2D）或( z,y,x)（3D）

    # 读取PNG/JPG图像（2D）
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 灰度图读取
        return img  # 格式：(y,x)（2D）

    else:
        raise ValueError(f"不支持的文件格式：{os.path.basename(file_path)}")


def single_file_convert(
        input_path: str,
        output_path: Optional[str] = None
) -> None:
    # 自动生成输出路径（同目录+原文件名.npz）
    if output_path is None:
        file_dir = os.path.dirname(input_path)  # 原文件所在目录（输出目录）
        file_name = os.path.splitext(os.path.basename(input_path))[0]  # 原文件名（不含后缀）
        output_path = os.path.join(file_dir, file_name)  # 保留原文件名，后续自动加.npz

    # 读取+保存
    data = read_single_file(input_path)
    save_to_npz(data, output_path)


def batch_file_convert(
        input_dir: str,
        target_formats: tuple = ('.nii', '.nii.gz', '.dcm', '.png', '.jpg')
) -> None:
    total_files = 0
    success_count = 0

    # 遍历所有目标格式文件
    for fmt in target_formats:
        files = glob(os.path.join(input_dir, f'**/*{fmt}'), recursive=True)
        for file in files:
            total_files += 1
            # 自动生成输出路径（同目录+原文件名.npz）
            file_dir = os.path.dirname(file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            output_path = os.path.join(file_dir, file_name)

            try:
                data = read_single_file(file)
                save_to_npz(data, output_path)
                success_count += 1
            except Exception as e:
                print(f"抱歉，处理失败")

    # 输出统计信息
    print(f"\n 批量处理完成："
          f"\n共找到{total_files}个文件"
          f"\n成功转换{success_count}个")

def save_to_npz(
        data: np.ndarray,
        output_path: str,
        compress_level: int = 3  # 压缩级别（0-9，越高压缩率越高）
) -> None:
    # 确保输出路径后缀正确
    if not output_path.endswith('.npz'):
        output_path += '.npz'


    np.savez_compressed(output_path, data=data, compresslevel=compress_level)
    print(f"输出NPZ文件路径：{output_path}")

if __name__ == "__main__":
    # 下面为主函数，根据实际路径修改！
    # single（单个文件）/ batch（批量文件）
    mode = "single"  # 建议为single
    # 单个文件模式配置
    single_input = "G:\mry1\TOM500\data preprocess\png/1_png/png_dcm008.png"  # 输入文件路径
    single_output = "G:\mry1\TOM500\data preprocess/npzoutput/1_png"  # 手动指定输出路径（None=自动生成：同目录+原文件名.npz）

    # 批量文件模式配置
    batch_input = "G:\mry1\TOM500\data preprocess\png/1_png"
    batch_output= None   #输出至存储PNG或JPG的输入文件夹

    try:
        if mode == "single":
            print("开始单个文件转换...")
            single_file_convert(single_input, single_output)

        elif mode == "batch":
            print("开始批量文件转换...")
            batch_file_convert(batch_input,batch_output)

        print("成功转换为NPZ文件！\n")

        # 可选：验证第一个生成的NPZ（以单个文件为例）
        if mode == "single" and single_output is None:
            test_npz_path = os.path.join(os.path.dirname(single_input),
                                         f"{os.path.splitext(os.path.basename(single_input))[0]}.npz")
            if os.path.exists(test_npz_path):
                test_data = np.load(test_npz_path)
                print(f"NPZ文件包含数组维度：{test_data['data'].shape}")

    except Exception as e:
        print(f"抱歉，转换失败")
