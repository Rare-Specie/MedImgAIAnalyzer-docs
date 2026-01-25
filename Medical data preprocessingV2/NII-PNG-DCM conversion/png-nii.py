#!/user/bin/env python3
# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import cv2
import os
from glob import glob
import re
from typing import Optional,Tuple
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def extract_number_from_folder(folder_name: str) -> str:
    # 用正则表达式匹配文件夹名中的所有数字
    numbers = re.findall(r'\d+', folder_name)
    if not numbers:
        raise ValueError(f"无法从文件夹名{folder_name}中提取数字，请检查命名格式")
    # 返回第一个匹配到的数字（默认文件夹名中核心数字唯一）
    return numbers[0]

def png_to_nii(
        png_dir: str,
        output_dir: Optional[str] = None,
        output_nii_path: Optional[str] = None,
        original_nii_path: Optional[str] = None,
        slice_axis: int = 2,
        pixel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # 原始空间分辨率（x,y,z轴，单位mm/像素）
        image_orientation: Tuple[float, ...] = (1, 0, 0, 0, 1, 0)  # DICOM标准方向矩阵（默认横断位）
) -> None:
    if not os.path.exists(png_dir):
        raise FileNotFoundError(f"PNG文件夹不存在：{png_dir}")

    png_folder_name = os.path.basename(png_dir)
    core_number = extract_number_from_folder(png_folder_name)
    source_format = "png"  # 固定转换前格式（根据文件夹名后缀_png确定）
    #格式输出主要在这里修改
    auto_nii_name = f"nii_{source_format}{core_number}.nii"  # 自动生成NII文件名
    print(f"生成NII文件名：{auto_nii_name}")

    if output_nii_path is not None:
        # 完全自定义路径（确保后缀正确）
        if not (output_nii_path.endswith('.nii') or output_nii_path.endswith('.nii.gz')):
            output_nii_path += '.nii'
        final_output_path = output_nii_path
        print(f"使用自定义NII路径：{final_output_path}")
    else:
        # 指定输出文件夹（新增功能）
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"创建NII输出文件夹：{output_dir}")
            final_output_path = os.path.join(output_dir, auto_nii_name)
            print(f"使用指定输出文件夹：{final_output_path}")
        else:
            # 默认与PNG文件夹同级
            png_parent_dir = os.path.dirname(png_dir)
            final_output_path = os.path.join(png_parent_dir, auto_nii_name)
            print(f"使用默认输出路径（与PNG同级）：{final_output_path}")

    png_files = sorted(glob(os.path.join(png_dir, "*.png")), key=natural_sort_key)
    if len(png_files) == 0:
        raise ValueError(f"PNG文件夹中未找到任何.png文件：{png_dir}")
    print(f"找到 {len(png_files)} 张PNG图片（不检测内部文件名规范）")

    first_png = cv2.imread(png_files[0], cv2.IMREAD_GRAYSCALE)
    if first_png is None:
        raise RuntimeError(f"无法读取PNG文件：{png_files[0]}")
    height, width = first_png.shape
    slice_count = len(png_files)
    print(f"读取PNG信息：尺寸{width}×{height}，切片数{slice_count}")

    nii_data = np.zeros(( width,height, slice_count), dtype=np.float32)

    for idx, png_file in enumerate(png_files):
        png_data = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        if png_data.shape != (height, width):
            raise ValueError(
                f"PNG尺寸不一致：{png_file}（应为{height}x{width}，实际为{png_data.shape[0]}x{png_data.shape[1]}）")
        if slice_axis == 0:  # x轴堆叠
            nii_data[idx, :, :] = np.transpose(png_data, (1, 0))
        elif slice_axis == 1:  # y轴堆叠
            nii_data[:, idx, :] = np.transpose(png_data, (1, 0))
        else:  # z轴堆叠（默认）
            nii_data[:, :, idx] = np.transpose(png_data, (1, 0))

    if original_nii_path and os.path.exists(original_nii_path):
        affine = nib.load(original_nii_path).affine
        print(f" 已加载原始NII的affine空间矩阵")
    else:
        affine = np.eye(4)
        # 设置像素间距（x,y,z轴分辨率，对应空间尺度）
        affine[0, 0] = pixel_spacing[0]  # x轴像素间距（mm/像素）
        affine[1, 1] = pixel_spacing[1]  # y轴像素间距（mm/像素）
        affine[2, 2] = pixel_spacing[2]  # z轴像素间距（mm/像素，即切片厚度）

        # 设置图像方向（DICOM标准：前6个元素定义x/y轴方向，避免空间姿态错误）
        affine[0:3, 0] = image_orientation[0:3]  # x轴方向向量
        affine[0:3, 1] = image_orientation[3:6]  # y轴方向向量
        affine[0:3, 2] = np.cross(  # z轴方向向量（x×y）
            image_orientation[0:3], image_orientation[3:6]
        )
        print(f" 像素间距：{pixel_spacing}mm，方向矩阵：{image_orientation}")

    # 创建NII图像并保存
    nii_image = nib.Nifti1Image(nii_data, affine=affine)
    nib.save(nii_image, final_output_path)
    print(f"\n PNG已转换成NII！")
    print(f"输入PNG：{png_dir}（共{slice_count}张PNG）")
    print(f"输出NII：{final_output_path}")
    print(f"NII维度：{nii_data.shape}（x×y×z）")
    print(f"NII像素间距：{pixel_spacing}mm")
    print(f"NII方向矩阵：{image_orientation}")


#下面的路径需要自己去调整！！！
if __name__ == "__main__":
    # 关键配置！
    png_dir = "G:/mry1/TOM500/data preprocess/png/26_png"  # 输入PNG文件夹路径
    output_dir = "G:/mry1/TOM500/data preprocess/niioutput/"  #指定NII输出文件夹
    # output_nii_path = "G:/mry1/TOM500/data preprocess/niioutput/nii_png39.nii"
    original_nii_path="G:/mry1/TOM500/data preprocess/mask2/39.nii"  # 若有原始NII，可填写路径
    pixel_spacing = (0.312, 0.009, 0.111)  # 根据实际图像分辨率调整（如CT常用0.5mm）
    # CT图像（0.625, 0.625, 1.0），MRI图像（1.0, 1.0, 2.0）
    # 横断位：(1,0,0, 0,1,0)
    # 冠状位：(1,0,0, 0,0,-1)
    # 矢状位：(0,1,0, 0,0,-1)
    try:
        png_to_nii(
            png_dir=png_dir,
            output_dir=output_dir,  # 启用指定输出文件夹
            # output_nii_path=output_nii_path,  # 若需完全自定义路径，取消注释
            original_nii_path=original_nii_path,
            slice_axis=2,
            pixel_spacing=pixel_spacing
        )
    except Exception as e:
        print(f"\n 转换失败：{str(e)}")