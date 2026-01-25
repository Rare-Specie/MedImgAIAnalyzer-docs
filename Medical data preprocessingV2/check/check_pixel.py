#!/user/bin/env python3
# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
from typing import Tuple
import os


def get_nii_spatial_info(nii_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    if not (nii_file_path.endswith('.nii') or nii_file_path.endswith('.nii.gz')):
        raise ValueError("文件必须是.nii或.nii.gz格式")
    if not os.path.exists(nii_file_path):
        raise FileNotFoundError(f"NII文件不存在：{nii_file_path}")

    # 尝试加载文件，判断是否损坏
    try:
        nii_img = nib.load(nii_file_path)
    except Exception as e:
        raise RuntimeError(f"NII文件损坏或无法解析：{str(e)}")


    affine = nii_img.affine  # 4x4 affine矩阵：存储方向、间距、原点信息


    pixel_spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])


    image_origin = affine[:3, 3]


    x_dir = affine[:3, 0]  # x轴方向向量
    y_dir = affine[:3, 1]  # y轴方向向量
    # 归一化方向向量（确保长度为1，符合DICOM标准）
    x_dir = x_dir / np.linalg.norm(x_dir) if np.linalg.norm(x_dir) != 0 else x_dir
    y_dir = y_dir / np.linalg.norm(y_dir) if np.linalg.norm(y_dir) != 0 else y_dir
    image_orientation = np.concatenate([x_dir, y_dir])  # 拼接为6元素

    return pixel_spacing, image_origin, image_orientation

def print_spatial_info(
        pixel_spacing: np.ndarray,
        image_origin: np.ndarray,
        image_orientation: np.ndarray,
        nii_file_path: str
) -> None:
    """格式化输出空间信息（便于阅读）"""
    print("=" * 60)
    print(f"NII文件空间信息提取结果：{nii_file_path}")
    print("=" * 60)
    print(f"像素间距：")
    print(f"- X轴：{pixel_spacing[0]:.3f} mm/像素")
    print(f"- Y轴：{pixel_spacing[1]:.3f} mm/像素")
    print(f"- Z轴：{pixel_spacing[2]:.3f} mm/像素（切片厚度）")
    print("-" * 60)
    print(f"图像原点：")
    print(f"- X轴坐标：{image_origin[0]:.3f} mm")
    print(f"- Y轴坐标：{image_origin[1]:.3f} mm")
    print(f"- Z轴坐标：{image_origin[2]:.3f} mm")
    print("-" * 60)
    print(f"图像方向矩阵：")
    print(f"格式：[x1,y1,z1, x2,y2,z2]（x轴方向 + y轴方向）")
    print(f"数值：{np.round(image_orientation, 6)}")
    print("=" * 60)

#下面的路径需要自己去调整！！！
if __name__ == "__main__":
    NII_file_path = "G:/mry1/TOM500/data preprocess/mask1/26.nii" # 输入NII文件路径

    try:
        # 提取空间信息
        pixel_spacing, image_origin, image_orientation = get_nii_spatial_info(NII_file_path)
        # 格式化输出
        print_spatial_info(pixel_spacing, image_origin, image_orientation, NII_file_path)

    except Exception as e:
        print(f"\n提取失败：{str(e)}")