#!/user/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import nibabel as nib
import os
from glob import glob
from typing import Optional, Tuple
def crop_medical_image(
        img: np.ndarray,
        crop_ratio: float = 0.8,
        is_random: bool = True,
        keep_original_size: bool = True  # 是否保持原始尺寸
) -> np.ndarray:

    height, width = img.shape[:2]
    original_size = (width, height)  # 记录原始尺寸

    crop_h = int(height * crop_ratio)
    crop_w = int(width * crop_ratio)


    crop_h = max(crop_h, int(height * 0.5))
    crop_w = max(crop_w, int(width * 0.5))

    if is_random:
        x_start = np.random.randint(0, width - crop_w + 1)
        y_start = np.random.randint(0, height - crop_h + 1)
    else:
        x_start = (width - crop_w) // 2
        y_start = (height - crop_h) // 2


    cropped_img = img[y_start:y_start + crop_h, x_start:x_start + crop_w]
    # 裁剪后resize回原始尺寸
    if keep_original_size and (cropped_img.shape[0] != height or cropped_img.shape[1] != width):
        cropped_img = cv2.resize(cropped_img, original_size, interpolation=cv2.INTER_LINEAR)
    return cropped_img


def rotate_medical_image(
        img: np.ndarray,
        angle: Optional[float] = None,
        angle_range: Tuple[float, float] = (-30, 30),
        keep_size: bool = True,
        keep_original_size: bool = True  # 强制保持原始输入尺寸
) -> np.ndarray:

    height, width = img.shape[:2]
    original_size = (width, height)  # 记录原始尺寸（width, height）
    if angle is None:
        angle = np.random.uniform(angle_range[0], angle_range[1])


    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    if keep_size:

        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    else:

        cos_theta = abs(rotation_matrix[0, 0])
        sin_theta = abs(rotation_matrix[0, 1])
        new_width = int(height * sin_theta + width * cos_theta)
        new_height = int(height * cos_theta + width * sin_theta)

        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    if keep_original_size and (rotated_img.shape[0] != height or rotated_img.shape[1] != width):
        rotated_img = cv2.resize(rotated_img, original_size, interpolation=cv2.INTER_LINEAR)
    return rotated_img


def augment_single_image(
        img_path: str,
        output_path: str,
        crop_ratio: float = 0.8,
        is_random_crop: bool = True,
        rotate_angle: Optional[float] = None,
        angle_range: Tuple[float, float] = (-30, 30),
        keep_size: bool = True,
        is_nii: bool = False,
        slice_axis: int = 2
) -> None:

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"输入文件不存在：{img_path}")


    if is_nii:

        nii_img = nib.load(img_path)
        nii_data = nii_img.get_fdata()
        slice_idx = nii_data.shape[slice_axis] // 2

        if slice_axis == 0:
            img = nii_data[slice_idx, :, :]
        elif slice_axis == 1:
            img = nii_data[:, slice_idx, :]
        else:
            img = nii_data[:, :, slice_idx]
        original_slice_size = img.shape  # 记录原始切片尺寸（height, width）
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)
    else:

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取PNG文件：{img_path}")
        original_slice_size = img.shape  # 记录原始切片尺寸（height, width）

    augmented_img = crop_medical_image(
        img, crop_ratio, is_random_crop, keep_original_size=True  # 强制保持原始尺寸
    )
    augmented_img = rotate_medical_image(
        augmented_img, rotate_angle, angle_range, keep_size, keep_original_size=True  # 强制保持原始尺寸
    )

    if is_nii:
        if augmented_img.shape != original_slice_size:
            augmented_img = cv2.resize(augmented_img, (original_slice_size[1], original_slice_size[0]),
                                       interpolation=cv2.INTER_LINEAR)
        nii_data_aug = np.zeros_like(nii_data)
        if slice_axis == 0:
            nii_data_aug[slice_idx, :, :] = augmented_img
        elif slice_axis == 1:
            nii_data_aug[:, slice_idx, :] = augmented_img
        else:
            nii_data_aug[:, :, slice_idx] = augmented_img

        nii_aug_img = nib.Nifti1Image(nii_data_aug, nii_img.affine)
        nib.save(nii_aug_img, output_path)
    else:

        cv2.imwrite(output_path, augmented_img)

    print(f" 增强完成：{img_path} → {output_path},尺寸：{augmented_img.shape}")


def batch_augment_images(
        input_dir: str,
        output_dir: str,
        crop_ratio: float = 0.8,
        is_random_crop: bool = True,
        rotate_angle: Optional[float] = None,
        angle_range: Tuple[float, float] = (-30, 30),
        keep_size: bool = True,
        is_nii: bool = False,
        slice_axis: int = 2
) -> None:


    os.makedirs(output_dir, exist_ok=True)


    if is_nii:
        file_paths = sorted(glob(os.path.join(input_dir, "*.nii")) + glob(os.path.join(input_dir, "*.nii.gz")))
    else:
        file_paths = sorted(glob(os.path.join(input_dir, "*.png")))

    if len(file_paths) == 0:
        raise ValueError(f"{input_dir}中未找到{'NII' if is_nii else 'PNG'}文件")


    for idx, file_path in enumerate(file_paths):

        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        if ext == ".gz":  # 处理.nii.gz格式
            name = os.path.splitext(name)[0]
            output_filename = f"{name}_aug.nii.gz"
        else:
            output_filename = f"{name}_aug{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # 执行增强
        try:
            augment_single_image(
                file_path, output_path, crop_ratio, is_random_crop,
                rotate_angle, angle_range, keep_size, is_nii, slice_axis
            )
        except Exception as e:
            print(f" 处理失败 {file_path}：{str(e)}")

    print(f"\n 批量增强完成！共处理 {len(file_paths)} 个文件，输出至：{output_dir}")


#下面的板块需要自己去调整！！！
if __name__ == "__main__":
    INPUT_DIR = "G:/mry1/TOM500/data preprocess/mask2"  # 输入文件夹（PNG或NII文件）路径
    OUTPUT_DIR = "G:/mry1/TOM500/data preprocess/output/enhance"  # 输出文件夹路径（没有会自动创建）
    IS_NII = True  # TRUE处理NII文件,FALSE不处理NII文件
    CROP_RATIO = 0.9  # 默认裁剪比例为0.8
    IS_RANDOM_CROP = True  # TRUE随机裁剪,False中心裁剪
    ROTATE_ANGLE = 10  # 固定旋转角度,默认±30度

    # 下面为批量增强
    try:
        batch_augment_images(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            crop_ratio=CROP_RATIO,
            is_random_crop=IS_RANDOM_CROP,
            rotate_angle=ROTATE_ANGLE,
            is_nii=IS_NII
        )
    except Exception as e:
        print(f"批量处理失败：{str(e)}")