#!/user/bin/env python3
# -*- coding: utf-8 -*-
import pydicom
import os
import cv2
import numpy as np

# 根据自己的输入输出路径进行修改！
input_folder = ("G:/mry1/TOM500/data preprocess/dicom/39_dcm")  # 输入DICOM文件夹路径
output_root = "G:/mry1/TOM500/data preprocess/png"  # PNG输出，填到根路径就可以
file_ext = "png"  # 输出格式（png/jpg）
digit_length = 3  # 编号位数（如3表示001, 002...）
source_format = "dcm"  # 转换前格式
# 自动生成输出文件夹名
input_folder_name = os.path.basename(input_folder)
if input_folder_name.lower().endswith("_dcm"):
    output_folder_name = input_folder_name[:-4] + "_png"
else:
    output_folder_name = input_folder_name + "_png"

output_folder = os.path.join(output_root, output_folder_name)
os.makedirs(output_folder, exist_ok=True)

# 获取所有DICOM文件
all_files = os.listdir(input_folder)
dcm_files = [f for f in all_files if f.lower().endswith(('.dcm', '.dicom'))]

if not dcm_files:
    print(f"在文件夹 {input_folder} 中未找到DICOM文件")
    exit()

print(f"找到 {len(dcm_files)} 个DICOM文件，开始转换...")
print(f"输出路径：{output_folder}")
print(f"文件名格式：{file_ext}_{source_format}001.{file_ext}、{file_ext}_{source_format}002.{file_ext}...")

total_slices = 0  # 统计总切片数
converted_slices = 0  # 统计成功转换的切片数

for dcm_idx, dcm_filename in enumerate(dcm_files, 1):
    dcm_path = os.path.join(input_folder, dcm_filename)

    try:
        # 读取DICOM文件（保留元数据，确保空间信息不丢失）
        ds = pydicom.dcmread(dcm_path, force=True)
        pixel_array = ds.pixel_array
        # 3D数据（z, h, w）
        if len(pixel_array.shape) == 3:
            z_slices, height, width = pixel_array.shape
            print(f"\n处理DICOM文件：{dcm_filename}")
            print(f"数据维度：{pixel_array.shape} (切片数×高度×宽度)")

            # 遍历每个切片
            for slice_idx in range(z_slices):
                # 获取当前切片数据
                slice_data = pixel_array[slice_idx, :, :]

                # 数据归一化（保持原始数据分布，避免信息丢失）
                if slice_data.dtype in [np.uint16, np.int16, np.float32, np.float64]:
                    # 使用min-max归一化到0-255
                    slice_min = np.min(slice_data)
                    slice_max = np.max(slice_data)
                    if slice_max > slice_min:  # 避免除零错误
                        slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                    else:
                        slice_data = np.zeros_like(slice_data, dtype=np.uint8)

                # 处理RGB转灰度（如果是彩色DICOM）
                if len(slice_data.shape) == 3 and slice_data.shape[2] == 3:
                    slice_data = cv2.cvtColor(slice_data, cv2.COLOR_RGB2GRAY)

                # 生成统一格式的文件名（png001, png002...）
                total_slices += 1
                png_filename = f"{file_ext}_{source_format}{total_slices:0{digit_length}d}.{file_ext}"
                png_path = os.path.join(output_folder, png_filename)

                success = cv2.imwrite(png_path, slice_data, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # 保存PNG文件（使用cv2.IMWRITE_PNG_COMPRESSION控制压缩质量）
                if success:
                    converted_slices += 1
                    print(f"  切片 {slice_idx + 1}/{z_slices} → {png_filename} (成功)")
                else:
                    print(f"  切片 {slice_idx + 1}/{z_slices} → {png_filename} (保存失败)")

        # 2D数据（h, w）
        elif len(pixel_array.shape) == 2:
            height, width = pixel_array.shape
            print(f"\n处理DICOM文件：{dcm_filename}")
            print(f"数据维度：{pixel_array.shape}")

            # 数据归一化
            if pixel_array.dtype in [np.uint16, np.int16, np.float32, np.float64]:
                slice_min = np.min(pixel_array)
                slice_max = np.max(pixel_array)
                if slice_max > slice_min:
                    pixel_array = ((pixel_array - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

            # RGB转灰度
            if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)

            # 生成文件名
            total_slices += 1
            png_filename = f"{file_ext}_{source_format}{total_slices:0{digit_length}d}.{file_ext}"
            png_path = os.path.join(output_folder, png_filename)

            # 保存文件
            success = cv2.imwrite(png_path, pixel_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            if success:
                converted_slices += 1
                print(f"  → {png_filename} (成功)")
            else:
                print(f"  → {png_filename} (保存失败)")


        else:
            print(f"\n跳过DICOM文件：{dcm_filename}")
            print(f"不支持的数据维度：{pixel_array.shape}")

    except Exception as e:
        print(f"\n处理DICOM文件 {dcm_filename} 时出错：{str(e)}")
        continue

# 输出转换总结
print(f"DCM成功转换为PNG！")
print(f"处理的DICOM文件数：{len(dcm_files)}")
print(f"成功转换的切片数：{converted_slices}")
