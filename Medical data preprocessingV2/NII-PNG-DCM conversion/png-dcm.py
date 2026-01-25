#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
import cv2

#下面的路径需要自己去调整！！！
INPUT_FOLDER = "G:/mry1/TOM500/data preprocess/png/39_png"  # 输入文件夹路径(PNG/JPG)
OUTPUT_ROOT = "G:/mry1/TOM500/data preprocess/dicom"  # DCM输出根路径
DCM_UID_PREFIX = "1.2.826.0.1.3680043.8.498."  # DCM唯一标识符前缀
PIXEL_SPACING = [0.312,0.312]  # 像素间距（关键参数）
MODALITY = "CT"  # 模态（CT、MRI、XR）
STUDY_DESCRIPTION = "PNG_to_DCM_Conversion"
DIGIT_LENGTH = 3  # 编号位数（3→001-999）
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg')  # 支持的输入格式
TARGET_FORMAT = "dcm" # 转换后格式（用于文件名拼接）
SOURCE_FORMAT = "png" # 转换前格式（用于文件名拼接）
PATIENT_NAME = "Unknown"
def create_dcm_dataset(
        pixel_array: np.ndarray,
        series_instance_uid: str,
        instance_number: int,
        pixel_spacing: list[float] = [0.312,0.312],
        modality: str = "CT"
) -> FileDataset:
    # 生成唯一标识符（UID）
    sop_instance_uid = pydicom.uid.generate_uid(prefix=DCM_UID_PREFIX)
    study_instance_uid = pydicom.uid.generate_uid(prefix=DCM_UID_PREFIX)
    frame_of_reference_uid = pydicom.uid.generate_uid(prefix=DCM_UID_PREFIX)

    # 创建DCM数据集
    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage  # 根据模态调整
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian  # 传输语法（兼容大多数软件）
    ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid(prefix=DCM_UID_PREFIX)

    # 核心元数据（确保医学影像软件可识别）
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.FrameOfReferenceUID = frame_of_reference_uid
    ds.InstanceNumber = instance_number  # 切片序号（关键：与文件名编号一致）
    ds.Modality = modality
    ds.StudyDescription = STUDY_DESCRIPTION
    ds.SeriesDescription = f"{modality}_Series"
    ds.BodyPartExamined = "Unknown"

    # 图像相关参数（保持空间信息准确）
    ds.Rows, ds.Columns = pixel_array.shape[:2]  # 图像尺寸（高度×宽度）
    ds.PixelSpacing = pixel_spacing  # 像素间距（mm/像素，医学影像核心空间参数）
    ds.SliceThickness = pixel_spacing[0]  # 切片厚度（默认与像素间距一致，可自定义）
    ds.SpacingBetweenSlices = pixel_spacing[0]

    # 像素数据参数
    if len(pixel_array.shape) == 3:  # 彩色图像（RGB）
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
    else:  # 灰度图像
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        pixel_array = (pixel_array.astype(np.uint16) << 8) | pixel_array.astype(np.uint16)

    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0.0  # 灰度转换截距
    ds.RescaleSlope = 1.0  # 灰度转换斜率

    # 设置像素数据
    ds.PixelData = pixel_array.tobytes()

    # 创建FileDataset（完整DCM文件对象）
    file_ds = FileDataset(
        "", ds, file_meta=ds.file_meta, preamble=b"\x00" * 128
    )
    file_ds.is_little_endian = True
    file_ds.is_implicit_VR = False

    return file_ds


def png_jpg_to_dcm(
        input_folder: str,
        output_root: str,
        target_format: str = "dcm",
        source_format: str = "png",
        digit_length: int = 3
) -> None:
    #验证输入文件夹
    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹不存在 → {input_folder}")
        return

    #自动生成输出文件夹（如1_png → 1_dcm）
    input_folder_name = os.path.basename(input_folder)
    if input_folder_name.lower().endswith(f"_{source_format}"):
        output_folder_name = input_folder_name[:-len(f"_{source_format}")] + f"_{target_format}"
    else:
        output_folder_name = input_folder_name + f"_{target_format}"
    output_folder = os.path.join(output_root, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"输出DCM文件夹：{output_folder}")

    #获取所有支持的图像文件并排序（确保编号顺序一致）
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]
    # 按文件名自然排序（避免10.png排在2.png前面）
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)

    if not image_files:
        print(f"警告：在输入文件夹中未找到PNG/JPG文件 → {input_folder}")
        return

    print(f"找到 {len(image_files)} 个图像文件，开始转换...")
    print(
        f"文件名格式：{target_format}_{source_format}001."
        f"{target_format}、{target_format}_{source_format}002.{target_format}...")

    # 生成统一的Series Instance UID（同文件夹下的图像属于同一序列）
    series_instance_uid = pydicom.uid.generate_uid(prefix=DCM_UID_PREFIX)
    converted_count = 0

    # 批量转换图像
    for idx, image_filename in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, image_filename)
        try:
            # 读取图像（cv2默认BGR，转换为RGB以符合DCM标准）
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"跳过：无法读取图像 → {image_filename}")
                continue

            # 处理通道顺序（BGR→RGB）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 处理Alpha通道（如果有）
            elif len(img.shape) == 4:
                img = img[:, :, :3]  # 移除Alpha通道，保留RGB

            # 创建DCM数据集
            dcm_dataset = create_dcm_dataset(
                pixel_array=img,
                series_instance_uid=series_instance_uid,
                instance_number=idx,
                pixel_spacing=PIXEL_SPACING,
                modality=MODALITY
            )

            # 生成DCM文件名（dcm001.dcm, dcm002.dcm...）
            dcm_filename = f"{target_format}_{source_format}{idx:0{digit_length}d}.{target_format}"
            dcm_path = os.path.join(output_folder, dcm_filename)

            # 保存DCM文件（使用Explicit VR Little Endian格式，兼容性最好）
            dcm_dataset.save_as(dcm_path, write_like_original=False)
            converted_count += 1
            print(f"成功：{image_filename} → {dcm_filename}")

        except Exception as e:
            print(f"失败：{image_filename} → 错误原因：{str(e)}")
            continue

    # 输出转换总结
    print(f"PNG/JPG成功转换为DCM！")
    print(f"总图像数：{len(image_files)}")
    print(f"成功转换数：{converted_count}")


if __name__ == "__main__":
    # 执行转换
    png_jpg_to_dcm(
        input_folder=INPUT_FOLDER,
        output_root=OUTPUT_ROOT,
        target_format=TARGET_FORMAT,
        source_format=SOURCE_FORMAT,
        digit_length=DIGIT_LENGTH
    )



