#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def create_base_dicom_metadata():
    ds = Dataset()
    ds.PatientName = "Anonymous"
    ds.PatientID = "000001"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    return ds


def save_slice_as_dicom(slice_array, output_path, instance_number):
    ds = create_base_dicom_metadata()

    #必要信息
    ds.SOPInstanceUID = generate_uid()
    ds.InstanceNumber = instance_number
    ds.Modality = "OT"
    ds.SeriesNumber = 1
    ds.ImageType = ["ORIGINAL", "PRIMARY"]

    #图像数据
    ds.Rows, ds.Columns = slice_array.shape
    ds.PixelSpacing = [1, 1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = slice_array.astype(np.int16).tobytes()

    #文件信息
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta

    #保存为DICOM
    dicom_file = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dicom_file.update(ds)
    dicom_file.save_as(output_path, write_like_original=False)


def convert_single_nii_to_dcm(nii_path: str, output_root: str = None):
    nii_name = os.path.splitext(os.path.basename(nii_path))[0]
    final_root = output_root if output_root else "."
    ensure_dir(final_root)
    output_folder = os.path.join(output_root if output_root else ".", f"{nii_name}_dcm")
    ensure_dir(output_folder)

    #加载NII
    nii_img = nib.load(nii_path)
    volume = nii_img.get_fdata()  # 三维形状
    num_slices = volume.shape[2]

    #读取切片信息
    for i in range(num_slices):
        slice_img = volume[:, :, i]
        filename = f"dcm_nii{i+1:03d}.dcm"
        output_path = os.path.join(output_folder, filename)
        save_slice_as_dicom(slice_img, output_path, i + 1)

    print(f"输入NII文件路径：{nii_path}\n "
          f"输出DCM文件路径：{output_folder}\n"
          f"一共生成{num_slices}切片")


def batch_convert_nii_to_dcm(nii_folder: str, output_root: str = None):
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii")]

    if not nii_files:
        print("没有找到.nii文件！")
        return

    for nii_file in nii_files:
        nii_path = os.path.join(nii_folder, nii_file)
        convert_single_nii_to_dcm(nii_path, output_root)

    print("多个nii文件成功转为多个dicom序列！")

if __name__ == "__main__":
    #括号内的路径都需要修改
    #下面是将单个nii文件转为dicom序列
    convert_single_nii_to_dcm(
        "G:/mry1/TOM500/data preprocess/mask2/29.nii",
        output_root="G:/mry1/TOM500/data preprocess/dicom")

    #下面是将多个nii文件转为多个dicom序列
    #batch_convert_nii_to_dcm(
    # "G:/mry1/TOM500/data preprocess/mask2",
    # output_root="G:/mry1/TOM500/data preprocess/dicom")
    pass
