#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
from skimage import measure
import matplotlib.pyplot as plt
import trimesh
import pydicom
import cv2
# 1. 读取DICOM序列
def load_dicom_series(dicom_dir):
    """
    读取DICOM影像序列
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise RuntimeError("目录中未发现DICOM影像序列")

    series_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(series_files)

    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    spacing = image.GetSpacing()
    spacing = (spacing[2], spacing[1], spacing[0])

    return volume, spacing, image, series_files

def find_rtstruct(dicom_dir):
    for f in os.listdir(dicom_dir):
        if f.upper().startswith("RTSTRUCT") and f.lower().endswith(".dcm"):
            return os.path.join(dicom_dir, f)
    raise FileNotFoundError("未找到RTSTRUCT文件")
def world_to_voxel(coord, origin, spacing, direction):
    """
    将物理坐标(mm)转换为体素索引
    """
    coord = np.array(coord)
    origin = np.array(origin)
    spacing = np.array(spacing)

    relative = coord - origin
    voxel = np.linalg.inv(np.array(direction).reshape(3, 3)).dot(relative)
    voxel = voxel / spacing

    return np.round(voxel).astype(int)
# 2. 读取GTV标签（DICOM或NII）
def extract_gtv_mask(rtstruct_path, sitk_img, volume_shape):
    """
    从RTSTRUCT中解析GTV并生成三维mask
    """
    ds = pydicom.dcmread(rtstruct_path)

    # ROI名称映射
    roi_number_to_name = {
        roi.ROINumber: roi.ROIName
        for roi in ds.StructureSetROISequence
    }

    # 找GTV
    gtv_roi_number = None
    for num, name in roi_number_to_name.items():
        if name.lower().startswith("gtv"):
            gtv_roi_number = num
            print("使用的GTV名称:", name)
            break

    if gtv_roi_number is None:
        raise ValueError("RTSTRUCT中未找到GTV")

    origin = sitk_img.GetOrigin()
    spacing = sitk_img.GetSpacing()
    direction = sitk_img.GetDirection()

    mask = np.zeros(volume_shape, dtype=np.uint8)

    for roi_contour in ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber != gtv_roi_number:
            continue

        for contour in roi_contour.ContourSequence:
            coords = np.array(contour.ContourData).reshape(-1, 3)

            # 取Z层
            z_world = coords[0, 2]
            z_index = world_to_voxel(
                [0, 0, z_world],
                origin,
                spacing,
                direction
            )[2]

            polygon = []
            for pt in coords:
                voxel = world_to_voxel(pt, origin, spacing, direction)
                polygon.append([voxel[0], voxel[1]])

            polygon = np.array(polygon, dtype=np.int32)

            if 0 <= z_index < volume_shape[0]:
                slice_mask = np.zeros(volume_shape[1:], dtype=np.uint8)
                cv2.fillPoly(slice_mask, [polygon], 1)
                mask[z_index] |= slice_mask

    return mask
# 3. 提取肿瘤区域并重建3D表面
def reconstruct_3d_surface(mask, spacing):
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        spacing=spacing
    )
    return verts, faces

# 4. 3D可视化
def visualize_3d(verts, faces):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        color="red",
        alpha=0.85
    )
    ax.set_title("GTV 三维重建")
    plt.show()

def export_stl(verts, faces, path):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(path)
# 6. 根据实际路径修改！
if __name__ == "__main__":
    dicom_dir = r"G:\mry1\TOM500\tumor dataset\guoshusen1153\Structure\20191153_guoshusen1153" #原始DICOM数据目录
    output_stl = r"G:\mry1\TOM500\tumor dataset\guoshusen1153\output"   #输出STL文件路径

    volume, spacing, sitk_img, _ = load_dicom_series(dicom_dir)
    rtstruct_path = find_rtstruct(dicom_dir)

    gtv_mask = extract_gtv_mask(
        rtstruct_path,
        sitk_img,
        volume.shape
    )

    verts, faces = reconstruct_3d_surface(gtv_mask, spacing)
    visualize_3d(verts, faces)
    export_stl(verts, faces, output_stl)

    print("肿瘤区域三维重建已完成")
