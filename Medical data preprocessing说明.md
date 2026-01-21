# Medical data preprocessing说明（DoWhat）

> 说明：按文件夹逐一列出并简要说明各 Python 脚本的功能、输入输出与依赖要点。请根据实际路径在脚本中的 `__main__` 区块调整参数（文件夹/文件路径）。 ✅

---

## 1. check 文件夹 ✅

- `check_document.py`  
  - 功能：读取 NIfTI 文件（`.nii` / `.nii.gz`）并提取空间信息（像素间距、图像原点、方向向量/方向矩阵），以便与 DICOM 空间信息对应。
  - 主要函数：`get_nii_spatial_info`, `print_spatial_info`。
  - 依赖：`nibabel`, `numpy`。

- `check_pixel.py`  
  - 功能：另一版 NII 空间信息提取脚本，返回 DICOM 风格的字段（PixelSpacing、SliceThickness、ImageOrientationPatient、ImagePositionPatient 等），并打印 NII 的体积形状。
  - 主要函数：`get_nii_spatial_info`（返回 dict 格式的信息）。
  - 依赖：`nibabel`, `numpy`。

- `check-slice.py`  
  - 功能：演示如何加载 `.npz` 文件并查看包含的条目（示例短脚本，主要用于快速检查 `.npz` 内部结构）。
  - 用法：直接 `np.load` 并打印 `files` 列表与 `affine` 字段。
  - 依赖：`numpy`。

- `read npz.py`  
  - 功能：简单的示例脚本，用于测试 `.npz` 文件读取并打印内部数组（示例路径需用户修改）。
  - 依赖：`numpy`。

---

## 2. enhance 文件夹 ✅

- `data-enhance.py`  
  - 功能：医学影像切片增强工具，包含：裁剪（`crop_medical_image`）、旋转（`rotate_medical_image`）、单张增强（支持 PNG 或从 NII 中抽取切片）和批量增强（`augment_single_image`, `batch_augment_images`）。
  - 特点：支持对 NII 抽取指定轴(slice_axis)的中间切片并保存为增强后的 NII 或 PNG；可保持原始尺寸或按需 resize。
  - 依赖：`cv2`（OpenCV）、`numpy`、`nibabel`。

---

## 3. NII-PNG-DCM conversion 文件夹 ✅

- `dcm-nii.py`  
  - 功能：将 DICOM 序列转换为 NIfTI（`.nii`），并使用“原始 NII 的 affine”来保证几何信息一致（常用于保证与已有 NII 对齐）。支持单序列与批量处理。
  - 主要行为：按 `InstanceNumber` 排序 DICOM 切片，stack 成体积并保存为 `nii_dcm<id>.nii`。
  - 依赖：`pydicom`, `nibabel`, `numpy`。

- `dcm-npz.py`  
  - 功能：把 DICOM 序列保存为 `.npz`（包含 image、affine、spacing、source_type、source_name）。方便后续快速加载与处理。
  - 依赖：`pydicom`, `numpy`。

- `dcm-png.py`  
  - 功能：把 DICOM（单文件或含多切片）导出为 PNG（或 JPG）；处理 2D/3D 情况，做像素值归一化、RGB→灰度转换等，按统一命名格式输出切片图像。
  - 依赖：`pydicom`, `cv2`, `numpy`。

- `nii-dcm.py`  
  - 功能：把单个 NIfTI 文件转换为 DICOM 切片序列（为每个切片创建简单 DICOM metadata 并保存为 `.dcm`）。支持批量将多个 `.nii` 转为多个 DICOM 文件夹。
  - 依赖：`nibabel`, `pydicom`, `numpy`。

- `nii-npz.py`  
  - 功能：将 NIfTI 转为 `.npz`（保存 image、affine、spacing 等），方便后续无需 NIfTI 库即可快速读取数据。
  - 依赖：`nibabel`, `numpy`。

- `nii-png.py`  
  - 功能：从 PNG 文件夹恢复成 NIfTI；支持自动生成 NII 名称、指定输出路径、可使用原始 NII 的 affine 或手动设置像素间距与方向。
  - 特点：支持 slice_axis 设置（决定如何在三轴堆叠 PNG 为体积），会校验 PNG 尺寸一致性。
  - 依赖：`opencv`, `nibabel`, `numpy`。

- `png-dcm.py`  
  - 功能：把 PNG/JPG 批量转换为 DICOM 序列，生成符合基本 DICOM 元数据（Series UID、InstanceNumber、PixelSpacing等）的 `.dcm` 文件。
  - 依赖：`pydicom`, `cv2`, `numpy`。

- `png-nii.py`  
  - 功能：把按顺序的 PNG 切片合并为 NIfTI（支持从文件夹名中提取数字生成编号），并可设置像素间距/方向或借用已有 NII 的 affine。
  - 依赖：`cv2`, `nibabel`, `numpy`。

- `png-npz.py`  
  - 功能：将单个文件（支持 NII/DICOM/PNG）或批量目录中的文件转换为 `.npz`（存储为 `data`），适用于多种输入格式的统一转换。
  - 依赖：`SimpleITK`, `cv2`, `pydicom`, `numpy`。

---

## 4. resample 文件夹 ✅

- `resample.py`  
  - 功能：使用 SimpleITK 对体积图像进行重采样（主要调整 Z 方向的 spacing），计算新的尺寸并执行重采样（线性插值），保存新 NIfTI。
  - 用法示例：设置 `new_spacing_z_mm`（例如 10.0），脚本会计算新的体积尺寸并输出重采样结果。
  - 依赖：`SimpleITK`, `numpy`。

---

## 5. Three-dimensional reconstruction 文件夹 ✅

- `reconstruction.py`  
  - 功能：读取 NIfTI 或 DICOM 文件夹，基于阈值进行简单肿瘤分割（threshold），进行各向同性（isotropic）重采样（通过 `scipy.ndimage.zoom`），并可视化：展示三轴的 2D 切片以及使用 `mayavi` 对分割体绘制 3D 表面（`contour3d`）。
  - 依赖：`SimpleITK`, `pydicom`, `numpy`, `scipy`, `matplotlib`, `mayavi`。

---

## 统一注意事项 & 依赖列表 🔧

- 常见依赖：`numpy`, `nibabel`, `pydicom`, `SimpleITK`, `opencv-python` (`cv2`), `matplotlib`, `scipy`, `mayavi`, `SimpleITK`。
- 路径配置：这些脚本多数在 `__main__` 部分包含示例路径，需要按实际环境调整后再运行。
- 数据一致性：图像尺寸、像素间距、方向等空间参数需要在不同格式间转换时注意保持一致（脚本中多数有读取/设置 affine 或 PixelSpacing 的步骤）。

---

如果你希望，我可以：
1. 将每个脚本头部添加简短注释（1-2 行）说明作用（自动修改脚本）；
2. 为这些脚本生成一个统一的 README 或运行示例脚本（批量执行/测试用）；
3. 检查哪些脚本缺少异常处理并进行改进。

请选择下一步（1 / 2 / 3 / 或直接“完成”）。 ✨
