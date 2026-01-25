#!/user/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np

def resample_volume(input_path, output_path, new_spacing_z_mm):

    # 读取nii图像
    img = sitk.ReadImage(input_path)

    # 获取原始图像的空间和大小
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()

    print("原始间距:", orig_spacing)
    print("原始尺寸:", orig_size)


    new_spacing = (orig_spacing[0], orig_spacing[1], new_spacing_z_mm)

    #计算新的空间维度
    new_size = [
        orig_size[0],
        orig_size[1],
        int(np.ceil(orig_size[2] * (orig_spacing[2] / new_spacing_z_mm)))
    ]

    print("新间距:", new_spacing)
    print("新尺寸:", new_size)

    # 开始重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)


    resampled = resampler.Execute(img)

    # 输出重采样的文件以及对应的地址
    sitk.WriteImage(resampled, output_path)
    print("重采样文件保存地址:", output_path)
    print("重采样后的间距:", resampled.GetSpacing())
    print("重采样后的尺寸:", resampled.GetSize())

#下面的路径需要根据实际路径进行修改！!!
if __name__ == "__main__":
    #分别填入输入路径和输出路径
    resample_volume("G:/mry1/TOM500/data preprocess/resample/1665867_0000.nii.gz",
                    "G:/mry1/TOM500/data preprocess/resample/outputen_10mm.nii",
                    new_spacing_z_mm=10.0)#这边填入想要转换成的mm数！想转换成10mm就填10


