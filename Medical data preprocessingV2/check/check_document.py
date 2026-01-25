#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
#填写需要检查的文件路径
file_path = 'G:\mry1\TOM500\data preprocess\dicom\eye29\slice_007.dcm'
#如果运行结果没有任何显示，就是文件存在没有问题！！！
if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
    else:
        print(f"目录存在，但文件不存在")
        files = os.listdir(directory)
        print("目录中的文件:")
        for f in files:
            print(f"  - {f}")
