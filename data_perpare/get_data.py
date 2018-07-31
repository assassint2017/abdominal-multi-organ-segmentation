"""
30例训练集中 3mm有17个，2.5mm有1个，5mm有12个
因此决定把轴向的sapcing统一到3mm，然后进行灰度值截断
对于图像的预处理，目前也就包括这两项
"""

import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

ct_path = '/home/zcy/Desktop/dataset/multi-organ/train/CT'
seg_path = '/home/zcy/Desktop/dataset/multi-organ/train/GT'

new_ct_path = '/home/zcy/Desktop/train/CT/'
new_seg_path = '/home/zcy/Desktop/train/GT/'

# 新产生的训练数据存储路径
if os.path.exists('/home/zcy/Desktop/train/') is True:
    shutil.rmtree('/home/zcy/Desktop/train/')
os.mkdir('/home/zcy/Desktop/train/')
os.mkdir(new_ct_path)
os.mkdir(new_seg_path)

upper = 350
lower = -upper
slice_thickness = 3
down_scale = 0.5
expand_slice = 10

file_index = 0  # 用于记录新产生的数据的序号
start_time = time()

for ct_file in os.listdir(ct_path):

    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_path, ct_file.replace('img', 'label')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 对CT和金标准使用插值算法进行插值来统一轴向的spacing，插值之后的array依然是int类型
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)

    # 对金标准插值不应该使用高级插值方式，这样会破坏边界部分，总之这次错误也说明了检查数据的重要性
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=0)

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 找到包含器官的slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # 两个方向上各扩张slice
    if start_slice - expand_slice < 0:
        start_slice = 0
    else:
        start_slice -= expand_slice

    if end_slice + expand_slice >= seg_array.shape[0]:
        end_slice = seg_array.shape[0] - 1
    else:
        end_slice += expand_slice

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    print('file name:', ct_file)
    print('shape:', ct_array.shape)

    # 保存数据
    new_ct_array = ct_array
    new_seg_array = seg_array

    new_ct = sitk.GetImageFromArray(new_ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(
        (ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

    new_seg = sitk.GetImageFromArray(new_seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing(
        (ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

    new_ct_name = 'img-' + str(file_index) + '.nii'
    new_seg_name = 'label-' + str(file_index) + '.nii'

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, new_ct_name))
    sitk.WriteImage(new_seg, os.path.join(new_seg_path, new_seg_name))

    # 每处理完一个数据，打印一次已经使用的时间
    print('already use {:.3f} min'.format((time() - start_time) / 60))
    print('-----------')

    file_index += 1
