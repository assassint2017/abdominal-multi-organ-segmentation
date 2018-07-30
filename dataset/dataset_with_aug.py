"""
和dataset.py脚本完全一样
只不过带有数据增强
"""

import os
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset as dataset

import SimpleITK as sitk
import scipy.ndimage as ndimage

on_server = True
size = 48
lower = -350


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('img', 'label'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([B, 1, 48, 256, 256]) torch.Size([B, 48, 256, 256])
        """

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # 在slice平面内随机选取48张slice
        start_slice = random.randint(0, ct_array.shape[0] - size)
        end_slice = start_slice + size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # 以0.5的概率在5度的范围内随机旋转
        # 角度为负数是顺时针旋转，角度为正数是逆时针旋转
        if random.uniform(0, 1) >= 0.5:
            angle = random.uniform(-5, 5)
            ct_array = ndimage.rotate(ct_array, angle, axes=(1, 2), reshape=False, cval=lower)
            seg_array = ndimage.rotate(seg_array, angle, axes=(1, 2), reshape=False, cval=0)

        # 有0.5的概率不进行任何修修改，剩下0.5随机挑选0.8-0.5大小的patch放大到256*256
        if random.uniform(0, 1) >= 0.5:
            ct_array, seg_array = self.zoom(ct_array, seg_array, patch_size=random.uniform(0.5, 0.8))

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)

    def zoom(self, ct_array, seg_array, patch_size):

        length = int(256 * patch_size)

        x1 = int(random.uniform(0, 255 - length))
        y1 = int(random.uniform(0, 255 - length))

        x2 = x1 + length
        y2 = y1 + length

        ct_array = ct_array[:, x1:x2 + 1, y1:y2 + 1]
        seg_array = seg_array[:, x1:x2 + 1, y1:y2 + 1]

        with torch.no_grad():

            ct_array = torch.FloatTensor(ct_array).unsqueeze(dim=0).unsqueeze(dim=0)
            ct_array = Variable(ct_array)
            ct_array = F.upsample(ct_array, (size, 256, 256), mode='trilinear').squeeze().detach().numpy()

            seg_array = torch.FloatTensor(seg_array).unsqueeze(dim=0).unsqueeze(dim=0)
            seg_array = Variable(seg_array)
            seg_array = F.upsample(seg_array, (size, 256, 256), mode='trilinear').squeeze().detach().numpy()

            return ct_array, seg_array


ct_dir = '/home/zcy/Desktop/train/CT/' \
    if on_server is False else './train/CT/'
seg_dir = '/home/zcy/Desktop/train/GT/' \
    if on_server is False else './train/GT/'

train_ds = Dataset(ct_dir, seg_dir)


# # 测试代码
# from torch.utils.data import DataLoader
# train_dl = DataLoader(train_ds, 6, True)
# for index, (ct, seg) in enumerate(train_dl):
#
#     print(index, ct.size(), seg.size())
#     print('----------------')