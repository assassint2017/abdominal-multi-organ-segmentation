"""
多分类Dice loss
使用最简单的策略：多个dice取平均
"""

import torch
import torch.nn as nn

num_organ = 13


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_stage1, pred_stage2, target):
        """
        :param pred_stage1: 经过放大之后(B, 14, 48, 256, 256)
        :param pred_stage2: (B, 14, 48, 256, 256)
        :param target: (B, 48, 256, 256)
        :return: Dice距离
        """

        # 首先将金标准拆开
        organ_target = torch.zeros((target.size(0), num_organ, 48, 256, 256))

        for organ_index in range(1, num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index - 1, :, :, :] = temp_target
            # organ_target: (B, 13, 48, 128, 128)

        organ_target = organ_target.cuda()

        # 计算第一阶段的loss
        dice_stage1 = 0.0

        for organ_index in range(1, num_organ + 1):
            dice_stage1 += 2 * (pred_stage1[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(
                dim=1) / (pred_stage1[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

        dice_stage1 /= num_organ

        # 计算第二阶段的loss
        dice_stage2 = 0.0

        for organ_index in range(1, num_organ + 1):

            dice_stage2 += 2 * (pred_stage2[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred_stage2[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

        dice_stage2 /= num_organ

        # 将两部分的loss加在一起
        dice = dice_stage1 + dice_stage2

        # 返回的是dice距离
        return (2 - dice).mean()