"""
多分类Dice loss
对于简单的多个dice取平均的扩展
共13种器官＋背景
(0) 背景
(1) spleen 脾
(2) right kidney 右肾
(3) left kidney 左肾
(4) gallbladder 胆囊
(5) esophagus 食管
(6) liver 肝脏
(7) stomach 胃
(8) aorta 大动脉
(9) inferior vena cava 下腔静脉
(10) portal vein and splenic vein 门静脉和脾静脉
(11) pancreas 胰腺
(12) right adrenal gland 右肾上腺
(13) left adrenal gland 左肾上腺
"""

import torch
import torch.nn as nn

num_organ = 13

# 每一种器官的权重，权重的把握原则就是，比较容易分割的器官的权重保持为１不变，分割效果不好的器官的权重要大一些
organ_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


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
        dice_stage1_numerator = 0.0  # dice系数的分子
        dice_stage1_denominator = 0.0  # dice系数的分母

        for organ_index in range(1, num_organ + 1):

            dice_stage1_numerator += 2 * (pred_stage1[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1)

            dice_stage1_numerator *= organ_weight[organ_index - 1]

            dice_stage1_denominator += (pred_stage1[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

            dice_stage1_denominator *= organ_weight[organ_index - 1]

        dice_stage1 = (dice_stage1_numerator / dice_stage1_denominator)

        # 计算第二阶段的loss
        dice_stage2_numerator = 0.0  # dice系数的分子
        dice_stage2_denominator = 0.0  # dice系数的分母

        for organ_index in range(1, num_organ + 1):

            dice_stage2_numerator += 2 * (pred_stage2[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1)

            dice_stage2_numerator *= organ_weight[organ_index - 1]

            dice_stage2_denominator += (pred_stage2[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

            dice_stage2_denominator *= organ_weight[organ_index - 1]

        dice_stage2 = (dice_stage2_numerator / dice_stage2_denominator)

        # 将两部分的loss加在一起
        dice = dice_stage1 + dice_stage2

        # 返回的是dice距离
        return (2 - dice).mean()