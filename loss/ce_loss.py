"""

基本的交叉熵损失函数
"""

import torch.nn as nn

num_organ = 13


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred_stage1, pred_stage2, target):
        """

        :param pred_stage1: (B, 14, 48, 256, 256)
        :param pred_stage2: (B, 14, 48, 256, 256)
        :param target: (B, 48, 256, 256)
        """

        # 计算交叉熵损失值
        loss_stage1 = self.loss(pred_stage1, target)
        loss_stage2 = self.loss(pred_stage2, target)

        # 最终的损失值由两部分组成
        loss = loss_stage1 + loss_stage2

        return loss
