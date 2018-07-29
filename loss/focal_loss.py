"""

focal loss
对交叉熵损失函数的扩展，更善于处理数据不平衡的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

num_organ = 13


class CELoss(nn.Module):
    def __init__(self, alpha=2):
        """

        :param alpha: focal loss中的指数项的次数
        """
        super().__init__()

        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred_stage1, pred_stage2, target):
        """

        :param pred_stage1: (B, 14, 48, 256, 256)
        :param pred_stage2: (B, 14, 48, 256, 256)
        :param target: (B, 48, 256, 256)
        """

        # 计算正样本的数量，这里所谓正样本就是属于器官的体素的个数
        num_target = (target > 0).type(torch.cuda.FloatTensor).sum()

        # 计算交叉熵损失值
        loss_stage1 = self.loss(pred_stage1, target)
        loss_stage2 = self.loss(pred_stage2, target)

        # 对已经可以良好分类的数据的损失值进行衰减
        exponential_term_stage1 = (1 - F.softmax(pred_stage1, dim=1).max(dim=1)[0]) ** self.alpha
        exponential_term_stage2 = (1 - F.softmax(pred_stage2, dim=1).max(dim=1)[0]) ** self.alpha

        loss_stage1 *= exponential_term_stage1
        loss_stage2 *= exponential_term_stage2

        # 最终的损失值由两部分组成
        loss = loss_stage1 + loss_stage2

        # 如果这一批数据中没有正样本，(虽然这样的概率非常小，但是还是要避免一下)
        if num_target == 0:
            # 则使用全部样本的数量进行归一化，和正常的CE损失一样
            loss = loss.mean()
        else:
            # 否侧用正样本的数量对损失值进行归一化
            loss = loss.sum() / num_target

        return loss
