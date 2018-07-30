"""
和ResUnet_dice没有什么区别
用于CE损失的网络脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.3
num_organ = 13


# 定义单个3D FCN
class ResUNet(nn.Module):
    """

    共9332094个可训练的参数, 九百三十万左右
    """
    def __init__(self, training, inchannel, stage):
        """

        :param training: 标志网络是属于训练阶段还是测试阶段
        :param inchannel 网络最开始的输入通道数量
        :param stage 标志网络属于第一阶段，还是第二阶段
        """
        super().__init__()

        self.training = training
        self.stage = stage

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.ELU()
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.ELU()
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.ELU(),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.ELU()
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.ELU(),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.ELU(),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.ELU()
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.ELU()
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ELU()
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.ELU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ELU()
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.ELU()
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.ELU()
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.ELU()
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.ELU()
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.ELU()
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.ELU()
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.ELU()
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ELU()
        )

        self.map = nn.Conv3d(32, num_organ + 1, 1)

    def forward(self, inputs):

        if self.stage is 'stage1':
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        outputs = self.map(outputs)

        # 返回不经过softmax归一化的概率图
        return outputs


# 定义最终的级连3D FCN
class Net(nn.Module):
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.stage1 = ResUNet(training=training, inchannel=1, stage='stage1')
        self.stage2 = ResUNet(training=training, inchannel=num_organ + 2, stage='stage2')

    def forward(self, inputs):
        """

        首先将输入数据在轴向上缩小一倍，然后送入第一阶段网络中
        得到一个粗糙尺度下的分割结果
        然后将原始尺度大小的数据与第一步中得到的分割结果进行拼接，共同送入第二阶段网络中
        得到最终的分割结果

        共18656348个可训练的参数，一千八百万左右
        """
        # 首先将输入缩小一倍
        inputs_stage1 = F.upsample(inputs, (48, 128, 128), mode='trilinear')

        # 得到第一阶段的结果
        output_stage1 = self.stage1(inputs_stage1)
        temp = F.upsample(output_stage1, (48, 256, 256), mode='trilinear')
        output_stage1 = F.upsample(output_stage1, (48, 512, 512), mode='trilinear')

        # 将第一阶段的结果经过softmax归一化成概率图之后再与原始输入进行拼接
        temp = F.softmax(temp, dim=1)

        # 将第一阶段的结果与原始输入数据进行拼接作为第二阶段的输入
        inputs_stage2 = torch.cat((temp, inputs), dim=1)

        # 得到第二阶段的结果
        output_stage2 = self.stage2(inputs_stage2)
        output_stage2 = F.upsample(output_stage2, (48, 512, 512), mode='trilinear')

        if self.training is True:
            return output_stage1, output_stage2
        else:
            return output_stage2


# 网络参数初始化函数
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = Net(training=True)
net.apply(init)

# # 输出数据维度检查
# net = net.cuda()
# data = torch.randn((1, 1, 48, 256, 256)).cuda()
#
# with torch.no_grad():
#     res = net(data)
#
# for item in res:
#     print(item.size())
#
# # 计算网络参数
# num_parameter = .0
# for item in net.modules():
#
#     if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
#         num_parameter += (item.weight.size(0) * item.weight.size(1) *
#                           item.weight.size(2) * item.weight.size(3) * item.weight.size(4))
#
#         if item.bias is not None:
#             num_parameter += item.bias.size(0)
#
#     elif isinstance(item, nn.PReLU):
#         num_parameter += item.num_parameters
#
#
# print(num_parameter)

