
import torch.nn as nn
import torch


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, dataset_dict, config_dict, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        # 调用父类的构造函数进行初始化
        super(CustomCrossEntropyLoss, self).__init__(
            weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target):
        # 这里可以添加额外的处理逻辑
        # 例如，对输入或者目标进行一些变换
        # 这里简单打印输入和目标的形状作为示例
        print(f"Input shape: {input.shape}")
        print(f"Target shape: {target.shape}")

        # 调用父类的 forward 方法进行损失计算
        return super(CustomCrossEntropyLoss, self).forward(input, target)
