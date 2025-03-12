
import torch.nn as nn
from typing import Dict, Any


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, dataset_dict, config_dict, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        # 调用父类的构造函数进行初始化
        super(CustomCrossEntropyLoss, self).__init__(
            weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target, task_configs=None):
        # 这里可以添加额外的处理逻辑
        # 例如，对输入或者目标进行一些变换
        # 这里简单打印输入和目标的形状作为示例
        print(f"Input shape: {input.shape}")
        print(f"Target shape: {target.shape}")

        # 调用父类的 forward 方法进行损失计算
        return super(CustomCrossEntropyLoss, self).forward(input, target)


class CrossBCEWithLogitsLoss(nn.Module):
    def __init__(self, dataset_dict: Dict[str, Any], config_dict: Dict[str, Any]):
        """
        初始化多任务损失类，支持任意任务组合的交叉熵损失和二分类交叉熵损失。
        """
        super(CrossBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_weights = config_dict["bce_weight"]
        self.ce_weights = 1 - self.bce_weights

    def forward(self, logits_dict, labels_dict, task_configs):
        """
        计算多任务损失。

        Args:
            logits_dict (dict): 任务 logits，键为任务名称，值为张量 (e.g., {"task1": [batch_size, ...]})
            labels_dict (dict): 任务标签，键为任务名称，值为张量 (e.g., {"task1": [batch_size, ...]})
            task_configs (list): 任务配置，包含 {"name": str, "type": str ("bce" 或 "ce"), "weight": float}

        Returns:
            torch.Tensor: 总损失值
        """
        total_loss = 0.0

        for config in task_configs:
            task_name = config["name"]
            loss_type = config["type"]
            weight = 0.0
            if task_name not in logits_dict or task_name not in labels_dict:
                raise ValueError(
                    f"Task {task_name} not found in logits_dict or labels_dict")

            logits = logits_dict[task_name]
            labels = labels_dict[task_name]

            if loss_type == "bce":
                # 二分类交叉熵损失，适用于 [batch_size, num_classes] 形状
                loss = self.bce_loss(logits, labels)
                weight = self.bce_weights
            elif loss_type == "ce":
                # 交叉熵损失，适用于 [batch_size, num_classes] logits 和 [batch_size] 标签
                if logits.dim() == 2 and labels.dim() == 1:
                    loss = self.ce_loss(logits, labels.long())
                    weight = self.ce_weights
                else:
                    raise ValueError(
                        f"Invalid shape for CE loss: logits {logits.shape}, labels {labels.shape}")
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            total_loss += weight * loss

        return total_loss
