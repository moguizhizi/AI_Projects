import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from langdetect import detect
from sklearn.preprocessing import MultiLabelBinarizer


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def is_english_by_langdetect(text: str) -> bool:
    """
    使用 langdetect 判断文本是否为英文。

    Args:
        text (str): 输入文本

    Returns:
        bool: 是否为英文
    """
    try:
        # 检测语言
        lang = detect(text)
        return lang == "en"
    except Exception as e:
        print(f"Language detection failed: {e}")
        return False

def to_multi_hot_sklearn(category_ids, num_classes):
    """
    将类别 ID 列表转换为多热编码，支持单样本和多样本输入。

    参数:
        category_ids (list): 类别 ID 列表，可以是单样本（如 [0, 2, 3]）或多样本（如 [[0, 2], [1, 3, 4]]）。
        num_classes (int): 总类别数量。

    返回:
        multi_hot (list): 多热编码结果，单样本返回一个列表，多样本返回嵌套列表。
    """
    # 初始化 MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=range(num_classes))

    # 检查输入是否为嵌套列表（多样本）
    if isinstance(category_ids[0], list):
        # 多样本，直接传入嵌套列表
        multi_hot = mlb.fit_transform(category_ids)
    else:
        # 单样本，包装为嵌套列表
        multi_hot = mlb.fit_transform([category_ids])

    # 转换为 Python 列表
    multi_hot = multi_hot.tolist()

    # 如果是单样本，返回第一个样本的结果（保持与原函数一致）
    if not isinstance(category_ids[0], list):
        return multi_hot[0]
    return multi_hot

__all__ = ["seed_everything", "is_english_by_langdetect", "to_multi_hot_sklearn"]
