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
    mlb = MultiLabelBinarizer(classes=range(num_classes))
    multi_hot = mlb.fit_transform([category_ids])[0]
    return multi_hot.tolist()


__all__ = ["seed_everything", "is_english_by_langdetect", "to_multi_hot_sklearn"]
