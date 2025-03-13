from typing import Union
import yaml
import json
import os
import torch
import shutil
from typing import Optional


def read_yaml_file(file_path):
    """
    读取 YAML 文件并返回解析后的 Python 对象。

    :param file_path: YAML 文件的路径
    :return: 解析后的 Python 对象（通常是字典或列表），如果读取失败则返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用 yaml.safe_load 安全地加载 YAML 文件内容
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        raise FileNotFoundError
    except yaml.YAMLError as e:
        print(f"错误：解析 YAML 文件 {file_path} 时出错 - {e}")
        raise e
    except yaml.parser.ParserError as e:
        print(f"错误：YAML 文件 {file_path} 解析错误 - {e}")
        raise e
    except Exception as e:
        print(f"发生未知错误：{e}")
        raise e


def load_json_data(file_path, default=None):
    """从文件加载JSON数据，如果文件不存在则返回默认值。"""
    try:
        if os.path.exists(file_path):
            print(f"Use cache from: {file_path}")
            with open(file_path, 'r') as file:
                return json.load(file)
    except Exception as e:
        print(f"Error loading JSON data from {file_path}: {e}")
    return default


def save_json_data(data, file_path: str) -> None:
    """
    将数据保存到 JSON 文件，自动创建目录（如果不存在）。

    Args:
        data: 要保存的数据
        file_path (str): JSON 文件的保存路径

    Raises:
        ValueError: 如果保存 JSON 文件时发生错误
    """
    try:
        # 获取文件所在的目录
        directory = os.path.dirname(file_path)

        # 如果目录不为空且不存在，则创建目录
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

        # 保存 JSON 文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Data has been converted to JSON and saved to {file_path}")

    except Exception as e:
        raise ValueError(f"Error saving JSON data to {file_path}: {e}")


def save_model_weights(model, optimizer=None, epoch=None, loss=None, save_dir='./checkpoints',
                       file_prefix='model', save_type='weights'):
    """
    保存模型权重或检查点的函数

    :param model: 要保存权重的模型
    :param optimizer: 优化器（如果需要保存优化器状态，默认为None）
    :param epoch: 当前训练的epoch数（如果需要保存相关信息，默认为None）
    :param loss: 当前epoch的损失值（如果需要保存相关信息，默认为None）
    :param save_dir: 保存文件的目录，默认为'./checkpoints'
    :param file_prefix: 保存文件的前缀，默认为'model'
    :param save_type: 保存类型，可选'weights'（仅保存模型权重）或'checkpoint'（保存模型权重、优化器状态等信息）
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_type == 'weights':
        file_path = f"{save_dir}/{file_prefix}_weights.pth"
        torch.save(model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")
    elif save_type == 'checkpoint':
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss

        file_path = f"{save_dir}/{file_prefix}_checkpoint.pth"
        torch.save(checkpoint, file_path)
        print(f"Model checkpoint saved to {file_path}")
    else:
        raise ValueError(
            "save_type should be either 'weights' or 'checkpoint'")

def load_model_weights(model, load_path, optimizer=None, device=None):
    """
    加载模型权重或检查点的函数，并支持指定设备加载

    :param model: 要加载权重的模型
    :param load_path: 模型权重或检查点的加载路径
    :param optimizer: 优化器（如果加载检查点且需要恢复优化器状态，默认为None）
    :param device: 设备（'cpu' 或 'cuda' 或 torch.device 对象，默认为None，表示使用模型当前设备）
    :return: 加载权重后的模型，可能还有加载状态后的优化器、epoch和loss（如果加载的是检查点）
    """
    # 检查文件是否存在
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"The file {load_path} does not exist.")

    # 如果 device 未指定，使用模型当前的设备
    if device is None:
        device = next(model.parameters()).device
        print(f"Using device: {device}")
    else:
        device = torch.device(device)
        print(f"Using specified device: {device}")

    # 加载检查点
    checkpoint = torch.load(load_path, map_location=device)
    # 将模型移到指定设备
    model.to(device)
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from {load_path} on {device}")

    # 如果有优化器状态且优化器不为空，加载优化器状态
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 确保优化器参数与模型在同一设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Optimizer state loaded from {load_path} on {device}")

    # 加载 epoch 和 loss（如果存在）
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        print(f"Epoch information loaded: {epoch}")
    else:
        epoch = None

    if 'loss' in checkpoint:
        loss = checkpoint['loss']
        print(f"Loss information loaded: {loss}")
    else:
        loss = None

    if optimizer is not None:
        return model, optimizer, epoch, loss
    else:
        return model


def clear_directory(save_dir='./checkpoints'):
    """
    清空指定目录下的所有文件和子目录，但保留目录本身。

    :param save_dir: 要清空的目录路径，默认为 './checkpoints'
    """
    if os.path.exists(save_dir):
        for item in os.listdir(save_dir):
            item_path = os.path.join(save_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # 删除文件
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # 删除目录
                print(f"Removed: {item_path}")
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        print(f"Directory {save_dir} does not exist.")


def copy_file(
    source_path: str,
    destination_path: str,
) -> bool:
    """
    复制文件到目标路径，带日志记录和异常处理。

    Args:
        source_path (str): 源文件路径
        destination_path (str): 目标文件路径

    Returns:
        bool: 复制是否成功，成功返回 True，失败返回 False
    """

    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)

    try:
        shutil.copy2(source_path, destination_path)
        print(f"File copied successfully to {destination_path}")
        return True
    except FileNotFoundError as e:
        print(f"Failed to copy file: {source_path} not found. Error: {e}")
        return False
    except Exception as e:
        print(f"Failed to copy file to {destination_path}: {e}")
        return False


def is_valid_json(text: Union[str, bytes]) -> bool:
    """
    判断文本内容是否满足 JSON 格式。

    Args:
        text (Union[str, bytes]): 要检查的文本内容，可以是字符串或字节

    Returns:
        bool: 如果文本是有效的 JSON 格式，返回 True；否则返回 False
    """
    try:
        # 如果输入是 bytes，解码为字符串
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # 尝试解析 JSON
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
