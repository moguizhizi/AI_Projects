import yaml
import json
import os

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

def save_json_data(data, file_path):
    """将数据保存到JSON文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Data has been converted to JSON and saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON data to {file_path}: {e}")
        

import torch


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
    import os
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
        raise ValueError("save_type should be either 'weights' or 'checkpoint'")