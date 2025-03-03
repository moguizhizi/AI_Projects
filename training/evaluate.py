from collections import defaultdict
from training.metric import select_best_checkpoint
from utils.file_utils import copy_file, load_model_weights
from data import load_dataset, data_loader, load_collate
from models import load_model
from training import load_metric, load_print, load_comparison
import glob
import os
import torch


def test(dataset_dict, config_dict, args):
    dataset_name = args.dataset
    model_name = args.model_name

    test_config = config_dict[dataset_name]["test"]

    device = config_dict[dataset_name]["device"]

    dataset = load_dataset[dataset_name](
        dataset_dict[dataset_name]["processed"]["test"], config_dict[dataset_name], model_name)
    collate_fn = load_collate[dataset_name]
    dataloader = data_loader[dataset_name](dataset, test_config, collate_fn)

    metric = load_metric[dataset_name][model_name]
    print_callback = load_print[dataset_name][model_name]
    comparison = load_comparison[dataset_name][model_name]

    model = load_model[model_name](config_dict[dataset_name], args)
    model.to(device)

    checkpoints_dir = os.path.join(
        "AI_Projects", "checkpoints", "model_checkpoints", dataset_name, model_name, "normal")
    # 获取所有检查点文件的路径列表
    all_checkpoint_paths = glob.glob(
        os.path.join(checkpoints_dir, f"epoch*.pth"))
    # 按修改时间倒序排序
    all_checkpoint_paths.sort(key=os.path.getmtime, reverse=True)

    # 判断是否为空
    if not all_checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoints_dir}")

    # 评估所有检查点
    checkpoint_metrics = []
    for single_checkpoint_path in all_checkpoint_paths:
        model = load_model_weights(model, single_checkpoint_path)
        model.eval()

        eval_metirc = evaluate_checkpoint(model, dataloader, metric,
                                          single_checkpoint_path, device, print_callback)

        checkpoint_metrics.append((single_checkpoint_path, eval_metirc))

    best_checkpoint, best_metric = select_best_checkpoint(
        checkpoint_metrics, comparison)
    for key, value in best_metric.items():
        print(f"best {key}:{value}")

    print(f"best checkpoint name: {os.path.basename(best_checkpoint)}")

    save_dir = os.path.join(
        "AI_Projects", "checkpoints", "model_checkpoints", dataset_name, model_name, "best")
    os.makedirs(save_dir, exist_ok=True)

    destination_path = os.path.join(save_dir, "best.pth")

    copy_file(best_checkpoint, destination_path)


def evaluate_checkpoint(model, dataloader, metric_func, checkpoint_path, device, print_callback=None):
    """
    评估单个检查点的性能，支持动态指标。

    Args:
        model: 推理模型
        dataloader: 数据加载器
        metric_func: 评估函数，返回字典形式的指标
        checkpoint_path: 检查点文件路径
        device: 推理设备
        print_callback: 自定义打印回调（可选）

    Returns:
        dict: 统计结果
    """
    # 加载检查点权重
    try:
        model = load_model_weights(model, checkpoint_path)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")

    model.to(device)
    model.eval()

    # 初始化统计
    stats = defaultdict(list)

    # 推理
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            try:
                # 移动 batch 到设备
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(
                        v) else v for k, v in batch.items()}
                elif torch.is_tensor(batch):
                    batch = batch.to(device)

                # 模型推理
                results = model.inference(batch)
                metrics = metric_func(results=results)

                # 验证指标
                if not isinstance(metrics, dict):
                    raise ValueError(
                        "metric_func must return a dict with 'success' and 'failed' keys")

                for key, value in metrics.items():
                    stats[key].append(value)

            except Exception as e:
                raise ValueError(f"Batch {batch_index} failed: {e}")

    eval_metirc = print_callback(stats)

    return eval_metirc
