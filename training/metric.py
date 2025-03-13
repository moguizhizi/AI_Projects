from collections import defaultdict
from typing import Tuple, Dict, Optional, List, Any
import torch


def step_actions_bert_classifier_metric(results: Dict):
    predict = results["predict"]
    label = results["label"]

    metric_count = defaultdict(int)

    num_success = int(torch.sum(predict == label).item())
    num_failed = len(predict) - num_success
    print(f"success:{num_success}")
    print(f"failed:{num_failed}")

    num_zero = int(torch.sum(predict == 0).item())
    num_one = int(torch.sum(predict == 1).item())
    num_two = int(torch.sum(predict == 2).item())
    num_three = int(torch.sum(predict == 3).item())
    num_four = int(torch.sum(predict == 4).item())

    metric_count["success"] = num_success
    metric_count["failed"] = num_failed
    metric_count["循环操作"] = num_zero
    metric_count["取数"] = num_one
    metric_count["等待时长"] = num_two
    metric_count["多app"] = num_three
    metric_count["其他"] = num_four

    return metric_count


def step_actions_bert_classifier_print(state: Dict):
    all_dict = defaultdict(int)
    for key, value in state.items():
        all_dict[key] = sum(value)

    all_success = all_dict["success"]
    all_failed = all_dict["failed"]
    all_zero = all_dict["循环操作"]
    all_one = all_dict["取数"]
    all_two = all_dict["等待时长"]
    all_three = all_dict["多app"]
    all_four = all_dict["其他"]

    total_samples = all_success + all_failed

    # 输出统计结果
    print(f"总样本数: {all_success+all_failed}")
    print(f"成功数量: {all_success}, 成功率: {all_success / total_samples * 100:.2f}%")
    print(f"失败数量: {all_failed}, 失败率: {all_failed / total_samples * 100:.2f}%")
    print(f"类别 循环操作 数量: {all_zero}, 占比: {all_zero / total_samples * 100:.2f}%")
    print(f"类别 取数 数量: {all_one}, 占比: {all_one / total_samples * 100:.2f}%")
    print(f"类别 等待时长 数量: {all_two}, 占比: {all_two / total_samples * 100:.2f}%")
    print(
        f"类别 多app 数量: {all_three}, 占比: {all_three / total_samples * 100:.2f}%")
    print(f"类别 其他 数量: {all_four}, 占比: {all_four / total_samples * 100:.2f}%")

    return {"accuracy": round(all_success / total_samples, 2)}


def step_actions_bert_classifier_comparison(metric1: Dict, metric2: Dict) -> bool:
    if metric1["accuracy"] > metric2["accuracy"]:
        return True
    else:
        return False


def step_actions_en_bert_classifier_metric(results: Dict):
    predict = results["predict"]
    label = results["label"]

    metric_count = defaultdict(int)

    num_success = int(torch.sum(predict == label).item())
    num_failed = len(predict) - num_success
    print(f"success:{num_success}")
    print(f"failed:{num_failed}")

    num_zero = int(torch.sum(predict == 0).item())
    num_one = int(torch.sum(predict == 1).item())
    num_two = int(torch.sum(predict == 2).item())
    num_three = int(torch.sum(predict == 3).item())
    num_four = int(torch.sum(predict == 4).item())

    metric_count["success"] = num_success
    metric_count["failed"] = num_failed
    metric_count["循环操作"] = num_zero
    metric_count["取数"] = num_one
    metric_count["等待时长"] = num_two
    metric_count["多app"] = num_three
    metric_count["其他"] = num_four

    return metric_count


def automotive_user_opinions_bert_multi_task_classifier(results: Dict):
    predictions = results["predict"]
    labels = results["label"]

    pred_category_id = predictions["multi-label"]
    pred_emotion_id = predictions["single-label"]

    category_id = labels["multi-label"]
    emotion_id = labels["single-label"]

    num_category_hit = torch.sum(pred_category_id & category_id, dim=0)
    num_emotion_hit = int(torch.sum(pred_emotion_id == emotion_id))
    num_batch_category = torch.sum(category_id, dim=0)
    num_batch_emotion = len(emotion_id)

    metric_count = {}
    metric_count["category_hit"] = num_category_hit.tolist()
    metric_count["emotion_hit"] = num_emotion_hit
    metric_count["category"] = num_batch_category.tolist()
    metric_count["emotion"] = num_batch_emotion

    return metric_count


def step_actions_en_bert_classifier_print(state: Dict):
    all_dict = defaultdict(int)
    for key, value in state.items():
        all_dict[key] = sum(value)

    all_success = all_dict["success"]
    all_failed = all_dict["failed"]
    all_zero = all_dict["循环操作"]
    all_one = all_dict["取数"]
    all_two = all_dict["等待时长"]
    all_three = all_dict["多app"]
    all_four = all_dict["其他"]

    total_samples = all_success + all_failed

    # 输出统计结果
    print(f"总样本数: {all_success+all_failed}")
    print(f"成功数量: {all_success}, 成功率: {all_success / total_samples * 100:.2f}%")
    print(f"失败数量: {all_failed}, 失败率: {all_failed / total_samples * 100:.2f}%")
    print(f"类别 循环操作 数量: {all_zero}, 占比: {all_zero / total_samples * 100:.2f}%")
    print(f"类别 取数 数量: {all_one}, 占比: {all_one / total_samples * 100:.2f}%")
    print(f"类别 等待时长 数量: {all_two}, 占比: {all_two / total_samples * 100:.2f}%")
    print(
        f"类别 多app 数量: {all_three}, 占比: {all_three / total_samples * 100:.2f}%")
    print(f"类别 其他 数量: {all_four}, 占比: {all_four / total_samples * 100:.2f}%")

    return {"accuracy": round(all_success / total_samples, 2)}


def automotive_user_opinions_bert_multi_task_print(state: Dict):

    all_category_hit = torch.LongTensor(state["category_hit"])
    all_category_hit = torch.sum(all_category_hit, dim=0)
    all_category = torch.LongTensor(state["category"])
    all_category = torch.sum(all_category, dim=0)

    all_emotion_hit = sum(state["emotion_hit"])
    all_emotion = sum(state["emotion"])

    # 避免除以零：当 all_category 为 0 时，返回 0
    mask = all_category != 0
    result = torch.where(mask, all_category_hit /
                         all_category, torch.tensor(0.0))
    category_accuracy = round(float(torch.mean(result)), 2)

    emotion_accuracy = round(all_emotion_hit / all_emotion, 2)

    return {"accuracy": round((category_accuracy + emotion_accuracy) / 2, 2)}


def step_actions_en_bert_classifier_comparison(metric1: Dict, metric2: Dict) -> bool:
    if metric1["accuracy"] > metric2["accuracy"]:
        return True
    else:
        return False
    
def automotive_user_opinions_bert_multi_task_comparison(metric1: Dict, metric2: Dict) -> bool:
    if metric1["accuracy"] > metric2["accuracy"]:
        return True
    else:
        return False


def select_best_checkpoint(checkpoint_metrics, comparison_func) -> Tuple[Optional[str], Optional[Dict]]:
    """
    从多个检查点的评估结果中筛选最佳检查点。

    Args:
        checkpoint_metrics: 列表，元素为 (checkpoint_path, eval_metric)，其中 eval_metric 是评估结果字典
        comparison_func: 比较函数，接受两个 eval_metric 字典，返回 True 表示第一个优于第二个

    Returns:
        Tuple[str, Dict]: 最佳检查点路径和对应的评估结果，如果无有效结果则返回 (None, None)
    """
    if not checkpoint_metrics:
        print("No checkpoint metrics provided for selection.")
        return None, None

    # 初始最佳检查点
    best_checkpoint, best_metric = checkpoint_metrics[0]

    # 比较并筛选最佳检查点
    for checkpoint_path, metric in checkpoint_metrics[1:]:
        try:
            if comparison_func(metric, best_metric):
                best_checkpoint, best_metric = checkpoint_path, metric
        except Exception as e:
            print(f"Comparison failed for checkpoint {checkpoint_path}: {e}")
            continue

    return best_checkpoint, best_metric


def step_actions_bert_classifier_predict(batch: Dict[str, Any], results: Dict[str, Any], id2label: Dict[int, str]) -> List:
    predict = results["predict"]

    predict_list = []
    for id in predict.tolist():
        predict_list.append(id2label[id])

    return zip(predict_list, batch["index"])


def step_actions_en_bert_classifier_predict(batch: Dict[str, Any], results: Dict[str, Any], id2label: Dict[int, str]) -> List:
    predict = results["predict"]

    predict_list = []
    for id in predict.tolist():
        predict_list.append(id2label[id])

    return zip(predict_list, batch["index"])
