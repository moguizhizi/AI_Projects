import argparse
from typing import Dict, Callable

def parse_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    定义所有任务共享的通用参数。
    """
    return parser

# 任务特定的参数解析函数
def parse_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    为 model 任务定义特定参数。
    """
    parser = parse_common_args(parser)  # 加载通用参数
    parser.add_argument("--dataset", type=str,
                        required=True, default="StepActions")
    parser.add_argument("--model_name", type=str,
                        required=True, default="bert_classifier")
    return parser

def parse_network_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    为 network 任务定义特定参数。
    """
    parser = parse_common_args(parser)  # 加载通用参数
    parser.add_argument("--model_name", type=str,
                        required=True, default="bert_classifier")
    return parser

# 任务映射字典
TASK_ARGS_MAPPING: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    "model": parse_model_args,
    "network": parse_network_args
}

def parse_args() -> argparse.Namespace:
    """
    根据任务动态解析参数。

    Returns:
        argparse.Namespace: 解析后的参数
    """
    # 初始解析器
    parser = argparse.ArgumentParser(
        description="Dynamic argument parser for different tasks")

    # 添加任务参数
    parser.add_argument("--task", type=str, required=True,
                        choices=TASK_ARGS_MAPPING.keys(), help="Task name")

    # 解析任务参数（仅获取 task）
    args, remaining_args = parser.parse_known_args()

    # 根据任务加载对应的参数解析器
    task_parser_func = TASK_ARGS_MAPPING.get(args.task)
    if task_parser_func is None:
        raise ValueError(
            f"Unknown task: {args.task}. Supported tasks: {list(TASK_ARGS_MAPPING.keys())}")
    
    # 重新定义解析器，加载任务特定参数
    parser = argparse.ArgumentParser(
        description=f"Arguments for task {args.task}")
    parser = task_parser_func(parser)

    # 重新解析剩余参数
    return parser.parse_args(remaining_args)

def get_args() -> argparse.Namespace:
    """
    获取解析后的参数。

    Returns:
        argparse.Namespace: 解析后的参数
    """
    return parse_args()