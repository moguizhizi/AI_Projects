from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from utils.file_utils import load_model_weights, read_yaml_file
from models import load_model
import os
import torch
import copy
from data import load_dataset, data_loader, load_collate
from training import load_resultor
from web_service.requestor import ClassifierRequest
from web_service.responder import ClassifierResponse
from web_service import response_predictor_map, data_normalizer_map


config_dict = read_yaml_file("AI_Projects/configs/network.yaml")


# 全局模型管理（支持动态 dataset_name）
class ModelManager:
    _models = {}  # 缓存：dataset_name -> 模型
    _devices = {}  # 缓存：dataset_name -> 设备

    @classmethod
    def get_or_load_model(cls, dataset_name: str, model_name: str = "bert_classifier"):
        """
        获取或加载模型（基于 dataset_name）。

        Args:
            dataset_name (str): 数据集名称
            model_name (str): 模型名称，默认为 bert_classifier

        Returns:
            Tuple[model, device]: 模型和设备
        """
        if dataset_name not in cls._models.keys():
            try:
                # 获取设备
                device = config_dict[dataset_name]["device"]

                # 加载模型
                model = load_model[model_name](config_dict[dataset_name])

                # 加载最佳检查点
                best_dir = os.path.join(
                    "AI_Projects", "checkpoints", "model_checkpoints", dataset_name, model_name, "best")
                best_checkpoint = os.path.join(best_dir, "best.pth")

                try:
                    model = load_model_weights(model, best_checkpoint)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load checkpoint {best_checkpoint}: {e}")

                # 移动模型到设备并设置为评估模式
                model.to(device)
                model.eval()

                # 缓存模型和设备
                cls._models[dataset_name] = model
                cls._devices[dataset_name] = device

                print(
                    f"Model {model_name} for dataset {dataset_name} loaded and moved to {device}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model for dataset {dataset_name}: {e}")

        return cls._models[dataset_name], cls._devices[dataset_name]


class ResponseData(BaseModel):
    code: int
    msg: str
    data: List[Dict[str, Any]]


def process_data_and_predict(
    original_data: List[Dict[str, Any]],
    normalized_data: List[Dict[str, Any]],
    dataset_name: str,
    model_name: str,
    model: Any,
    device: torch.device,
    id2label: Dict[int, str],
    config_dict: Dict[str, Any],
    load_dataset: Dict[str, Any],
    load_collate: Dict[str, Any],
    data_loader: Dict[str, Any],
    load_resultor: Dict[str, Any],
    response_predictor_map: Dict[str, Any]
) -> ClassifierResponse:
    """
    处理数据并进行预测，返回分类响应。

    Args:
        data (List[Dict[str, Any]]): 输入数据
        dataset_name (str): 数据集名称
        model_name (str): 模型名称
        model (Any): 预加载的模型
        device (torch.device): 设备（CPU 或 GPU）
        id2label (Dict[int, str]): ID 到标签的映射
        config_dict (Dict[str, Any]): 配置字典
        load_dataset (Dict[str, Any]): 数据集加载函数映射
        load_collate (Dict[str, Any]): 批处理函数映射
        data_loader (Dict[str, Any]): DataLoader 加载函数映射
        load_resultor (Dict[str, Any]): 结果处理器映射
        response_predictor_map (Dict[str, Any]): 响应预测器映射

    Returns:
        ClassifierResponse: 分类响应对象

    Raises:
        ValueError: 如果批处理失败
    """
    try:
        # 创建数据集
        dataset = load_dataset[dataset_name](
            config_dict=config_dict[dataset_name],
            model_name=model_name,
            data=normalized_data
        )

        # 获取批处理函数和 DataLoader
        collate_fn = load_collate[dataset_name]
        dataloader = data_loader[dataset_name](
            dataset,
            config_dict[dataset_name],
            collate_fn
        )

        # 获取结果处理器和响应构造器
        resultor = load_resultor[dataset_name][model_name]
        constructor = response_predictor_map[dataset_name][model_name]
        

        # 预测结果列表
        predict_list = []
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

                    # 处理预测结果
                    predict_list.extend(resultor(batch, results, id2label))

                except Exception as e:
                    print(f"Batch {batch_index} failed: {e}")
                    predict_list = []

        # 构造响应
        result = constructor(original_data, predict_list)
        
        if result["success"] is True:
            response = ClassifierResponse(
                code=200, msg="success", data=result["data"])
        else:
            response = ClassifierResponse(
                code=500, msg="failed", data=result["data"])
        return response

    except Exception as e:
        print(f"Error in processing data and predicting: {e}")
        response = ClassifierResponse(code=500, msg="failed", data=[])


def create_ai_app():
    app = FastAPI()

    @app.post("/ai/api/classifier")
    async def classifier_case(request: ClassifierRequest):
        try:
            print("="*50)
            print(f"Received data num: {len(request.data)}")
            
            request_copy = copy.deepcopy(request)

            data = request_copy.data
            dataset_name = request_copy.DataSet
            normalized_data = data_normalizer_map[dataset_name](config_dict, data)
            if dataset_name == "StepActions":
                dataset_name = "StepActions_En"
            id2label = config_dict[dataset_name]["id2label"]
            model_name = "bert_classifier"

            # 获取或加载模型
            model, device = ModelManager.get_or_load_model(
                dataset_name, model_name)            
            

            # 调用提取的函数
            response = process_data_and_predict(
                original_data=request.data,
                normalized_data = normalized_data,
                dataset_name=dataset_name,
                model_name=model_name,
                model=model,
                device=device,
                id2label=id2label,
                config_dict=config_dict,
                load_dataset=load_dataset,
                load_collate=load_collate,
                data_loader=data_loader,
                load_resultor=load_resultor,
                response_predictor_map=response_predictor_map
            )
            return response

        except Exception as e:
            print(f"Error occurred: {e}")
            return ClassifierResponse(code=500, msg="failed", data=[])

    return app
