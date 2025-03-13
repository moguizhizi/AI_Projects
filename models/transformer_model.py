import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, BertModel
from utils import to_multi_hot_sklearn


class BertClassifier(nn.Module):
    def __init__(self, config_dict, args=None):
        super(BertClassifier, self).__init__()
        self.device = config_dict["device"]

        self.model_ckpt = config_dict["model_ckpt"].get("bert_classifier")

        if self.model_ckpt != None:
            self.bert = BertForSequenceClassification.from_pretrained(
                self.model_ckpt, num_labels=config_dict["num_class"])

        else:
            # 抛出异常提示用户指定模型检查点
            raise ValueError(
                "Please specify a valid model checkpoint in the config_dict['model_ckpt']['bert_classifier'].")

    def forward(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(
            batch["attention_mask"]).to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        return logits, labels, None

    def inference(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(
            batch["attention_mask"]).to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        return {"predict": predictions, "label": labels}


class BertMultiTask(nn.Module):
    def __init__(self, config_dict, args=None):
        super(BertMultiTask, self).__init__()

        self.device = config_dict["device"]
        self.model_ckpt = config_dict["model_ckpt"].get(
            "bert_multi_task_classifier")
        self.num_categories = config_dict["num_category_class"]
        self.num_emotions = config_dict["num_emotion_class"]
        self.threshold = config_dict["multi_label_threshold"]
        self.num_category_class = config_dict["num_category_class"]

        if self.model_ckpt != None:
            self.bert = BertModel.from_pretrained(self.model_ckpt)

        else:
            # 抛出异常提示用户指定模型检查点
            raise ValueError(
                "Please specify a valid model checkpoint in the config_dict['model_ckpt']['bert_classifier'].")

        self.dropout = nn.Dropout(0.1)
        self.category_classifier = nn.Linear(768, self.num_categories)
        self.emotion_classifier = nn.Linear(768, self.num_emotions)

    def forward(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(
            batch["attention_mask"]).to(self.device)
        category_id = torch.FloatTensor(batch["category_id"]).to(self.device)
        emotion_id = torch.FloatTensor(batch["emotion_id"]).to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)

        logits = {"multi-label": category_logits,
                  "single-label": emotion_logits}
        labels = {"multi-label": category_id, "single-label": emotion_id}
        task_config = [{"name": "multi-label", "type": "bce"},
                       {"name": "single-label", "type": "ce"}]

        return logits, labels, task_config,

    def inference(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(
            batch["attention_mask"]).to(self.device)
        category_id = torch.LongTensor(batch["category_id"]).to(self.device)
        emotion_id = torch.LongTensor(batch["emotion_id"]).to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)

        pred_emotion_id = torch.argmax(emotion_logits, dim=-1)
        pred_category_id = self.get_category_indices(
            category_logits, self.threshold)
        pred_category_id = to_multi_hot_sklearn(
            pred_category_id, self.num_category_class)
        pred_category_id = torch.LongTensor(pred_category_id).to(self.device)

        predictions = {}
        predictions["multi-label"] = pred_category_id
        predictions["single-label"] = pred_emotion_id

        labels = {}
        labels["multi-label"] = category_id
        labels["single-label"] = emotion_id

        return {"predict": predictions, "label": labels}

    def get_category_indices(self, category_logits, threshold=0.5):
        """
        计算 category_logits 中每行中大于指定阈值的索引，如果没有则返回最大值的索引。

        参数:
            category_logits (torch.Tensor): 输入的 logits 张量，形状为 (N, C)，N 为样本数，C 为类别数。
            threshold (float, optional): 阈值，默认值为 0.5。

        返回:
            result_list (list): 每个样本的索引列表，如果有大于阈值的索引则返回这些索引，否则返回最大值的索引。
        """
        # 应用 softmax，dim=-1 表示在最后一个维度（类别维度）上归一化
        category_logits = F.softmax(category_logits, dim=-1)

        # 找到每行中大于 threshold 的索引
        mask = category_logits > threshold
        indices = torch.where(mask)

        # 按行组织大于 threshold 的索引
        greater_than_threshold = {}
        for row, col in zip(indices[0], indices[1]):
            row = row.item()
            col = col.item()
            if row not in greater_than_threshold:
                greater_than_threshold[row] = []
            greater_than_threshold[row].append(col)

        # 找到每行最大值的索引
        max_indices = torch.argmax(category_logits, dim=-1)

        # 构造最终结果：如果有大于 threshold 的值，使用这些索引；否则使用最大值的索引
        result = {}
        for row in range(category_logits.size(0)):
            if row in greater_than_threshold:
                # 如果有大于 threshold 的值，使用这些索引
                result[row] = greater_than_threshold[row]
            else:
                # 否则，使用最大值的索引
                result[row] = [max_indices[row].item()]

        # 输出每行的结果
        # print("\n每行的索引（大于 {:.1f} 或最大值）：".format(threshold))
        # for row in range(category_logits.size(0)):
        #     print(f"第 {row} 行：{result[row]}")

        # 转换为列表形式的结果
        result_list = [result[row] for row in range(category_logits.size(0))]
        print("\n结果列表：", result_list)

        return result_list
