import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, BertModel


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
        return {"multi-label": category_logits, "single-label": emotion_logits}, {"multi-label": category_id, "single-label": emotion_id}, [{"name": "multi-label", "type": "bce"}, {"name": "single-label", "type": "ce"}],
