import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, config_dict, args):
        super(BertClassifier, self).__init__()
        self.device = config_dict["device"]

        self.model_ckpt = config_dict["model_ckpt"].get("bert_classifier")

        if self.model_ckpt != None:
            self.bert = BertForSequenceClassification.from_pretrained(
                self.model_ckpt, num_labels=config_dict["num_class"])
        else:
            # �׳��쳣��ʾ�û�ָ��ģ�ͼ���
            raise ValueError(
                "Please specify a valid model checkpoint in the config_dict['model_ckpt']['bert_classifier'].")

    def forward(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(
            batch["attention_mask"]).to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        return logits, labels

    def inference(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(
            batch["attention_mask"]).to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        return {"predict": predictions, "label": labels}
