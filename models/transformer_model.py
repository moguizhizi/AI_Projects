import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    def __init__(self, config_dict, args):
        super(BertClassifier, self).__init__()
        self.config = config_dict[args.dataset]
        self.device = self.config["device"]
        # self.bert = AutoModel.from_pretrained("bert-base-chinese")
        self.bert = AutoModel.from_pretrained("/data1/huggingface/models--bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size,
                            self.config["num_class"])

    def forward(self, batch):
        input_ids = torch.LongTensor(batch["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(batch["attention_mask"]).to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(self.dropout(pooled_output)), labels
