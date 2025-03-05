from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.file_utils import load_json_data
from typing import List, Dict, Any


class StepActionsDataset(Dataset):
    def __init__(self, dataset_path="", config_dict=None, model_name="", data: List[Dict[str, Any]] = []):
        super().__init__()
        if len(dataset_path) != 0:
            self.dataset = load_json_data(dataset_path)
        else:
            self.dataset = data
        self.model_name = model_name

        self.model_ckpt = config_dict["model_ckpt"].get(model_name)

        if self.model_ckpt != None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
            self.max_length = self.tokenizer.model_max_length
        else:
            self.tokenizer = None
            self.max_length = 0

    def __getitem__(self, index):
        test_entry = self.dataset[index]
        if self.model_ckpt != None:
            encoding = self.tokenizer(
                test_entry["StepActions"], max_length=self.max_length, truncation=True, padding="max_length")

            return {"index": index, "input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"], "label": test_entry["class_id"]}
        else:
            return test_entry

    def __len__(self):
        return len(self.dataset)


class StepActionsEnDataset(Dataset):
    def __init__(self, dataset_path="", config_dict=None, model_name="", data: List[Dict[str, Any]] = []):
        super().__init__()
        if len(dataset_path) != 0:
            self.dataset = load_json_data(dataset_path)
        else:
            self.dataset = data
        self.model_name = model_name

        self.model_ckpt = config_dict["model_ckpt"].get(model_name)

        if self.model_ckpt != None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
            self.max_length = self.tokenizer.model_max_length
        else:
            self.tokenizer = None
            self.max_length = 0

    def __getitem__(self, index):
        test_entry = self.dataset[index]
        if self.model_ckpt != None:
            encoding = self.tokenizer(
                test_entry["StepActions"], max_length=self.max_length, truncation=True, padding="max_length")

            return {"index": index, "input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"], "label": test_entry["class_id"]}
        else:
            return test_entry

    def __len__(self):
        return len(self.dataset)
