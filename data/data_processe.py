from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.file_utils import load_json_data


class StepActionsDataset(Dataset):
    def __init__(self, dataset_path, config_dict, model_name=""):
        super().__init__()
        self.dataset = load_json_data(dataset_path)
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

            return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"], "label": test_entry["class_id"]}
        else:
            return test_entry

    def __len__(self):
        return len(self.dataset)
