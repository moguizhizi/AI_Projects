
from utils.file_utils import load_json_data, save_json_data
from typing import Dict, List


class BasePreprocessor:
    def __init__(self):
        pass

    def preprocess(self):
        raise NotImplementedError("Subclasses should implement this method.")


class StepActionsPreprocessor(BasePreprocessor):
    def __init__(self, dataset_dict):
        super().__init__()
        self.dataset_dict = dataset_dict

    def preprocess(self):
        raw_train_path = self.dataset_dict["StepActions"]["raw"]["train"]
        raw_test_path = self.dataset_dict["StepActions"]["raw"]["test"]
        processed_train_path = self.dataset_dict["StepActions"]["processed"]["train"]
        processed_test_path = self.dataset_dict["StepActions"]["processed"]["test"]

        raw_train = load_json_data(raw_train_path)
        raw_test = load_json_data(raw_test_path)

        id2label = self.dataset_dict["StepActions"]["id2label"]
        label2id = {value: key for key, value in id2label.items()}

        processed_train = []
        for _, test_content in enumerate(raw_train):
            processed_test_content = {}
            class_name = test_content["class_name"]
            step_actions = test_content["StepActions"]
            summary = test_content["Summary"]
            class_id = label2id[class_name]

            processed_test_content["class_name"] = class_name
            processed_test_content["StepActions"] = step_actions
            processed_test_content["Summary"] = summary
            processed_test_content["class_id"] = class_id

            processed_train.append(processed_test_content)

        save_json_data(processed_train, processed_train_path)

        processed_test = []
        for _, test_content in enumerate(raw_test):
            processed_test_content = {}
            class_name = test_content["class_name"]
            step_actions = test_content["StepActions"]
            summary = test_content["Summary"]
            class_id = label2id[class_name]

            processed_test_content["class_name"] = class_name
            processed_test_content["StepActions"] = step_actions
            processed_test_content["Summary"] = summary
            processed_test_content["class_id"] = class_id

            processed_test.append(processed_test_content)

        save_json_data(processed_test, processed_test_path)


class StepActionsEnPreprocessor(BasePreprocessor):
    def __init__(self, dataset_dict):
        super().__init__()
        self.dataset_dict = dataset_dict

    def preprocess(self):
        raw_train_path = self.dataset_dict["StepActions_En"]["raw"]["train"]
        raw_test_path = self.dataset_dict["StepActions_En"]["raw"]["test"]
        processed_train_path = self.dataset_dict["StepActions_En"]["processed"]["train"]
        processed_test_path = self.dataset_dict["StepActions_En"]["processed"]["test"]

        raw_train = load_json_data(raw_train_path)
        raw_test = load_json_data(raw_test_path)

        id2label = self.dataset_dict["StepActions_En"]["id2label"]
        label2id = {value: key for key, value in id2label.items()}

        processed_train = []
        for _, test_content in enumerate(raw_train):
            processed_test_content = {}
            class_name = test_content["class_name"]
            step_actions = test_content["StepActions"]
            summary = test_content["Summary"]
            class_id = label2id[class_name]

            processed_test_content["class_name"] = class_name
            processed_test_content["StepActions"] = step_actions
            processed_test_content["Summary"] = summary
            processed_test_content["class_id"] = class_id

            processed_train.append(processed_test_content)

        save_json_data(processed_train, processed_train_path)

        processed_test = []
        for _, test_content in enumerate(raw_test):
            processed_test_content = {}
            class_name = test_content["class_name"]
            step_actions = test_content["StepActions"]
            summary = test_content["Summary"]
            class_id = label2id[class_name]

            processed_test_content["class_name"] = class_name
            processed_test_content["StepActions"] = step_actions
            processed_test_content["Summary"] = summary
            processed_test_content["class_id"] = class_id

            processed_test.append(processed_test_content)

        save_json_data(processed_test, processed_test_path)


class AutomotiveUserOpinionsPreprocessor(BasePreprocessor):
    def __init__(self, dataset_dict):
        super().__init__()
        self.dataset_dict = dataset_dict

    def preprocess(self):
        raw_train_path = self.dataset_dict["AutomotiveUserOpinions"]["raw"]["train"]
        raw_test_path = self.dataset_dict["AutomotiveUserOpinions"]["raw"]["test"]
        processed_train_path = self.dataset_dict["AutomotiveUserOpinions"]["processed"]["train"]
        processed_test_path = self.dataset_dict["AutomotiveUserOpinions"]["processed"]["test"]

        categoryid2label = self.dataset_dict["AutomotiveUserOpinions"]["id2label"]["category"]
        label2categoryid = {value: key for key,
                            value in categoryid2label.items()}

        train_content_with_ids = self.convert_to_id(
            label2categoryid, raw_train_path)
        test_content_with_ids = self.convert_to_id(
            label2categoryid, raw_test_path)

        save_json_data(train_content_with_ids, processed_train_path)
        save_json_data(test_content_with_ids, processed_test_path)

    def convert_to_id(self, label2categoryid: Dict[str, int], filename: str):
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_dict = {}
                chunk_list = line.strip().split("\t")
                categoryid_list = []
                emotion_id = -1
                for idx, chunk in enumerate(chunk_list):
                    if idx == 0:
                        chunk_dict["text"] = chunk
                    else:
                        categoryid = label2categoryid[chunk.split("#")[0]]
                        emotion_id = int(chunk.split("#")[1])+1
                        categoryid_list.append(categoryid)
                
                chunk_dict["category_id"] = categoryid_list
                chunk_dict["emotion_id"] = emotion_id
                
                data.append(chunk_dict)

        return data
