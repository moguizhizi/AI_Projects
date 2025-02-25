from utils.file_utils import load_json_data, save_json_data


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
