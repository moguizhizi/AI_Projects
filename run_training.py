from configs.args import get_args
from utils.file_utils import read_yaml_file
from training.train import train

if __name__ == "__main__":

    dataset_dict = read_yaml_file(
        "AI_Projects/configs/dataset.yaml")
    config_dict = read_yaml_file("AI_Projects/configs/config.yaml")
    args = get_args()

    train(dataset_dict, config_dict, args)
