from web_service.network import start_web_service
from configs.args import get_args
from utils.file_utils import read_yaml_file

if __name__ == "__main__":

    network_dict = read_yaml_file("AI_Projects/configs/network.yaml")
    args = get_args()

    start_web_service(network_dict, args)
