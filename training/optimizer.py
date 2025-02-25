from torch.optim import AdamW


class CustomAdamW(AdamW):
    def __init__(self, dataset_dict, config_dict, params):

        optimizer_type = config_dict["optimizer_type"]
        learning_rate = config_dict["Optimizer"][optimizer_type]["lr"]
        weight_decay = config_dict["Optimizer"][optimizer_type]["weight_decay"]

        super().__init__(params, lr=learning_rate, weight_decay=weight_decay)
