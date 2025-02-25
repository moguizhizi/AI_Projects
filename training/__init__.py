
from training.optimizer import CustomAdamW
from training.loss_function import CustomCrossEntropyLoss


load_loss_func = {
    "cross_entropy_loss": CustomCrossEntropyLoss
}

load_optimizer = {
    "adamW": CustomAdamW
}
