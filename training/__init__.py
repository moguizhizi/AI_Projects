
from training.optimizer import CustomAdamW
from training.loss_function import CustomCrossEntropyLoss
from training.metric import *


load_loss_func = {
    "cross_entropy_loss": CustomCrossEntropyLoss
}

load_optimizer = {
    "adamW": CustomAdamW
}

load_metric = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_metric},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_metric},
}


load_print = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_print},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_print},
}

load_comparison = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_comparison},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_comparison},
}
