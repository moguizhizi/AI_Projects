
from training.optimizer import CustomAdamW
from training.loss_function import CustomCrossEntropyLoss, CrossBCEWithLogitsLoss
from training.metric import *


load_loss_func = {
    "cross_entropy_loss": CustomCrossEntropyLoss,
    "cross_bce_loss": CrossBCEWithLogitsLoss,
}

load_optimizer = {
    "adamW": CustomAdamW,
}

load_metric = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_metric},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_metric},
    "AutomotiveUserOpinions":
        {"bert_multi_task_classifier": automotive_user_opinions_bert_multi_task_classifier},
}


load_print = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_print},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_print},
    "AutomotiveUserOpinions":
        {"bert_multi_task_classifier": automotive_user_opinions_bert_multi_task_print},
}

load_comparison = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_comparison},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_comparison},
    "AutomotiveUserOpinions":
        {"bert_multi_task_classifier": automotive_user_opinions_bert_multi_task_comparison},
}

load_resultor = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_predict},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_predict},
}
