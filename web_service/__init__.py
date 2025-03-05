from web_service.data_handler import step_actions_normalize, step_actions_en_normalize
from web_service.responder import step_actions_bert_classifier_response, step_actions_en_bert_classifier_response


response_predictor_map = {
    "StepActions":
        {"bert_classifier": step_actions_bert_classifier_response},
    "StepActions_En":
        {"bert_classifier": step_actions_en_bert_classifier_response},
}

data_normalizer_map = {
    "StepActions": step_actions_normalize,

    "StepActions_En": step_actions_en_normalize
}
