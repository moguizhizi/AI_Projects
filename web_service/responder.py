from pydantic import BaseModel
from typing import List, Dict, Any, Tuple





class ClassifierResponse(BaseModel):
    code: int
    msg: str
    data: List[Dict[str, Any]]
    
class DeepSeek_R1Response():
    def __init__(self, response:Dict[str, Any]):
        self.response = response
        
class Qwen2_72Response():
    def __init__(self, response:Dict[str, Any]):
        self.response = response
    
    def get_content(self):
        return self.response["choices"][0]["message"]["content"]
        
class Qwen2vlResponse():
    def __init__(self, response:Dict[str, Any]):
        self.response = response
        
class DeepSeekQwenResponse():
    def __init__(self, response:Dict[str, Any]):
        self.response = response
        
response_parser_map = {
    "qwen2_72b": Qwen2_72Response,
    "qwen2_vl": Qwen2vlResponse,
    "deepseek-qwen": DeepSeekQwenResponse,
    "DeepSeek-R1": DeepSeek_R1Response,
}


def step_actions_bert_classifier_response(original: List[Dict[str, Any]], response: List[Tuple]) -> Dict[str, Any]:
    response_dict = {}

    response_list = []
    for predict, index in response:
        single = original[index]
        single["class_name"] = predict
        response_list.append(single)

    response_dict["data"] = response_list
    if len(response_list) != 0:
        response_dict["success"] = True

    else:
        response_dict["success"] = False

    return response_dict


def step_actions_en_bert_classifier_response(original: List[Dict[str, Any]], response: List[Tuple]) -> Dict[str, List[Dict[str, Any]]]:
    response_dict = {}

    response_list = []
    for predict, index in response:
        single = original[index]
        single["class_name"] = predict
        response_list.append(single)

    response_dict["data"] = response_list
    if len(response_list) != 0:
        response_dict["success"] = True

    else:
        response_dict["success"] = False

    return response_dict
