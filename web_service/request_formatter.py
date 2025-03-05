from typing import Dict, Any


def build_qwen2_72b_request(template: Dict[str, Any], content: str) -> Dict[str, Any]:
    messages = template["messages"]
    messages[1]["content"] = content
    
    template["messages"] = messages
    
    return template


def build_deepseek_qwen_request(template: Dict[str, Any], content: str) -> Dict[str, Any]:
    messages = template["messages"]
    messages[1]["content"] = content
    
    template["messages"] = messages
    
    return template

def build_DeepSeek_R1_request(template: Dict[str, Any], content: str) -> Dict[str, Any]:
    messages = template["messages"]
    messages[1]["content"] = content
    
    template["messages"] = messages
    
    return template

def build_qwen2_v1_request(template: Dict[str, Any], content: Any) -> Dict[str, Any]:
    pass


request_format_map = {
    "DeepSeek-R1": build_DeepSeek_R1_request,
    "qwen2_72b": build_qwen2_72b_request,
    "qwen2_vl": build_qwen2_v1_request,
    "deepseek-qwen": build_deepseek_qwen_request,
}