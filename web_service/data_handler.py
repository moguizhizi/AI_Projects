from utils.http_utils import send_chat_request
from utils import is_english_by_langdetect
from web_service.request_formatter import request_format_map
from web_service.responder import response_parser_map


from typing import Dict, Any, List
from http import HTTPStatus
from tenacity import retry, stop_after_attempt, wait_fixed


def validate_inputs(config_dict: Dict[str, Any], data: List[Dict[str, str]]) -> None:
    """验证输入参数"""
    required_keys = {"LLM_Type", "LLM_Request_URL", "request_template"}
    if not all(key in config_dict for key in required_keys):
        raise ValueError("配置字典缺少必要键")
    if not isinstance(data, list):
        raise TypeError("数据必须是列表")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_chat_request_with_retry(url: str, content: str):
    """带重试机制的聊天请求"""
    return send_chat_request(url, content)


def step_actions_normalize(config_dict: Dict[str, Any], data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """规范化步骤动作数据，将非英文内容翻译为英文"""
    validate_inputs(config_dict, data)

    llm_type = config_dict["LLM_Type"]
    request_url = config_dict["LLM_Request_URL"][llm_type]
    formatter = request_format_map[llm_type]
    single_template = config_dict["request_template"][llm_type]
    TRANSLATION_PROMPT = "\n请将以上语句翻译为英文。"

    normalized_data = []

    for content in data:
        normalized_content = content.copy()
        normalized_content["class_id"] = -1
        step_actions = normalized_content["StepActions"]

        if not is_english_by_langdetect(step_actions):
            try:
                formatted_request = formatter(
                    single_template,
                    step_actions + TRANSLATION_PROMPT
                )
                response = send_chat_request_with_retry(
                    request_url, formatted_request)

                if response and response.status_code == HTTPStatus.OK:
                    parser = response_parser_map[llm_type]
                    translated_content = parser(response.json()).get_content()
                    normalized_content["StepActions"] = translated_content
                else:
                    print(
                        f"获取有效响应失败: {response.status_code if response else '无响应'}")

            except Exception as e:
                print(f"翻译失败，内容: {step_actions}。错误: {str(e)}")

        normalized_data.append(normalized_content)

    return normalized_data


def step_actions_en_normalize():
    pass
