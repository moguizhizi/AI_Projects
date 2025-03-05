import requests
from typing import Dict, Any, Optional

def send_chat_request(
    url: str,
    data: Dict[str, Any],
    timeout: float = 100.0
) -> Optional[Dict[str, Any]]:
    """
    向聊天 API 发送 POST 请求，并打印操作结果。

    Args:
        url (str): API 端点 URL
        data (Dict[str, Any]): 要发送的数据，JSON 格式
        timeout (float, optional): 请求超时时间，单位秒，默认 5.0

    Returns:
        Optional[Dict[str, Any]]: 成功时返回响应的 JSON 数据，失败时返回 None
    """
    
    try:
        # 发送 POST 请求
        response = requests.post(
            url,
            json=data,  # 自动将字典转换为 JSON 格式
            timeout=timeout
        )

        # 检查响应状态码
        if response.status_code == 200:
            print("请求成功！")
            print("响应内容：")
            response_data = response.json()  # 解析 JSON 响应
            print(response_data)
            return response
        else:
            print(f"请求失败，状态码：{response.status_code}")
            print(f"响应内容：{response.text}")
            return response

    except requests.exceptions.RequestException as e:
        print(f"请求发生异常：{str(e)}")
        return None
    except ValueError:
        print("响应内容不是有效的 JSON 格式")
        return None