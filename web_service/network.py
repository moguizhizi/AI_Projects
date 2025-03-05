import uvicorn
from typing import Dict, Any

from web_service.classifier_service import create_ai_app


# 服务映射字典：port_id -> FastAPI 应用
SERVICE_MAPPING = {
    6578: create_ai_app(),  # 端口 6578 运行 ai 服务
}


def start_web_service(network_dict: Dict[str, Any], args: Any) -> None:
    service_dict = network_dict[args.model_name]
    port_id = service_dict["port"]

    # 根据 port_id 获取对应的 FastAPI 应用
    app = SERVICE_MAPPING.get(port_id)
    if app is None:
        raise ValueError(
            f"No service defined for port {port_id}. Supported ports: {list(SERVICE_MAPPING.keys())}")

    print(f"Starting service on port {port_id}...")
    uvicorn.run(app, host="0.0.0.0", port=port_id)
