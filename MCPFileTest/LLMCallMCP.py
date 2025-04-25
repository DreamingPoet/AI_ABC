# 使用本地 ollama 模型调用 mcp-server-fetch 服务

import ollama
import requests
import json
import os
from typing import Dict, Any, List, Optional

# 配置参数
OLLAMA_MODEL = "deepseek-r1:7b"
MCP_SERVER_URL = "http://localhost:8000"  # mcp-server-fetch 默认地址，根据实际部署修改
MCP_SERVER_API_KEY = os.environ.get("MCP_SERVER_API_KEY", "")  # 如果需要认证

def get_ollama_response(prompt: str) -> str:
    """从本地Ollama模型获取响应"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ])
    return response['message']['content']

def call_mcp_server(query: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """调用本地的mcp-server-fetch服务"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # 如果配置了API密钥，则添加到请求头
    if MCP_SERVER_API_KEY:
        headers["Authorization"] = f"Bearer {MCP_SERVER_API_KEY}"
    
    payload = {
        "query": query
    }
    
    if context:
        payload["context"] = context
    
    # 调用mcp-server-fetch的API端点
    response = requests.post(f"{MCP_SERVER_URL}/api/mcp", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"MCP服务调用失败: {response.status_code} - {response.text}")

def process_with_local_and_mcp(user_query: str) -> Dict[str, Any]:
    """结合本地模型和MCP服务处理用户查询"""
    # 首先使用本地模型生成初步回答
    local_response = get_ollama_response(user_query)
    
    # 准备MCP服务的上下文
    context = [
        {
            "role": "assistant",
            "content": local_response
        }
    ]
    
    # 通过MCP服务增强本地模型的回答
    final_response = call_mcp_server(user_query, context)
    
    return {
        "local_model_response": local_response,
        "mcp_enhanced_response": final_response
    }

def check_mcp_server_status() -> bool:
    """检查mcp-server-fetch是否正在运行"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    # 检查MCP服务器是否可用
    if not check_mcp_server_status():
        print("警告: 无法连接到mcp-server-fetch服务。请确保服务已启动并运行在配置的地址上。")
        print(f"当前配置的服务地址: {MCP_SERVER_URL}")
        print("\n是否仅使用本地模型继续? (y/n)")
        choice = input().strip().lower()
        if choice != "y":
            print("退出程序")
            exit()
        
    # 使用示例
    user_query = input("请输入您的问题: ")
    
    try:
        # 使用本地模型获取响应
        local_response = get_ollama_response(user_query)
        print("\n本地模型响应:")
        print(local_response)
        
        # 如果MCP服务器可用，则使用MCP服务增强响应
        if check_mcp_server_status():
            try:
                result = process_with_local_and_mcp(user_query)
                print("\nMCP增强后的响应:")
                print(json.dumps(result["mcp_enhanced_response"], ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"\nMCP服务处理出错: {e}")
                
    except Exception as e:
        print(f"发生错误: {e}")

# 使用说明：
# 1. 首先确保mcp-server-fetch服务已经在本地启动
# 2. 根据实际情况修改MCP_SERVER_URL变量，设置为mcp-server-fetch的实际地址
# 3. 如果mcp-server-fetch需要API密钥，请设置环境变量MCP_SERVER_API_KEY