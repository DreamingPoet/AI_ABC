import requests
import json
from typing import Dict, Any

# 配置
OLLAMA_API_URL = "http://192.168.0.245:11434/api/chat"  # Ollama 默认端口
FILE_SERVICE_URL = "http://localhost:5000/api/files"    # 您的文件 MCP 服务地址

# 文件操作工具函数
class FileServiceClient:
    @staticmethod
    def list_files(path: str = "") -> Dict[str, Any]:
        """列出目录内容"""
        response = requests.get(f"{FILE_SERVICE_URL}", params={"path": path})
        return response.json()

    @staticmethod
    def read_file(path: str) -> Dict[str, Any]:
        """读取文件内容"""
        response = requests.get(f"{FILE_SERVICE_URL}/content", params={"path": path})
        return response.json()

    @staticmethod
    def create_file(path: str, content: str) -> Dict[str, Any]:
        """创建新文件"""
        response = requests.post(FILE_SERVICE_URL, data={"path": path, "content": content})
        return response.json()

    @staticmethod
    def delete_file(path: str) -> Dict[str, Any]:
        """删除文件"""
        response = requests.delete(FILE_SERVICE_URL, params={"path": path})
        return response.json()

# Ollama 交互函数
class OllamaClient:
    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.model_name = model_name
        
    def generate_response(self, prompt: str, context: str = "") -> str:
        """与Ollama大模型交互"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "context": context,
            "stream": False
        }
        
        response = requests.post(
            OLLAMA_API_URL,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama API error: {response.text}")

# 协调层 - 解析大模型指令并执行文件操作
class FileOperationOrchestrator:
    def __init__(self):
        self.ollama = OllamaClient()
        self.file_service = FileServiceClient()
        
    def parse_and_execute(self, natural_language_command: str) -> Dict[str, Any]:
        """解析自然语言指令并执行相应文件操作"""
        # 第一步：让大模型理解指令意图
        prompt = f"""
        请分析以下用户指令，判断用户想要执行什么文件操作，并以JSON格式返回。
        可用的操作类型: list_files, read_file, create_file, delete_file
        示例输出: {{"operation": "list_files", "path": "documents/"}}
        
        用户指令: {natural_language_command}
        """
        
        # 获取大模型的分析结果
        analysis = self.ollama.generate_response(prompt)
        
        try:
            # 尝试解析大模型的JSON响应
            command = json.loads(analysis.strip())
            operation = command.get("operation")
            path = command.get("path", "")
            content = command.get("content", "")
            
            # 执行相应操作
            if operation == "list_files":
                return self.file_service.list_files(path)
            elif operation == "read_file":
                return self.file_service.read_file(path)
            elif operation == "create_file":
                return self.file_service.create_file(path, content)
            elif operation == "delete_file":
                return self.file_service.delete_file(path)
            else:
                return {"error": "Unsupported operation"}
                
        except json.JSONDecodeError:
            # 如果大模型没有返回有效JSON，尝试直接执行
            return self.fallback_execution(natural_language_command)
    
    def fallback_execution(self, command: str) -> Dict[str, Any]:
        """当大模型无法返回结构化响应时的备用方案"""
        # 简单关键词匹配
        if "列出" in command or "显示" in command or "查看目录" in command:
            path = command.split(" ")[-1] if len(command.split(" ")) > 1 else ""
            return self.file_service.list_files(path)
        elif "读取" in command or "查看文件" in command:
            path = command.split(" ")[-1]
            return self.file_service.read_file(path)
        elif "创建" in command or "新建" in command:
            parts = command.split(" ")
            path = parts[-2] if len(parts) > 2 else ""
            content = parts[-1] if len(parts) > 1 else ""
            return self.file_service.create_file(path, content)
        elif "删除" in command or "移除" in command:
            path = command.split(" ")[-1]
            return self.file_service.delete_file(path)
        else:
            return {"error": "无法理解指令"}

# 使用示例
if __name__ == "__main__":
    orchestrator = FileOperationOrchestrator()
    
    # 示例1: 列出文件
    print("=== 列出文件示例 ===")
    result = orchestrator.parse_and_execute("请帮我创建一个 memo.txt 文件，并写入内容 '重要会议记录'")
    print(result)
    
    # # 示例2: 读取文件
    # print("\n=== 读取文件示例 ===")
    # result = orchestrator.parse_and_execute("请读取report.txt文件的内容")
    # print(result)
    
    # # 示例3: 创建文件
    # print("\n=== 创建文件示例 ===")
    # result = orchestrator.parse_and_execute("在notes目录下创建memo.txt文件，内容为'重要会议记录'")
    # print(result)
    
    # # 示例4: 删除文件
    # print("\n=== 删除文件示例 ===")
    # result = orchestrator.parse_and_execute("请删除temp/old_file.txt")
    # print(result)