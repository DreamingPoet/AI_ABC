# 基于本地 ollama：  model_name: str = "deepseek-r1:7b", base_url: str = "http://192.168.0.245:11434"
# 开发一个 智能问答机器人，能够支持多轮对话的记忆存储，并具备搜索互联网的能力。

import requests
import json
import os
import time
from typing import List, Dict, Any, Optional
import datetime

class ChatWithMemory:
    def __init__(
        self, 
        model_name: str = "deepseek-r1:7b", 
        base_url: str = "http://192.168.0.245:11434",
        memory_file: str = "chat_memory.json",
        debug_mode: bool = False
    ):
        """初始化聊天机器人

        Args:
            model_name: Ollama模型名称
            base_url: Ollama API基础URL
            memory_file: 记忆存储文件路径
            debug_mode: 是否启用调试模式
        """
        self.model_name = model_name
        self.base_url = base_url
        self.memory_file = memory_file
        self.debug_mode = debug_mode
        self.memory = self._load_memory()
        
        # 检查Ollama服务是否可用
        self._check_ollama_availability()
        
    def _check_ollama_availability(self) -> None:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                if self.debug_mode:
                    print(f"Ollama服务正常，可用模型: {', '.join([model['name'] for model in response.json().get('models', [])])}")
            else:
                print(f"警告: Ollama服务接口返回异常状态码: {response.status_code}")
        except Exception as e:
            print(f"警告: 无法连接到Ollama服务 ({self.base_url}): {str(e)}")
            print("请确保Ollama服务正在运行，并且基础URL正确。")

    def _load_memory(self) -> Dict[str, List[Dict[str, str]]]:
        """加载对话记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载记忆文件失败: {e}")
                return {"conversations": []}
        else:
            return {"conversations": []}

    def _save_memory(self) -> None:
        """保存对话记忆"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆文件失败: {e}")

    def _get_recent_messages(self, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """获取最近的对话消息"""
        for conv in self.memory["conversations"]:
            if conv["id"] == conversation_id:
                return conv["messages"][-limit:] if len(conv["messages"]) > limit else conv["messages"]
        return []

    def search_internet(self, query: str) -> str:
        """搜索互联网获取信息
        
        这里使用了一个简化的搜索实现，在实际应用中可以替换为真实的搜索API
        如Google Custom Search API, Bing Search API等
        """
        try:
            search_url = f"https://ddg-api.herokuapp.com/search?query={query}&limit=3"
            response = requests.get(search_url)
            
            if response.status_code == 200:
                results = response.json()
                return "\n".join([f"标题: {r['title']}\n链接: {r['link']}\n摘要: {r['snippet']}\n" 
                                for r in results])
            else:
                return f"搜索请求失败，状态码: {response.status_code}"
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"

    def ask(self, user_message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """向聊天机器人提问
        
        Args:
            user_message: 用户消息
            conversation_id: 对话ID，如果为None则创建新对话
            
        Returns:
            包含回复和对话ID的字典
        """
        # 如果没有会话ID或会话ID不存在，创建一个新会话
        if conversation_id is None or not any(conv["id"] == conversation_id for conv in self.memory["conversations"]):
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.memory["conversations"].append({
                "id": conversation_id,
                "created_at": datetime.datetime.now().isoformat(),
                "messages": []
            })
        
        # 检查是否需要搜索网络
        need_search = "搜索" in user_message or "查询" in user_message
        search_results = ""
        
        if need_search:
            search_query = user_message.replace("搜索", "").replace("查询", "").strip()
            search_results = self.search_internet(search_query)
            user_message += f"\n\n[搜索结果]: {search_results}"
        
        # 获取历史对话
        messages = self._get_recent_messages(conversation_id)
        
        # 准备发送给模型的消息
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 添加用户新消息
        ollama_messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 调用Ollama API - 使用流式处理
        try:
            # 使用stream=True启用流式处理
            with requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": ollama_messages,
                    "stream": True,  # 启用流式处理
                },
                stream=True
            ) as response:
                
                if response.status_code == 200:
                    # 收集完整的回答
                    full_response = ""
                    last_json = None
                    
                    # 逐行处理流式响应
                    for line in response.iter_lines():
                        if not line:
                            continue
                            
                        try:
                            # 解析当前行的JSON
                            json_data = json.loads(line)
                            last_json = json_data
                            
                            # 获取当前片段内容
                            if "message" in json_data and "content" in json_data["message"]:
                                content = json_data["message"]["content"]
                                full_response += content
                                
                                # 如果启用了调试模式，可以显示流式内容
                                # if self.debug_mode:
                                #     print(f"收到片段: {content}", end="", flush=True)
                            
                            # 检查是否为最后一个响应
                            if json_data.get("done", False):
                                break
                                
                        except json.JSONDecodeError as je:
                            if self.debug_mode:
                                print(f"跳过非JSON行: {line}")
                            continue
                    
                    if full_response:
                        # 存储用户消息和机器人回复
                        for conv in self.memory["conversations"]:
                            if conv["id"] == conversation_id:
                                conv["messages"].append({
                                    "role": "user",
                                    "content": user_message,
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                                conv["messages"].append({
                                    "role": "assistant",
                                    "content": full_response,
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                                break
                        
                        self._save_memory()
                        
                        return {
                            "response": full_response,
                            "conversation_id": conversation_id,
                            "searched": need_search,
                            "search_results": search_results if need_search else None
                        }
                    else:
                        error_msg = "模型未返回任何内容"
                        return {"error": error_msg, "conversation_id": conversation_id}
                else:
                    error_msg = f"API请求失败，状态码: {response.status_code}, 响应内容: {response.text[:300]}..."
                    return {"error": error_msg, "conversation_id": conversation_id}
        
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            return {"error": error_msg, "conversation_id": conversation_id}

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取完整的对话历史"""
        for conv in self.memory["conversations"]:
            if conv["id"] == conversation_id:
                return conv["messages"]
        return []

    def list_conversations(self) -> List[Dict[str, Any]]:
        """列出所有对话"""
        return [{
            "id": conv["id"],
            "created_at": conv["created_at"],
            "message_count": len(conv["messages"])
        } for conv in self.memory["conversations"]]

# 示例使用代码
if __name__ == "__main__":
    # 启用调试模式以获取更多日志信息
    bot = ChatWithMemory(debug_mode=True)
    
    print("欢迎使用智能问答机器人！输入'退出'结束对话。")
    conversation_id = None
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break
        
        # 添加显示"思考中..."的提示
        print("思考中...", end="", flush=True)
        result = bot.ask(user_input, conversation_id)
        print("\r", end="")  # 清除"思考中..."提示
        
        if "error" in result:
            print(f"错误: {result}")
        else:
            print(f"\n机器人: {result['response']}")
            conversation_id = result["conversation_id"]
            
            if result.get("searched"):
                print("\n[搜索结果摘要]:")
                print(result["search_results"])
    
    print("感谢使用，再见！")