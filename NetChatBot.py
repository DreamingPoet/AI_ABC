# ollama 本地大模型 地址 http://192.168.0.245:11434

import requests
from typing import List, Dict
import json
import os
from datetime import datetime
from duckduckgo_search import DDGS

class NetChatBot:
    def __init__(self, model_name: str = "deepseek-r1:7b", base_url: str = "http://192.168.0.245:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.ddgs = DDGS()
        
        # 测试 Ollama 连接
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                print(f"警告：无法连接到 Ollama 服务，请确保服务正在运行。状态码：{response.status_code}")
        except requests.exceptions.ConnectionError:
            print("错误：无法连接到 Ollama 服务，请确保服务正在运行。")
        
    def search_web(self, query: str) -> List[Dict]:
        """使用 DuckDuckGo 进行网络搜索"""
        try:
            results = []
            # 获取搜索结果
            for r in self.ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "link": r.get("link", "")
                })
            return results
        except Exception as e:
            print(f"搜索出错: {str(e)}")
            return []

    def generate_response(self, prompt: str, search_results: List[Dict] = None) -> str:
        """使用Ollama生成回复"""
        messages = [
            {"role": "system", "content": "你是一个智能助手，可以回答用户的问题。请基于搜索结果提供准确的信息。"},
            {"role": "user", "content": prompt}
        ]
        
        if search_results:
            context = "\n".join([f"标题: {result.get('title', '')}\n内容: {result.get('snippet', '')}\n链接: {result.get('link', '')}" 
                               for result in search_results])
            messages.append({"role": "system", "content": f"以下是相关搜索结果：\n{context}"})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return "错误：无法连接到 Ollama 服务，请确保服务正在运行。"
        except requests.exceptions.HTTPError as e:
            return f"HTTP错误：{str(e)}"
        except Exception as e:
            return f"生成回复时出错: {str(e)}"

    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复"""
        # 首先进行网络搜索
        search_results = self.search_web(user_input)
        
        # 生成回复
        response = self.generate_response(user_input, search_results)
        return response

def main():
    # 创建聊天机器人实例
    bot = NetChatBot()
    
    print("聊天机器人已启动！输入 'exit' 退出对话。")
    print("提示：请确保 Ollama 服务正在运行。")
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            break
            
        response = bot.chat(user_input)
        print(f"\n助手: {response}")

if __name__ == "__main__":
    main()
