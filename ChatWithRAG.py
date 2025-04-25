# ollama 本地大模型 地址 http://192.168.0.245:11434
# 完成 基于 RAG 的大语言模型对话

### 使用
# OllamaEmbeddings(
#             model="nomic-embed-text",
#             base_url="http://192.168.0.245:11434"
#         )
###

# 使用  milvus_host: str = "192.168.0.245", milvus_port: str = "19530"

import ollama
from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings import OllamaEmbeddings
from pymilvus import connections, Collection, utility
import logging
from typing import List, Dict, Optional, Any
from langchain_ollama import OllamaEmbeddings
import fitz  # PyMuPDF

# 配置参数
EMBEDDING_MODEL = "nomic-embed-text"
MILVUS_HOST = "192.168.0.245"
MILVUS_PORT = "19530"
COLLECTION_NAME = "doc_embeddings"
OLLAMA_MODEL = "deepseek-r1:7b"  # 或其他你本地安装的模型

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocSearch:
    def __init__(self, milvus_host: str = "192.168.0.245", milvus_port: str = "19530"):
        # 初始化 Milvus 连接
        connections.connect(host=milvus_host, port=milvus_port)
        
        # 初始化 Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://192.168.0.245:11434"
        )
        
        # 获取集合
        self.collection_name = "doc_embeddings"
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        # 检查集合中的实体数量
        collection = Collection("doc_embeddings")
        print(f"集合中的实体数量: {collection.num_entities}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索与查询文本最相关的文档片段
        
        Args:
            query: 搜索文本
            top_k: 返回最相关的前 k 个结果
            
        Returns:
            包含搜索结果信息的列表，每个结果包含：
            - text: 匹配的文本片段
            - source: 来源文档
            - score: 相似度分数
        """
        # 将查询文本转换为向量
        query_embedding = self.embeddings.embed_query(query)
        # 检查向量维度
        print(f"向量维度: {len(query_embedding)}")

        # 准备搜索参数
        # 较低的nprobe值：搜索速度更快，但可能降低召回率(recall)
        # 较高的nprobe值：搜索更彻底，召回率更高，但计算成本增加
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # 执行向量搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"]
        )
        print(f"搜索结果: {results}")
        # 处理搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score": hit.score
                }
                search_results.append(result)
        
        return search_results
    
    
class RAGSystem:
    def __init__(self):
        # 初始化嵌入模型
        logger.info("Loading embedding model...")
        # 初始化 Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://192.168.0.245:11434"
        )
        
        # 连接Milvus
        logger.info("Connecting to Milvus...")
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        # 检查集合是否存在
        if not utility.has_collection(COLLECTION_NAME):
            raise ValueError(f"Collection {COLLECTION_NAME} does not exist in Milvus")
        
        # 加载集合
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()
        
        logger.info("RAG system initialized successfully")
    
    def format_context(self, documents: List[Dict]) -> str:
        """将检索到的文档格式化为上下文"""
        context = "Retrieved documents:\n"
        for i, doc in enumerate(documents, 1):
            context += f"\nDocument {i}:\n{doc['text']}\n"
            if doc.get('source'):
                context += f"source: {doc['source']}\n"
        return context
    
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """使用Ollama生成响应"""
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

User question: {query}

{context if context else 'No additional context provided.'}

Please provide a helpful and accurate response:"""
        
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=False
        )
        
        return response['response']
    
    def chat(self, query: str) -> str:
        """完整的RAG对话流程"""

        # 执行搜索
        doc_search = DocSearch()
        retrieved_docs = doc_search.search(query)
        
        
        # 如果有检索结果，添加到上下文
        context = None
        if retrieved_docs:
            context = self.format_context(retrieved_docs)
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
        
        # 生成响应
        response = self.generate_response(query, context)
        return response
    
    def close(self):
        """清理资源"""
        connections.disconnect()

# 示例使用
if __name__ == "__main__":
    rag = RAGSystem()
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = rag.chat(user_input)
            print(f"AI: {response}")
    finally:
        rag.close()