# 文档 搜索 的实现

import os
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from pymilvus import connections, Collection, utility
import fitz  # PyMuPDF

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
    
    def get_context_from_pdf(self, pdf_path: str, text: str) -> Dict[str, Any]:
        """
        从 PDF 文件中获取文本片段的上下文信息
        
        Args:
            pdf_path: PDF 文件路径
            text: 要查找的文本片段
            
        Returns:
            包含上下文信息的字典：
            - page_number: 页码
            - surrounding_text: 周围文本
        """
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if text in page_text:
                # 获取文本在页面中的位置
                text_instances = page.search_for(text)
                if text_instances:
                    # 获取文本周围的上下文
                    rect = text_instances[0]
                    context_rect = fitz.Rect(
                        rect.x0 - 50,  # 左边扩展
                        rect.y0 - 50,  # 上边扩展
                        rect.x1 + 50,  # 右边扩展
                        rect.y1 + 50   # 下边扩展
                    )
                    surrounding_text = page.get_textbox(context_rect)
                    
                    return {
                        "page_number": page_num + 1,
                        "surrounding_text": surrounding_text
                    }
        return {"page_number": None, "surrounding_text": None}

if __name__ == "__main__":
    # 使用示例
    doc_search = DocSearch()
    
    # 执行搜索
    query = "王铁宏"
    results = doc_search.search(query)
    
    # 打印结果
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"相似度分数: {result['score']}")
        print(f"来源文档: {result['source']}")
        print(f"匹配文本: {result['text']}")
        
        # 获取上下文信息
        pdf_path = os.path.join("docs", result["source"])
        if os.path.exists(pdf_path):
            context = doc_search.get_context_from_pdf(pdf_path, result["text"])
            if context["page_number"]:
                print(f"页码: {context['page_number']}")
                print(f"上下文: {context['surrounding_text']}")



