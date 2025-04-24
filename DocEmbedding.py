# 文档 Embedding 的实现
import os
from typing import List
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np

class DocEmbedding:
    def __init__(self, milvus_host: str = "192.168.0.245", milvus_port: str = "19530"):
        # 初始化 Milvus 连接
        connections.connect(host=milvus_host, port=milvus_port)
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        
        # 初始化 Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://192.168.0.245:11434"
        )
        
        # 创建或获取集合
        self.collection_name = "doc_embeddings"
        self._setup_collection()
    
    def _setup_collection(self):
        """设置或获取 Milvus 集合"""
        # 如果集合存在且维度不匹配，则删除它
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            if collection.schema.fields[2].params["dim"] != 768:  # 检查 embedding 字段的维度
                utility.drop_collection(self.collection_name)
                print("已删除维度不匹配的旧集合")
                
        if not utility.has_collection(self.collection_name):
            # 定义集合的字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # nomic-embed-text 模型的向量维度是 768
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            ]
            
            # 创建集合
            schema = CollectionSchema(fields=fields, description="文档向量存储")
            collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
        else:
            collection = Collection(self.collection_name)
        
        self.collection = collection
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从 PDF 文件中提取文本"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def process_document(self, file_path: str):
        """处理单个文档"""
        # 提取文本
        text = self.extract_text_from_pdf(file_path)
        
        # 分割文本
        chunks = self.text_splitter.split_text(text)
        
        # 生成嵌入向量
        embeddings = self.embeddings.embed_documents(chunks)
        
        # 确保向量维度正确
        for i, emb in enumerate(embeddings):
            if len(emb) != 768:  # nomic-embed-text 模型的向量维度是 768
                print(f"Warning: Embedding dimension mismatch at chunk {i}. Expected 768, got {len(emb)}")
                continue
        
        # 准备数据
        data = [
            chunks,  # text
            embeddings,  # embedding
            [os.path.basename(file_path)] * len(chunks)  # source
        ]
        
        # 插入数据到 Milvus
        try:
            self.collection.insert(data)
            self.collection.flush()
            print(f"Successfully inserted {len(chunks)} chunks from {file_path}")
        except Exception as e:
            print(f"Error inserting data: {str(e)}")
            print(f"Number of chunks: {len(chunks)}")
            print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 'None'}")
    
    def process_directory(self, directory_path: str):
        """处理目录中的所有 PDF 文件"""
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                print(f"Processing {filename}...")
                self.process_document(file_path)
                print(f"Completed processing {filename}")

if __name__ == "__main__":
    # 使用示例
    doc_embedding = DocEmbedding()
    doc_embedding.process_directory("docs")