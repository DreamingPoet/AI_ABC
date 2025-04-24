from pymilvus import connections, utility

# 连接Milvus
def connect_to_milvus(host='localhost', port='19530'):
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f"成功连接到Milvus服务器: {host}:{port}")
        print(f"Milvus版本: {utility.get_server_version()}")
    except Exception as e:
        print(f"连接失败: {e}")

if __name__ == "__main__":
    # 替换为你的局域网Milvus服务器IP
    connect_to_milvus(host='192.168.0.245', port='19530')