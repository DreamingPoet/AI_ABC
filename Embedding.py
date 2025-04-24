# import os
# # 配置 ollama 的地址
# os.environ['OLLAMA_HOST'] = '192.168.0.245:11434'

# 必须 在设置环境变量后才生效
import ollama

# # 使用配置好的地址进行嵌入
res = ollama.embeddings(model='nomic-embed-text', prompt='The sky is blue because of rayleigh scattering')
print(res)




