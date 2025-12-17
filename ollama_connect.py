from ollama import Client

# 1. 初始化客户端，指定服务器地址和端口
client = Client(host='http://10.15.102.186:9000')

# -------------------------
# 场景 A: 对话 (Chat)
# -------------------------
model_name = "qwen3:30b-a3b-instruct-2507-q8_0" # 必须严格复制图片中的模型名

response = client.chat(model=model_name, messages=[
  {
    'role': 'user',
    'content': '你好，请介绍一下RAG技术。',
  },
])
print("大模型回复:", response['message']['content'])

# -------------------------
# 场景 B: 生成向量 (Embedding)
# -------------------------
embed_model = "bge-m3:latest"

embedding_response = client.embeddings(
    model=embed_model,
    prompt="这里是需要向量化的文本"
)
print("向量维度:", len(embedding_response['embedding']))
# print(embedding_response['embedding'])