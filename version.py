import importlib.metadata

# 定义你想检查的组件列表
packages = [
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-openai",  # 如果你使用了OpenAI
    "langchain-experimental", # 如果你使用了实验性功能
    "langgraph", # 如果你使用了LangGraph
    "langserve"  # 如果你使用了LangServe
]

print(f"{'组件名称':<25} {'版本号'}")
print("-" * 35)

for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package:<25} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package:<25} 未安装")