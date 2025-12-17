共有5个文件，介绍如下：

1. `process_bibs.py`: 文献清洗，当初开发时摸索了多种方式。现在推荐最快捷的方法是用Medeley自动抓取文献的元数据，导出成bib文件，再写个python脚本提取出来，写入MongoDB数据库；实际处理单个文档时，利用python库`pdf2doi`获取到论文的doi，再按doi从MongoDB里检索获取相应的其它元数据。代码中可以看到利用`pdf2doi`或`pdf2bib`处理论文的示例。一篇论文的元数据至少包括标题、摘要、期刊、年份
2. `vectorize_bibs.py`：文献矢量化。从中可以看到调用`BGE-M3`模型将文献矢量化以及将结果写入Milvus数据库的示例。
3. `rag.py`：实现RAG各个过程，包括文献的加载与拆分，检索与生成等过程。提示：**LangChain**的版本迭代较快，可能需要参照最新手册
4. `chatxfel_app.py`：利用`streamlit`写的一个UI界面，回答输出未实现打字机模式，也未添加历史对话功能；
5. `naive.pt`：提示词示例。