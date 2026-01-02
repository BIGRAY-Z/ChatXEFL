#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zhangxf2@shanghaitech.edu.cn
# Date: Mar, 29 2024

'''
Define the functions for RAG pipeline
'''

import os
import sys
from langchain_community.document_loaders import (PyPDFLoader, PDFPlumberLoader, 
        UnstructuredMarkdownLoader, BSHTMLLoader, JSONLoader, CSVLoader, DirectoryLoader)
from langchain_community.vectorstores import Milvus 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceBgeEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatOllama
import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将其加入到系统路径中
sys.path.append(current_dir)

def load(file_name, file_type='pdf', pdf_loader='pypdf'):
    '''
    load documents by following loader:
    pdf: PyPDFLoader or PDFPlumberLoader
    markdown: UnstructedMarkdownLoader
    html: BSHTMLLoader
    json: JSONLoader
    csv: CSVLoader
    
    Args:
        file_name: file name to be load
        file_type: pdf, markdown, html, json, csv
        loader: specify document loader
        split: load or load_and_split
    '''
    if not os.path.exists(file_name):
        print(f'ERROR: {file_name} does not exist')
        return []

    doc = []
    if file_type.lower() == 'pdf':
        if pdf_loader == 'pypdf':
            #loader = PyPDFLoader(file_name, extract_images=True)
            loader = PyPDFLoader(file_name)
            doc = loader.load()
        elif pdf_loader == 'pdfplumber':
            loader = PDFPlumberLoader(file_name)
            doc = loader.load()
        else:
            print('pdf_loader should be one of pypdf or pdfplumber')
    elif file_type.lower() == 'markdwon':
        loader = UnstructuredMarkdownLoader(file_name)
        doc = loader.load()
    elif file_type.lower() == 'html':
        loader = BSHTMLLoader(file_name)
        doc = loader.load()
    elif file_type.lower() == 'json':
        loader = JSONLoader(file_name)
        doc = loader.load()
    elif file_type.lower() == 'csv':
        loader = CSVLoader(file_name)
        doc = loader.load()
    else:
        print(f'Unsupported file type.')
        print('Supported file types are: pdf, markdown, html, json, csv')

    return doc

def load_pdf_directory(file_dir, recursive=True, multitread=True):
    kwargs = {'extract_images':True}
    loader = DirectoryLoader(file_dir, glob='**/*.pdf', loader_cls=PyPDFLoader, recursive=recursive,
                             loader_kwargs=kwargs, show_progress=True, use_multithreading=multitread)
    docs = loader.load()
    return docs

def split(docs, size=2000, overlap=200, length_func=len, sep=None, is_regex=False): 
    '''
    only recursively split by character is used now.
    '''
    if type(docs) is not list:
        print(f'{docs} should be a list.')
        return []
    if sep != None:
        separator = sep
    else:
        separator = ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = size,
        chunk_overlap = overlap,
        length_function = length_func,
        is_separator_regex=is_regex,
        add_start_index = True,
        separators=separator)

    texts = splitter.split_documents(docs)
    return texts

def get_embedding_bge(model_kwargs=None, encode_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {'device':'cuda'}

    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings':True}
    model_name = 'BAAI/bge-m3'
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embedding

def get_embedding(model_name, n_gpu_layers=-1, n_ctx=4096):
    '''
    Supported models: llama, gpt
    '''
    embedding = None
    if 'llama' in model_name.lower():
        embedding = LlamaCppEmbeddings(
            model_path=model_name,
            n_gpu_layers = n_gpu_layers,
            n_ctx=n_ctx
        )
    elif 'gpt' in model_name.lower():
        print('Support for GPT models is TBD')
    else:
        print('Only gpt or llama are supported')

    return embedding

def restore_vector(docs, connection_args, col_name, embedding, desc=''):
    _ = Milvus(embedding_function=embedding,
                          connection_args=connection_args,
                          collection_name=col_name,
                          drop_old=True
                         ).from_documents(
                             docs,
                             embedding=embedding,
                             connection_args=connection_args,
                             collection_description=desc,
                             collection_name=col_name
                         )
    return _


def get_retriever(connection_args, col_name, embedding, vector_field='vector', use_rerank=False, 
                  top_n=4, filters=None, return_as_retreiever=True):
    search_kwargs = {'k':10, 'params': {'ef': 20}}
    if filters:
        search_kwargs['filter'] = filters
    retriever = Milvus(embedding_function=embedding,
                       connection_args=connection_args,
                       collection_name=col_name,
                       vector_field=vector_field)
    if use_rerank:
        rerank_model = HuggingFaceCrossEncoder(
            #model_name = '/data-10gb/data/llm/bge-reranker-v2-m3')
            model_name = 'BAAI/bge-reranker-v2-m3')
        compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                   base_retriever=retriever.as_retriever(search_kwargs=search_kwargs))
        return compression_retriever
    else:
        if return_as_retreiever:
            return retriever.as_retriever(search_kwargs=search_kwargs)
        else:
            return retriever
# --- 需要新增的 import ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pymilvus import Collection, AnnSearchRequest, RRFRanker, WeightedRanker, connections
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from typing import List, Any

# --- 自定义混合检索器 ---
# rag.py 中的更新代码

class MilvusHybridRetriever(BaseRetriever):
    """
    支持 BGE-M3 Dense + Sparse 混合检索的自定义 Retriever
    """
    collection_name: str
    connection_args: dict
    embedding_function: Any
    top_k: int = 10
    # 新增：用于存储动态过滤表达式
    current_filter: str = "" 

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 1. 确保连接
        try:
            connections.connect(alias="default", **self.connection_args)
        except Exception:
            pass

        # 2. 编码 Query
        query_embeddings = self.embedding_function.encode_queries([query])
        dense_vec = query_embeddings['dense'][0].tolist()

        raw_sparse = query_embeddings['sparse']
        
        # 尝试将整个矩阵转为 CSR（这一步保留，有利于性能）
        if hasattr(raw_sparse, 'tocsr'):
            sparse_matrix = raw_sparse.tocsr()
        else:
            sparse_matrix = raw_sparse
            
        # 获取第一行
        sparse_row = sparse_matrix[0]
        
        # [关键修复]：再次检查并确保行向量是 CSR 格式
        # 解决 AttributeError: 'coo_array' object has no attribute 'indices'
        try:
            if not hasattr(sparse_row, "indices") and hasattr(sparse_row, "tocsr"):
                sparse_row = sparse_row.tocsr()
        except Exception:
            # 防御性编程：如果上述转换失败，假设它是某种类数组对象
            pass

        # 构建 Milvus 接受的字典格式
        # 注意：现在 sparse_row 一定有 indices 和 data
        sparse_vec = {int(k): float(v) for k, v in zip(sparse_row.indices, sparse_row.data)}
        # 3. 准备过滤表达式
        # 混合检索的 AnnSearchRequest 支持传入 expr
        search_expr = self.current_filter if self.current_filter else None

        # 4. 构建搜索请求 (传入 expr)
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=self.top_k,
            expr=search_expr  # <--- 关键修改：传入过滤条件
        )

        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, 
            limit=self.top_k,
            expr=search_expr  # <--- 关键修改：传入过滤条件
        )

        rerank = RRFRanker(k=60) 
        col = Collection(self.collection_name)
        
        # 5. 执行搜索
        try:
            res = col.hybrid_search(
                [dense_req, sparse_req],
                rerank=rerank,
                limit=self.top_k,
                output_fields=["text", "title", "doi", "journal", "year", "page"]
            )
        except Exception as e:
            print(f"Hybrid search failed: {e}")
            return []

        # 6. 转换结果
        documents = []
        for hits in res:
            for hit in hits:
                metadata = {
                    "title": hit.entity.get("title"),
                    "doi": hit.entity.get("doi"),
                    "journal": hit.entity.get("journal"),
                    "year": hit.entity.get("year"),
                    "page": hit.entity.get("page"),
                    "score": hit.score
                }
                documents.append(Document(page_content=hit.entity.get("text"), metadata=metadata))

        return documents

# --- 新增一个工厂函数来获取这个 Retriever ---
def get_hybrid_retriever(connection_args, col_name, top_k=10):
    # 初始化 BGE-M3 模型 (用于 Query 编码)
    # 注意：这里的 device 和路径需要和您的环境一致
    embedding = BGEM3EmbeddingFunction(
        model_name = 'BAAI/bge-m3', # 请确认路径
        device='cuda', 
        use_fp16=True
    )
    
    retriever = MilvusHybridRetriever(
        collection_name=col_name,
        connection_args=connection_args,
        embedding_function=embedding,
        top_k=top_k
    )
    return retriever

# 在 rag.py 中添加以下 imports (如果尚未存在)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def rewrite_query(question, llm):
    """
    Rewrite the user question to be more suitable for retrieval.
    """
    # 定义重写提示词
    template = """You are an AI assistant for X-ray Free Electron Laser Facility(XFEL). 
    Your task is to rephrase the user's question to make it more precise and suitable for retrieving technical documents.
    
    Rules:
    1. Expand abbreviations if necessary (e.g., 'SPI' -> 'Single Particle Imaging').
    2. Focus on keywords relevant to physics, engineering, or XFEL operations.
    3. Do not change the original intent of the question.
    4. Return ONLY the rewritten question, no explanation.

    Original Question: {question}
    Rewritten Question:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 构建 Chain: Prompt -> LLM -> OutputParser
    rewrite_chain = prompt | llm | StrOutputParser()
    
    # 执行重写
    rewritten_query = rewrite_chain.invoke({"question": question})
    return rewritten_query.strip()

# 在 rag.py 中修改或添加
def rewrite_query_with_feedback(original_question, current_rewrite, feedback, llm):
    """
    根据用户反馈进一步优化重写的问题
    """
    template = """You are an AI assistant helping a researcher refine their search query for X-ray Free Electron Laser Facility(XFEL).
    
    Original User Intent: {original_question}
    Current Rewritten Version: {current_rewrite}
    User Feedback: {feedback}
    
    Task: Based on the feedback, provide an improved version of the rewritten query. 
    Focus on technical precision, XFEL terminology, and clarity.
    Return ONLY the new rewritten question.

    New Rewritten Question:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    new_rewrite = chain.invoke({
        "original_question": original_question,
        "current_rewrite": current_rewrite,
        "feedback": feedback
    })
    return new_rewrite.strip()

def get_prompt(prompt='', return_format=True):
    if prompt == '':
        Prompt = """Use the following pieces of context to answer the question at the end.
                    You should answer the question in detail as far as possible.
                    If you cannot find anwser in the context, just say that you don't know, don't try to make up an answer.

                    {context}

                    Question: {question}

                    Helpful Answer:
                """
    if return_format:
        Prompt = PromptTemplate.from_template(prompt)
    return Prompt

def get_llm_LLaMA(model_name, model_path, n_batch=2048, n_ctx=8192, verbose=False, 
                  streaming=True, max_tokens=8192, temperature=0.8):
    if model_name == 'LLaMA3-8B':
        tokenizer = AutoTokenizer.from_pretrained('/data-10gb/data/llm/llama3/Meta-Llama-3-8B-Instruct-hf')
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        model_kwargs = {'do_sample':True, 'eos_token_id':terminators, 'max_new_tokens':8192, 'pad_token_id':128001}
        llm = LlamaCpp(model_path=model_path, 
                       n_gpu_layers=-1,
                       n_ctx=8192, 
                       n_batch=n_batch, 
                       f16_kv=True,
                       verbose=verbose,
                       streaming=streaming, 
                       temperature=temperature,
                       model_kwargs=model_kwargs)
        llm.client.verbose=False
    elif model_name == 'LLaMA2-7B':
        llm = LlamaCpp(model_path=model_path,
                       n_gpu_layers=-1,
                       n_ctx=n_ctx,
                       n_batch=n_batch,
                       f16_kv=True,
                       verbose=verbose,
                       streaming=streaming,
                       temperature=temperature,
                       max_tokens=max_tokens)
    llm.client.verbose=False
    return llm

def get_llm_ollama(model_name, num_predict, num_ctx=8192, keep_alive=600, temperature=0.1, base_url='http://10.15.85.78:11434'):
    model = 'qwen3:30b-a3b-instruct-2507-q8_0'
    llm = ChatOllama(model=model, num_ctx=num_ctx, keep_alive=keep_alive, num_predict=num_predict, 
                     temperature=temperature, base_url=base_url, num_thread=2)
    return llm

def get_contextualize_question(llm, history_prompt_template, input_: dict):
    history_context = None
    history_chain = history_prompt_template | llm | StrOutputParser()
    if input_.get('chat_history'):
        history_context = history_chain
    else:
        history_context = input_['question']
    return history_context

def retrieve_generate(question, llm, prompt, retriever, history=None, return_source=True, return_chain=False): 
    if return_source:
        rag_source = (RunnablePassthrough.assign(
            context=(lambda x: utils.format_docs(x['context'])))
            | prompt
            | llm
            | StrOutputParser()
        )

        if history:
            rag_chain = RunnableParallel(
                {'context':retriever, 'history':history, 'question':RunnablePassthrough()}).assign(
                    answer=rag_source)
        else:
            rag_chain = RunnableParallel(
                {'context':retriever, 'question':RunnablePassthrough()}).assign(
                    answer=rag_source)

    else:
        if history:
            rag_chain = ({'context':retriever, 'history':history, 'question':RunnablePassthrough()}
                         | prompt | llm)
        else:
            rag_chain = ({'context':retriever, 'question':RunnablePassthrough()}
                         | prompt | llm)

    if return_chain:
        return rag_chain
    else:
        answer = rag_chain.invoke(question)
        return answer


class MilvusRoutingRetriever(BaseRetriever):
    """
    路由检索器：
    1. 先在 abstract_collection (摘要库) 中进行混合检索，找出最相关的文献 DOI。
    2. 使用这些 DOI 作为过滤条件，在 fulltext_collection (全文库) 中进行第二次混合检索。
    """
    abstract_collection: str = "xfel_abstracts_v1" # 固定的摘要库名称
    fulltext_collection: str # 动态传入的全文库名称
    connection_args: dict
    embedding_function: Any
    top_k_abstracts: int = 5   # 第一步：检索多少篇最相关的摘要
    top_k_fulltext: int = 10   # 第二步：最终返回多少个全文切片
    current_filter: str = ""   # 外部传入的年份/关键词过滤

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 0. 确保连接
        try:
            connections.connect(alias="default", **self.connection_args)
        except Exception:
            pass

        # -------------------------------------------------------
        # 第一步：对 Query 编码 (Dense + Sparse)
        # -------------------------------------------------------
        query_embeddings = self.embedding_function.encode_queries([query])
        dense_vec = query_embeddings['dense'][0].tolist()
        
        # 处理 Sparse 向量
        raw_sparse = query_embeddings['sparse']
        if hasattr(raw_sparse, 'tocsr'):
            sparse_matrix = raw_sparse.tocsr()
        else:
            sparse_matrix = raw_sparse
        sparse_row = sparse_matrix[0]
        # 确保 CSR 格式
        try:
            if not hasattr(sparse_row, "indices") and hasattr(sparse_row, "tocsr"):
                sparse_row = sparse_row.tocsr()
        except Exception:
            pass
        sparse_vec = {int(k): float(v) for k, v in zip(sparse_row.indices, sparse_row.data)}

        # -------------------------------------------------------
        # 第二步：检索摘要库 (xfel_abstracts_v1)
        # -------------------------------------------------------
        # 准备摘要库的搜索请求
        # 注意：这里也应该应用外部传入的 current_filter (比如年份过滤)，以确保找出的摘要符合年份要求
        abstract_expr = self.current_filter if self.current_filter else ""

        abs_dense_req = AnnSearchRequest(
            data=[dense_vec], anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=self.top_k_abstracts, expr=abstract_expr
        )
        abs_sparse_req = AnnSearchRequest(
            data=[sparse_vec], anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, 
            limit=self.top_k_abstracts, expr=abstract_expr
        )
        
        rerank = RRFRanker(k=60)
        
        try:
            abs_col = Collection(self.abstract_collection)
            abs_res = abs_col.hybrid_search(
                [abs_dense_req, abs_sparse_req], rerank=rerank, limit=self.top_k_abstracts,
                output_fields=["doi"] # 只需要 DOI
            )
        except Exception as e:
            print(f"Abstract search failed: {e}")
            return []

        # 提取相关文献的 DOI
        target_dois = []
        for hits in abs_res:
            for hit in hits:
                doi = hit.entity.get("doi")
                if doi and doi not in target_dois:
                    target_dois.append(doi)
        
        if not target_dois:
            print("No relevant abstracts found.")
            return []

        # -------------------------------------------------------
        # 第三步：构建全文库的过滤条件 (DOI Routing)
        # -------------------------------------------------------
        # 格式: doi in ["10.111", "10.222"]
        formatted_dois = [f'"{d}"' for d in target_dois]
        doi_filter = f'doi in [{", ".join(formatted_dois)}]'
        
        # 合并外部过滤器 (年份等) 和 DOI 过滤器
        if self.current_filter:
            final_expr = f"({self.current_filter}) and ({doi_filter})"
        else:
            final_expr = doi_filter

        # -------------------------------------------------------
        # 第四步：检索全文库 (Target Collection)
        # -------------------------------------------------------
        full_dense_req = AnnSearchRequest(
            data=[dense_vec], anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=self.top_k_fulltext, expr=final_expr
        )
        full_sparse_req = AnnSearchRequest(
            data=[sparse_vec], anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, 
            limit=self.top_k_fulltext, expr=final_expr
        )

        try:
            full_col = Collection(self.fulltext_collection)
            full_res = full_col.hybrid_search(
                [full_dense_req, full_sparse_req], rerank=rerank, limit=self.top_k_fulltext,
                output_fields=["text", "title", "doi", "journal", "year", "page"]
            )
        except Exception as e:
            print(f"Fulltext search failed: {e}")
            return []

        # 转换结果为 Document 对象
        documents = []
        for hits in full_res:
            for hit in hits:
                metadata = {
                    "title": hit.entity.get("title"),
                    "doi": hit.entity.get("doi"),
                    "journal": hit.entity.get("journal"),
                    "year": hit.entity.get("year"),
                    "page": hit.entity.get("page"),
                    "score": hit.score,
                    "source_collection": self.fulltext_collection
                }
                documents.append(Document(page_content=hit.entity.get("text"), metadata=metadata))

        return documents

def get_routing_retriever(connection_args, fulltext_col_name, top_k=10):
    """
    工厂函数：获取路由检索器
    """
    # 初始化 embedding (与 get_hybrid_retriever 保持一致)
    embedding = BGEM3EmbeddingFunction(
        model_name = 'BAAI/bge-m3', 
        device='cuda', 
        use_fp16=True
    )
    
    retriever = MilvusRoutingRetriever(
        abstract_collection="xfel_abstracts_v1", # 硬编码摘要库
        fulltext_collection=fulltext_col_name,   # 动态全文库
        connection_args=connection_args,
        embedding_function=embedding,
        top_k_fulltext=top_k
    )
    return retriever

def grade_relevance(question, docs, llm):
    """验证检索文档的相关性"""
    prompt = ChatPromptTemplate.from_template("""
    Assessment: Is the following context relevant to the user's question? 
    Context: {context}
    Question: {question}
    Answer only 'yes' or 'no'.
    """)
    chain = prompt | llm | StrOutputParser()
    context_str = "\n\n".join([d.page_content for d in docs])
    return chain.invoke({"question": question, "context": context_str}).strip().lower()

def grade_hallucination(answer, docs, llm):
    """验证是否存在幻觉（是否基于文档回答）"""
    prompt = ChatPromptTemplate.from_template("""
    Is the following answer supported BY the provided context?
    Context: {context}
    Answer: {answer}
    Answer only 'yes' or 'no'.
    """)
    chain = prompt | llm | StrOutputParser()
    context_str = "\n\n".join([d.page_content for d in docs])
    return chain.invoke({"answer": answer, "context": context_str}).strip().lower()

def grade_utility(question, answer, llm):
    """验证回答是否有用（是否解决了问题而非回避）"""
    prompt = ChatPromptTemplate.from_template("""
    Does the following answer resolve the user's question effectively? 
    If the answer says 'I don't know' or similar, answer 'no'. 
    Otherwise, answer 'yes'.
    Question: {question}
    Answer: {answer}
    """)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "answer": answer}).strip().lower()

def rewrite_query_with_feedback(original_question, current_rewrite, feedback, llm):
    """
    根据用户提供的反馈（feedback），对当前的重写结果（current_rewrite）进行再次优化。
    """
    template = """You are an AI assistant helping a researcher refine their search query for an XFEL database.
    
    Original User Intent: {original_question}
    Current Rewritten Version: {current_rewrite}
    User Feedback: {feedback}
    
    Task: Based on the feedback, provide an improved version of the rewritten query. 
    Focus on technical precision and XFEL terminology.
    Return ONLY the new rewritten question, no explanation.

    New Rewritten Question:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    new_rewrite = chain.invoke({
        "original_question": original_question,
        "current_rewrite": current_rewrite,
        "feedback": feedback
    })
    return new_rewrite.strip()