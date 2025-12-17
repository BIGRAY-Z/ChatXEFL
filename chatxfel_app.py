from onnx import ModelProto
import streamlit as st
import sys
import time
from datetime import datetime
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.chat_models import ChatOllama
from streamlit import session_state as ss
from streamlit.runtime.scriptrunner import get_script_run_ctx
import os
import ui_utils  # <--- 【新增 1】引入打字机
import chat_manager # 【新增】引入新的对话管理器
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将其加入到系统路径中
sys.path.append(current_dir)
import rag
import utils

# App title
#st.set_page_config(page_title="ChatXFEL", layout='wide')
st.set_page_config(page_title="ChatXFEL Beta 1.0", page_icon='./draw/logo.png')

st.header('ChatXFEL: Q & A System for XFEL')
if 'agree' not in ss:
    ss['agree'] = False
def update_agree():
    ss['agree'] = True
    
if not ss['agree']:
    with st.empty():
        msg = '''This page is an intelligent system to answer the questions in the field of XFEL. If you click the **agree box** below, 
        :red[you IP and the time will be recorded]. If you don't agree with that, please close the page. 
        This note will appear again when you refresh the page.''' 
        st.markdown(msg)
    agree = st.checkbox('Agree', key='read', value=False, on_change=update_agree)
    while True:
        time.sleep(3600)

def reset_retriever_cache():
    try:
        get_retriever.clear()
        get_retriever_runtime.clear()
    except Exception as e:
        pass

with st.sidebar:
    st.title('ChatXFEL Beta 1.0')
    
    # --- 【新增/修改】多对话管理区域 Start ---
    # 1. 初始化 Session
    chat_manager.init_session()
    
    # 2. 新建对话按钮
    if st.button('➕ New Chat', use_container_width=True):
        chat_manager.create_new_chat()
        st.rerun() # 强制刷新页面以更新右侧聊天区

    # 3. 历史对话列表 (使用 Expander 折叠)
    with st.expander("🕒 Chat History", expanded=True):
        if not st.session_state.chat_history:
            st.write("No history yet.")
        else:
            for chat in st.session_state.chat_history:
                # 给当前选中的对话加个视觉标记
                label = chat['title']
                if chat['id'] == st.session_state.current_chat_id:
                    label = f"🟢 {label}"
                
                # 点击历史记录切换
                if st.button(label, key=f"hist_{chat['id']}", use_container_width=True):
                    chat_manager.switch_chat(chat['id'])
                    st.rerun()
    
    st.divider() # 加个分割线美观一点
    # --- 【新增/修改】多对话管理区域 End ---
    
    #st.markdown('[About ChatXFEL](https://confluence.cts.shanghaitech.edu.cn/pages/viewpage.action?pageId=129762874)')
    st.markdown('[ChatXFEL简介与提问技巧](https://confluence.cts.shanghaitech.edu.cn/pages/viewpage.action?pageId=129762874)')
    #st.write(':red[You have agreed the recording of your IP and access time.]')
    #st.markdown('**IMPORTANT: The answers given by ChatXFEL are for informational purposes only, please consult the references in the source.**')
    st.markdown('**重要提示：大模型的回答仅供参考，点击Sources查看参考文献**')
    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    #st.subheader('Models and parameters')
    #model_list = ['LLaMA3.1-8B', 'Qwen2.5-7B']
    model_list = ['Qwen3-30B']
    col_list = ['xfel_bibs_collection', 'xfel_bibs_collection_with_abstract', 'xfel_imported_v1','fix_with_abstract_only']
    embedding_list = ['BGE-M3']

    selected_model = st.sidebar.selectbox('LLM model', model_list, index=0, key='selected_model')
    
    n_recall = 6 if selected_model.startswith('Q') else 5
    #if selected_model == 'LLaMA3-8B':
    #    #model_path = '/data-10gb/data/llm/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf'
    #    n_recall = 5
    #elif selected_model == 'LLaMA3.1-8B':
    #    #model_path = '/data-10gb/data/llm/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf'
    #    n_recall = 5
    #elif selected_model == 'Qwen2.5-7B':
    #    #model_path = '/data'
    #    n_recall = 5
    #elif selected_model == 'Qwen2.5-14B':
    #    #model_path = '/data-10gb/data/llm/qwen/qwen2-7b-instruct-q8_0.gguf'
    #    n_recall = 5

    selected_em = st.sidebar.selectbox('Embedding model', embedding_list, key='selected_em')
    if selected_em == 'llama2-7b':
        col_list.append('llama2_7b')
    elif selected_em == 'llama3-8b':
        col_list.append('llama3_8b')
    #selected_col = st.sidebar.selectbox('Bibliography collection', col_list, key='select_col', on_change=reset_retriever_cache)
    selected_col = st.sidebar.selectbox('Bibliography collection', col_list, key='select_col', on_change=reset_retriever_cache)
    col_name = selected_col
    with st.popover('About the collection'):
        if col_name == 'book':
            msg = '''This collection now only contains some theses from EuXFEL.'''
            st.markdown(msg)
        if col_name == 'chatxfel':
            msg = '''This collection contains 3000+ publications of wordwide XFEL facilities, so ChatXFEL may be slower than other collections'''
            st.markdown(msg)
        if col_name == 'report':
            msg = '''This collection only contains unpulished references, e.g CDR, TDR, engineering reports.'''
            st.markdown(msg)

    filter_year = st.sidebar.checkbox('Filter papers by year', key='filter_year', value=True)
    if filter_year:
        min_year = 1949
        max_year = datetime.now().year
        year1, year2 = st.columns([1,1])
        #year_start = year1.selectbox('Start', list(range(min_year, max_year+1))[::-1], key='year_start', index=max_year-min_year)
        year_start = year1.selectbox('Start', list(range(min_year, max_year+1))[::-1], key='year_start', index=max_year-2000)
        year_end = year2.selectbox('End', list(range(year_start, max_year+1))[::-1], key='year_end')
    filter_keyword = st.sidebar.checkbox('Filter by keywords', key='filter_keyword', value=False)
    keyword_expr = ""

    if filter_keyword:
        key_input = st.sidebar.text_input('Keywords in title', key='key_title', placeholder='e.g. XFEL, laser')
        if key_input:
            # 支持逗号分隔的多个关键词，逻辑可以是 OR 或 AND，这里演示 OR
            keywords = [k.strip() for k in key_input.split(',') if k.strip()]
            if keywords:
                # 构建类似 (title like "%XFEL%" or title like "%laser%") 的表达式
                # 注意：Milvus 的 like 语法支持 % 通配符
                sub_exprs = [f'title like "%{k}%"' for k in keywords]
                keyword_expr = f"({' or '.join(sub_exprs)})"
    
    filters = {}
    expr_parts = []

    # 1. 添加年份过滤
    if filter_year:
        expr_parts.append(f'(year >= {year_start} and year <= {year_end})')

    # 2. 添加关键词过滤
    if keyword_expr:
        expr_parts.append(keyword_expr)

    # 3. 组合表达式
    if expr_parts:
        filters['expr'] = " and ".join(expr_parts)

    enable_abstract_routing = st.sidebar.checkbox(
    'Enable Abstract Routing', 
    value=False, 
    help="First search abstracts to find relevant papers, then retrieve detailed content."
)
    n_batch, n_ctx, max_tokens = 512, 8192, 8192 
    #return_source = st.sidebar.checkbox('Return Source', key='source', value=True)
    return_source = True
    use_mongo = True
    enable_log = st.sidebar.checkbox('Enable log', key='log', value=True)
    use_monog = False
    if enable_log:
        with st.popover(':warning: :red[About the log]'):
            msg = '''All the questions, answers, retrieved documents, and the question time will be logged. 
            The logs would only be used for the development of ChatXFEL. \n\nIf you don't like the log, just uncheck the box "Enable log" above.
            \n\n**Your IP address will always be recorded.**''' 
            st.markdown(msg)

@st.cache_resource
def get_embedding(embedding_model, n_ctx, n_gpu_layers=1):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting embedding...")
    # Get embedding
    if embedding_model == 'BGE-M3':
        embedding = rag.get_embedding_bge()
    return embedding
embedding = get_embedding(embedding_model=selected_em, n_ctx=n_ctx)
#print(f'Embedding: {embedding}')

#@st.cache_resource
#def get_llm(model_name, model_path, n_batch, n_ctx, max_tokens):
#    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting llm...")
#    # Get llm
#    llm = rag.get_llm_llama(model_name=model_name, model_path=model_path, n_batch=n_batch,n_ctx=n_ctx,verbose=False,
#                            streaming=True,max_tokens=max_tokens, temperature=0.8)
#    return llm
#llm = get_llm(selected_model, model_path, n_batch, n_ctx, max_tokens)

#def get_llm_ollama(model_name, num_predict, num_ctx=8192, keep_alive=-1, temperature=0.1, base_url='http://10.15.85.78:11434'):
@st.cache_resource
def get_llm(model_name, num_predict, keep_alive, num_ctx=8192, temperature=0.0):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting LLM...")
    llm = rag.get_llm_ollama(model_name=model_name, num_predict=num_predict, 
                             keep_alive=keep_alive, num_ctx=num_ctx, temperature=temperature, base_url='http://10.15.102.186:9000')
    return llm
llm = get_llm(model_name=selected_model, num_predict=2048, keep_alive=-1)

#You should answer the question in detail as far as possible. Do not make up questions by yourself.
#If you cannot find anwser in the context, just say that you don't know, don't try to make up an answer.
#Please remember some common abbrevations: SFX is short for serial femtosecond crystallography, SPI is 
#short for single particle imaging. 
#
#{context}
#
#Question: {question}
#Helpful Answer:"""
with open('naive.pt', 'r') as f:
    prompt_template = f.read()

@st.cache_data
def get_prompt_template(template):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting prompt...")
    prompt = rag.get_prompt(template)
    return prompt
prompt = get_prompt_template(template=prompt_template)

@st.cache_resource
def get_rerank_model(model_name='', top_n=n_recall):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting rerank model...")
    if model_name == '':
        model_name = 'BAAI/bge-reranker-v2-m3'
    rerank_model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)
    return compressor

connection_args = utils.get_milvus_connection()
@st.cache_resource
def get_retriever(connection_args, col_name, _embedding):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting retriever...")
    if selected_em in ['llama2-7b', 'llama3-8b']:
        retriever = rag.get_retriever(connection_args=connection_args, col_name=col_name,
                                      embedding=_embedding, use_rerank=False, return_as_retreiever=False)
    else:
        retriever = rag.get_retriever(connection_args=connection_args, col_name=col_name,
                                      embedding=_embedding, vector_field='dense_vector',
                                      use_rerank=False, return_as_retreiever=False)
    return retriever

# chatxfel_app.py 中的更新代码

@st.cache_resource
def get_retriever_runtime(_retriever_obj, _compressor, filters=None):
    # print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting retriever at runtime...")
    
    base_retriever = None

    # 情况 A: 标准 LangChain VectorStore (如 LLaMA2/3 模式)
    # 这种对象不是 Retriever，需要调用 .as_retriever() 转换
    if hasattr(_retriever_obj, "as_retriever"):
        search_kwargs = {'k': 10}
        if filters:
            search_kwargs = {**search_kwargs, **filters} # 将 filters 合并进 search_kwargs
        base_retriever = _retriever_obj.as_retriever(search_kwargs=search_kwargs)
    
    # 情况 B: 我们自定义的 MilvusHybridRetriever
    # 它本身就是 Retriever，直接使用即可。我们需要手动把 filters 塞给它。
    else:
        # 如果有过滤条件 (如 year 范围或 title 关键词)
        if filters and "expr" in filters:
            # 将表达式赋值给我们在 rag.py 中新增的 current_filter 属性
            _retriever_obj.current_filter = filters["expr"]
        else:
            _retriever_obj.current_filter = ""
            
        base_retriever = _retriever_obj

    # 最后的重排序包装
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

@st.cache_resource
def get_hybrid_retriever_obj(connection_args, col_name):
    # 调用我们在 rag.py 中新写的函数
    return rag.get_hybrid_retriever(connection_args, col_name, top_k=10)

@st.cache_resource
def get_routing_retriever_obj(connection_args, col_name):
    # 调用 rag.py 中新写的工厂函数
    return rag.get_routing_retriever(connection_args, col_name, top_k=10)

# 实例化
if selected_em == 'BGE-M3': 
    if enable_abstract_routing:
        # 使用新的路由检索器
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Using Abstract Routing Retriever...")
        retriever_obj = get_routing_retriever_obj(connection_args, selected_col)
    else:
        # 使用原有的混合检索器
        retriever_obj = get_hybrid_retriever_obj(connection_args, selected_col)
else:
    # 其它模型(Llama等)继续使用旧逻辑
    retriever_obj = get_retriever(connection_args, selected_col, embedding)
#retriever_obj = get_retriever(connection_args, selected_col, embedding)
compressor = get_rerank_model(top_n=n_recall)
filters = {}
if filter_year:
    #filters['expr'] = f'year >= {year_start} and year <= {year_end}'
    filters['expr'] = f'{year_start} <= year <= {year_end}'
#if filter_title:
#    expr_title = ''
#    for i, word in enumerate(keywords):
#        if i ==  len(keywords) -1:
#            expr_title += f'\"{word}\" in title'
#        else:
#            expr_title += f'\"{word}\" in title and '
#    if 'expr' in filters.keys():
#        filters['expr'] += ' and ' + expr_title
#    else:
#        filters['expr'] = expr_title
    
retriever = get_retriever_runtime(retriever_obj, compressor, filters=filters)

initial_message = {"role": "assistant", "content": "What do you want to know about XFEL?"}
# Store LLM generated responses
if "messages" not in ss.keys():
    # ss.messages = [initial_message]
    chat_manager.create_new_chat(reset_ui=True)

def log_feedback(feedback:dict, use_mongo):
    if feedback.get('Feedback', '') == '':
        feedback['Feedback'] = ss['feedback']+1
    utils.log_rag(feedback, use_mongo=use_mongo)

for message in ss.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        #try:
        c = st.columns([8,2.5])
        if 'source' in message.keys():
            with c[0].popover('Sources'):
                st.markdown(message['source'])
            if message == ss.messages[-1]:
                if 'feedback' in ss:
                    ss['feedback'] = None
                with c[1]:
                    feedback = st.feedback('stars', key='feedback', on_change=log_feedback, args=({'Feedback':''}, use_mongo,))
                    #if feedback is not None:
                    #    log_feedback({'Feedback':str(feedback+1)}, use_mongo=use_mongo)
                #with c[1]:
                #    good = st.button(':thumbsup:', key='feedback_good_1', on_click=log_feedback, args=({'Feedback':'Good'},use_mongo,))
                #with c[2]:
                #    bad = st.button(':thumbsdown:', key='feedback_bad_1', on_click=log_feedback, args=({'Feedback':'Bad'}, use_mongo,))
        #except Exception as e:
        #    pass
        #num += 1
        #if 'messages' in ss.keys():
        #    ele = st.columns([3,1,1,1,1])
        #    if 'source' in message.keys():
        #        with ele[0]:
        #            with st.popover('Show source'):
        #                st.write(message['source'])
        #        if message['role'] == 'assistant':
        #            ele[1].button('Like', key=f'01{num}')
        #            ele[2].button('Dislike', key=f'02{num}')
        #            ele[3].button('Retry', key=f'03{num}')
        #            ele[4].button('Modify', key=f'04{num}')

def clear_chat_history():
    #ss.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    ss.messages = [initial_message]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(question):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. \
        You only respond once as 'Assistant'."
    for dict_message in ss.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    #output = replicate.run(llm, 
    #                       input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
    #                              "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    output = rag.retrieve_generate(question=question, llm=llm, prompt=prompt,retriever=retriever,
                                  return_source=return_source, return_chain=False)

    return output

@st.cache_data
def log_ip_time(session_id):
    ip = session.request.remote_ip
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {ip} connected or refreshed!", flush=True)

ctx = get_script_run_ctx()
client_ip = ''
if ctx:
    session = st.runtime.get_instance().get_client(ctx.session_id)
    client_ip = session.request.remote_ip
    log_ip_time(ctx.session_id)

# User-provided prompt
question_time = ''
if question:= st.chat_input():
    if enable_log:
        question_time = time.strftime('%Y-%m-%d %H:%M:%S')
    ss.messages.append({"role": "user", "content": question})
    # 【新增】关键点：用户输入完问题后，立即保存状态
    # 这样 chat_manager 就能把 "New Chat" 的标题改成这个问题的内容
    chat_manager.save_current_chat()
    with st.chat_message("user"):
        st.write(question)

# Generate a new response if last message is not from assistant
if 'feedback_good' not in ss:
    ss['feedback_good'] = None
if 'feedback_bad' not in ss:
    ss['feedback_bad'] = None

if ss.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rewritten_question = rag.rewrite_query(question, llm)
            #print(f"Original: {question} -> Rewritten: {rewritten_question}")
            # response = rag.retrieve_generate(
            #     question=rewritten_question, 
            #     llm=llm, 
            #     prompt=prompt,
            #     retriever=retriever,
            #     return_source=return_source
            # )
            p = ' Please answer the question as detailed as possible and make up you answer in markdown format.'
            final_question = f"{rewritten_question}{p}"
            response = rag.retrieve_generate(
                question=final_question, 
                llm=llm, 
                prompt=prompt,
                retriever=retriever, 
                return_source=return_source
            )
            #response = generate_llama2_response(question)
            placeholder = st.empty()
            full_response = ''
            source = ''
            if return_source:
                # 1. 获取纯文本回答
                answer_text = response['answer']
                
                # 2. 使用新工具进行打字机效果输出
                # 注意：这里只对 answer_text 进行流式输出
                ui_utils.stream_output(placeholder, answer_text)
                
                full_response += answer_text
                # full_response += response['answer']
                # placeholder.markdown(full_response)
                # #full_response += '\nContext: \n'
                for i, c in enumerate(response['context']):
                    source += f'{c.page_content}'
                    title = c.metadata.get('title') if 'title' in c.metadata.keys() else c.metadata.get('source')
                    doi = c.metadata.get('doi', '')
                    journal = c.metadata.get('journal', '')
                    year = c.metadata.get('year', '')
                    page = c.metadata.get('page')
                    if doi == '':
                        source += f'\n\n**Ref. {i+1}**: {title}, {journal}, {year}, page {page}'
                    else:
                        source += f'\n\n**Ref. {i+1}**: {title}, {journal}, {year}, [{doi}](http://dx.doi.org/{doi}), page {page}'
                    if i != len(response['context'])-1:
                        source += '\n\n'
                    #placeholder.markdown(source)
                    if i == len(response['context'])-1:
                        c = st.columns([8,3])
                        with c[0].popover('Source'):
                            st.markdown(source)
                        #with c[1]:
                        #    feedback = st.feedback('stars', key='feedback')
                        #    if feedback is not None:
                        #        log_feedback({'Feedback':str(feedback+1)}, use_mongo=use_mongo)
                        #with c[1]:
                        #    good = st.button(':thumbsup:', key='feedback_good', on_click=log_feedback, args=({'Feedback':'Good'},use_mongo,))
                        #with c[2]:
                        #    bad = st.button(':thumbsdown:', key='feedback_bad', on_click=log_feedback, args=({'Feedback':'Bad'}, use_mongo,))
            else:
                full_response = response.content
                #for item in response:
                #    full_response += item
                #    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
            #placeholder.markdown(full_response)

    if return_source:
        message = {"role": "assistant", "content": full_response, "source":source}
        #if enable_log:
            #utils.log_rag(client_ip, question_time, question, full_response, source, use_mongo=False)
    else:
        message = {"role": "assistant", "content": full_response}
        #if enable_log:
        #    utils.log_rag(client_ip, question_time, question, full_response, use_mongo=False)
    if enable_log:
        logs = {'IP':client_ip, 'Time':question_time, 'Model':selected_model, 'Question': question, 'Answer':full_response, 'Source':source}
        utils.log_rag(logs, use_mongo=use_mongo)
    ss.messages.append(message)
    # 【新增】关键点：AI 回答完毕后，再次保存状态
    # 确保刚才生成的回答被存入 chat_history 列表
    chat_manager.save_current_chat()
    st.rerun()
#c = st.columns([8,2.5])
#feedback = st.feedback('stars', key='feedback')
#with c[0]:
#    pass
#with c[1]:
#    if feedback is not None:
#        log_feedback({'Feedback':str(feedback+1)}, use_mongo=use_mongo)
    #history.append({'role':'user','content':question})
    #history.append({'role':'assistant', 'content':full_response})
