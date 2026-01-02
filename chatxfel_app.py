import re
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
import ui_utils
import chat_manager

current_dir = os.path.dirname(os.path.abspath(__file__))
# 将其加入到系统路径中
sys.path.append(current_dir)
import rag
import utils

# --- App Configuration ---
st.set_page_config(page_title="ChatXFEL Beta 1.0", page_icon='./draw/logo.png', layout='wide')

# --- Header ---
st.header('ChatXFEL: Q & A System for XFEL')

# --- CSS Styling (保持 new_chatxfel_app 的样式优化) ---
st.markdown(
    """
    <style>
    /* 调整侧边栏宽度 */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        width: 320px !important;
    }
    /* 侧边栏按钮微调 */
    [data-testid="stSidebar"] button {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if 'agree' not in ss:
    ss['agree'] = False
if 'rewrite_stage' not in ss:
    ss['rewrite_stage'] = False      # 标识当前是否处于“等待用户确认Query”的状态
if 'temp_query' not in ss:
    ss['temp_query'] = ""            # 存储中间生成的重写结果
if 'confirmed_query' not in ss:
    ss['confirmed_query'] = ""       # 存储用户最终确认的重写结果

# --- Agreement Logic ---
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

def clear_chat_history():
    # 清空当前对话及相关状态
    ss.messages = [{"role": "assistant", "content": "What do you want to know about XFEL?"}]
    ss.rewrite_stage = False
    ss.temp_query = ""
    ss.confirmed_query = ""

# --- Dialogs (保留 new_chatxfel_app 的弹窗逻辑) ---
@st.dialog("⚠️ Confirm Deletion")
def open_delete_dialog(chat):
    st.write(f"Are you sure you want to permanently delete **{chat['title']}**?")
    st.warning("This action cannot be undone.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun() 
            
    with col2:
        if st.button("Delete", type="primary", use_container_width=True):
            chat_id_to_delete = chat['id']
            current_id = st.session_state.current_chat_id
            
            # 1. 物理移除
            st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat_id_to_delete]
            
            # 2. 判断逻辑
            if chat_id_to_delete == current_id:
                if not st.session_state.chat_history:
                    chat_manager.create_new_chat()
                else:
                    new_target_id = st.session_state.chat_history[0]['id']
                    chat_manager.switch_chat(new_target_id)
            
            # 3. 刷新页面
            st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.title('ChatXFEL Beta 1.0')
    st.markdown('[ChatXFEL简介与提问技巧](https://confluence.cts.shanghaitech.edu.cn/pages/viewpage.action?pageId=129762874)')
    st.markdown('**重要提示：大模型的回答仅供参考，点击Sources查看参考文献**')
    
    # --- Settings & Filters (保留 new_chatxfel_app 的折叠设计) ---
    with st.expander("⚙️ Settings & Filters", expanded=False):
        st.caption("Configure Model & Search")
        
        model_list = ['Qwen3-30B']
        col_list = ['xfel_bibs_collection', 'xfel_bibs_collection_with_abstract', 'xfel_imported_v1','fix_with_abstract_only']
        embedding_list = ['BGE-M3']

        selected_model = st.selectbox('LLM model', model_list, index=0, key='selected_model')
        n_recall = 6 if selected_model.startswith('Q') else 5

        selected_em = st.selectbox('Embedding model', embedding_list, key='selected_em')
        if selected_em == 'llama2-7b':
            col_list.append('llama2_7b')
        elif selected_em == 'llama3-8b':
            col_list.append('llama3_8b')
        selected_col = st.selectbox('Bibliography collection', col_list, key='select_col', on_change=reset_retriever_cache)
        col_name = selected_col
        
        if col_name == 'book':
            st.info('Collection: Theses from EuXFEL.')
        if col_name == 'chatxfel':
            st.info('Collection: 3000+ publications (slower).')
        if col_name == 'report':
            st.info('Collection: Unpublished references (CDR, TDR).')

        st.caption("Filters")
        filter_year = st.checkbox('Filter by year', key='filter_year', value=True)
        year_start = 1949
        year_end = datetime.now().year
        
        if filter_year:
            min_year = 1949
            max_year = datetime.now().year
            c_y1, c_y2 = st.columns([1,1])
            year_start = c_y1.selectbox('Start', list(range(min_year, max_year+1))[::-1], key='year_start', index=max_year-2000)
            year_end = c_y2.selectbox('End', list(range(year_start, max_year+1))[::-1], key='year_end')
            
        filter_keyword = st.checkbox('Filter by keywords', key='filter_keyword', value=False)
        keyword_expr = ""

        if filter_keyword:
            key_input = st.text_input('Keywords in title', key='key_title', placeholder='e.g. XFEL, laser')
            if key_input:
                keywords = [k.strip() for k in key_input.split(',') if k.strip()]
                if keywords:
                    sub_exprs = [f'title like "%{k}%"' for k in keywords]
                    keyword_expr = f"({' or '.join(sub_exprs)})"
        
        # Filters 逻辑构建
        filters = {}
        expr_parts = []
        if filter_year:
            expr_parts.append(f'(year >= {year_start} and year <= {year_end})')
        if keyword_expr:
            expr_parts.append(keyword_expr)
        if expr_parts:
            filters['expr'] = " and ".join(expr_parts)

        enable_abstract_routing = st.checkbox('Abstract Routing', value=False, help="First search abstracts to find relevant papers.")
        n_batch, n_ctx, max_tokens = 512, 8192, 8192 
        return_source = True
        use_mongo = True
        enable_log = st.checkbox('Enable log', key='log', value=True)
        use_monog = False
        
        # Response Mode
        response_mode = st.select_slider(
            'Response Mode',
            options=['Strict (Rigorous)', 'Balanced', 'Creative (Flexible)'],
            value='Balanced',
            help="Strict: Only answers from papers. Creative: Uses AI knowledge."
        )

    # --- Chat Management ---
    chat_manager.init_session()
    
    if st.button('➕ New Chat', use_container_width=True):
        chat_manager.create_new_chat()
        # Reset rewrite states on new chat
        ss.rewrite_stage = False
        ss.temp_query = ""
        ss.confirmed_query = ""
        st.rerun() 

    with st.expander("🕒 Chat History", expanded=True):
        if not st.session_state.chat_history:
            st.write("No history yet.")
        else:
            for i, chat in enumerate(st.session_state.chat_history):
                col_title, col_del = st.columns([0.8, 0.2])
                label = chat['title']
                if chat['id'] == st.session_state.current_chat_id:
                    label = f"🟢 {label}"
                
                with col_title:
                    if st.button(label, key=f"hist_{chat['id']}", use_container_width=True):
                        chat_manager.switch_chat(chat['id'])
                        ss.rewrite_stage = False # 切换对话时退出重写状态
                        st.rerun()
                
                with col_del:
                    if st.button("🗑️", key=f"del_btn_{chat['id']}"):
                        open_delete_dialog(chat)

    st.button('Clear Current Chat', on_click=clear_chat_history, use_container_width=True)
    st.divider() 

# --- Backend Resources (Cache) ---
@st.cache_resource
def get_embedding(embedding_model, n_ctx, n_gpu_layers=1):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting embedding...")
    if embedding_model == 'BGE-M3':
        embedding = rag.get_embedding_bge()
    return embedding
embedding = get_embedding(embedding_model=selected_em, n_ctx=n_ctx)

@st.cache_resource
def get_llm(model_name, num_predict, keep_alive, num_ctx=8192, temperature=0.0):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting LLM...")
    llm = rag.get_llm_ollama(model_name=model_name, num_predict=num_predict, 
                             keep_alive=keep_alive, num_ctx=num_ctx, temperature=temperature, base_url='http://10.15.102.186:9000')
    return llm
llm = get_llm(model_name=selected_model, num_predict=2048, keep_alive=-1)

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

@st.cache_resource
def get_retriever_runtime(_retriever_obj, _compressor, filters=None):
    base_retriever = None
    if hasattr(_retriever_obj, "as_retriever"):
        search_kwargs = {'k': 10}
        if filters:
            search_kwargs = {**search_kwargs, **filters}
        base_retriever = _retriever_obj.as_retriever(search_kwargs=search_kwargs)
    else:
        if filters and "expr" in filters:
            _retriever_obj.current_filter = filters["expr"]
        else:
            _retriever_obj.current_filter = ""
        base_retriever = _retriever_obj

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

@st.cache_resource
def get_hybrid_retriever_obj(connection_args, col_name):
    return rag.get_hybrid_retriever(connection_args, col_name, top_k=10)

@st.cache_resource
def get_routing_retriever_obj(connection_args, col_name):
    return rag.get_routing_retriever(connection_args, col_name, top_k=10)

if selected_em == 'BGE-M3': 
    if enable_abstract_routing:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Using Abstract Routing Retriever...")
        retriever_obj = get_routing_retriever_obj(connection_args, selected_col)
    else:
        retriever_obj = get_hybrid_retriever_obj(connection_args, selected_col)
else:
    retriever_obj = get_retriever(connection_args, selected_col, embedding)

compressor = get_rerank_model(top_n=n_recall)
retriever = get_retriever_runtime(retriever_obj, compressor, filters=filters)

initial_message = {"role": "assistant", "content": "What do you want to know about XFEL?"}

# --- Load History ---
if "messages" not in ss.keys():
    current_history = ss.get('chat_history', [])
    current_id = ss.get('current_chat_id', None)
    target_chat = next((c for c in current_history if c['id'] == current_id), None)
    if target_chat:
        ss['messages'] = target_chat['messages']
    else:
        chat_manager.create_new_chat(reset_ui=True)

# --- Feedback Function ---
def log_feedback(feedback:dict, use_mongo):
    if feedback.get('Feedback', '') == '':
        feedback['Feedback'] = ss['feedback']+1
    utils.log_rag(feedback, use_mongo=use_mongo)

# --- Message Rendering (使用 new_chatxfel_app 的高级样式) ---
for message in ss.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        c = st.columns([1.2, 1.2, 7.6]) 
        
        if 'source' in message.keys():
            # 1. Source 按钮
            with c[0].popover('Sources'):
                st.markdown(message['source'])
            
            # 2. Copy 按钮 (保留高级复制功能)
            with c[1].popover("Copy"):
                st.caption("**Markdown (Original)**")
                st.code(message['content'], language='markdown')
                st.caption("**Plain Text (Cleaned)**")
                raw_text = message['content']
                plain = re.sub(r'\$\$[\s\S]*?\$\$', '', raw_text)
                plain = re.sub(r'\$.*?\$', '', plain)
                plain = re.sub(r'\*\*|__|\*|_|`|^#+\s*', '', plain, flags=re.MULTILINE)
                plain = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', plain)
                st.code(plain.strip(), language=None)

            # 3. Feedback 按钮
            if message == ss.messages[-1]:
                if 'feedback' in ss:
                    ss['feedback'] = None
                with c[2]:
                    feedback = st.feedback('stars', key='feedback', on_change=log_feedback, args=({'Feedback':''}, use_mongo,))

# --- Logging Utils ---
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

# --- Input Handling & Rewrite Initiation ---
question_time = ''
if question:= st.chat_input():
    if enable_log:
        question_time = time.strftime('%Y-%m-%d %H:%M:%S')
    ss.messages.append({"role": "user", "content": question})
    
    # 整合逻辑：触发 Interactive Rewrite (来自 chatxfel_app)
    with st.spinner("Optimizing your query for XFEL database..."):
        ss.temp_query = rag.rewrite_query(question, llm) 
        ss.rewrite_stage = True
        ss.confirmed_query = "" # 重置确认状态

    chat_manager.save_current_chat()
    with st.chat_message("user"):
        st.write(question)
    st.rerun()

# Feedback session init
if 'feedback_good' not in ss:
    ss['feedback_good'] = None
if 'feedback_bad' not in ss:
    ss['feedback_bad'] = None

# --- Interactive Rewrite Stage (来自 chatxfel_app) ---
if ss.rewrite_stage:
    with st.chat_message("assistant", avatar="🔍"):
        st.info("I have rewritten your query to improve search results. You can refine it further:")
        
        # 1. 允许手动修改 Query
        ss.temp_query = st.text_area(
            "Refined Search Query (Full View):", 
            value=ss.temp_query,
            height=120,
            help="You can manually edit this text to precisely match your needs."
        )
        
        # 2. 接收用户反馈进行再次 AI 修改
        user_feedback = st.text_input("Provide feedback to AI for better rewriting (optional):", 
                                      placeholder="e.g. 'Focus on the detector part', 'Expand abbreviations'")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✅ Confirm & Search", type="primary"):
                ss.confirmed_query = ss.temp_query # 保存用户确认的 Query
                ss.rewrite_stage = False # 结束重写阶段
                st.rerun()
        with col2:
            if st.button("🔄 Refine with AI"):
                if user_feedback:
                    with st.spinner("Refining..."):
                        ss.temp_query = rag.rewrite_query_with_feedback(
                            ss.messages[-1]["content"], ss.temp_query, user_feedback, llm
                        )
                    st.rerun()
                else:
                    st.warning("Please enter feedback first.")
    # 阻塞后续代码，直到用户确认
    st.stop()

# --- Response Generation Logic (整合后) ---
if ss.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ''
        source = ''
        source_docs = []
        
        # 恢复原始问题文本
        original_question = ss.messages[-1]["content"]

        with st.status("Thinking...", expanded=(response_mode == 'Strict')) as status:
            
            # 【关键整合点】：使用 Interactive Rewrite 确认的结果，或者自动生成
            if ss.confirmed_query:
                rewritten_question = ss.confirmed_query
                status.write("✅ Using confirmed rewritten query.")
            else:
                rewritten_question = rag.rewrite_query(original_question, llm)
                
            p = ' Please answer the question as detailed as possible and make up you answer in markdown format.'
            final_question = f"{rewritten_question}{p}"
            
            # --- 核心检索与生成逻辑 ---
            if response_mode == 'Strict':
                max_retries = 2
                current_q = final_question
                for i in range(max_retries + 1):
                    status.write(f"🔍 Retrieval Attempt {i+1}...")
                    res_raw = retriever.invoke(current_q)
                    rel = rag.grade_relevance(original_question, res_raw, llm)
                    
                    if rel == 'yes':
                        source_docs = res_raw
                        status.write("✅ Relevant evidence found.")
                        break
                    elif i < max_retries:
                        status.write("⚠️ Low relevance. Rewriting query...")
                        # 严格模式下自动重试
                        current_q = rag.rewrite_query(f"Focus on factual details of: {original_question}", llm) + p
                    else:
                        source_docs = res_raw 
                
                status.write("✍️ Generating response...")
                response_data = rag.retrieve_generate(final_question, llm, prompt, retriever, return_source=True)
                
                status.write("🛡️ Checking for hallucinations...")
                hal = rag.grade_hallucination(response_data['answer'], source_docs, llm)
                if hal == 'no':
                    full_response = "⚠️ [Self-Correction] Based on the references, I cannot fully confirm the previous thought. " + response_data['answer']
                else:
                    full_response = response_data['answer']
                source_docs = response_data['context']

            elif response_mode == 'Creative':
                response_data = rag.retrieve_generate(final_question, llm, prompt, retriever, return_source=True)
                util = rag.grade_utility(response_data['answer'], llm)
                if util == 'no':
                    status.write("💡 No info in papers. Switching to internal knowledge...")
                    fallback_prompt = f"The following question cannot be answered by specific XFEL papers. Please answer using your internal scientific knowledge: {original_question}"
                    fallback_res = llm.invoke(fallback_prompt)
                    full_response = "💡 **Note: Based on internal AI knowledge (not found in current papers):**\n\n" + fallback_res.content
                    source_docs = []
                else:
                    full_response = response_data['answer']
                    source_docs = response_data['context']
            
            else: # Balanced
                response_data = rag.retrieve_generate(final_question, llm, prompt, retriever, return_source=True)
                full_response = response_data['answer']
                source_docs = response_data['context']
            
            status.update(label="Response Generated!", state="complete", expanded=False)

        # 2. 流式输出
        ui_utils.stream_output(placeholder, full_response)

        # 3. Source 处理 (保留 new_chatxfel_app 的美观版)
        if return_source and source_docs:
            for i, c in enumerate(source_docs):
                source += f'{c.page_content}'
                title = c.metadata.get('title', c.metadata.get('source', 'Unknown Title'))
                doi = c.metadata.get('doi', '')
                journal = c.metadata.get('journal', '')
                year = c.metadata.get('year', '')
                page = c.metadata.get('page', '')
                
                if doi == '':
                    source += f'\n\n**Ref. {i+1}**: {title}, {journal}, {year}, page {page}'
                else:
                    source += f'\n\n**Ref. {i+1}**: {title}, {journal}, {year}, [{doi}](http://dx.doi.org/{doi}), page {page}'
                
                if i != len(source_docs)-1:
                    source += '\n\n'
            
            cols = st.columns([8,3])
            with cols[0].popover('Source'):
                st.markdown(source)

    # 4. 保存与日志
    if return_source:
        message = {"role": "assistant", "content": full_response, "source": source}
    else:
        message = {"role": "assistant", "content": full_response}
        
    if enable_log:
        logs = {'IP': client_ip, 'Time': question_time, 'Model': selected_model, 
                'Mode': response_mode, 'Question': original_question, 'Answer': full_response, 'Source': source}
        utils.log_rag(logs, use_mongo=use_mongo)
    
    ss.messages.append(message)
    # 完成一次对话后，清理 rewrite 状态以防万一
    ss.confirmed_query = ""
    chat_manager.save_current_chat()
    st.rerun()