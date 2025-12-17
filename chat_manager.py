# chat_manager.py
import streamlit as st
import uuid

# 默认的欢迎语，保持和你主程序一致
INITIAL_MESSAGE = {"role": "assistant", "content": "What do you want to know about XFEL?"}

def init_session():
    """
    初始化 Session State 中的对话管理变量
    """
    # 1. 存储所有的对话历史列表
    # 结构示例: 
    # [
    #   {'id': 'uuid-1', 'title': 'XFEL原理...', 'messages': [...]}, 
    #   {'id': 'uuid-2', 'title': 'New Chat', 'messages': [...]}
    # ]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 2. 记录当前正在查看的对话 ID
    if 'current_chat_id' not in st.session_state:
        # 如果没有当前对话，就创建一个新的
        create_new_chat(reset_ui=False)

def create_new_chat(reset_ui=True):
    """
    创建一个新的对话 Session
    """
    # 如果当前已经有正在进行的对话，先保存它（防止未同步）
    if 'current_chat_id' in st.session_state:
        save_current_chat()

    new_id = str(uuid.uuid4())
    new_chat = {
        'id': new_id,
        'title': 'New Chat', # 默认标题
        'messages': [INITIAL_MESSAGE] # 初始消息
    }
    
    # 将新对话插入到历史记录的最前面
    st.session_state.chat_history.insert(0, new_chat)
    
    # 更新当前 ID
    st.session_state.current_chat_id = new_id
    
    # 如果需要立即重置 UI 显示的消息列表 (通常为 True)
    if reset_ui:
        st.session_state.messages = new_chat['messages']

def switch_chat(chat_id):
    """
    切换到指定的历史对话
    """
    # 1. 切换前，先保存当前界面的对话状态
    save_current_chat()
    
    # 2. 找到目标对话
    target_chat = next((c for c in st.session_state.chat_history if c['id'] == chat_id), None)
    
    if target_chat:
        # 3. 更新当前 ID
        st.session_state.current_chat_id = chat_id
        # 4. 这一步很关键：将 Session State 的 messages 替换为历史记录里的 messages
        st.session_state.messages = target_chat['messages']

def save_current_chat():
    """
    将当前 st.session_state.messages 同步回 chat_history 列表
    并根据第一个问题自动更新标题
    """
    if 'current_chat_id' not in st.session_state or 'messages' not in st.session_state:
        return

    current_id = st.session_state.current_chat_id
    current_msgs = st.session_state.messages
    
    # 找到当前对话在列表中的索引
    chat_index = -1
    for i, chat in enumerate(st.session_state.chat_history):
        if chat['id'] == current_id:
            chat_index = i
            break
    
    if chat_index != -1:
        # 1. 同步消息内容
        st.session_state.chat_history[chat_index]['messages'] = current_msgs
        
        # 2. 自动生成标题 (如果还是默认标题，且有了用户提问)
        # 逻辑：取第一条 role 为 user 的消息作为标题
        current_title = st.session_state.chat_history[chat_index]['title']
        if current_title == 'New Chat' and len(current_msgs) > 1:
            for msg in current_msgs:
                if msg['role'] == 'user':
                    # 截取前 20 个字符作为标题
                    new_title = msg['content'][:20].strip()
                    if len(msg['content']) > 20:
                        new_title += "..."
                    st.session_state.chat_history[chat_index]['title'] = new_title
                    break