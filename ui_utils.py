# ui_utils.py
import time

def stream_output(placeholder, text, delay=0.02):
    """
    在 Streamlit 的 placeholder 中实现打字机效果输出文本。
    
    Args:
        placeholder: Streamlit 的 st.empty() 或容器对象
        text: 需要输出的完整字符串
        delay: 每个字/词之间的间隔时间（秒）
    """
    full_text = ""
    # 为了避免过于频繁的刷新导致前端卡顿，我们按“词”或者“字”分割
    # 这里使用简单的空格分割，既保留了英文单词完整性，也能处理中文
    
    # 如果是纯中文环境，可以使用 chunks = list(text) 按字符分割
    chunks = text.split(' ') 
    
    for i, chunk in enumerate(chunks):
        full_text += chunk + " "
        # 更新 placeholder，并加上一个模拟的光标 "▌"
        placeholder.markdown(full_text + "▌")
        time.sleep(delay)
        
    # 最后移除光标，显示最终文本
    placeholder.markdown(full_text)