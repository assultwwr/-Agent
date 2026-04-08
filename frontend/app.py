"""
智扫通机器人智能客服 - Streamlit 前端入口
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from agent.react_agent import ReactAgent

# 页面配置
st.set_page_config(
    page_title="智扫通机器人智能客服",
    page_icon="🤖",
    layout="wide"
)

# 初始化 session state
if "agent" not in st.session_state:
    with st.spinner("正在初始化智能客服系统..."):
        st.session_state.agent = ReactAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 自定义 CSS 样式
st.markdown("""
<style>
.stChatMessage {
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.user-message {
    background-color: #e3f2fd;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.assistant-message {
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# 标题区域
st.title("🤖 智扫通机器人智能客服")
st.markdown("您的智能扫地机器人助手，随时为您解答问题")

# 功能介绍
with st.expander("📋 功能介绍", expanded=False):
    st.markdown("""
    - ✅ RAG 知识库问答
    - ✅ 天气查询
    - ✅ 用户位置获取
    - ✅ 使用报告生成
    - ✅ 外部数据查询
    """)

# 聊天历史显示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成 AI 响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 流式显示响应
        for chunk in st.session_state.agent.execute_stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # 保存 AI 响应到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 侧边栏 - 清空聊天
with st.sidebar:
    st.title("⚙️ 设置")
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 关于")
    st.markdown("**智扫通机器人智能客服 v1.0**")
    st.markdown("基于 LangGraph + Milvus 构建")
