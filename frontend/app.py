"""
智能问答 Agent - Streamlit 前端入口
"""
import sys
import os
import warnings
from pathlib import Path

# 在导入任何其他模块之前过滤警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import json
import requests
import logging

logger = logging.getLogger(__name__)

# 后端 API 地址配置
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# 页面配置
st.set_page_config(
    page_title="智能问答 Agent",
    page_icon="🤖",
    layout="centered"
)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # 当前会话 ID

if "user_ip" not in st.session_state:
    st.session_state.user_ip = ""

# 自定义 CSS 样式
st.markdown("""
<style>
/* 限制主容器宽度 */
.main .block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* 聊天消息样式 */
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
.tool-call-box {
    background-color: #e8f5e9;
    border-left: 4px solid #4CAF50;
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.9em;
}
.tool-result-box {
    background-color: #fff3e0;
    border-left: 4px solid #FF9800;
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.9em;
}

/* 加载动画样式 */
.thinking-indicator {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    vertical-align: middle;
    margin-right: 5px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 显示侧边栏展开按钮 */
[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    visibility: visible !important;
}

/* 隐藏 Streamlit 默认菜单，但保留侧边栏控制 */
#MainMenu {visibility: hidden;}
header {padding-top: 0px; background-color: transparent;}
</style>
""", unsafe_allow_html=True)

# 标题区域
st.title("🤖 智能问答 Agent")

st.markdown("您的智能问答助手，随时为您解答问题")

# 功能介绍
with st.expander("📋 功能介绍", expanded=False):
    st.markdown("""
    - ✅ RAG 知识库问答
    - ✅ 天气查询
    - ✅ 用户位置获取
    - ✅ 使用报告生成
    - ✅ 外部数据查询
    """)

# 聊天历史显示（仅展示文本消息）
for message in st.session_state.messages:
    if message["type"] == "text":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 显示 AI 思考中的加载状态
    with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown('<div class="thinking-indicator"></div> 正在思考中...', unsafe_allow_html=True)
        
        message_placeholder = st.empty()
        full_content = ""
        
        try:
            # 调用后端流式 API，传递 thread_id
            response = requests.post(
                f"{API_BASE_URL}/chat/stream",
                json={
                    "message": prompt,
                    "thread_id": st.session_state.thread_id
                },
                stream=True,
                timeout=120  # 增加到 120 秒，适应长时间的工具调用
            )
            response.raise_for_status()
            
            # 获取后端返回的 thread_id
            new_thread_id = response.headers.get("X-Thread-ID")
            if new_thread_id:
                st.session_state.thread_id = new_thread_id
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    
                    try:
                        parsed = json.loads(line_str)
                        msg_type = parsed.get("type")
                        data = parsed.get("data", {})
                        
                        if msg_type == "content":
                            # 纯文本内容
                            full_content += str(data)
                            message_placeholder.markdown(full_content + "▌")
                        elif msg_type == "tool_call":
                            # 工具调用 - 仅更新加载提示，不存储到消息历史
                            tool_name = data.get("tool_name", "unknown")
                            loading_placeholder.markdown(f'<div class="thinking-indicator"></div> 正在调用 {tool_name} 获取信息...', unsafe_allow_html=True)
                        elif msg_type == "tool_result":
                            # 工具执行结果 - 仅更新加载提示，不存储到消息历史
                            tool_name = data.get("tool_name", "unknown")
                            loading_placeholder.markdown(f'<div class="thinking-indicator"></div> {tool_name} 执行完成，正在生成回答...', unsafe_allow_html=True)
                        elif msg_type == "error":
                            # 后端返回的错误
                            error_msg = f"❌ {data}"
                            message_placeholder.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})
                            loading_placeholder.empty()
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {line_str[:100]}")
                        continue
            
            # 保存最终消息
            if full_content:
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": full_content})
                # 清除 placeholder，避免与历史消息重复渲染
                message_placeholder.empty()
                loading_placeholder.empty()
                # 强制重新渲染，显示历史消息中的 AI 回复
                st.rerun()
        
        except requests.exceptions.ConnectionError:
            error_msg = "❌ 无法连接到后端服务，请确保后端已启动"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})
        except requests.exceptions.Timeout:
            error_msg = "⏱️ 请求超时，请稍后重试"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})
        except Exception as e:
            error_msg = f"❌ 请求失败: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})

# 侧边栏 - 清空聊天
with st.sidebar:
    st.title("⚙️ 设置")
    
    # 显示当前会话 ID
    if st.session_state.thread_id:
        st.markdown(f"**当前会话:** `{st.session_state.thread_id[:8]}...`")
    
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = None  # 重置会话 ID
        st.rerun()
    
    if st.button("🔄 新建对话", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = None  # 清空会话 ID，下次请求自动创建新会话
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 关于")
    st.markdown("**智能问答 Agent v1.0**")
    st.markdown(f"后端地址: `{API_BASE_URL}`")
    st.markdown("基于 LangGraph + Milvus + MongoDB 构建")
