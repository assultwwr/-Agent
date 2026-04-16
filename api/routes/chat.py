"""
聊天相关路由
"""
import json
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from utils.logger_handler import logger
from utils.chat_history import get_chat_history_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # 会话ID，为空时自动创建新会话

# 全局agent实例（由app.py注入）
agent_instance = None


async def generate_stream(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    """生成流式响应（支持多轮对话）"""
    if agent_instance is None:
        logger.error("Agent 实例未初始化")
        yield json.dumps({"type": "error", "data": "服务未就绪，请稍后重试"}) + "\n"
        return
    
    chat_history = get_chat_history_service()
    
    full_response = ""  # 用于收集完整的 AI 回复
    
    try:
        # 先获取历史消息（此时当前用户消息还未保存，不会重复）
        history_messages = chat_history.format_messages_for_agent(thread_id, max_history=10)
        
        # 再保存用户消息到 MongoDB
        chat_history.add_message(thread_id, "user", query)
        
        # execute_stream 已返回 JSON 格式，直接透传
        for chunk in agent_instance.execute_stream(query, history=history_messages):
            yield chunk
            
            # 收集完整的 content 类型消息
            try:
                parsed = json.loads(chunk)
                if parsed.get("type") == "content":
                    full_response += str(parsed.get("data", ""))
            except json.JSONDecodeError:
                # 如果不是 JSON，直接当作 content
                full_response += chunk
        
        # 流式响应结束后，保存完整的 AI 回复到 MongoDB
        if full_response:
            chat_history.add_message(thread_id, "assistant", full_response)
            logger.info(f"保存 AI 回复到会话 {thread_id}: {len(full_response)} 字符")
            
    except Exception as e:
        logger.error(f"流式响应错误: {str(e)}", exc_info=True)
        yield json.dumps({"type": "error", "data": f"处理请求时出错: {str(e)}"}) + "\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    if not request.message or not request.message.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "消息不能为空"}
        )
    
    # 处理 thread_id：为空时创建新会话
    thread_id = request.thread_id
    if not thread_id:
        thread_id = str(uuid.uuid4())
        logger.info(f"创建新会话: {thread_id}")
    else:
        logger.info(f"使用已有会话: {thread_id}")
    
    logger.info(f"收到流式聊天请求: {request.message[:50]}...")
    return StreamingResponse(
        generate_stream(request.message, thread_id),
        media_type="application/x-ndjson",
        headers={"X-Thread-ID": thread_id}  # 返回 thread_id 给前端
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """同步聊天接口"""
    if agent_instance is None:
        logger.error("Agent 实例未初始化")
        return JSONResponse(
            status_code=503,
            content={"error": "服务未就绪"}
        )
    
    if not request.message or not request.message.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "消息不能为空"}
        )
    
    # 处理 thread_id
    thread_id = request.thread_id
    if not thread_id:
        thread_id = str(uuid.uuid4())
        logger.info(f"创建新会话: {thread_id}")
    
    try:
        logger.info(f"收到同步聊天请求: {request.message[:50]}...")
        
        # 保存用户消息
        chat_history = get_chat_history_service()
        chat_history.add_message(thread_id, "user", request.message)
        
        # 获取历史消息
        history_messages = chat_history.format_messages_for_agent(thread_id, max_history=10)
        
        full_response = ""
        for chunk in agent_instance.execute_stream(request.message, history=history_messages):
            try:
                parsed = json.loads(chunk)
                if parsed.get("type") == "content":
                    full_response += str(parsed.get("data", ""))
            except json.JSONDecodeError:
                # 如果不是 JSON，直接当作 content
                full_response += chunk
        
        # 保存 AI 回复到 MongoDB
        chat_history.add_message(thread_id, "assistant", full_response)
        
        return {"response": full_response, "thread_id": thread_id}
    except Exception as e:
        logger.error(f"聊天接口错误: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"处理请求时出错: {str(e)}"}
        )
