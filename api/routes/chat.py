"""
聊天相关路由
"""
import json
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from utils.logger_handler import logger

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

# 全局agent实例（由app.py注入）
agent_instance = None


async def generate_stream(query: str) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    try:
        for chunk in agent_instance.execute_stream(query):
            yield json.dumps({"type": "content", "data": chunk}) + "\n"
    except Exception as e:
        logger.error(f"流式响应错误: {str(e)}")
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    return StreamingResponse(
        generate_stream(request.message),
        media_type="application/x-ndjson"
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """同步聊天接口"""
    try:
        full_response = ""
        for chunk in agent_instance.execute_stream(request.message):
            full_response += chunk
        return {"response": full_response, "thread_id": request.thread_id}
    except Exception as e:
        logger.error(f"聊天接口错误: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
