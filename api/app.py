import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.react_agent import ReactAgent
from utils.logger_handler import logger
from api.routes import router as api_router
from api.routes import chat

# 初始化 FastAPI 应用
app = FastAPI(
    title="智能问答 Agent API",
    description="基于 RAG + ReAct Agent 的智能问答系统",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 ReactAgent
agent_instance = ReactAgent()

# 注入 agent 实例到路由模块
chat.agent_instance = agent_instance

# 注册路由
app.include_router(api_router)

if __name__ == "__main__":
    logger.info("启动智能问答 Agent API 服务...")
    uvicorn.run(
        "api.app:app",  # 使用模块路径
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # 开发模式自动重载
    )