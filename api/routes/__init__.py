"""
路由聚合模块
"""
from fastapi import APIRouter
from api.routes.health import router as health_router
from api.routes.chat import router as chat_router

# 创建主路由器并包含子路由
router = APIRouter()
router.include_router(health_router)
router.include_router(chat_router)
