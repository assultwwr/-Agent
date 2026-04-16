"""
健康检查路由

API 使用说明：
=============
这些接口通过 HTTP 请求调用，用于系统监控和运维。

1. GET /health
   - 功能：检查所有服务状态（MongoDB、Milvus、Ollama）
   - 使用场景：Docker 健康检查、监控系统、部署后验证
   - 示例：curl http://localhost:8000/health
   - 返回：{"status": "healthy/degraded", "services": {...}}

2. POST /init/vector-store?drop_existing=true/false
   - 功能：初始化向量数据库，加载知识库文档
   - 使用场景：首次部署、更新知识库、重建索引
   - 示例：curl -X POST http://localhost:8000/init/vector-store
   - 返回：{"status": "success", "documents": 100}

3. GET /status/vector-store
   - 功能：查询向量数据库当前状态
   - 使用场景：监控文档数量、检查 Milvus 连接
   - 示例：curl http://localhost:8000/status/vector-store
   - 返回：{"status": "healthy", "documents": 100}

使用方式：
- Postman/Apifox：直接发送 HTTP 请求
- 终端：curl 命令
- Docker：容器启动后自动调用健康检查
- CI/CD：部署流程中验证服务可用性
"""
from fastapi import APIRouter
from utils.logger_handler import logger
from utils.config_handler import milvus_config, rag_config
from utils.chat_history import get_chat_history_service
import requests
from typing import Dict

router = APIRouter()

def check_mongodb() -> Dict:
    """检查 MongoDB 连接状态"""
    try:
        chat_history = get_chat_history_service()
        stats = chat_history.get_session_stats()
        if "error" in stats:
            return {"status": "unhealthy", "error": stats["error"]}
        return {
            "status": "healthy",
            "database": stats.get("database"),
            "sessions": stats.get("total_sessions", 0)
        }
    except Exception as e:
        logger.error(f"MongoDB 健康检查失败: {e}")
        return {"status": "unhealthy", "error": str(e)}

def check_milvus() -> Dict:
    """检查 Milvus 连接状态"""
    try:
        from rag.vector_store import VectorStoreService
        vs = VectorStoreService()
        count = vs.count_documents()
        return {
            "status": "healthy",
            "collection": milvus_config.get("collection_name"),
            "documents": count
        }
    except Exception as e:
        logger.error(f"Milvus 健康检查失败: {e}")
        return {"status": "unhealthy", "error": str(e)}

def check_ollama() -> Dict:
    """检查 Ollama 服务状态"""
    try:
        ollama_url = rag_config.get("ollama_base_url", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "status": "healthy",
                "url": ollama_url,
                "models_count": len(models),
                "models": [m.get("name") for m in models[:5]]  # 最多显示 5 个模型
            }
        return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.error(f"Ollama 健康检查失败: {e}")
        return {"status": "unhealthy", "error": str(e)}

@router.get("/health")
async def health_check():
    """完整健康检查"""
    mongodb_status = check_mongodb()
    milvus_status = check_milvus()
    ollama_status = check_ollama()
    
    # 整体状态：所有服务都正常才算 healthy
    overall_status = "healthy" if all(
        s["status"] == "healthy" 
        for s in [mongodb_status, milvus_status, ollama_status]
    ) else "degraded"
    
    return {
        "status": overall_status,
        "service": "customer-service",
        "services": {
            "mongodb": mongodb_status,
            "milvus": milvus_status,
            "ollama": ollama_status
        }
    }

@router.post("/init/vector-store")
async def init_vector_store(drop_existing: bool = False):
    """初始化向量数据库，加载知识库文档"""
    try:
        from rag.vector_store import VectorStoreService
        logger.info(f"开始初始化向量库 (drop_existing={drop_existing})...")
        vs = VectorStoreService(drop_existing=drop_existing, auto_load=True)
        count = vs.count_documents()
        logger.info(f"向量库初始化完成，共 {count} 个文档")
        return {
            "status": "success",
            "documents": count,
            "collection": milvus_config.get("collection_name")
        }
    except Exception as e:
        logger.error(f"向量库初始化失败: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/status/vector-store")
async def vector_store_status():
    """查询向量数据库当前状态"""
    try:
        from rag.vector_store import VectorStoreService
        vs = VectorStoreService(auto_load=False)  # 不自动加载，仅查询状态
        count = vs.count_documents()
        return {
            "status": "healthy",
            "collection": milvus_config.get("collection_name"),
            "documents": count
        }
    except Exception as e:
        logger.error(f"查询向量库状态失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
