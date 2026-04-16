"""
MongoDB 聊天记录存储服务
支持多轮对话上下文管理和自动过期清理
"""
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.logger_handler import logger
import os
import uuid


class ChatHistoryService:
    def __init__(self, uri: str = None, db_name: str = "agent_chat"):
        """
        初始化 MongoDB 连接
        
        Args:
            uri: MongoDB 连接字符串，默认 mongodb://localhost:27017
            db_name: 数据库名称
        """
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name
        self.client = None
        self.db = None
        self.collection = None
        
        self._connect()
        self._ensure_indexes()
    
    def _connect(self):
        """建立 MongoDB 连接"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # 测试连接
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db["chat_sessions"]
            logger.info(f"MongoDB 连接成功: {self.uri}/{self.db_name}")
        except ConnectionFailure as e:
            logger.error(f"MongoDB 连接失败: {e}")
            raise
        except Exception as e:
            logger.error(f"MongoDB 初始化失败: {e}")
            raise
    
    def _ensure_indexes(self):
        """确保索引存在，包括 TTL 索引"""
        try:
            # TTL 索引：自动删除 30 天前的记录
            self.collection.create_index(
                "expire_at",
                expireAfterSeconds=0,
                name="ttl_expire_at"
            )
            
            # thread_id 索引：加速会话查询
            self.collection.create_index(
                "thread_id",
                unique=True,
                name="idx_thread_id"
            )
            
            # updated_at 索引：加速排序查询
            self.collection.create_index(
                "updated_at",
                name="idx_updated_at"
            )
            
            logger.info("MongoDB 索引创建完成（TTL: 30天）")
        except Exception as e:
            logger.error(f"MongoDB 索引创建失败: {e}")
    
    def create_session(self, thread_id: str = None, user_id: str = None) -> str:
        """
        创建新的对话会话
        
        Args:
            thread_id: 会话ID，如为空则自动生成
            user_id: 用户ID（可选）
        
        Returns:
            thread_id: 会话ID
        """
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        now = datetime.utcnow()
        expire_at = now + timedelta(days=30)
        
        session_doc = {
            "thread_id": thread_id,
            "user_id": user_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "expire_at": expire_at,
            "message_count": 0
        }
        
        try:
            self.collection.insert_one(session_doc)
            logger.info(f"创建新会话: {thread_id}")
            return thread_id
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            raise
    
    def get_session(self, thread_id: str) -> Optional[Dict]:
        """
        获取会话记录
        
        Args:
            thread_id: 会话ID
        
        Returns:
            会话文档，不存在返回 None
        """
        try:
            session = self.collection.find_one({"thread_id": thread_id})
            if session:
                # 更新访问时间
                self.collection.update_one(
                    {"thread_id": thread_id},
                    {"$set": {"updated_at": datetime.utcnow()}}
                )
            return session
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            return None
    
    def add_message(self, thread_id: str, role: str, content: str, 
                   metadata: Dict = None) -> bool:
        """
        添加消息到会话
        
        Args:
            thread_id: 会话ID
            role: 消息角色 (user/assistant/tool)
            content: 消息内容
            metadata: 额外元数据（可选）
        
        Returns:
            是否成功
        """
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            }
            if metadata:
                message["metadata"] = metadata
            
            result = self.collection.update_one(
                {"thread_id": thread_id},
                {
                    "$push": {"messages": message},
                    "$set": {
                        "updated_at": datetime.utcnow(),
                        "created_at": {"$ifNull": ["$created_at", datetime.utcnow()]}
                    },
                    "$inc": {"message_count": 1}
                },
                upsert=True  # 关键：不存在时自动创建新会话
            )
            
            if result.modified_count > 0 or result.upserted_id:
                logger.debug(f"添加消息到会话 {thread_id}: {role}")
                return True
            else:
                logger.warning(f"添加消息到会话 {thread_id} 失败")
                return False
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            return False
    
    def get_messages(self, thread_id: str, limit: int = None) -> List[Dict]:
        """
        获取会话的历史消息
        
        Args:
            thread_id: 会话ID
            limit: 返回最近 N 条消息（可选）
        
        Returns:
            消息列表
        """
        try:
            session = self.collection.find_one({"thread_id": thread_id})
            if not session:
                return []
            
            messages = session.get("messages", [])
            
            # 如果指定 limit，返回最近 N 条
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            return messages
        except Exception as e:
            logger.error(f"获取消息失败: {e}")
            return []
    
    def format_messages_for_agent(self, thread_id: str, max_history: int = 10, max_tokens: int = 3000) -> List[Dict]:
        """
        获取格式化的消息列表（用于传递给 Agent）
        
        Args:
            thread_id: 会话ID
            max_history: 最大历史记录数
            max_tokens: 最大 token 数限制（估算值，中文字符约 1 token/字）
        
        Returns:
            格式化的消息列表 [{"role": "user", "content": "..."}, ...]
        """
        messages = self.get_messages(thread_id, limit=max_history)
        
        # 转换为 LangChain 消息格式
        formatted = []
        total_chars = 0  # 使用中文字符数估算 token 数
        
        # 从后往前遍历，保留最近的消息
        for msg in reversed(messages):
            content = msg["content"]
            char_count = len(content)
            
            # 检查是否超出 token 限制
            if total_chars + char_count > max_tokens:
                logger.warning(f"历史消息超过 {max_tokens} tokens 限制，裁剪到 {len(formatted)} 条")
                break
            
            formatted.insert(0, {
                "role": msg["role"],
                "content": content
            })
            total_chars += char_count
        
        return formatted
    
    def delete_session(self, thread_id: str) -> bool:
        """
        删除会话
        
        Args:
            thread_id: 会话ID
        
        Returns:
            是否成功
        """
        try:
            result = self.collection.delete_one({"thread_id": thread_id})
            if result.deleted_count > 0:
                logger.info(f"删除会话: {thread_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False
    
    def get_session_stats(self) -> Dict:
        """
        获取会话统计信息
        
        Returns:
            统计信息字典
        """
        try:
            total_sessions = self.collection.count_documents({})
            total_messages = sum(
                doc.get("message_count", 0) 
                for doc in self.collection.find({}, {"message_count": 1})
            )
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "database": self.db_name
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def close(self):
        """关闭 MongoDB 连接"""
        if self.client:
            self.client.close()
            logger.info("MongoDB 连接已关闭")


# 全局实例（延迟初始化）
_chat_history_service = None


def get_chat_history_service() -> ChatHistoryService:
    """获取全局 ChatHistoryService 实例"""
    global _chat_history_service
    if _chat_history_service is None:
        _chat_history_service = ChatHistoryService()
    return _chat_history_service
