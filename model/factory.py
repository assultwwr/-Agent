from abc import ABC, abstractmethod
from typing import Optional
import threading
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi, BaseChatModel
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from utils.config_handler import rag_config
from utils.logger_handler import logger


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_config.get("chat_model_provider")
        model_name = rag_config["chat_model_name"]

        if provider == "ollama":
            return ChatOllama(
                model=model_name,
                base_url=rag_config.get("ollama_base_url"),
                num_ctx=8192  # 显式设置上下文窗口为8192 tokens
            )
        elif provider == "aliyun":
            return ChatTongyi(model=model_name)
        else:
            raise ValueError(f"不支持的聊天模型提供商：{provider}")


class EmbeddingsFactory(BaseModelFactory):
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_config.get("embedding_model_provider")
        model_name = rag_config["embedding_model_name"]

        if provider == "ollama":
            return OllamaEmbeddings(
                model=model_name,
                base_url=rag_config.get("ollama_base_url")
            )
        elif provider == "aliyun":
            return DashScopeEmbeddings(model=model_name)
        else:
            raise ValueError(f"不支持的嵌入模型提供商：{provider}")


# 全局单例实例（延迟初始化 + 线程安全）
_chat_model = None
_embedding_model = None
_lock = threading.Lock()

def get_chat_model():
    """获取聊天模型单例（线程安全的延迟初始化）"""
    global _chat_model
    if _chat_model is None:
        with _lock:
            if _chat_model is None:  # 双重检查锁定
                logger.info("初始化聊天模型...")
                _chat_model = ChatModelFactory.get_instance().generator()
    return _chat_model

def get_embedding_model():
    """获取嵌入模型单例（线程安全的延迟初始化）"""
    global _embedding_model
    if _embedding_model is None:
        with _lock:
            if _embedding_model is None:  # 双重检查锁定
                logger.info("初始化嵌入模型...")
                _embedding_model = EmbeddingsFactory.get_instance().generator()
    return _embedding_model

# 向后兼容：支持直接导入 chat_model 和 embedding_model
def __getattr__(name):
    if name == "chat_model":
        return get_chat_model()
    elif name == "embedding_model":
        return get_embedding_model()
    raise AttributeError(f"module {__name__} has no attribute {name}")
