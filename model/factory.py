from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi, BaseChatModel
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from utils.config_handler import rag_config


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_config.get("chat_model_provider")
        model_name = rag_config["chat_model_name"]

        if provider == "ollama":
            return ChatOllama(
                model=model_name,
                base_url=rag_config.get("ollama_base_url")
            )
        elif provider == "aliyun":
            return ChatTongyi(model=model_name)
        else:
            raise ValueError(f"不支持的聊天模型提供商：{provider}")


class EmbeddingsFactory(BaseModelFactory):
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


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
