import os
import yaml
from dotenv import load_dotenv
from utils.path_tool import get_abs_path

# 加载 .env 文件
load_dotenv(get_abs_path(".env"))

def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 环境变量覆盖
    if os.getenv("OLLAMA_BASE_URL"):
        config["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL")
    if os.getenv("CHAT_MODEL_PROVIDER"):
        config["chat_model_provider"] = os.getenv("CHAT_MODEL_PROVIDER")
    if os.getenv("CHAT_MODEL_NAME"):
        config["chat_model_name"] = os.getenv("CHAT_MODEL_NAME")
    if os.getenv("EMBEDDING_MODEL_PROVIDER"):
        config["embedding_model_provider"] = os.getenv("EMBEDDING_MODEL_PROVIDER")
    if os.getenv("EMBEDDING_MODEL_NAME"):
        config["embedding_model_name"] = os.getenv("EMBEDDING_MODEL_NAME")
    if os.getenv("RERANK_DEVICE"):
        config["rerank_device"] = os.getenv("RERANK_DEVICE")
    if os.getenv("RERANK_MODEL_PATH"):
        config["rerank_model_path"] = os.getenv("RERANK_MODEL_PATH")
    
    return config

def load_milvus_config(config_path: str=get_abs_path("config/milvus.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 环境变量覆盖
    if os.getenv("MILVUS_URI"):
        config["uri"] = os.getenv("MILVUS_URI")
    if os.getenv("MILVUS_COLLECTION_NAME"):
        config["collection_name"] = os.getenv("MILVUS_COLLECTION_NAME")
    
    return config

def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 环境变量覆盖
    if os.getenv("AMAP_API_KEY"):
        config["amap_api_key"] = os.getenv("AMAP_API_KEY")
    
    return config

rag_config = load_rag_config()
milvus_config = load_milvus_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()