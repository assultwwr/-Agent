import yaml
from utils.path_tool import get_abs_path

def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_milvus_config(config_path: str=get_abs_path("config/milvus.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config = load_rag_config()
milvus_config = load_milvus_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()