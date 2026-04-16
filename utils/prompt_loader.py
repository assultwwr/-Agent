from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def _load_prompt(prompt_key: str) -> str:
    """通用提示词加载器
    
    Args:
        prompt_key: 配置键名 (main_prompt_path/rag_summarize_prompt_path/report_prompt_path)
    
    Returns:
        提示词内容字符串
    """
    try:
        prompt_path = get_abs_path(prompts_config[prompt_key])
    except KeyError as e:
        logger.error(f"[load_prompt]在yaml配置中未找到{prompt_key}字段")
        raise e

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_prompt]解析提示词出错 ({prompt_key}): {str(e)}")
        raise e

def load_system_prompts():
    """加载系统提示词"""
    return _load_prompt('main_prompt_path')

def load_rag_prompts():
    """加载RAG总结提示词"""
    return _load_prompt('rag_summarize_prompt_path')

def load_report_prompts():
    """加载报告生成提示词"""
    return _load_prompt('report_prompt_path')