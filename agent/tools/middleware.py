from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger

# 尝试导入，如果失败则创建模拟函数
try:
    from langchain.agents.middleware import (
        AgentState,
        wrap_tool_call,
        before_model,
        dynamic_prompt,
        ModelRequest,
        ToolCallRequest
    )
    from langgraph.runtime import Runtime

    HAS_MIDDLEWARE = True
except ImportError:
    HAS_MIDDLEWARE = False
    logger.warning("langchain.agents.middleware 不可用，使用模拟装饰器")


    # 创建模拟的装饰器和类型
    def wrap_tool_call(func):
        return func


    def before_model(func):
        return func


    def dynamic_prompt(func):
        return func


    # 模拟类型
    class AgentState:
        pass


    class ModelRequest:
        pass


    class ToolCallRequest:
        pass


    class Runtime:
        pass

if HAS_MIDDLEWARE:
    @wrap_tool_call
    def monitor_tool(
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
        logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

        try:
            result = handler(request)
            logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

            if request.tool_call['name'] == "fill_context_for_report":
                request.runtime.context["report"] = True

            return result
        except Exception as e:
            logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
            raise e
else:
    # 如果没有 middleware，创建空函数
    def monitor_tool(request, handler):
        result = handler(request)
        return result

if HAS_MIDDLEWARE:
    @before_model
    def log_before_model(
            state: AgentState,
            runtime: Runtime,
    ):
        logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
        if hasattr(state, 'messages') and state['messages']:
            logger.debug(
                f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
        return None
else:
    def log_before_model(state, runtime):
        logger.info("[log_before_model]模型调用前（简化模式）")
        return None

if HAS_MIDDLEWARE:
    @dynamic_prompt
    def report_prompt_switch(request: ModelRequest):
        is_report = request.runtime.context.get("report", False)
        if is_report:
            return load_report_prompts()
        return load_system_prompts()
else:
    def report_prompt_switch(request):
        # 简化版本：总是返回系统提示词
        return load_system_prompts()
