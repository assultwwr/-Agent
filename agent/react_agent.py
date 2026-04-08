from langgraph.prebuilt import create_react_agent as create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location,
                                             get_user_id, get_current_month, fetch_external_data,
                                             fill_context_for_report)
import asyncio


class ReactAgent:
    def __init__(self):
        self.agent = None
        self.mcp_client = None
        self._init_mcp_client()
        self.agent = self._create_agent()
        logger.info("ReactAgent初始化完成")

    def _init_mcp_client(self):
        """初始化MCP客户端"""
        try:
            # MCP服务器配置（使用本地Python模块方式）
            self.mcp_client = MultiServerMCPClient(
                {
                    "智扫通工具": {
                        "command": "python",
                        "args": ["-m", "agent.tools.mcp_server"],
                    }
                }
            )
            logger.info("MCP客户端初始化成功")
        except Exception as e:
            logger.error(f"MCP客户端初始化失败: {e}")
            self.mcp_client = None

    def _get_tools(self):
        """从MCP服务器获取工具"""
        if self.mcp_client:
            try:
                return self.mcp_client.get_tools()
            except Exception as e:
                logger.error(f"获取MCP工具失败: {e}")
        
        # 降级方案：使用传统@tool工具
        logger.warning("使用传统@tool工具降级方案")

        return [rag_summarize, get_weather, get_user_location, get_user_id,
                get_current_month, fetch_external_data, fill_context_for_report]

    def _create_agent(self):
        """创建agent，使用MCP工具"""
        tools = self._get_tools()
        system_prompt = load_system_prompts()

        # 尝试不同的参数组合
        try:
            # 新版本使用 state_modifier
            agent = create_agent(
                model=chat_model,
                tools=tools,
                state_modifier=system_prompt,
            )
            logger.info("使用state_modifier参数初始化agent")
            return agent
        except TypeError as e:
            logger.debug(f"state_modifier参数失败: {e}")

        try:
            # 旧版本使用 system_prompt
            agent = create_agent(
                model=chat_model,
                tools=tools,
                system_prompt=system_prompt,
            )
            logger.info("使用system_prompt参数初始化agent")
            return agent
        except TypeError as e:
            logger.debug(f"system_prompt参数失败: {e}")

        try:
            # 更旧版本可能使用 prompt
            agent = create_agent(
                model=chat_model,
                tools=tools,
                prompt=system_prompt,
            )
            logger.info("使用prompt参数初始化agent")
            return agent
        except TypeError as e:
            logger.debug(f"prompt参数失败: {e}")

        # 最后的备选方案：不传递系统提示词
        try:
            agent = create_agent(
                model=chat_model,
                tools=tools,
            )
            logger.warning("使用无系统提示词的默认配置初始化 agent")
            return agent
        except Exception as e:
            logger.error(f"无法初始化agent: {e}")
            raise

    def execute_stream(self, query: str):
        """执行流式响应"""
        if self.agent is None:
            logger.error("Agent未初始化")
            yield "Agent未初始化，请检查配置\n"
            return

        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        try:
            # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                latest_message = chunk["messages"][-1]
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"
        except Exception as e:
            logger.error(f"流式执行错误: {e}")
            yield f"执行出错: {str(e)}\n"

    def get_langgraph_agent(self):
        """返回LangGraph agent实例"""
        return self.agent


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
