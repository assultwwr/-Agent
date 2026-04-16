from langgraph.prebuilt import create_react_agent
from model.factory import ChatModelFactory
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioConnection
from utils.path_tool import get_abs_path
import asyncio
import os
import sys
import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

# Windows 平台必须使用 ProactorEventLoop 支持异步子进程
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class ReactAgent:
    def __init__(self):
        self.mcp_client = None
        self._init_mcp_client()
        logger.info("ReactAgent 初始化完成")

    def _init_mcp_client(self):
        """初始化 MCP 客户端配置"""
        try:
            project_root = get_abs_path(".")
            self.mcp_client = MultiServerMCPClient(
                connections={
                    "智能问答工具": StdioConnection(
                        command=sys.executable,  # 使用当前环境的 Python 解释器
                        args=["-m", "agent.tools.mcp_server"],
                        env={**os.environ, "PYTHONPATH": project_root},
                        transport="stdio"
                    )
                }
            )
            logger.info("MCP 客户端配置成功")
        except Exception as e:
            logger.error(f"MCP 客户端配置失败: {e}")
            self.mcp_client = None

    def execute_stream(self, query: str, history: List[Dict[str, str]] = None):
        """执行流式响应（含工具调用可视化）
        
        Args:
            query: 当前用户查询
            history: 历史消息列表，格式 [{"role": "user/assistant", "content": "..."}]
        """
        if self.mcp_client is None:
            yield json.dumps({"type": "error", "data": "MCP 客户端未初始化"}) + "\n"
            return

        # 构建消息列表：历史消息 + 当前查询
        messages = []
        if history:
            messages.extend(history)
            logger.info(f"加载历史消息: {len(history)} 条")
        messages.append({"role": "user", "content": query})
        
        input_dict = {"messages": messages}
        config = {"configurable": {"report": False}, "recursion_limit": 50}
        system_prompt = load_system_prompts()

        async def _run_session():
            # 每次请求创建新的模型实例，避免全局单例与临时事件循环冲突
            current_chat_model = ChatModelFactory().generator()
            
            # 直接获取工具（无需 async with 上下文管理器）
            tools = await self.mcp_client.get_tools()
            logger.info(f"获取到 {len(tools)} 个工具")
            
            agent = create_react_agent(
                model=current_chat_model,
                tools=tools,
                state_modifier=system_prompt,
            )
            
            # 使用 stream_mode="updates" 捕获中间步骤
            results = []
            seen_message_ids = set()  # 追踪已处理的消息，避免重复
            
            async for step_data in agent.astream(input_dict, stream_mode="updates", config=config):
                logger.info(f"[stream] step_data keys: {list(step_data.keys())}")
                for node, node_data in step_data.items():
                    if "messages" not in node_data:
                        continue
                    
                    for message in node_data["messages"]:
                        # 生成消息唯一标识，避免重复处理
                        msg_id = id(message)
                        if msg_id in seen_message_ids:
                            continue
                        seen_message_ids.add(msg_id)
                        
                        # 处理 AI 消息（包含工具调用）
                        if isinstance(message, AIMessage):
                            # 检查是否有工具调用
                            if message.tool_calls:
                                for tool_call in message.tool_calls:
                                    tool_info = {
                                        "tool_name": tool_call.get("name", "unknown"),
                                        "tool_args": tool_call.get("args", {})
                                    }
                                    logger.info(f"[工具调用] {tool_info}")
                                    results.append(json.dumps({"type": "tool_call", "data": tool_info}) + "\n")
                                # 有工具调用时，跳过 content 输出
                                continue
                            
                            # 如果 tool_calls 为空，但 content 是 JSON 格式的工具调用
                            content_str = str(message.content).strip() if message.content else ""
                            if content_str and content_str.startswith("{"):
                                try:
                                    parsed_json = json.loads(content_str)
                                    if "name" in parsed_json and "arguments" in parsed_json:
                                        tool_info = {
                                            "tool_name": parsed_json["name"],
                                            "tool_args": parsed_json["arguments"]
                                        }
                                        logger.info(f"[工具调用（JSON解析）] {tool_info}")
                                        results.append(json.dumps({"type": "tool_call", "data": tool_info}) + "\n")
                                        continue
                                except json.JSONDecodeError:
                                    pass  # 不是 JSON 格式，继续处理为普通文本
                            
                            # 仅处理纯文本响应（最终回复）
                            # 修复：只要内容存在且不为空，就捕获作为回复
                            if content_str:
                                logger.info(f"[AI回复] {content_str[:50]}...")
                                results.append(json.dumps({"type": "content", "data": content_str}) + "\n")
                        
                        # 处理工具执行结果
                        elif isinstance(message, ToolMessage):
                            tool_name = getattr(message, "name", "unknown")
                            # 尝试从 content 中提取结果
                            content_str = str(message.content)
                            tool_result = {
                                "tool_name": tool_name,
                                "result": content_str[:200]
                            }
                            logger.info(f"[工具结果] name={tool_name}, content={content_str[:50]}")
                            results.append(json.dumps({"type": "tool_result", "data": tool_result}) + "\n")
                        else:
                            logger.info(f"[未知消息类型] {type(message)}")
            
            return results

        # 在独立线程中运行异步任务，避免与 FastAPI 主事件循环冲突
        import concurrent.futures
        
        def run_async_task():
            """在线程中运行异步任务"""
            # 创建新的事件循环
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                future = asyncio.ensure_future(_run_session(), loop=new_loop)
                new_loop.run_until_complete(future)
                return future.result()
            finally:
                # 安全关闭循环
                try:
                    pending = asyncio.all_tasks(new_loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except:
                    pass
                finally:
                    new_loop.close()
        
        # 在线程池中执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_task)
            try:
                chunks = future.result(timeout=120)  # 2 分钟超时
            except concurrent.futures.TimeoutError:
                logger.error("Agent 执行超时")
                yield f"执行超时\n"
                return
            except Exception as e:
                logger.error(f"执行错误: {e}")
                import traceback
                traceback.print_exc()
                yield f"执行出错: {str(e)}\n"
                return
        
        for chunk in chunks:
            yield chunk


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
