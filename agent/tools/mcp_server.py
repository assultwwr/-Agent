from mcp.server.fastmcp import FastMCP
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
import os
import random

# 初始化 FastMCP 服务器
mcp = FastMCP(name="智扫通工具服务", version="1.0.0")

# 初始化 RAG 服务
rag = RagSummarizeService()

# 模拟数据
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

external_data = {}

@mcp.tool()
def rag_summarize(query: str) -> str:
    """从向量存储中检索参考资料。

    Args:
        query: 用户查询内容

    Returns:
        检索到的参考资料摘要
    """
    logger.info(f"[MCP]rag_summarize called with query: {query}")
    return rag.rag_summarize(query)

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    logger.info(f"[MCP]get_weather called for city: {city}")
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"

@mcp.tool()
def get_user_location() -> str:
    """获取用户所在城市的名称。

    Returns:
        城市名称
    """
    logger.info("[MCP]get_user_location called")
    return random.choice(["深圳", "合肥", "杭州"])

@mcp.tool()
def get_user_id() -> str:
    """获取用户的ID。

    Returns:
        用户ID字符串
    """
    logger.info("[MCP]get_user_id called")
    return random.choice(user_ids)

@mcp.tool()
def get_current_month() -> str:
    """获取当前月份。

    Returns:
        月份字符串 (格式: YYYY-MM)
    """
    logger.info("[MCP]get_current_month called")
    return random.choice(month_arr)

def load_external_data():
    """加载外部数据文件"""
    global external_data
    if external_data:
        return

    external_data_path = get_abs_path(agent_config.get("external_data_path", "data/external/records.csv"))

    if not os.path.exists(external_data_path):
        logger.warning(f"[MCP]外部数据文件不存在: {external_data_path}")
        return

    try:
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")
                if len(arr) < 6:
                    continue

                user_id = arr[0].replace('"', "")
                external_data.setdefault(user_id, {})[arr[5].replace('"', "")] = {
                    "特征": arr[1].replace('"', ""),
                    "效率": arr[2].replace('"', ""),
                    "耗材": arr[3].replace('"', ""),
                    "对比": arr[4].replace('"', ""),
                }
        logger.info(f"[MCP]成功加载外部数据: {len(external_data)} 个用户")
    except Exception as e:
        logger.error(f"[MCP]加载外部数据失败: {e}")

@mcp.tool()
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录。

    Args:
        user_id: 用户ID
        month: 月份 (格式: YYYY-MM)

    Returns:
        使用记录数据，未找到返回空字符串
    """
    logger.info(f"[MCP]fetch_external_data called for user: {user_id}, month: {month}")
    load_external_data()

    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(f"[MCP]未检索到用户{user_id}在{month}的数据")
        return ""

@mcp.tool()
def fill_context_for_report() -> str:
    """触发报告生成场景的上下文切换。

    无入参，调用后为后续提示词切换提供上下文标记。

    Returns:
        确认字符串
    """
    logger.info("[MCP]fill_context_for_report called")
    return "fill_context_for_report已调用"

# 启动 MCP 服务器
if __name__ == "__main__":
    logger.info("启动智扫通MCP工具服务...")
    mcp.run()
