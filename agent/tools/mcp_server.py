from mcp.server.fastmcp import FastMCP
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
import os
import random
import requests
import json
from datetime import datetime

# 初始化 FastMCP 服务器
mcp = FastMCP(name="智能问答工具服务")

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
    """获取指定城市的实时天气信息。

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    logger.info(f"[MCP]get_weather called for city: {city}")
    
    api_key = agent_config.get("amap_api_key", "")
    if not api_key or api_key == "YOUR_AMAP_API_KEY_HERE":
        logger.warning("[MCP]高德 API Key 未配置，使用 Mock 数据")
        return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"
    
    try:
        # 1. 获取城市 adcode
        geo_url = "https://restapi.amap.com/v3/config/district"
        geo_params = {
            "key": api_key,
            "keywords": city,
            "subdistrict": "0"
        }
        geo_resp = requests.get(geo_url, params=geo_params, timeout=5)
        geo_data = geo_resp.json()
        
        if geo_data.get("status") == "1" and geo_data.get("districts"):
            adcode = geo_data["districts"][0]["adcode"]
            
            # 2. 根据 adcode 获取天气
            weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
            weather_params = {
                "key": api_key,
                "city": adcode,
                "extensions": "base"  # base 为实况天气
            }
            weather_resp = requests.get(weather_url, params=weather_params, timeout=5)
            weather_data = weather_resp.json()
            
            if weather_data.get("status") == "1" and weather_data.get("lives"):
                live = weather_data["lives"][0]
                result = (
                    f"城市：{live['province']}{live['city']}\n"
                    f"天气：{live['weather']}\n"
                    f"温度：{live['temperature']}℃\n"
                    f"湿度：{live['humidity']}%\n"
                    f"风向：{live['winddirection']}风 {live['windpower']}级\n"
                    f"发布时间：{live['reporttime']}"
                )
                logger.info(f"[MCP]成功获取 {city} 天气：{live['weather']}")
                return result
    except Exception as e:
        logger.error(f"[MCP]获取天气失败：{e}")
    
    return f"无法获取{city}的天气信息，请稍后重试"

@mcp.tool()
def get_user_location(user_ip: str = "") -> str:
    """获取用户所在城市的名称。

    Args:
        user_ip: (可选) 用户公网 IP。若前端已获取，Agent 应传入此参数以提高精度。

    Returns:
        城市名称
    """
    logger.info(f"[MCP]get_user_location called with ip: {user_ip or 'Auto'}")
    
    api_key = agent_config.get("amap_api_key", "")
    if not api_key or api_key == "YOUR_AMAP_API_KEY_HERE":
        logger.warning("[MCP]高德 API Key 未配置")
        return "无法获取当前位置"
    
    try:
        ip_url = "https://restapi.amap.com/v3/ip"
        ip_params = {
            "key": api_key,
            "ip": user_ip  # 传入前端获取的 IP，若为空则高德自动识别
        }
        resp = requests.get(ip_url, params=ip_params, timeout=3)
        data = resp.json()
        
        if data.get("status") == "1":
            city = data.get("city") or data.get("province") or ""
            if city:
                city = city.replace("市", "")
                logger.info(f"[MCP]基于 IP {user_ip or 'Auto'} 定位到：{city}")
                return city
        
        logger.warning(f"[MCP]高德定位失败")
        return "无法获取当前位置"
    except Exception as e:
        logger.error(f"[MCP]获取位置失败：{e}")
        return "无法获取当前位置"

@mcp.tool()
def get_user_id() -> str:
    """获取用户的ID。

    Returns:
        用户ID字符串
    """
    logger.info("[MCP]get_user_id called")
    return random.choice(user_ids)

@mcp.tool()
def get_current_time() -> str:
    """获取当前的实时时间信息。

    Returns:
        当前时间字符串，包含日期、时间、星期
    """
    now = datetime.now()
    weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    result = (
        f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"星期：{weekdays[now.weekday()]}"
    )
    logger.info(f"[MCP]get_current_time called, result: {result}")
    return result

@mcp.tool()
def get_current_month() -> str:
    """获取当前月份。

    Returns:
        月份字符串 (格式: YYYY-MM)
    """
    logger.info("[MCP]get_current_month called")
    return datetime.now().strftime('%Y-%m')

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
    logger.info("启动智能问答 MCP 工具服务...")
    mcp.run()
