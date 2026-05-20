"""长期记忆工具 - 供 Agent 搜索历史记忆

Agent 在旅行规划时可以调用此工具搜索已有的记忆，
了解用户之前探索过的城市和景点信息。
"""

from langchain_core.tools import tool
from ..services.memory_service import get_memory_service
from ..services.rag_service import get_rag_service


@tool
def memory_search(query: str) -> str:
    """
    搜索历史记忆 - 查找之前探索过的城市和景点信息。
    当你需要了解某个城市或景点的信息时，先搜索记忆看看有没有已有记录。

    Args:
        query: 搜索关键词，如城市名、景点名、美食名等

    Returns:
        匹配的记忆信息文本
    """
    memory = get_memory_service()

    # 优先用 RAG 做语义搜索（如果已初始化）
    try:
        rag = get_rag_service()
        docs = rag.retrieve(query, k=3)
        memory_hits = [d for d in docs if "memory" in d.metadata.get("source", "")]
        if memory_hits:
            parts = []
            for d in memory_hits:
                source = d.metadata.get("source", "记忆")
                parts.append(f"[{source}]\n{d.page_content[:300]}")
            if parts:
                return "\n\n".join(parts)
    except Exception:
        pass

    # 后备：关键词搜索
    results = memory.search_memories(query)
    if not results:
        return f"记忆库中未找到与「{query}」相关的记录。"

    parts = []
    for r in results[:5]:
        parts.append(f"📄 {r['title']}\n   {r['excerpt']}")

    return "找到以下相关记忆:\n" + "\n\n".join(parts)


@tool
def save_search_memory(city: str, attractions_json: str) -> str:
    """
    保存搜索结果到长期记忆 - 将搜索到的城市和景点信息保存下来，
    以便下次查询时可以直接使用。

    Args:
        city: 城市名称
        attractions_json: 景点信息的 JSON 字符串，格式:
            [{"name": "景点名", "address": "地址", "category": "类别", "description": "描述"}]

    Returns:
        保存结果信息
    """
    import json

    memory = get_memory_service()

    try:
        attractions = json.loads(attractions_json)
    except json.JSONDecodeError:
        return f"❌ 景点信息解析失败，请确保传入有效的 JSON 格式"

    if not attractions:
        return "⚠️  无景点信息需要保存"

    memory.save_city_memory(
        city=city,
        attractions=attractions,
        source="高德地图搜索",
    )
    memory.save_attractions_memory(attractions, city=city, source="高德地图搜索")

    return f"✅ 已保存 {city} 的 {len(attractions)} 个景点信息到长期记忆"


def get_memory_tools() -> list:
    """获取所有记忆工具"""
    return [memory_search, save_search_memory]
