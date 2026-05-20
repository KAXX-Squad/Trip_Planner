"""HelloAgents 智能旅行助手 - 命令行入口

提供完整的命令行交互能力，覆盖所有核心功能。

用法:
    python cli.py plan-trip 北京 --days 3 --preferences 历史文化 美食
    python cli.py rag-search "北京美食推荐"
    python cli.py weather 北京
    python cli.py poi-search 故宫 --city 北京
    python cli.py route "天安门" "故宫"
    python cli.py serve
    python cli.py health
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

# 将项目根目录加入 Python 路径，确保能正确导入后端模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_plan_trip(args: argparse.Namespace):
    """旅行规划命令 - 核心功能"""
    from app.models.schemas import TripRequest
    from app.agents.trip_planner_agent import get_trip_planner_agent
    from app.config import validate_config, get_settings

    # 验证配置
    try:
        validate_config()
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        sys.exit(1)

    # 解析日期
    start_date = args.start_date
    travel_days = args.days

    from datetime import datetime, timedelta
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(days=travel_days - 1)
    end_date = end.strftime("%Y-%m-%d")
 
    # 构建请求
    request = TripRequest(
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        travel_days=travel_days,
        transportation=args.transportation,
        accommodation=args.accommodation,
        preferences=args.preferences or [],
        free_text_input=args.free_text or "",
    )

    # 打印请求概览
    print(f"\n{'='*60}")
    print(f"🌍 HelloAgents 智能旅行规划")
    print(f"{'='*60}")
    print(f"  目的地:    {request.city}")
    print(f"  日期:      {request.start_date} → {request.end_date}")
    print(f"  天数:      {request.travel_days} 天")
    print(f"  交通:      {request.transportation}")
    print(f"  住宿:      {request.accommodation}")
    if request.preferences:
        print(f"  偏好:      {', '.join(request.preferences)}")
    if request.free_text_input:
        print(f"  额外要求:  {request.free_text_input}")
    print(f"{'='*60}\n")

    try:
        agent = get_trip_planner_agent()
        trip_plan = agent.plan_trip(request)

        print(f"\n{'='*60}")
        print(f"✅ 行程规划完成!")
        print(f"{'='*60}")
        print(f"📍 {trip_plan.city} | {trip_plan.start_date} → {trip_plan.end_date}")
        print()

        for day in trip_plan.days:
            print(f"  📅 第 {day.day_index + 1} 天 ({day.date})")
            print(f"     {day.description}")
            print(f"     🏨 {day.hotel.name if day.hotel else day.accommodation}")
            for attr in day.attractions:
                print(f"     🏛  {attr.name} ({attr.visit_duration}分钟)")
                if attr.ticket_price:
                    print(f"        门票: {attr.ticket_price} 元")
            for meal in day.meals:
                icon = {"breakfast": "🌅", "lunch": "☀️", "dinner": "🌙", "snack": "🍪"}
                print(f"     {icon.get(meal.type, '🍽')}  {meal.name}")
            print()

        if trip_plan.weather_info:
            print(f"  🌤  天气信息:")
            for w in trip_plan.weather_info:
                print(f"     {w.date}: {w.day_weather} {w.day_temp}°C / {w.night_weather} {w.night_temp}°C")
            print()

        print(f"  💡 总体建议:")
        print(f"     {trip_plan.overall_suggestions}")

        if trip_plan.budget:
            b = trip_plan.budget
            print()
            print(f"  💰 预算概览:")
            print(f"     门票:    {b.total_attractions:>6} 元")
            print(f"     住宿:    {b.total_hotels:>6} 元")
            print(f"     餐饮:    {b.total_meals:>6} 元")
            print(f"     交通:    {b.total_transportation:>6} 元")
            print(f"     {'─'*18}")
            print(f"     总计:    {b.total:>6} 元")

        # 可选导出 JSON
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(
                trip_plan.model_dump_json(indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"\n  📄 已导出到: {output_path.resolve()}")
        elif args.json_output:
            print()
            print(json.dumps(
                trip_plan.model_dump() if hasattr(trip_plan, 'model_dump') else trip_plan,
                indent=2, ensure_ascii=False
            ))

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"\n❌ 旅行规划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_rag_search(args: argparse.Namespace):
    """RAG 知识库检索命令"""
    from app.services.rag_service import get_rag_service
    from app.config import get_settings

    settings = get_settings()
    if not settings.rag_enabled:
        print("❌ RAG 功能未启用 (RAG_ENABLED=false)")
        sys.exit(1)

    try:
        rag = get_rag_service()
        results = rag.retrieve(args.query, k=args.top_k)

        if not results:
            print(f"\n⚠️  未找到与「{args.query}」相关的知识")
            return

        print(f"\n{'='*60}")
        print(f"🔍 RAG 知识检索: 「{args.query}」")
        print(f"   共找到 {len(results)} 条相关结果 (Top-{args.top_k})")
        print(f"{'='*60}\n")

        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content.strip()
            print(f"  [{i+1}] 来源: {source}")
            print(f"      {content[:300]}{'...' if len(content) > 300 else ''}")
            print()

        if args.output_json:
            output = []
            for doc in results:
                output.append({
                    "source": doc.metadata.get("source", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })
            print(json.dumps(output, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ RAG 检索失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_weather(args: argparse.Namespace):
    """天气查询命令"""
    from app.tools.amap_tools import maps_weather

    print(f"\n🌤  正在查询「{args.city}」天气...\n")
    result = maps_weather.invoke({"city": args.city})
    print(result)

    if args.json_output:
        pass


def cmd_poi_search(args: argparse.Namespace):
    """POI 搜索命令"""
    from app.tools.amap_tools import maps_text_search

    print(f"\n🔍 正在搜索「{args.keywords}」(城市: {args.city})...\n")
    result = maps_text_search.invoke({
        "keywords": args.keywords,
        "city": args.city,
    })
    print(result)


def _is_coordinate(text: str) -> bool:
    """判断输入是否已经是经纬度坐标（如 116.397128,39.916527）"""
    import re
    return bool(re.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', text.strip()))


def _geocode(address: str, city: str = "") -> str:
    """将地名解析为经纬度坐标

    使用高德地理编码 API，返回 "经度,纬度" 格式的字符串。
    解析失败时返回 None。
    """
    result = _geocode_full(address, city)
    return result["location"] if result else None


def _geocode_full(address: str, city: str = "") -> dict:
    """将地名解析为详细地理信息

    返回包含 location, city, adcode 等信息的字典，解析失败时返回 None。
    """
    from app.config import get_settings
    import requests

    settings = get_settings()
    if not settings.amap_api_key:
        print(f"⚠️  高德 API Key 未配置，无法解析地名")
        return None

    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": settings.amap_api_key,
        "address": address,
        "city": city,
        "output": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "1" and data.get("geocodes"):
            geo = data["geocodes"][0]
            location = geo.get("location", "")
            if location:
                return {
                    "location": location,
                    "city": geo.get("city", ""),
                    "adcode": geo.get("adcode", ""),
                    "formatted_address": geo.get("formatted_address", address),
                }
        return None
    except Exception as e:
        print(f"⚠️  地名解析失败 ({address}): {str(e)}")
        return None


def _resolve_route(args: argparse.Namespace):
    """解析路线规划所需的坐标和信息"""
    origin_info = args.origin if _is_coordinate(args.origin) else _geocode_full(args.origin)
    dest_info = args.destination if _is_coordinate(args.destination) else _geocode_full(args.destination)

    if not origin_info:
        print(f"❌ 无法解析起点「{args.origin}」的坐标")
        print(f"   建议使用经纬度坐标，格式: 经度,纬度")
        return None

    if not dest_info:
        print(f"❌ 无法解析终点「{args.destination}」的坐标")
        print(f"   建议使用经纬度坐标，格式: 经度,纬度")
        return None

    origin_coord = origin_info if _is_coordinate(args.origin) else origin_info["location"]
    dest_coord = dest_info if _is_coordinate(args.destination) else dest_info["location"]
    origin_city = origin_info["city"] if isinstance(origin_info, dict) else ""

    origin_display = args.origin
    dest_display = args.destination
    if _is_coordinate(origin_display):
        origin_display = f"坐标 {origin_display}"
    if _is_coordinate(dest_display):
        dest_display = f"坐标 {dest_display}"

    return {
        "origin_coord": origin_coord,
        "dest_coord": dest_coord,
        "origin_city": origin_city,
        "origin_display": origin_display,
        "dest_display": dest_display,
    }


def cmd_route(args: argparse.Namespace):
    """路线规划命令（支持驾车和公交/地铁）"""
    from app.tools.amap_tools import maps_direction_driving, maps_direction_transit

    info = _resolve_route(args)
    if not info:
        return

    origin_city = info["origin_city"]

    if args.mode == "transit":
        print(f"\n🗺  正在规划公交/地铁路线 ...")
        print(f"   起点: {info['origin_display']}")
        print(f"   终点: {info['dest_display']}")
        print(f"   路线计算中...\n")
        print(maps_direction_transit.invoke({
            "origin": info["origin_coord"],
            "destination": info["dest_coord"],
            "city": origin_city,
        }))
    elif args.mode == "driving":
        print(f"\n🗺  正在规划驾车路线 ...")
        print(f"   起点: {info['origin_display']}")
        print(f"   终点: {info['dest_display']}")
        print(f"   路线计算中...\n")
        print(maps_direction_driving.invoke({
            "origin": info["origin_coord"],
            "destination": info["dest_coord"],
        }))
    else:
        print(f"\n🗺  路线规划（全部方案）")
        print(f"   起点: {info['origin_display']}")
        print(f"   终点: {info['dest_display']}")
        print(f"   {'='*40}")

        print(f"\n🚗 驾车方案:")
        print(maps_direction_driving.invoke({
            "origin": info["origin_coord"],
            "destination": info["dest_coord"],
        }))

        print(f"\n🚇 公交/地铁方案:")
        print(maps_direction_transit.invoke({
            "origin": info["origin_coord"],
            "destination": info["dest_coord"],
            "city": origin_city,
        }))


def cmd_serve(args: argparse.Namespace):
    """启动 Web 服务器"""
    import uvicorn
    from app.config import get_settings

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    print(f"\n{'='*60}")
    print(f"🚀 启动 HelloAgents 智能旅行助手 Web 服务")
    print(f"{'='*60}")
    print(f"   地址: http://{host}:{port}")
    print(f"   API:  http://{host}:{port}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


def cmd_health(args: argparse.Namespace):
    """服务健康检查"""
    from app.config import validate_config, get_settings

    print(f"\n{'='*60}")
    print(f"🏥 HelloAgents 旅行助手 - 系统诊断")
    print(f"{'='*60}")

    settings = get_settings()
    print(f"\n  应用:    {settings.app_name} v{settings.app_version}")

    # 配置验证
    try:
        validate_config()
        print(f"  配置:    ✅ 验证通过")
    except ValueError as e:
        print(f"  配置:    ❌ {e}")

    # LLM 检查
    from app.services.llm_service import get_llm
    try:
        llm = get_llm()
        print(f"  LLM:     ✅ 已就绪 ({len(llm.llm_instances)} 个实例)")
        for i, (name, inst) in enumerate(zip(llm.llm_names, llm.llm_instances)):
            print(f"           ├─ [{i+1}] {name}")
    except Exception as e:
        print(f"  LLM:     ❌ {str(e)}")

    # 高德地图检查
    if settings.amap_api_key:
        print(f"  高德地图: ✅ API Key 已配置")
    else:
        print(f"  高德地图: ⚠️  未配置 API Key")

    # RAG 检查
    from app.services.rag_service import get_rag_service
    try:
        rag = get_rag_service()
        if rag._initialized:
            print(f"  RAG:     ✅ 已就绪 (Embedding: {rag.embedding_provider})")
        else:
            print(f"  RAG:     ⚠️  未初始化")
    except Exception as e:
        print(f"  RAG:     ❌ {str(e)}")

    # 智能体检查
    try:
        from app.agents.trip_planner_agent import get_trip_planner_agent
        agent = get_trip_planner_agent()
        print(f"  智能体:  ✅ 已就绪 ({len(agent.tools)} 个工具)")
    except Exception as e:
        print(f"  智能体:  ❌ {str(e)}")

    print(f"\n{'='*60}\n")


def cmd_kg(args: argparse.Namespace):
    """知识图谱命令"""
    from app.services.knowledge_graph import get_kg_service

    kg = get_kg_service()

    if not kg.is_connected():
        print("❌ 无法连接到 Neo4j，请检查 NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD 配置")
        return

    if args.kg_action == "build":
        kg.build_from_files()
    elif args.kg_action == "info":
        stats = kg.query.get_stats()
        print(f"\n{'='*60}")
        print(f"📊 旅行知识图谱统计")
        print(f"{'='*60}")
        print(f"\n  节点类型:")
        for label, count in sorted(stats["node_counts"].items(), key=lambda x: -x[1]):
            print(f"    {label:15s} {count:>4} 个")
        print(f"\n  关系类型:")
        for rel_type, count in sorted(stats["relationship_counts"].items(), key=lambda x: -x[1]):
            print(f"    {rel_type:20s} {count:>4} 条")
        total_nodes = sum(stats["node_counts"].values())
        total_rels = sum(stats["relationship_counts"].values())
        print(f"\n  总计: {total_nodes} 个节点, {total_rels} 条关系\n")
    elif args.kg_action == "query":
        if args.city:
            info = kg.query.get_city_info(args.city)
            if info:
                print(f"\n{'='*60}")
                print(f"📍 {info['name']}")
                print(f"{'='*60}")
                if info.get("budget"):
                    print(f"  预算: {info['budget']}")
                if info.get("suggest_days"):
                    print(f"  建议天数: {info['suggest_days']}")
                if info.get("transportation"):
                    print(f"  交通: {info['transportation']}")
                if info.get("accommodation"):
                    print(f"  住宿: {info['accommodation']}")
                if info.get("attractions"):
                    print(f"\n  🏛  必游景点:")
                    for a in info["attractions"]:
                        print(f"     - {a}")
                if info.get("cuisines"):
                    print(f"\n  🍜 特色美食:")
                    for c in info["cuisines"]:
                        print(f"     - {c}")
                if info.get("best_seasons"):
                    print(f"\n  🌤  最佳季节:")
                    for s in info["best_seasons"]:
                        print(f"     - {s}")
            else:
                print(f"❌ 未找到城市「{args.city}」的信息")
        elif args.season:
            cities = kg.query.recommend_by_season(args.season)
            if cities:
                print(f"\n{'='*60}")
                print(f"🌤  {args.season}推荐目的地")
                print(f"{'='*60}")
                for c in cities:
                    print(f"\n  📍 {c['name']} ({c.get('budget', '')})")
                    if c.get("attractions"):
                        print(f"     景点: {', '.join(c['attractions'][:4])}")
                    if c.get("cuisines"):
                        print(f"     美食: {', '.join(c['cuisines'][:4])}")
            else:
                print(f"未找到 {args.season} 的推荐目的地")
        else:
            print("⚠️  请指定 --city 或 --season 参数查询")
            print("   示例: python cli.py kg query --city 北京")
            print("         python cli.py kg query --season 春季")

    kg.close()


def cmd_memory(args: argparse.Namespace):
    """长期记忆管理命令"""
    from app.services.memory_service import get_memory_service

    memory = get_memory_service()

    if args.mem_action == "list":
        mems = memory.get_all_memories()
        if not mems:
            print("\n📭 记忆库为空，进行旅行规划后会自动生成记忆。")
            return
        print(f"\n{'='*60}")
        print(f"💾 长期记忆 ({len(mems)} 条)")
        print(f"{'='*60}")
        for m in mems:
            icon = "🏙" if m["file"].startswith("city_") else "🏛"
            print(f"  {icon} {m['title']}")
            print(f"     文件: {m['file']}")
            print(f"     来源: {m['source']} | 时间: {m['time']}")
            print()

    elif args.mem_action == "search":
        results = memory.search_memories(args.query)
        if not results:
            print(f"\n🔍 未找到与「{args.query}」相关的记忆")
            return
        print(f"\n{'='*60}")
        print(f"🔍 搜索「{args.query}」({len(results)} 条匹配)")
        print(f"{'='*60}")
        for r in results[:10]:
            print(f"\n  📄 {r['title']}")
            print(f"     {r['excerpt'][:120]}...")

    elif args.mem_action == "rebuild":
        memory.rebuild_index()
        print("\n✅ 记忆索引已重建")

    elif args.mem_action == "guide":
        guide = memory.collect_knowledge_into_guide()
        if guide:
            print(f"\n📖 已生成用户城市指南:")
            print(guide[:500] + "\n...")
        else:
            print("\n📭 暂无记忆可生成指南")

    memory = None


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="helloagents-trip",
        description="🌍 HelloAgents 智能旅行助手 - 命令行版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 旅行规划
  python cli.py plan-trip 北京 --days 3 --preferences 历史文化 美食
  
  # RAG 知识检索
  python cli.py rag-search "北京美食推荐"
  
  # 天气查询
  python cli.py weather 北京
  
  # POI 搜索
  python cli.py poi-search 故宫 --city 北京
  
  # 路线规划（默认同时展示驾车和公交/地铁方案）
  python cli.py route "五一广场" "中南大学"
  
  # 仅查看驾车方案
  python cli.py route "天安门" "故宫" --mode driving
  
  # 仅查看公交/地铁路线
  python cli.py route "五一广场" "中南大学" --mode transit
  
  # 启动 Web 服务
  python cli.py serve
  
  # 系统诊断
  python cli.py health
  
  # 知识图谱：构建
  python cli.py kg build
  
  # 知识图谱：查询城市信息
  python cli.py kg query --city 杭州
  
  # 知识图谱：按季节推荐
  python cli.py kg query --season 春季
  
  # 知识图谱：统计信息
  python cli.py kg info
  
  # 长期记忆：查看所有记忆
  python cli.py memory list
  
  # 长期记忆：搜索记忆
  python cli.py memory search 成都
  
  # 长期记忆：生成用户城市指南
  python cli.py memory guide
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="可用命令",
        description="运行 python cli.py <命令> --help 查看详细用法",
    )

    # ===== plan-trip =====
    p_plan = subparsers.add_parser(
        "plan-trip", aliases=["plan"],
        help="🎯 智能旅行规划（核心功能）",
        description="基于 LangGraph 多智能体的智能旅行规划，自动搜索景点、查询天气、推荐酒店，生成完整行程。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_plan.add_argument("city", help="目的地城市，如：北京、上海、成都")
    p_plan.add_argument("--start-date", default=None,
                        help="出发日期 (YYYY-MM-DD)，默认为明天")
    p_plan.add_argument("--days", type=int, default=3,
                        help="旅行天数 (默认: 3)")
    p_plan.add_argument("--transportation", default="公共交通",
                        choices=["公共交通", "自驾", "打车", "步行"],
                        help="交通方式 (默认: 公共交通)")
    p_plan.add_argument("--accommodation", default="经济型酒店",
                        choices=["经济型酒店", "舒适型酒店", "豪华酒店", "民宿"],
                        help="住宿偏好 (默认: 经济型酒店)")
    p_plan.add_argument("--preferences", nargs="+", default=[],
                        help="旅行偏好标签，如：历史文化 美食 自然风光")
    p_plan.add_argument("--free-text", default="",
                        help="额外的个性化要求，如：希望多安排博物馆")
    p_plan.add_argument("--output", "-o", default="",
                        help="导出为 JSON 文件路径")
    p_plan.add_argument("--json", dest="json_output", action="store_true",
                        help="同时以 JSON 格式打印结果")

    # ===== rag-search =====
    p_rag = subparsers.add_parser(
        "rag-search", aliases=["rag"],
        help="📚 RAG 知识库检索",
        description="基于向量检索的旅行知识库搜索，支持语义理解。",
    )
    p_rag.add_argument("query", help="检索关键词，如：北京美食推荐、省钱技巧")
    p_rag.add_argument("--top-k", type=int, default=5,
                       help="返回结果数量 (默认: 5)")
    p_rag.add_argument("--json", dest="output_json", action="store_true",
                       help="以 JSON 格式输出")

    # ===== weather =====
    p_weather = subparsers.add_parser(
        "weather", aliases=["w"],
        help="🌤  天气查询",
        description="查询指定城市未来几天天气预报。",
    )
    p_weather.add_argument("city", help="城市名称，如：北京")
    p_weather.add_argument("--json", dest="json_output", action="store_true",
                           help="以 JSON 格式输出")

    # ===== poi-search =====
    p_poi = subparsers.add_parser(
        "poi-search", aliases=["poi"],
        help="🔍 POI 兴趣点搜索",
        description="搜索指定城市的景点、餐厅、酒店等兴趣点信息。",
    )
    p_poi.add_argument("keywords", help="搜索关键词，如：故宫、火锅、博物馆")
    p_poi.add_argument("--city", default="北京", help="城市名称 (默认: 北京)")
    p_poi.add_argument("--json", dest="json_output", action="store_true",
                       help="以 JSON 格式输出")

    # ===== route =====
    p_route = subparsers.add_parser(
        "route", aliases=["r"],
        help="🗺  路线规划（驾车 / 公交地铁）",
        description="规划两点之间的出行路线，支持驾车和公交/地铁两种方式。不指定 --mode 时同时展示全部方案。",
    )
    p_route.add_argument("origin", help="起点（支持中文地名或经纬度坐标），如：天安门 或 116.397128,39.916527")
    p_route.add_argument("destination", help="终点（支持中文地名或经纬度坐标），如：故宫 或 116.391276,39.906217")
    p_route.add_argument("--mode", default=None,
                         choices=["driving", "transit"],
                         help="出行方式: driving=驾车, transit=公交/地铁 (不指定则同时展示全部方案)")

    # ===== serve =====
    p_serve = subparsers.add_parser(
        "serve", aliases=["server", "web"],
        help="🚀 启动 Web 服务",
        description="启动 FastAPI Web 服务，提供 API 和交互式文档。",
    )
    p_serve.add_argument("--host", default=None,
                         help="监听地址 (默认: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=None,
                         help="监听端口 (默认: 8000)")
    p_serve.add_argument("--reload", action="store_true",
                         help="启用热重载（开发模式）")
    p_serve.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="日志级别 (默认: INFO)")

    # ===== health =====
    p_health = subparsers.add_parser(
        "health", aliases=["diag", "status"],
        help="🏥 系统诊断",
        description="检查所有服务的运行状态和配置完整性。",
    )

    # ===== kg =====
    p_kg = subparsers.add_parser(
        "kg", aliases=["knowledge-graph"],
        help="🔗 知识图谱（基于 Neo4j）",
        description="基于 Neo4j 的旅行知识图谱，支持城市、景点、美食、季节关系查询。",
    )
    p_kg.add_argument("kg_action", choices=["build", "query", "info"],
                       help="操作: build=构建图谱, query=查询, info=统计信息")
    p_kg.add_argument("--city", default="", help="查询城市信息，如：北京")
    p_kg.add_argument("--season", default="", help="按季节推荐目的地，如：春季")

    # ===== memory =====
    p_mem = subparsers.add_parser(
        "memory", aliases=["mem"],
        help="💾 长期记忆管理",
        description="管理系统的长期记忆。旅行规划后自动保存城市和景点信息，下次可直接检索。",
    )
    p_mem.add_argument("mem_action", choices=["list", "search", "rebuild", "guide"],
                        help="操作: list=列出记忆, search=搜索记忆, rebuild=重建索引, guide=生成城市指南")
    p_mem.add_argument("query", nargs="?", default="",
                        help="搜索关键词（仅 search 操作需要）")

    return parser


def main():
    """CLI 入口函数"""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n\n💡 提示: 运行 python cli.py <命令> --help 查看详细用法")
        print("   例如: python cli.py plan-trip --help")
        sys.exit(0)

    # 为 plan-trip 设置默认 start_date
    if args.command in ("plan-trip", "plan") and not args.start_date:
        from datetime import datetime, timedelta
        args.start_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 命令路由
    command_map = {
        "plan-trip": cmd_plan_trip,
        "plan": cmd_plan_trip,
        "rag-search": cmd_rag_search,
        "rag": cmd_rag_search,
        "weather": cmd_weather,
        "w": cmd_weather,
        "poi-search": cmd_poi_search,
        "poi": cmd_poi_search,
        "route": cmd_route,
        "r": cmd_route,
        "serve": cmd_serve,
        "server": cmd_serve,
        "web": cmd_serve,
        "health": cmd_health,
        "diag": cmd_health,
        "status": cmd_health,
        "kg": cmd_kg,
        "knowledge-graph": cmd_kg,
        "memory": cmd_memory,
        "mem": cmd_memory,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
