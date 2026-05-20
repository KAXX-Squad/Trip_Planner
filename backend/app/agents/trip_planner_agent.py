"""基于 LangGraph 的多智能体旅行规划系统 - 支持并行执行"""

import json
import asyncio
from typing import TypedDict, AsyncGenerator, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from ..services.llm_service import get_llm
from ..services.rag_service import get_rag_service
from ..tools.amap_tools import get_amap_tools
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings
from datetime import datetime, timedelta


# ============ 状态定义 ============

def progress_reducer(a: int, b: int) -> int:
    """进度值 reducer - 取最大值"""
    return max(a, b)

def messages_reducer(a: list, b: list) -> list:
    """消息列表 reducer - 合并列表"""
    return a + b

class AgentState(TypedDict):
    """定义智能体工作流的状态"""
    request: TripRequest
    attractions: str
    weather: str
    hotels: str
    plan: str
    # 使用 Annotated 类型支持并发更新
    messages: Annotated[list, messages_reducer]
    progress: Annotated[int, progress_reducer]
    error: str


# ============ Agent 提示词 ============

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

重要提示:
- 你必须使用提供的工具来搜索景点
- 不要自己编造景点信息
- 使用 maps_text_search 工具，参数包括 keywords 和 city
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

重要提示:
- 你必须使用提供的工具来查询天气
- 不要自己编造天气信息
- 使用 maps_weather 工具，参数是 city
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和用户偏好推荐合适的酒店。

重要提示:
- 你必须使用提供的工具来搜索酒店
- 不要自己编造酒店信息
- 使用 maps_text_search 工具，关键词使用"酒店"
"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息，生成详细的旅行计划。

请严格按照以下 JSON 格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第 1 天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500 元",
        "rating": "4.5",
        "distance": "距离景点 2 公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3 级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

重要提示:
1. weather_info 数组必须包含每一天的天气信息
2. 温度必须是纯数字 (不要带°C 等单位)
3. 每天安排 2-3 个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. 必须包含预算信息
"""


class LangGraphTripPlanner:
    """基于 LangGraph 的多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化 LangGraph 多智能体旅行规划系统...")

        try:
            settings = get_settings()
            self.llm = get_llm()
            self.tools = get_amap_tools()

            # 添加长期记忆工具
            try:
                from ..tools.memory_tools import get_memory_tools
                self.tools += get_memory_tools()
            except Exception:
                pass

            # 创建带工具的 LLM
            self.llm_with_tools = self._create_llm_with_tools()

            # 构建 LangGraph 工作流
            print("  - 构建 LangGraph 工作流...")
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile()

            print(f"✅ LangGraph 多智能体系统初始化成功")
            print(f"   工具数量：{len(self.tools)}")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败：{str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_llm_with_tools(self):
        """创建绑定工具的 LLM"""
        # 由于我们使用的是自定义 LLM 客户端，需要手动处理工具调用
        # 这里返回基础 LLM，工具调用在节点中手动处理
        return self.llm

    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流 - 支持并行执行"""
        # 创建状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("attraction_search", self.attraction_search_node)
        workflow.add_node("weather_search", self.weather_search_node)
        workflow.add_node("hotel_search", self.hotel_search_node)
        workflow.add_node("plan_generator", self.plan_generator_node)

        # 设置入口点 - 从景点搜索开始
        workflow.set_entry_point("attraction_search")
        
        # 使用并行分支：从景点搜索节点同时触发天气和酒店搜索
        # LangGraph 的 add_edge 支持从一个节点到多个节点的分支
        workflow.add_edge("attraction_search", "weather_search")
        workflow.add_edge("attraction_search", "hotel_search")
        
        # 汇聚：天气和酒店搜索都完成后，进入计划生成节点
        workflow.add_edge("weather_search", "plan_generator")
        workflow.add_edge("hotel_search", "plan_generator")
        
        workflow.add_edge("plan_generator", END)

        return workflow

    def attraction_search_node(self, state: AgentState) -> dict:
        """景点搜索节点 - 并行执行"""
        try:
            request = state["request"]
            print(f"📍 景点搜索节点：搜索{request.city}的景点")

            # 更新进度
            progress = state.get("progress", 0) + 15

            # 构建查询
            keywords = request.preferences[0] if request.preferences else "景点"
            
            # 直接使用工具
            from ..tools.amap_tools import maps_text_search
            result = maps_text_search.invoke({"keywords": keywords, "city": request.city})
            
            print(f"景点搜索结果：{result[:200]}...\n")

            return {
                "attractions": result,
                "progress": progress,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"搜索{request.city}的{keywords}景点：{result}")
                ]
            }
        except Exception as e:
            return {
                "attractions": f"景点搜索失败：{str(e)}",
                "error": str(e),
                "progress": state.get("progress", 0) + 15
            }

    def weather_search_node(self, state: AgentState) -> dict:
        """天气查询节点 - 并行执行"""
        try:
            request = state["request"]
            print(f"🌤️  天气查询节点：查询{request.city}的天气")

            # 更新进度
            progress = state.get("progress", 0) + 15

            # 使用天气工具
            from ..tools.amap_tools import maps_weather
            result = maps_weather.invoke({"city": request.city})
            
            print(f"天气查询结果：{result[:200]}...\n")

            return {
                "weather": result,
                "progress": progress,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"{request.city}天气：{result}")
                ]
            }
        except Exception as e:
            return {
                "weather": f"天气查询失败：{str(e)}",
                "error": str(e),
                "progress": state.get("progress", 0) + 15
            }

    def hotel_search_node(self, state: AgentState) -> dict:
        """酒店搜索节点 - 并行执行"""
        try:
            request = state["request"]
            print(f"🏨 酒店搜索节点：搜索{request.city}的酒店")

            # 更新进度
            progress = state.get("progress", 0) + 15

            # 使用工具
            from ..tools.amap_tools import maps_text_search
            result = maps_text_search.invoke({"keywords": "酒店", "city": request.city})
            
            print(f"酒店搜索结果：{result[:200]}...\n")

            return {
                "hotels": result,
                "progress": progress,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"搜索{request.city}的酒店：{result}")
                ]
            }
        except Exception as e:
            return {
                "hotels": f"酒店搜索失败：{str(e)}",
                "error": str(e),
                "progress": state.get("progress", 0) + 15
            }

    def plan_generator_node(self, state: AgentState) -> dict:
        """行程规划生成节点（集成 RAG 增强）"""
        try:
            request = state["request"]
            print(f"📋 行程规划节点：生成{request.city}的旅行计划")

            # 更新进度
            progress = state.get("progress", 0) + 40

            # RAG 检索：根据城市和偏好获取相关知识
            rag_context = self._retrieve_rag_context(request)
            if rag_context:
                print(f"  - RAG 检索到 {rag_context.count('知识')} 条相关知识")

            # 构建提示词
            query = self._build_planner_query(
                request,
                state["attractions"],
                state["weather"],
                state["hotels"],
                rag_context
            )

            # 调用 LLM
            messages = [
                {"role": "system", "content": PLANNER_AGENT_PROMPT},
                {"role": "user", "content": query}
            ]
            
            response = self.llm.invoke(messages)
            
            print(f"行程规划结果：{response[:300]}...\n")

            return {
                "plan": response,
                "progress": progress,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"生成的行程计划：{response}")
                ]
            }
        except Exception as e:
            return {
                "plan": f"行程规划失败：{str(e)}",
                "error": str(e),
                "progress": state.get("progress", 0) + 40
            }

    def _retrieve_rag_context(self, request: TripRequest) -> str:
        """RAG 检索相关知识

        根据城市、偏好等信息检索旅行知识库，返回格式化上下文。
        """
        settings = get_settings()
        if not settings.rag_enabled:
            return ""

        try:
            rag = get_rag_service()

            # 构建检索查询：结合城市和偏好
            query_parts = [request.city]
            if request.preferences:
                query_parts.extend(request.preferences)
            if request.free_text_input:
                query_parts.append(request.free_text_input)

            query = " ".join(query_parts)
            k = settings.rag_retrieval_k

            context = rag.retrieve_as_context(query, k=k)
            return context
        except Exception as e:
            print(f"⚠️  RAG 检索异常（已跳过）: {str(e)}")
            return ""

    def _build_planner_query(self, request: TripRequest, attractions: str, weather: str, hotels: str = "", rag_context: str = "") -> str:
        """构建行程规划查询（集成 RAG 知识）"""
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市：{request.city}
- 日期：{request.start_date} 至 {request.end_date}
- 天数：{request.travel_days}天
- 交通方式：{request.transportation}
- 住宿：{request.accommodation}
- 偏好：{', '.join(request.preferences) if request.preferences else '无'}

**景点信息:**
{attractions}

**天气信息:**
{weather}

**酒店信息:**
{hotels}
"""
        if rag_context:
            query += f"""
**旅行知识库参考（请结合这些知识优化行程）:**
{rag_context}

"""
        query += """**要求:**
1. 每天安排 2-3 个景点
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店 (从酒店信息中选择)
4. 考虑景点之间的距离和交通方式
5. 返回完整的 JSON 格式数据
6. 景点的经纬度坐标要真实准确
7. 结合旅行知识库中的建议，优化行程安排、餐饮推荐和预算规划
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用 LangGraph 工作流生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始 LangGraph 多智能体协作规划旅行...")
            print(f"目的地：{request.city}")
            print(f"日期：{request.start_date} 至 {request.end_date}")
            print(f"天数：{request.travel_days}天")
            print(f"{'='*60}\n")

            # 初始化状态
            initial_state = {
                "request": request,
                "attractions": "",
                "weather": "",
                "hotels": "",
                "plan": "",
                "messages": [
                    SystemMessage(content="你是一个智能旅行规划助手。"),
                    HumanMessage(content=f"请为{request.city}规划一个{request.travel_days}天的行程。")
                ],
                "progress": 0,
                "error": ""
            }

            # 执行工作流
            result = self.app.invoke(initial_state)

            # 解析最终计划
            trip_plan = self._parse_response(result["plan"], request)

            # 保存到长期记忆（提取稳定的城市和景点信息）
            try:
                from ..services.memory_service import get_memory_service
                memory = get_memory_service()
                memory.save_from_trip_plan(trip_plan)
            except Exception as mem_err:
                print(f"⚠️  保存记忆失败（不影响行程）: {str(mem_err)[:100]}")

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)

    async def plan_trip_stream(self, request: TripRequest) -> AsyncGenerator[dict, None]:
        """
        使用 LangGraph 原生流式执行，实时返回进度
        
        通过 LangGraph 的 astream 方法，可以在每个节点执行后获取更新，
        实现真正的流式执行和进度追踪。
        """
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始 LangGraph 流式执行...")
            print(f"目的地：{request.city}")
            print(f"天数：{request.travel_days}天")
            print(f"{'='*60}\n")

            yield {"type": "progress", "progress": 5, "message": "正在初始化 LangGraph 工作流..."}

            # 初始化状态
            initial_state = {
                "request": request,
                "attractions": "",
                "weather": "",
                "hotels": "",
                "plan": "",
                "messages": [
                    SystemMessage(content="你是一个智能旅行规划助手。"),
                    HumanMessage(content=f"请为{request.city}规划一个{request.travel_days}天的行程。")
                ],
                "progress": 0,
                "error": ""
            }

            # 使用 LangGraph 原生流式执行
            # stream_mode=["updates", "values"] 同时返回节点更新和最终状态
            progress_map = {
                "attraction_search": (15, "🔍 正在搜索景点信息..."),
                "weather_search": (25, "🌤️  正在查询天气信息..."),
                "hotel_search": (35, "🏨 正在搜索酒店信息..."),
                "plan_generator": (50, "📋 正在生成行程计划..."),
            }

            final_state = None

            # 使用多模式流式执行：updates 用于进度，values 用于最终状态
            async for mode, chunk in self.app.astream(initial_state, stream_mode=["updates", "values"]):
                if mode == "updates":
                    # 处理节点更新（用于进度显示）
                    for node_name, node_output in chunk.items():
                        print(f"✅ 节点完成：{node_name}")
                        
                        # 发送进度更新
                        if node_name in progress_map:
                            progress, message = progress_map[node_name]
                            yield {"type": "progress", "progress": progress, "message": message}
                            
                            # 如果有错误，也发送错误信息
                            if "error" in node_output:
                                yield {
                                    "type": "progress", 
                                    "progress": progress, 
                                    "message": f"⚠️ {node_name} 执行失败：{node_output.get('error', '未知错误')}"
                                }
                
                elif mode == "values":
                    # 处理完整状态（用于获取最终结果）
                    final_state = chunk

            # 检查是否获取到最终状态
            if not final_state:
                raise Exception("未能获取最终状态")
            
            yield {"type": "progress", "progress": 80, "message": "🔄 正在解析和优化行程数据..."}

            # 解析最终计划
            trip_plan = self._parse_response(final_state["plan"], request)

            # 保存到长期记忆
            try:
                from ..services.memory_service import get_memory_service
                memory = get_memory_service()
                memory.save_from_trip_plan(trip_plan)
            except Exception:
                pass

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            yield {"type": "progress", "progress": 100, "message": "✅ 完成!"}
            yield {"type": "result", "data": trip_plan.model_dump() if hasattr(trip_plan, 'model_dump') else trip_plan}

        except Exception as e:
            print(f"❌ 流式生成旅行计划失败：{str(e)}")
            import traceback
            traceback.print_exc()
            fallback_plan = self._create_fallback_plan(request)
            yield {"type": "progress", "progress": 100, "message": "⚠️ 使用备用方案生成"}
            yield {"type": "result", "data": fallback_plan.model_dump() if hasattr(fallback_plan, 'model_dump') else fallback_plan}

    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        解析 Agent 响应
        
        Args:
            response: Agent 响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取 JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到 JSON 数据")
            
            # 解析 JSON
            data = json.loads(json_str)
            
            # 转换为 TripPlan 对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            print(f"⚠️  解析响应失败：{str(e)}")
            return self._create_fallback_plan(request)

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划 (当 Agent 失败时)"""
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=116.4 + i*0.01 + j*0.005, latitude=39.9 + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程，建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_trip_planner = None


def get_trip_planner_agent() -> LangGraphTripPlanner:
    """获取多智能体旅行规划系统实例 (单例模式)"""
    global _trip_planner

    if _trip_planner is None:
        _trip_planner = LangGraphTripPlanner()

    return _trip_planner
