"""长期记忆服务 - 将搜索结果保存为 Markdown 记忆文档

每次旅行规划或搜索后，将稳定的知识（城市信息、景点信息）
持久化为 Markdown 文件，RAG 初始化时自动加载，实现"用过即学"。
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from ..config import get_settings


MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"


class MemoryService:
    """长期记忆服务

    将搜索结果中的稳定知识（城市信息、景点信息）保存为 Markdown 文件。
    所有记忆文件存放在 data/memory/ 目录下，RAG 初始化时自动加载。
    """

    def __init__(self):
        self.settings = get_settings()
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # ============ 保存记忆 ============

    def save_city_memory(self, city: str, attractions: List[dict],
                         cuisines: List[str] = None, seasons: List[str] = None,
                         transportation: str = "", source: str = "旅行规划") -> str:
        """保存城市信息记忆

        Args:
            city: 城市名称
            attractions: 景点列表，每个元素含 name, address, category, description 等
            cuisines: 特色美食列表
            seasons: 最佳季节列表
            transportation: 交通提示
            source: 信息来源

        Returns:
            保存的文件路径
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [f"# 城市记忆: {city}"]
        lines.append(f"- 来源：{source}")
        lines.append(f"- 记录时间：{now}")
        lines.append("")

        if attractions:
            lines.append("## 景点")
            for a in attractions:
                name = a.get("name", "")
                addr = a.get("address", "")
                cat = a.get("category", "")
                parts = [name]
                if addr:
                    parts.append(addr)
                if cat:
                    parts.append(cat)
                if len(parts) > 1:
                    detail = "，".join(parts[1:])
                    lines.append(f"- {parts[0]}（{detail}）")
                else:
                    lines.append(f"- {parts[0]}")

        if cuisines:
            lines.append("")
            lines.append("## 特色美食")
            for c in cuisines:
                lines.append(f"- {c}")

        if seasons:
            lines.append("")
            lines.append("## 最佳季节")
            for s in seasons:
                lines.append(f"- {s}")

        if transportation:
            lines.append("")
            lines.append("## 交通提示")
            lines.append(f"- {transportation}")

        lines.append("")
        content = "\n".join(lines)

        filename = f"city_{city}.md"
        filepath = MEMORY_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        print(f"💾 已保存城市记忆: {city} → {filepath.name}")

        # 同步到知识图谱
        try:
            from .knowledge_graph import get_kg_service
            kg = get_kg_service()
            if kg.is_connected():
                kg.sync_city_memory(city, attractions, cuisines)
        except Exception as e:
            print(f"  ⚠️  知识图谱同步跳过: {str(e)[:80]}")

        return str(filepath)

    def save_attractions_memory(self, attractions: List[dict],
                                city: str = "", source: str = "旅行规划"):
        """保存景点信息记忆

        Args:
            attractions: 景点列表
            city: 所在城市
            source: 信息来源

        Returns:
            保存的文件路径列表
        """
        saved = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        for a in attractions:
            name = a.get("name", "")
            if not name:
                continue

            lines = [f"# 景点: {name}"]
            lines.append(f"- 城市：{a.get('city', city)}")
            lines.append(f"- 来源：{source}")
            lines.append(f"- 记录时间：{now}")
            lines.append("")

            addr = a.get("address", "")
            if addr:
                lines.append(f"- 地址：{addr}")

            loc = a.get("location", {})
            if isinstance(loc, dict) and loc.get("longitude"):
                lines.append(f"- 坐标：{loc['longitude']},{loc['latitude']}")
            elif isinstance(loc, str) and "," in loc:
                lines.append(f"- 坐标：{loc}")

            cat = a.get("category", "")
            if cat:
                lines.append(f"- 类别：{cat}")

            desc = a.get("description", "")
            if desc:
                lines.append("")
                lines.append("## 描述")
                lines.append(desc)

            lines.append("")
            content = "\n".join(lines)

            safe_name = re.sub(r'[\\/:*?"<>|]', '_', name)
            filename = f"attraction_{safe_name}.md"
            filepath = MEMORY_DIR / filename
            filepath.write_text(content, encoding="utf-8")
            print(f"💾 已保存景点记忆: {name} → {filepath.name}")
            saved.append(str(filepath))

        return saved

    def save_from_trip_plan(self, trip_plan) -> List[str]:
        """从旅行计划中提取并保存稳定知识

        只保存城市信息和景点信息，不保存酒店、天气、行程等易变信息。

        Args:
            trip_plan: TripPlan 对象

        Returns:
            保存的文件路径列表
        """
        saved = []

        attractions_list = []
        cuisines_set = set()
        for day in trip_plan.days:
            for a in day.attractions:
                attr_dict = {
                    "name": a.name,
                    "address": a.address,
                    "location": {"longitude": a.location.longitude, "latitude": a.location.latitude} if a.location else "",
                    "category": a.category or "景点",
                    "description": a.description,
                    "city": trip_plan.city,
                }
                attractions_list.append(attr_dict)
            for m in day.meals:
                if m.name:
                    cuisines_set.add(m.name)

        if attractions_list:
            self.save_city_memory(
                city=trip_plan.city,
                attractions=attractions_list,
                cuisines=list(cuisines_set) if cuisines_set else None,
                source="旅行规划",
            )
            saved.extend(self.save_attractions_memory(attractions_list, city=trip_plan.city))

        return saved

    # ============ 查询记忆 ============

    def search_memories(self, query: str) -> List[dict]:
        """在记忆文件中做关键词搜索

        用 RAG 做语义搜索更准确，这个作为轻量级后备方案。
        """
        results = []
        query_lower = query.lower()

        md_files = glob.glob(str(MEMORY_DIR / "*.md"))
        for filepath in md_files:
            try:
                content = Path(filepath).read_text(encoding="utf-8")
                if query_lower in content.lower():
                    filename = os.path.basename(filepath)
                    # 提取标题和第一段作为摘要
                    title = ""
                    for line in content.split("\n"):
                        if line.startswith("# "):
                            title = line.strip("# ").strip()
                            break
                    excerpt = content[:300].replace("\n", " ").strip()
                    results.append({
                        "file": filename,
                        "title": title,
                        "excerpt": excerpt[:200],
                    })
            except Exception:
                continue

        return results

    def get_all_memories(self) -> List[dict]:
        """列出所有记忆文件"""
        results = []
        md_files = sorted(glob.glob(str(MEMORY_DIR / "*.md")))
        for filepath in md_files:
            try:
                content = Path(filepath).read_text(encoding="utf-8")
                title = ""
                source = ""
                time_str = ""
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line.strip("# ").strip()
                    elif line.startswith("- 来源：") or line.startswith("- 来源:"):
                        source = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    elif line.startswith("- 记录时间：") or line.startswith("- 记录时间:"):
                        time_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                results.append({
                    "file": os.path.basename(filepath),
                    "title": title or os.path.basename(filepath),
                    "source": source,
                    "time": time_str,
                    "size": len(content),
                })
            except Exception:
                continue
        return results

    def rebuild_index(self):
        """重建所有记忆文件索引

        调用此方法后，RAG 下次初始化时会重新加载 memory 目录的所有文件。
        这个方法主要是清理旧文件或重新组织。
        """
        self.collect_knowledge_into_guide()
        print("✅ 记忆索引重建完成")

    def collect_knowledge_into_guide(self) -> str:
        """将分散的记忆收集到一份城市指南中

        合并所有 city_*.md 和 attraction_*.md 文件，
        生成一份类似 city_guides.md 格式的完整指南。
        """
        cities = {}

        md_files = glob.glob(str(MEMORY_DIR / "*.md"))
        for filepath in md_files:
            try:
                content = Path(filepath).read_text(encoding="utf-8")
                filename = os.path.basename(filepath)

                if filename.startswith("city_"):
                    city_name = filename.replace("city_", "").replace(".md", "")
                    cities[city_name] = self._parse_city_memory(content, city_name)
                elif filename.startswith("attraction_"):
                    for line in content.split("\n"):
                        if line.startswith("- 城市："):
                            city_name = line.split("：", 1)[1].strip()
                            if city_name not in cities:
                                cities[city_name] = {"name": city_name, "attractions": [], "cuisines": [], "seasons": [], "transportation": ""}
                            break
            except Exception:
                continue

        if not cities:
            return ""

        lines = ["# 用户探索过的城市指南（来自记忆）", ""]
        for city_name in sorted(cities.keys()):
            city = cities[city_name]
            lines.append(f"## {city_name}")
            if city.get("attractions"):
                lines.append(f"- 探索过的景点：{'、'.join(city['attractions'][:10])}")
            if city.get("cuisines"):
                lines.append(f"- 当地美食：{'、'.join(city['cuisines'][:10])}")
            if city.get("seasons"):
                lines.append(f"- 最佳季节：{'、'.join(city['seasons'][:5])}")
            if city.get("transportation"):
                lines.append(f"- 交通：{city['transportation']}")
            lines.append("")

        content = "\n".join(lines)
        guide_path = MEMORY_DIR / "_user_cities_guide.md"
        guide_path.write_text(content, encoding="utf-8")
        print(f"📖 已生成用户城市指南: {guide_path.name}")
        return content

    def _parse_city_memory(self, content: str, city_name: str) -> dict:
        """解析城市记忆文件"""
        city = {"name": city_name, "attractions": [], "cuisines": [], "seasons": [], "transportation": ""}
        current_section = ""
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                current_section = line.strip("# ").strip()
            elif line.startswith("- ") and current_section:
                item = line.lstrip("- ").strip()
                if current_section == "景点":
                    city["attractions"].append(item.split("（")[0].strip() if "（" in item else item)
                elif current_section == "特色美食":
                    city["cuisines"].append(item)
                elif current_section == "最佳季节":
                    city["seasons"].append(item)
                elif current_section == "交通提示":
                    city["transportation"] = item
        return city


# 全局单例
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """获取记忆服务实例"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
