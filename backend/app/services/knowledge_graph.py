"""知识图谱服务 - 基于 Neo4j 的旅行知识图谱

从 Markdown 知识文档中抽取实体和关系，构建可查询的知识图谱。
支持城市、景点、美食、季节、活动等实体及其关系。
"""

import re
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional
from neo4j import GraphDatabase, Driver, Session
from ..config import get_settings


# ============ Markdown 解析器 ============

class CityGuideParser:
    """解析 city_guides.md，抽取城市-景点-美食关系"""

    def parse(self, content: str) -> List[dict]:
        """解析城市指南文档

        返回城市节点列表，每个城市包含景点、美食、季节等子实体。
        """
        cities = []
        # 按 ## 分割城市
        city_blocks = re.split(r'\n## ', content)

        for block in city_blocks:
            if not block.strip():
                continue

            city = self._parse_city(block)
            if city:
                cities.append(city)

        return cities

    def _parse_city(self, block: str) -> Optional[dict]:
        """解析单个城市信息块"""
        lines = block.strip().split('\n')
        if not lines:
            return None

        city_name = lines[0].strip().rstrip('：:')
        if not city_name or city_name.startswith('#') or city_name == '中国热门旅行城市指南':
            return None

        city = {
            "name": city_name,
            "attractions": [],
            "cuisines": [],
            "best_seasons": [],
            "transportation": "",
            "accommodation": "",
            "shopping": "",
            "suggest_days": "",
            "budget": "",
        }

        for line in lines[1:]:
            line = line.strip()
            if line.startswith('- 必游景点') or line.startswith('- 必游景点'):
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    raw = parts[1].strip()
                    # 先去掉括号里的补充内容，再按顿号分割
                    raw_no_paren = re.sub(r'[（(][^）)]*[）)]', '', raw)
                    attractions = [a.strip() for a in raw_no_paren.split('、') if a.strip()]
                    city["attractions"] = attractions
            elif line.startswith('- 特色美食') or line.startswith('- 特色美食'):
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    city["cuisines"] = [c.strip() for c in parts[1].split('、') if c.strip()]
            elif '最佳季节' in line:
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    seasons_text = parts[1]
                    seasons = []
                    for s in ['春季', '夏季', '秋季', '冬季']:
                        if s in seasons_text:
                            seasons.append(s)
                    city["best_seasons"] = seasons
            elif line.startswith('- 交通') or line.startswith('- 交通'):
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    city["transportation"] = parts[1].strip()
            elif '住宿' in line and ('推荐' in line or '建议' in line):
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    city["accommodation"] = parts[1].strip()
            elif line.startswith('- 建议天数') or line.startswith('- 建议天数'):
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    city["suggest_days"] = parts[1].strip()
            elif '预算' in line:
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    city["budget"] = parts[1].strip()

        return city


class SeasonalTravelParser:
    """解析 seasonal_travel.md，抽取季节-城市-活动关系"""

    def parse(self, content: str) -> List[dict]:
        """解析季节性旅行建议文档"""
        seasons = []
        season_blocks = re.split(r'\n## ', content)

        for block in season_blocks:
            if not block.strip():
                continue

            season = self._parse_season(block)
            if season:
                seasons.append(season)

        return seasons

    def _parse_season(self, block: str) -> Optional[dict]:
        """解析单个季节信息块"""
        lines = block.strip().split('\n')
        if not lines:
            return None

        season_name = lines[0].strip().rstrip('：:')
        # 去掉括号里的日期范围，如 "春季（3月-5月）" → "春季"
        season_name = re.sub(r'[（(].*?[）)]', '', season_name).strip()
        if not season_name or season_name.startswith('#'):
            return None
        # 只保留真正的季节名称
        valid_seasons = {"春季", "夏季", "秋季", "冬季"}
        if season_name not in valid_seasons:
            return None

        season = {
            "name": season_name,
            "suitable_cities": [],
            "activities": [],
            "clothing": "",
            "tips": "",
        }

        for line in lines[1:]:
            line = line.strip()
            if '适合目的地' in line:
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    cities = [c.strip() for c in parts[1].split('、') if c.strip()]
                    season["suitable_cities"] = cities
            elif '特色活动' in line:
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    raw = parts[1].strip()
                    raw_no_paren = re.sub(r'[（(][^）)]*[）)]', '', raw)
                    season["activities"] = [a.strip() for a in raw_no_paren.split('、') if a.strip()]
            elif '穿衣' in line:
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    season["clothing"] = parts[1].strip()
            elif '注意' in line:
                parts = line.split('：', 1) if '：' in line else line.split(':', 1)
                if len(parts) > 1:
                    season["tips"] = parts[1].strip()

        return season


# ============ 知识图谱构建器 ============

class KnowledgeGraphBuilder:
    """将解析后的数据写入 Neo4j"""

    def __init__(self, driver: Driver, database: str):
        self.driver = driver
        self.database = database

    def build(self, cities: List[dict], seasons: List[dict]):
        """构建完整知识图谱"""
        print(f"  - 准备写入 {len(cities)} 个城市, {len(seasons)} 个季节")

        with self.driver.session(database=self.database) as session:
            self._clear_graph(session)
            self._create_constraints(session)
            self._build_cities(session, cities)
            self._build_seasons(session, seasons)
            self._link_seasons_to_cities(session, cities, seasons)

    def _clear_graph(self, session: Session):
        """清空已有数据"""
        session.run("MATCH (n) DETACH DELETE n")
        print("  - 已清空旧数据")

    def _create_constraints(self, session: Session):
        """创建唯一约束"""
        for label, prop in [("City", "name"), ("Attraction", "name"),
                            ("Cuisine", "name"), ("Season", "name"),
                            ("Activity", "name")]:
            try:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
            except Exception as e:
                print(f"  ⚠️  约束创建跳过 ({label}): {str(e)[:60]}")
        print("  - 约束已创建")

    def _build_cities(self, session: Session, cities: List[dict]):
        """构建城市及关联节点"""
        for city in cities:
            session.run(
                """MERGE (c:City {name: $name})
                   SET c.budget = $budget,
                       c.suggest_days = $suggest_days,
                       c.transportation = $transportation,
                       c.accommodation = $accommodation,
                       c.shopping = $shopping""",
                name=city["name"],
                budget=city["budget"],
                suggest_days=city["suggest_days"],
                transportation=city["transportation"],
                accommodation=city["accommodation"],
                shopping=city["shopping"],
            )

            for attraction_name in city["attractions"]:
                session.run(
                    """MERGE (a:Attraction {name: $name})
                       SET a.category = $category
                       WITH a
                       MATCH (c:City {name: $city_name})
                       MERGE (c)-[:HAS_ATTRACTION]->(a)""",
                    name=attraction_name,
                    category="景点",
                    city_name=city["name"],
                )

            for cuisine_name in city["cuisines"]:
                session.run(
                    """MERGE (cu:Cuisine {name: $name})
                       WITH cu
                       MATCH (c:City {name: $city_name})
                       MERGE (c)-[:KNOWN_FOR]->(cu)""",
                    name=cuisine_name,
                    city_name=city["name"],
                )

            for season_name in city["best_seasons"]:
                session.run(
                    """MERGE (s:Season {name: $name})
                       WITH s
                       MATCH (c:City {name: $city_name})
                       MERGE (c)-[:BEST_IN]->(s)
                       MERGE (s)-[:RECOMMEND]->(c)""",
                    name=season_name,
                    city_name=city["name"],
                )

        print(f"  - 已构建 {len(cities)} 个城市节点")

    def _build_seasons(self, session: Session, seasons: List[dict]):
        """构建季节及活动节点"""
        for season in seasons:
            session.run(
                """MERGE (s:Season {name: $name})
                   SET s.clothing = $clothing,
                       s.tips = $tips""",
                name=season["name"],
                clothing=season["clothing"],
                tips=season["tips"],
            )

            for activity_name in season["activities"]:
                session.run(
                    """MERGE (a:Activity {name: $name})
                       WITH a
                       MATCH (s:Season {name: $season_name})
                       MERGE (s)-[:SUITABLE_FOR]->(a)""",
                    name=activity_name,
                    season_name=season["name"],
                )

        print(f"  - 已构建 {len(seasons)} 个季节节点")

    def _link_seasons_to_cities(self, session: Session, cities: List[dict], seasons: List[dict]):
        """确保季节-城市双向关联完整"""
        season_map = {s["name"]: s for s in seasons}

        for season in seasons:
            for city_name in season.get("suitable_cities", []):
                session.run(
                    """MATCH (s:Season {name: $season_name})
                       MATCH (c:City {name: $city_name})
                       MERGE (c)-[:BEST_IN]->(s)
                       MERGE (s)-[:RECOMMEND]->(c)""",
                    season_name=season["name"],
                    city_name=city_name,
                )

    def upsert_city(self, city_name: str, attractions: List[dict], cuisines: List[str] = None):
        """增量更新城市节点及其关联的景点和美食

        旅行规划后调用，将记忆同步到知识图谱，避免全量重建。
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                """MERGE (c:City {name: $name})
                   SET c.source = $source""",
                name=city_name,
                source="memory",
            )

            added_attr = 0
            added_cuisine = 0
            for a in attractions:
                a_name = a.get("name", "")
                a_cat = a.get("category", "景点")
                a_addr = a.get("address", "")
                if not a_name:
                    continue
                result = session.run(
                    """MERGE (a:Attraction {name: $name})
                       SET a.category = $category,
                           a.address = $address
                       WITH a
                       MATCH (c:City {name: $city_name})
                       MERGE (c)-[:HAS_ATTRACTION]->(a)
                       RETURN a.name AS name""",
                    name=a_name,
                    category=a_cat,
                    address=a_addr,
                    city_name=city_name,
                )
                if result.single():
                    added_attr += 1

            if cuisines:
                for c_name in cuisines:
                    if not c_name:
                        continue
                    session.run(
                        """MERGE (cu:Cuisine {name: $name})
                           WITH cu
                           MATCH (c:City {name: $city_name})
                           MERGE (c)-[:KNOWN_FOR]->(cu)""",
                        name=c_name,
                        city_name=city_name,
                    )
                    added_cuisine += 1

            print(f"  - 已同步到知识图谱: {city_name} ({added_attr} 景点, {added_cuisine} 美食)")


# ============ 知识图谱查询器 ============

class KnowledgeGraphQuery:
    """封装常用的知识图谱查询"""

    def __init__(self, driver: Driver, database: str):
        self.driver = driver
        self.database = database

    def get_city_info(self, city_name: str) -> Optional[dict]:
        """查询城市完整信息（景点、美食、季节）"""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """MATCH (c:City {name: $name})
                   OPTIONAL MATCH (c)-[:HAS_ATTRACTION]->(a:Attraction)
                   OPTIONAL MATCH (c)-[:KNOWN_FOR]->(cu:Cuisine)
                   OPTIONAL MATCH (c)-[:BEST_IN]->(s:Season)
                   RETURN c.name AS city,
                          c.budget AS budget,
                          c.suggest_days AS suggest_days,
                          c.transportation AS transportation,
                          c.accommodation AS accommodation,
                          collect(DISTINCT a.name) AS attractions,
                          collect(DISTINCT cu.name) AS cuisines,
                          collect(DISTINCT s.name) AS best_seasons""",
                name=city_name,
            )
            record = result.single()
            if record and record["city"]:
                return {
                    "name": record["city"],
                    "budget": record.get("budget", ""),
                    "suggest_days": record.get("suggest_days", ""),
                    "transportation": record.get("transportation", ""),
                    "accommodation": record.get("accommodation", ""),
                    "attractions": [a for a in record.get("attractions", []) if a],
                    "cuisines": [c for c in record.get("cuisines", []) if c],
                    "best_seasons": [s for s in record.get("best_seasons", []) if s],
                }
            return None

    def recommend_by_season(self, season_name: str) -> List[dict]:
        """按季节推荐目的地"""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """MATCH (s:Season {name: $name})-[r:RECOMMEND]->(c:City)
                   OPTIONAL MATCH (c)-[:HAS_ATTRACTION]->(a:Attraction)
                   OPTIONAL MATCH (c)-[:KNOWN_FOR]->(cu:Cuisine)
                   RETURN c.name AS city,
                          c.budget AS budget,
                          collect(DISTINCT a.name) AS attractions,
                          collect(DISTINCT cu.name) AS cuisines
                   ORDER BY c.name""",
                name=season_name,
            )
            return [
                {
                    "name": r["city"],
                    "budget": r.get("budget", ""),
                    "attractions": [a for a in r.get("attractions", []) if a],
                    "cuisines": [c for c in r.get("cuisines", []) if c],
                }
                for r in result
            ]

    def recommend_by_budget(self, budget_keyword: str) -> List[dict]:
        """按预算推荐目的地"""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """MATCH (c:City)
                   WHERE c.budget CONTAINS $keyword
                   OPTIONAL MATCH (c)-[:HAS_ATTRACTION]->(a:Attraction)
                   RETURN c.name AS city,
                          c.budget AS budget,
                          c.suggest_days AS suggest_days,
                          collect(DISTINCT a.name) AS attractions
                   ORDER BY c.name""",
                keyword=budget_keyword,
            )
            return [
                {
                    "name": r["city"],
                    "budget": r.get("budget", ""),
                    "suggest_days": r.get("suggest_days", ""),
                    "attractions": [a for a in r.get("attractions", []) if a],
                }
                for r in result
            ]

    def get_season_activities(self, season_name: str) -> List[str]:
        """查询某季节适合的活动"""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """MATCH (s:Season {name: $name})-[:SUITABLE_FOR]->(a:Activity)
                   RETURN a.name AS activity""",
                name=season_name,
            )
            return [r["activity"] for r in result]

    def get_all_cities(self) -> List[str]:
        """获取所有城市列表"""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (c:City) RETURN c.name AS city ORDER BY c.name")
            return [r["city"] for r in result]

    def get_all_seasons(self) -> List[str]:
        """获取所有季节列表"""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (s:Season) RETURN s.name AS season ORDER BY s.name")
            return [r["season"] for r in result]

    def query_cypher(self, cypher: str) -> List[dict]:
        """执行自定义 Cypher 查询"""
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher)
            return [record.data() for record in result]

    def get_stats(self) -> dict:
        """获取图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            nodes = session.run(
                """MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count
                   ORDER BY count DESC"""
            ).data()
            rels = session.run(
                """MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count
                   ORDER BY count DESC"""
            ).data()
            return {
                "node_counts": {r["label"]: r["count"] for r in nodes},
                "relationship_counts": {r["type"]: r["count"] for r in rels},
            }


# ============ 知识图谱服务（入口） ============

class KnowledgeGraphService:
    """知识图谱服务入口

    管理 Neo4j 连接、构建图谱、提供查询接口。
    """

    def __init__(self):
        self.settings = get_settings()
        self._driver: Optional[Driver] = None
        self._query: Optional[KnowledgeGraphQuery] = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
                max_connection_lifetime=3600,
            )
        return self._driver

    @property
    def query(self) -> KnowledgeGraphQuery:
        if self._query is None:
            self._query = KnowledgeGraphQuery(self.driver, self.settings.neo4j_database)
        return self._query

    def is_connected(self) -> bool:
        """验证连接是否正常"""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def build_from_files(self):
        """从 Markdown 文件构建知识图谱"""
        print("🔄 开始构建旅行知识图谱...")

        data_dir = Path(__file__).parent.parent / "data"

        # 1. 解析城市指南
        city_file = data_dir / "city_guides.md"
        if city_file.exists():
            print("  - 解析城市指南...")
            city_parser = CityGuideParser()
            cities = city_parser.parse(city_file.read_text(encoding="utf-8"))
            print(f"  - 找到 {len(cities)} 个城市")
        else:
            cities = []
            print("  ⚠️  未找到 city_guides.md")

        # 2. 解析季节性建议
        season_file = data_dir / "seasonal_travel.md"
        if season_file.exists():
            print("  - 解析季节性旅行建议...")
            season_parser = SeasonalTravelParser()
            seasons = season_parser.parse(season_file.read_text(encoding="utf-8"))
            print(f"  - 找到 {len(seasons)} 个季节")
        else:
            seasons = []
            print("  ⚠️  未找到 seasonal_travel.md")

        if not cities and not seasons:
            print("❌ 未找到任何可解析的知识文档")
            return False

        # 3. 写入 Neo4j
        print("  - 写入 Neo4j...")
        builder = KnowledgeGraphBuilder(self.driver, self.settings.neo4j_database)
        builder.build(cities, seasons)

        # 4. 打印统计
        stats = self.query.get_stats()
        total_nodes = sum(stats["node_counts"].values())
        total_rels = sum(stats["relationship_counts"].values())
        print(f"\n✅ 知识图谱构建完成!")
        print(f"   节点: {total_nodes} 个 ({', '.join(f'{k}: {v}' for k, v in stats['node_counts'].items())})")
        print(f"   关系: {total_rels} 条")
        return True

    def sync_city_memory(self, city_name: str, attractions: List[dict], cuisines: List[str] = None):
        """将长期记忆中的城市信息同步到知识图谱

        旅行规划后自动调用，无需全量重建。
        """
        if not self.is_connected():
            return False
        builder = KnowledgeGraphBuilder(self.driver, self.settings.neo4j_database)
        builder.upsert_city(city_name, attractions, cuisines)
        return True

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None


# 全局单例
_kg_service: Optional[KnowledgeGraphService] = None


def get_kg_service() -> KnowledgeGraphService:
    """获取知识图谱服务实例"""
    global _kg_service
    if _kg_service is None:
        _kg_service = KnowledgeGraphService()
    return _kg_service
