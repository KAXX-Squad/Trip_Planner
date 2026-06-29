"""RAG 检索质量评估 - 单路(向量) vs 多路(BM25+向量) 对比测试 (100 条查询)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import get_rag_service
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TestCase:
    query: str
    relevant_sources: List[str]
    category: str


class RAGEvaluator:
    """RAG 质量评估器"""

    def __init__(self):
        self.rag = get_rag_service(force_reinit=True)
        self.cases: List[TestCase] = []

    def load_100_queries(self):
        """基于现有 memory 文档生成 100 条测试查询"""
        c = self.cases

        # ========== 1-12：城市级查询 (12条) ==========
        for q, srcs in [
            ("西安有什么好玩的", ["city_西安.md"]),
            ("想去西安旅游", ["city_西安.md"]),
            ("成都旅游攻略", ["city_成都.md"]),
            ("成都值得去吗", ["city_成都.md"]),
            ("长沙旅行推荐", ["city_长沙.md"]),
            ("长沙好玩的地方", ["city_长沙.md"]),
            ("东京旅行攻略", ["city_东京.md"]),
            ("东京有什么景点", ["city_东京.md"]),
            ("鹤岗旅游", ["city_鹤岗.md"]),
            ("鹤岗好玩吗", ["city_鹤岗.md"]),
            ("佳木斯旅行", ["city_佳木斯.md"]),
            ("佳木斯景点推荐", ["city_佳木斯.md"]),
        ]:
            c.append(TestCase(q, srcs, "城市"))

        # ========== 13-72：景点级查询 (60条) ==========
        # --- 西安 (20条) ---
        xi_an_attrs = [
            ("大明宫遗址公园开放时间", ["attraction_大明宫国家遗址公园.md"]),
            ("大明宫值得去吗", ["attraction_大明宫国家遗址公园.md"]),
            ("大雁塔音乐喷泉", ["attraction_大雁塔北广场音乐喷泉.md"]),
            ("西安大雁塔北广场", ["attraction_大雁塔北广场音乐喷泉.md"]),
            ("半坡国际艺术区怎么样", ["attraction_半坡国际艺术区.md"]),
            ("西安回民街小吃", ["attraction_回民街.md"]),
            ("回民街有什么好吃的", ["attraction_回民街.md"]),
            ("宣平里景区", ["attraction_宣平里景区.md"]),
            ("开元公园在哪里", ["attraction_开元公园.md"]),
            ("文景公园好不好玩", ["attraction_文景公园.md"]),
            ("秦始皇兵马俑门票", ["attraction_秦始皇帝陵博物院(兵马俑).md", "attraction_秦始皇帝陵博物院.md"]),
            ("秦始皇帝陵博物院参观", ["attraction_秦始皇帝陵博物院.md", "attraction_秦始皇帝陵博物院(兵马俑).md"]),
            ("红旗铁路公园", ["attraction_红旗铁路公园.md"]),
            ("终南山古楼观", ["attraction_终南山古楼观历史文化景区.md"]),
            ("西安博物院值得看吗", ["attraction_西安博物院.md"]),
            ("西安城墙门票价格", ["attraction_西安城墙.md", "attraction_西安城墙永宁门段.md"]),
            ("西安钟鼓楼广场", ["attraction_西安钟楼.md", "attraction_钟鼓楼广场.md"]),
            ("西安美术馆展览", ["attraction_西安美术馆.md"]),
            ("陕西历史博物馆怎么预约", ["attraction_陕西历史博物馆.md"]),
            ("西安城市运动公园", ["attraction_西安城市运动公园.md"]),
        ]
        for q, srcs in xi_an_attrs:
            c.append(TestCase(q, srcs, "景点"))

        # --- 成都 (10条) ---
        chengdu_attrs = [
            ("宽窄巷子怎么去", ["attraction_宽窄巷子.md", "attraction_宽窄巷子历史文化街区.md"]),
            ("宽窄巷子历史文化", ["attraction_宽窄巷子历史文化街区.md", "attraction_宽窄巷子.md"]),
            ("成都大熊猫繁育基地门票", ["attraction_成都大熊猫繁育研究基地.md"]),
            ("去看大熊猫去哪里", ["attraction_成都大熊猫繁育研究基地.md"]),
            ("春熙路步行街", ["attraction_春熙路步行街_IFS国金中心.md"]),
            ("武侯祠在哪里", ["attraction_武侯祠.md", "attraction_武侯祠博物馆.md"]),
            ("武侯祠博物馆历史", ["attraction_武侯祠博物馆.md", "attraction_武侯祠.md"]),
            ("锦里古街有什么", ["attraction_锦里古街.md"]),
            ("锦里古街美食", ["attraction_锦里古街.md"]),
            ("成都武侯祠和锦里", ["attraction_武侯祠.md", "attraction_武侯祠博物馆.md", "attraction_锦里古街.md"]),
        ]
        for q, srcs in chengdu_attrs:
            c.append(TestCase(q, srcs, "景点"))

        # --- 长沙 (10条) ---
        changsha_attrs = [
            ("岳麓山怎么上去", ["attraction_岳麓山风景名胜区.md", "attraction_麓山景区-观景台.md"]),
            ("岳麓山风景名胜区", ["attraction_岳麓山风景名胜区.md"]),
            ("橘子洲头攻略", ["attraction_橘子洲头.md", "attraction_橘子洲风景名胜区.md"]),
            ("橘子洲景区介绍", ["attraction_橘子洲风景名胜区.md", "attraction_橘子洲头.md"]),
            ("湘江中路沿江风光", ["attraction_湘江中路沿江风光带.md"]),
            ("尖山湖公园怎么样", ["attraction_尖山湖公园.md"]),
            ("月亮岛好玩吗", ["attraction_月亮岛.md"]),
            ("麓山景区观景台", ["attraction_麓山景区-观景台.md", "attraction_岳麓山风景名胜区.md"]),
            ("长沙橘子洲烟花", ["attraction_橘子洲头.md", "attraction_橘子洲风景名胜区.md"]),
            ("长沙滨江风光带", ["attraction_湘江中路沿江风光带.md"]),
        ]
        for q, srcs in changsha_attrs:
            c.append(TestCase(q, srcs, "景点"))

        # --- 东京 (10条) ---
        tokyo_attrs = [
            ("东京晴空塔高多少", ["attraction_东京晴空塔.md"]),
            ("晴空塔怎么去", ["attraction_东京晴空塔.md"]),
            ("台场有什么好玩的", ["attraction_台场.md"]),
            ("新宿御苑门票", ["attraction_新宿御苑.md"]),
            ("明治神宫参观", ["attraction_明治神宫.md"]),
            ("浅草寺怎么样", ["attraction_浅草寺.md"]),
            ("涩谷十字路口", ["attraction_涩谷全向十字路口.md"]),
            ("日本皇居可以参观吗", ["attraction_皇居.md"]),
            ("秋叶原买电器", ["attraction_秋叶原电器街.md"]),
            ("筑地市场吃海鲜", ["attraction_筑地场外市场.md"]),
        ]
        for q, srcs in tokyo_attrs:
            c.append(TestCase(q, srcs, "景点"))

        # --- 鹤岗 (5条) ---
        hegang_attrs = [
            ("鹤岗国家矿山公园", ["attraction_鹤岗国家矿山公园.md"]),
            ("北山公园在哪里", ["attraction_北山公园.md"]),
            ("北普陀寺", ["attraction_北普陀寺.md"]),
            ("天水湖公园", ["attraction_天水湖公园.md"]),
            ("黑龙江桦川国家森林公园", ["attraction_黑龙江桦川国家森林公园万景山景区.md"]),
        ]
        for q, srcs in hegang_attrs:
            c.append(TestCase(q, srcs, "景点"))

        # --- 佳木斯 (3条) ---
        jiamusi_attrs = [
            ("佳木斯八方山", ["attraction_八方山.md"]),
            ("建国镇大雁迁徙", ["attraction_建国镇大雁迁徒地.md"]),
            ("松花江十里景观带", ["attraction_松花江十里景观带环保生态林.md"]),
        ]
        for q, srcs in jiamusi_attrs:
            c.append(TestCase(q, srcs, "景点"))

        # ========== 73-82：美食查询 (10条) ==========
        for q, srcs in [
            ("西安美食推荐", ["city_西安.md"]),
            ("西安特色小吃", ["city_西安.md"]),
            ("成都美食必吃", ["city_成都.md"]),
            ("成都火锅推荐", ["city_成都.md"]),
            ("长沙有什么好吃的", ["city_长沙.md"]),
            ("长沙特色美食街", ["city_长沙.md"]),
            ("东京美食攻略", ["city_东京.md"]),
            ("日本美食推荐", ["city_东京.md"]),
            ("西安回民街必吃小吃", ["attraction_回民街.md", "city_西安.md"]),
            ("锦里古街美食推荐", ["attraction_锦里古街.md"]),
        ]:
            c.append(TestCase(q, srcs, "美食"))

        # ========== 83-92：综合/模糊查询 (10条) ==========
        for q, srcs in [
            ("想去西安玩几天合适", ["city_西安.md"]),
            ("成都好玩还是西安好玩", ["city_成都.md", "city_西安.md"]),
            ("想去看大熊猫去哪里", ["attraction_成都大熊猫繁育研究基地.md", "city_成都.md"]),
            ("历史文化城市推荐", ["city_西安.md", "city_成都.md"]),
            ("爬山看风景去哪里", ["attraction_岳麓山风景名胜区.md", "attraction_终南山古楼观历史文化景区.md"]),
            ("带孩子去哪里玩", ["attraction_成都大熊猫繁育研究基地.md", "attraction_西安城墙.md"]),
            ("拍照打卡景点", ["attraction_大雁塔北广场音乐喷泉.md", "attraction_钟鼓楼广场.md", "attraction_宽窄巷子.md"]),
            ("有哪些博物馆可以逛", ["attraction_陕西历史博物馆.md", "attraction_西安博物院.md", "attraction_武侯祠博物馆.md"]),
            ("适合晚上去的景点", ["attraction_回民街.md", "attraction_钟鼓楼广场.md", "attraction_锦里古街.md"]),
            ("免费景点推荐", ["attraction_西安城墙.md", "attraction_大雁塔北广场音乐喷泉.md", "attraction_宽窄巷子.md"]),
        ]:
            c.append(TestCase(q, srcs, "综合"))

        # ========== 93-100：位置/地址查询 (8条) ==========
        for q, srcs in [
            ("陕西历史博物馆在哪个区", ["attraction_陕西历史博物馆.md"]),
            ("西安钟楼附近有什么", ["attraction_西安钟楼.md"]),
            ("武侯祠地址在哪里", ["attraction_武侯祠.md"]),
            ("大明宫遗址公园怎么去", ["attraction_大明宫国家遗址公园.md"]),
            ("鹤岗矿山公园在哪儿", ["attraction_鹤岗国家矿山公园.md", "city_鹤岗.md"]),
            ("锦里古街在成都哪里", ["attraction_锦里古街.md", "city_成都.md"]),
            ("橘子洲在长沙哪个区", ["attraction_橘子洲头.md", "city_长沙.md"]),
            ("宽窄巷子在成都哪里", ["attraction_宽窄巷子.md", "city_成都.md"]),
        ]:
            c.append(TestCase(q, srcs, "位置"))

        print(f"  已加载 {len(self.cases)} 条测试查询")

    def run_single(self, rag, query: str, k: int = 5):
        results = rag.retrieve(query, k=k)
        sources = [doc.metadata.get("source", "") for doc in results]
        return sources

    def run_single_hybrid(self, rag, query: str, k: int = 5):
        results = rag.retrieve_hybrid(query, k=k)
        sources = [doc.metadata.get("source", "") for doc in results]
        return sources

    def calc_metrics(self, retrieved_sources, relevant_sources):
        relevant_set = set(relevant_sources)
        retrieved_set = set(retrieved_sources)
        hits = relevant_set & retrieved_set
        recall = len(hits) / len(relevant_set) if relevant_set else 0
        precision = len(hits) / len(retrieved_set) if retrieved_set else 0
        mrr = 0.0
        for i, src in enumerate(retrieved_sources):
            if src in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        return {"hits": list(hits), "recall": recall, "precision": precision, "mrr": mrr}

    def run_comparison(self, k: int = 5):
        if not self.cases:
            self.load_100_queries()

        rag = self.rag
        rag._build_bm25_index()

        results_vec = []
        results_hyb = []

        n = len(self.cases)
        print(f"\n{'='*100}")
        print(f"  RAG 检索质量对比测试 (k={k}, 共 {n} 条查询)")
        print(f"  A: 向量检索（bge-m3）  B: 混合检索（BM25 + 向量 + RRF）")
        print(f"{'='*100}\n")

        diff_count = 0
        for i, tc in enumerate(self.cases, 1):
            vec_src = self.run_single(rag, tc.query, k=k)
            hyb_src = self.run_single_hybrid(rag, tc.query, k=k)

            m_vec = self.calc_metrics(vec_src, tc.relevant_sources)
            m_hyb = self.calc_metrics(hyb_src, tc.relevant_sources)

            results_vec.append(m_vec)
            results_hyb.append(m_hyb)

            if m_vec["recall"] != m_hyb["recall"] or m_vec["mrr"] != m_hyb["mrr"]:
                diff_count += 1
                v_icon = "✓" if m_vec["recall"] > 0 else "✗"
                h_icon = "✓" if m_hyb["recall"] > 0 else "✗"
                delta_r = m_hyb["recall"] - m_vec["recall"]
                delta_m = m_hyb["mrr"] - m_vec["mrr"]
                tag = ""
                if delta_r > 0:
                    tag = " <<< B更好 (召回↑)"
                elif delta_r < 0:
                    tag = "    A更好 (召回↓)"
                elif delta_m > 0:
                    tag = " <<< B更好 (排序↑)"
                elif delta_m < 0:
                    tag = "    A更好 (排序↓)"
                print(f"  [{i:03d}] {tc.category} | {tc.query}")
                print(f"       期望: {tc.relevant_sources}")
                print(f"       A向量: {v_icon} R={m_vec['recall']:.0%} MRR={m_vec['mrr']:.2f} | {vec_src}")
                print(f"       B混合: {h_icon} R={m_hyb['recall']:.0%} MRR={m_hyb['mrr']:.2f} | {hyb_src} {tag}")
                print()

        # 汇总
        print(f"\n{'='*100}")
        print(f"  汇总对比 (有差异的查询: {diff_count}/{n})")
        print(f"{'='*100}")
        print(f"  {'指标':<22} {'A: 向量检索':>20} {'B: 混合检索':>20} {'提升':>10}")
        print(f"  {'-'*72}")

        def avg(lst, key):
            return sum(d[key] for d in lst) / n

        def hit_rate(lst):
            return sum(1 for d in lst if d["recall"] > 0) / n

        rows = [
            ("命中率 (Hit@{})".format(k), hit_rate(results_vec), hit_rate(results_hyb)),
            ("平均召回率 (Recall)", avg(results_vec, "recall"), avg(results_hyb, "recall")),
            ("平均精确率 (Precision)", avg(results_vec, "precision"), avg(results_hyb, "precision")),
            ("平均 MRR", avg(results_vec, "mrr"), avg(results_hyb, "mrr")),
        ]

        for label, va, vb in rows:
            delta = vb - va
            if abs(delta) < 0.0001:
                delta_s = "   -  "
            elif delta > 0:
                delta_s = f" +{delta:.2%}"
            else:
                delta_s = f" {delta:.2%}"
            print(f"  {label:<22} {va:>20.2%} {vb:>20.2%} {delta_s:>10}")

        print(f"{'='*100}")

        # 按类别细分
        categories = ["城市", "景点", "美食", "综合", "位置"]
        print(f"\n{'='*100}")
        print(f"  按类别细分 - 召回率")
        print(f"{'='*100}")
        print(f"  {'类别':<10} {'查询数':>8} {'A: 向量':>15} {'B: 混合':>15} {'提升':>10}")
        print(f"  {'-'*58}")
        for cat in categories:
            cv = [r for r, tc in zip(results_vec, self.cases) if tc.category == cat]
            ch = [r for r, tc in zip(results_hyb, self.cases) if tc.category == cat]
            cn = len(cv)
            if cn == 0:
                continue
            va = sum(d["recall"] for d in cv) / cn
            vb = sum(d["recall"] for d in ch) / cn
            delta = vb - va
            if abs(delta) < 0.001:
                ds = "   -  "
            elif delta > 0:
                ds = f" +{delta:.0%}"
            else:
                ds = f" {delta:.0%}"
            hr_v = sum(1 for d in cv if d["recall"] > 0) / cn
            hr_h = sum(1 for d in ch if d["recall"] > 0) / cn
            print(f"  {cat:<10} {cn:>8}  R={va:>6.0%}(H={hr_v:.0%})  R={vb:>6.0%}(H={hr_h:.0%})  {ds:>10}")

        print(f"{'='*100}\n")

        # 失败案例
        failed = [(i + 1, tc, m_vec, m_hyb) for i, (tc, m_vec, m_hyb)
                  in enumerate(zip(self.cases, results_vec, results_hyb))
                  if m_vec["recall"] == 0 and m_hyb["recall"] == 0]
        if failed:
            print(f"\n  都未命中的查询 ({len(failed)} 条):")
            for idx, tc, _, _ in failed:
                print(f"    [{idx:03d}] {tc.query} → 期望: {tc.relevant_sources}")

        return {
            "vector": {"hit_rate": hit_rate(results_vec), "avg_recall": avg(results_vec, "recall"),
                       "avg_precision": avg(results_vec, "precision"), "avg_mrr": avg(results_vec, "mrr")},
            "hybrid": {"hit_rate": hit_rate(results_hyb), "avg_recall": avg(results_hyb, "recall"),
                       "avg_precision": avg(results_hyb, "precision"), "avg_mrr": avg(results_hyb, "mrr")},
        }


def main():
    eval = RAGEvaluator()
    summary = eval.run_comparison(k=5)
    return summary


if __name__ == "__main__":
    main()
