"""RAG 召回率测试工具

用于评估 RAG 检索系统的质量，包括：
1. 召回率 (Recall): 检索到的相关文档数 / 总相关文档数
2. 精确率 (Precision): 检索到的相关文档数 / 检索到的总文档数
3. MRR (Mean Reciprocal Rank): 平均倒数排名
4. NDCG (Normalized Discounted Cumulative Gain): 归一化折损累计增益
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import get_rag_service
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TestQuery:
    """测试查询数据"""
    query: str  # 查询文本
    relevant_doc_ids: List[str]  # 相关文档的 ID（来源文件名）
    description: str  # 查询描述


class RAGRecallTester:
    """RAG 召回率测试器"""

    def __init__(self):
        self.rag_service = get_rag_service()
        self.test_queries: List[TestQuery] = []
        self.results: List[Dict] = []

    def add_test_query(self, query: str, relevant_docs: List[str], description: str = ""):
        """添加测试查询
        
        Args:
            query: 查询文本
            relevant_docs: 相关文档列表（文档来源文件名，如 'city_guides.md'）
            description: 查询描述
        """
        self.test_queries.append(TestQuery(
            query=query,
            relevant_doc_ids=relevant_docs,
            description=description
        ))

    def load_default_test_queries(self):
        """加载默认测试查询集"""
        # 基于实际文档内容设计的测试查询
        self.add_test_query(
            query="北京旅游景点推荐",
            relevant_docs=["city_guides.md"],
            description="测试城市景点推荐"
        )
        self.add_test_query(
            query="旅行预算怎么控制",
            relevant_docs=["budget_tips.md"],
            description="测试预算控制建议"
        )
        self.add_test_query(
            query="什么时候去西藏旅游最好",
            relevant_docs=["seasonal_travel.md"],
            description="测试最佳旅行季节"
        )
        self.add_test_query(
            query="旅行打包技巧",
            relevant_docs=["travel_tips.md"],
            description="测试旅行打包建议"
        )
        self.add_test_query(
            query="上海美食推荐",
            relevant_docs=["city_guides.md"],
            description="测试城市美食推荐"
        )
        self.add_test_query(
            query="如何节省住宿费用",
            relevant_docs=["budget_tips.md"],
            description="测试省钱技巧"
        )
        self.add_test_query(
            query="雨季旅行注意事项",
            relevant_docs=["travel_tips.md", "seasonal_travel.md"],
            description="测试特殊天气建议"
        )
        self.add_test_query(
            query="云南旅游攻略",
            relevant_docs=["city_guides.md", "seasonal_travel.md"],
            description="测试综合旅游攻略"
        )

    def run_single_test(self, test_query: TestQuery, k: int = 5) -> Dict:
        """运行单个测试查询
        
        Args:
            test_query: 测试查询对象
            k: 检索的文档数量
            
        Returns:
            测试结果字典
        """
        # 执行检索
        results = self.rag_service.retrieve(test_query.query, k=k)
        
        # 获取检索到的文档 ID
        retrieved_doc_ids = [doc.metadata.get("source", "") for doc in results]
        
        # 计算指标
        relevant_set = set(test_query.relevant_doc_ids)
        retrieved_set = set(retrieved_doc_ids)
        
        # 召回率：检索到的相关文档数 / 总相关文档数
        hit_docs = relevant_set & retrieved_set
        recall = len(hit_docs) / len(relevant_set) if relevant_set else 0
        
        # 精确率：检索到的相关文档数 / 检索到的总文档数
        precision = len(hit_docs) / len(retrieved_set) if retrieved_set else 0
        
        # MRR: 计算第一个相关文档的排名
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG@k: 简化的二元相关性版本
        dcg = 0.0
        idcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_set:
                dcg += 1.0 / (i + 1)
        
        # 理想情况下的 DCG
        ideal_hits = min(len(relevant_set), k)
        for i in range(ideal_hits):
            idcg += 1.0 / (i + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            "query": test_query.query,
            "description": test_query.description,
            "k": k,
            "retrieved_docs": retrieved_doc_ids,
            "relevant_docs": test_query.relevant_doc_ids,
            "hit_docs": list(hit_docs),
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "ndcg": ndcg,
            "results_detail": [
                {
                    "rank": i + 1,
                    "doc_id": doc.metadata.get("source", ""),
                    "content_preview": doc.page_content[:100].replace("\n", " ")
                }
                for i, doc in enumerate(results)
            ]
        }

    def run_all_tests(self, k: int = 5) -> Dict:
        """运行所有测试
        
        Args:
            k: 每个查询检索的文档数量
            
        Returns:
            汇总的测试结果
        """
        if not self.test_queries:
            self.load_default_test_queries()
        
        print(f"\n{'='*80}")
        print(f"RAG 召回率测试 (k={k})")
        print(f"{'='*80}\n")
        
        individual_results = []
        for i, test_query in enumerate(self.test_queries, 1):
            print(f"[{i}/{len(self.test_queries)}] 测试：{test_query.description}")
            print(f"  查询：{test_query.query}")
            print(f"  期望相关文档：{test_query.relevant_doc_ids}")
            
            result = self.run_single_test(test_query, k)
            individual_results.append(result)
            
            print(f"  检索结果：{result['retrieved_docs']}")
            print(f"  命中：{result['hit_docs']}")
            print(f"  指标：Recall={result['recall']:.2f}, Precision={result['precision']:.2f}, "
                  f"MRR={result['mrr']:.2f}, NDCG={result['ndcg']:.2f}")
            print()
        
        # 计算平均指标
        avg_recall = sum(r["recall"] for r in individual_results) / len(individual_results)
        avg_precision = sum(r["precision"] for r in individual_results) / len(individual_results)
        avg_mrr = sum(r["mrr"] for r in individual_results) / len(individual_results)
        avg_ndcg = sum(r["ndcg"] for r in individual_results) / len(individual_results)
        
        # 计算命中率（至少召回一个相关文档的查询比例）
        hit_rate = sum(1 for r in individual_results if r["recall"] > 0) / len(individual_results)
        
        summary = {
            "total_queries": len(individual_results),
            "k": k,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "avg_mrr": avg_mrr,
            "avg_ndcg": avg_ndcg,
            "hit_rate": hit_rate,
            "individual_results": individual_results
        }
        
        self.results = individual_results
        
        # 打印汇总
        print(f"{'='*80}")
        print(f"测试汇总")
        print(f"{'='*80}")
        print(f"总查询数：{summary['total_queries']}")
        print(f"平均召回率 (Recall): {avg_recall:.2%}")
        print(f"平均精确率 (Precision): {avg_precision:.2%}")
        print(f"平均 MRR: {avg_mrr:.2f}")
        print(f"平均 NDCG: {avg_ndcg:.2f}")
        print(f"命中率：{hit_rate:.2%}")
        print(f"{'='*80}\n")
        
        return summary

    def analyze_failures(self):
        """分析失败的测试案例"""
        if not self.results:
            print("请先运行测试")
            return
        
        failed_tests = [r for r in self.results if r["recall"] == 0]
        
        if not failed_tests:
            print("✅ 所有测试都成功召回了相关文档！")
            return
        
        print(f"\n{'='*80}")
        print(f"失败案例分析 ({len(failed_tests)} 个)")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(failed_tests, 1):
            print(f"[{i}] 查询：{result['query']}")
            print(f"    描述：{result['description']}")
            print(f"    期望文档：{result['relevant_docs']}")
            print(f"    检索结果：{result['retrieved_docs']}")
            print(f"    分析：检索结果中没有期望的文档\n")

    def export_results(self, filepath: str = "rag_recall_results.txt"):
        """导出测试结果到文件"""
        if not self.results:
            print("请先运行测试")
            return
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("RAG 召回率测试报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 汇总统计
            summary = {
                "total_queries": len(self.results),
                "avg_recall": sum(r["recall"] for r in self.results) / len(self.results),
                "avg_precision": sum(r["precision"] for r in self.results) / len(self.results),
                "avg_mrr": sum(r["mrr"] for r in self.results) / len(self.results),
                "avg_ndcg": sum(r["ndcg"] for r in self.results) / len(self.results),
            }
            
            f.write(f"总查询数：{summary['total_queries']}\n")
            f.write(f"平均召回率：{summary['avg_recall']:.2%}\n")
            f.write(f"平均精确率：{summary['avg_precision']:.2%}\n")
            f.write(f"平均 MRR: {summary['avg_mrr']:.2f}\n")
            f.write(f"平均 NDCG: {summary['avg_ndcg']:.2f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("详细结果\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"[{i}] {result['description']}\n")
                f.write(f"    查询：{result['query']}\n")
                f.write(f"    Recall: {result['recall']:.2f}, Precision: {result['precision']:.2f}, "
                       f"MRR: {result['mrr']:.2f}, NDCG: {result['ndcg']:.2f}\n")
                f.write(f"    期望文档：{result['relevant_docs']}\n")
                f.write(f"    检索结果：{result['retrieved_docs']}\n")
                f.write(f"    命中文档：{result['hit_docs']}\n\n")
        
        print(f"测试结果已导出到：{filepath}")


def main():
    """主函数"""
    print("初始化 RAG 服务...")
    tester = RAGRecallTester()
    
    # 运行测试
    summary = tester.run_all_tests(k=5)
    
    # 分析失败案例
    tester.analyze_failures()
    
    # 导出结果
    tester.export_results("rag_recall_results.txt")
    
    return summary


if __name__ == "__main__":
    main()
