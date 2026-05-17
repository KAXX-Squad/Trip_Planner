"""RAG 快速测试脚本 - 用于快速测试单个查询的检索效果"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import get_rag_service


def test_query(query: str, k: int = 5):
    """测试单个查询的检索效果
    
    Args:
        query: 查询文本
        k: 返回的文档数量
    """
    print(f"\n{'='*80}")
    print(f"测试查询：{query}")
    print(f"{'='*80}\n")
    
    rag = get_rag_service()
    results = rag.retrieve(query, k=k)
    
    if not results:
        print("❌ 未检索到任何结果")
        return
    
    print(f"检索到 {len(results)} 个相关文档块:\n")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', '未知')
        content = doc.page_content.replace('\n', ' ').strip()
        
        print(f"[{i}] 来源：{source}")
        print(f"    内容：{content[:150]}...")
        print()


def compare_queries(queries: list, k: int = 3):
    """比较多个查询的检索效果"""
    print(f"\n{'='*80}")
    print(f"批量测试 {len(queries)} 个查询")
    print(f"{'='*80}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] ", end="")
        test_query(query, k)


if __name__ == "__main__":
    # 示例：测试单个查询
    # test_query("北京旅游景点")
    
    # 示例：批量测试
    test_queries = [
        "北京旅游攻略",
        "如何节省旅行预算",
        "什么时候去云南最好",
        "旅行打包清单",
    ]
    
    compare_queries(test_queries, k=3)
