# RAG 召回率测试指南

## 📊 测试结果概览

您的 RAG 系统当前表现：
- **平均召回率 (Recall)**: 100% ✅
- **平均精确率 (Precision)**: 51.04%
- **平均 MRR**: 0.88
- **平均 NDCG**: 1.49
- **命中率**: 100%

### 指标解释

1. **召回率 (Recall)**: 检索到的相关文档数 / 总相关文档数
   - 100% 表示所有相关文档都被找到了
   - 这是最重要的指标

2. **精确率 (Precision)**: 检索到的相关文档数 / 检索到的总文档数
   - 51% 表示检索结果中约一半是相关的
   - 受检索数量 k 影响

3. **MRR (平均倒数排名)**: 第一个相关文档的排名倒数
   - 0.88 表示相关文档通常排在第 1-2 位
   - 越接近 1 越好

4. **NDCG**: 考虑排名质量的综合指标
   - 值越大越好，考虑了相关性等级

## 🚀 如何运行测试

### 基本测试
```bash
cd backend
python tests\test_rag_recall.py
```

### 自定义 k 值（检索文档数量）
编辑 `tests\test_rag_recall.py` 文件，在 `main()` 函数中修改：
```python
summary = tester.run_all_tests(k=3)  # 改为 3 或 10 等其他值
```

## 📝 如何添加自定义测试用例

### 方法 1: 在代码中添加

编辑 `tests\test_rag_recall.py`，在 `load_default_test_queries()` 方法中添加：

```python
self.add_test_query(
    query="您的测试查询",
    relevant_docs=["期望的文档名.md"],  # 可以是多个
    description="测试描述"
)
```

### 方法 2: 使用 Python 脚本交互式添加

```python
from tests.test_rag_recall import RAGRecallTester

tester = RAGRecallTester()

# 添加自定义测试
tester.add_test_query(
    query="西安旅游攻略",
    relevant_docs=["city_guides.md"],
    description="测试西安相关"
)

# 运行测试
results = tester.run_all_tests(k=5)

# 分析失败案例
tester.analyze_failures()

# 导出结果
tester.export_results("my_test_results.txt")
```

## 📈 如何优化 RAG 性能

### 如果召回率低 (< 80%)

1. **调整文档分割参数**
   ```python
   # 在 config.py 或 .env 中
   RAG_CHUNK_SIZE=300  # 减小块大小
   RAG_CHUNK_OVERLAP=50  # 增加重叠
   ```

2. **改进 Embedding 模型**
   - 当前使用：text-embedding-v2
   - 可尝试：text-embedding-v1

3. **增加检索数量 k**
   ```python
   tester.run_all_tests(k=10)  # 增加 k 值
   ```

### 如果精确率低 (< 40%)

1. **减小 k 值**
   ```python
   tester.run_all_tests(k=3)  # 减少返回的文档数
   ```

2. **优化查询语句**
   - 使用更具体的关键词
   - 避免过于宽泛的查询

3. **调整文档分割**
   - 增加 chunk_size，让文档块更完整

## 🔍 详细分析工具

### 查看检索详情
```python
from app.services.rag_service import get_rag_service

rag = get_rag_service()
results = rag.retrieve("北京旅游", k=5)

for i, doc in enumerate(results, 1):
    print(f"{i}. 来源：{doc.metadata['source']}")
    print(f"   内容：{doc.page_content[:200]}...\n")
```

### 比较不同 k 值的效果
```python
tester = RAGRecallTester()
tester.load_default_test_queries()

for k in [3, 5, 10]:
    print(f"\n=== k={k} ===")
    tester.run_all_tests(k=k)
```

## 📊 测试结果文件

每次测试会生成 `rag_recall_results.txt` 文件，包含：
- 汇总统计
- 每个测试的详细结果
- 检索到的具体文档内容预览

## 💡 最佳实践

1. **定期测试**: 每次修改 RAG 配置后都运行测试
2. **多样化查询**: 添加各种类型的查询（城市、季节、预算等）
3. **关注趋势**: 比较不同配置下的指标变化
4. **平衡指标**: 在召回率和精确率之间找到平衡点

## 🎯 当前系统优势

✅ **召回率 100%** - 所有相关文档都能被找到
✅ **MRR 0.88** - 相关文档排名靠前
✅ **通义千问 Embedding** - 语义理解能力强
✅ **文档分割合理** - 11 个文档块覆盖全面

## 🔧 下一步优化建议

1. 增加测试查询数量（建议 20+ 个）
2. 添加细粒度的相关性标注（部分相关 vs 完全相关）
3. 测试不同查询长度对效果的影响
4. 对比 TF-IDF 和语义 Embedding 的效果差异
