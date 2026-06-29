"""RAG 检索增强生成服务 - 基于向量检索的旅行知识增强

支持三种 Embedding 方式（按优先级）:
1. HuggingFace 本地模型（默认，无需 API Key，需下载模型）
2. OpenAI 兼容 API（复用 LLM 配置，需 API 支持 Embedding）
3. TF-IDF 关键词检索（纯本地，零依赖，始终可用）

向量数据库使用 Qdrant：
- local 模式：本地持久化，无需外部服务
- memory 模式：仅内存，适合测试
- cloud 模式：连接远程 Qdrant 服务器

支持多路召回：
- 向量检索（语义相似度）
- BM25 检索（关键词精确匹配）
- 混合检索（RRF 融合 BM25 + 向量）
"""

import os
import glob
import re
from pathlib import Path
from typing import List, Optional
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from ..config import get_settings


class TfidfEmbedding(Embeddings):
    """基于 TF-IDF 的简单 Embedding（纯本地实现，无需下载任何模型）

    兼容 LangChain Embeddings 接口：embed_documents, embed_query
    注意：embed_documents 首次调用会训练（fit），之后调用仅转换（transform）。
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer="word",
            ngram_range=(1, 2),
        )
        self._fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            embeddings = self.vectorizer.fit_transform(texts).toarray().tolist()
            self._fitted = True
        else:
            embeddings = self.vectorizer.transform(texts).toarray().tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            return [0.0] * self.vectorizer.max_features
        embedding = self.vectorizer.transform([text]).toarray().tolist()[0]
        return embedding

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


class DashScopeEmbedding(Embeddings):
    """基于通义千问 DashScope API 的 Embedding

    兼容 LangChain Embeddings 接口
    """

    def __init__(self, model: str = "text-embedding-v2", api_key: str = "", base_url: str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com/api/v1"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import httpx
        # 批量处理，一次请求所有文本
        emb = self._embed_batch(texts)
        return emb

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        import httpx
        # 通义千问原生 Embedding API
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 通义千问原生 API 格式
        payload = {
            "model": self.model,
            "input": {"texts": texts}
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # 通义千问返回格式：{"output": {"embeddings": [{"text_index": 0, "embedding": [...}]}}
                if "output" in data and "embeddings" in data["output"]:
                    embeddings = [item["embedding"] for item in data["output"]["embeddings"]]
                    return embeddings
                # 兼容 OpenAI 格式
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                raise Exception(f"Unexpected response format: {data}")
        except Exception as e:
            raise Exception(f"DashScope Embedding 失败：{str(e)}")

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


class RAGService:
    """基于 Qdrant 向量数据库的 RAG 检索服务

    加载旅行知识文档，构建 Qdrant 向量索引，并提供语义检索能力。
    检索到的知识将被注入到 LLM 提示词中，增强生成质量。

    支持三种 Qdrant 运行模式:
    - local: 本地持久化（默认），数据存储在磁盘上
    - memory: 仅内存模式，适合测试
    - cloud: 连接远程 Qdrant 服务器
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = None
        self.embedding_provider = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self._initialized = False
        # BM25 相关
        self._bm25_index = None
        self._bm25_chunks: List[Document] = []
        self._bm25_tokenizer = None

    def _init_embeddings(self):
        """初始化 Embedding 模型

        尝试顺序:
        1. HuggingFace 本地模型（需下载，默认）
        2. OpenAI 兼容 API（需 API Key 支持 Embedding）
        3. TF-IDF 关键词检索（纯本地，始终可用）
        """
        provider = self.settings.rag_embedding_provider
        model_name = self.settings.rag_embedding_model

        if provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                print(f"  - 尝试 HuggingFace 本地 Embedding: {model_name}")
                embedding = HuggingFaceEmbeddings(
                    model_name=f"sentence-transformers/{model_name}",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                embedding.embed_query("验证")
                self.embedding_provider = "huggingface"
                print(f"  ✅ HuggingFace Embedding 加载成功")
                return embedding
            except Exception as e:
                print(f"  ⚠️ HuggingFace Embedding 加载失败: {str(e)}")

        has_api_key = bool(self.settings.openai_api_key or self.settings.rag_embedding_api_key)
        is_dashscope = "dashscope" in (self.settings.openai_base_url or "").lower()

        # 判断 Embedding 专属 API 配置（如 Ollama）
        embedding_base_url = self.settings.rag_embedding_base_url or self.settings.openai_base_url
        # Ollama 等本地服务不需要 API Key，但 OpenAI SDK 要求非空，用占位符
        embedding_api_key = self.settings.rag_embedding_api_key or self.settings.openai_api_key or "ollama-placeholder"

        # 如果用户配置 openai 或 tfidf，跳过 DashScope 原生 API
        # 否则如果是通义千问 API，先尝试原生 DashScope Embedding
        if provider not in ("openai", "tfidf") and is_dashscope and has_api_key:
            dashscope_models = ["text-embedding-v2", "text-embedding-v1"]
            for ds_model in dashscope_models:
                try:
                    print(f"  - 尝试 DashScope 原生 Embedding: {ds_model}")
                    embedding = DashScopeEmbedding(
                        model=ds_model,
                        api_key=self.settings.openai_api_key,
                        base_url="https://dashscope.aliyuncs.com/api/v1",
                    )
                    embedding.embed_query("验证")
                    self.embedding_provider = "dashscope"
                    print(f"  ✅ DashScope 原生 Embedding 加载成功：{ds_model}")
                    return embedding
                except Exception as e:
                    print(f"  ⚠️ DashScope 模型 {ds_model} 不可用：{str(e)[:100]}")
                    continue

        # 尝试 OpenAI 兼容 API
        if provider == "openai" or (has_api_key and provider == "huggingface"):
            # 如果有 RAG 专属的 Embedding URL/Key，使用用户指定的模型
            if self.settings.rag_embedding_base_url or provider == "openai":
                model_list = [model_name]
            else:
                # 否则尝试多个模型（兼容模式）
                model_list = [
                    model_name,
                    "text-embedding-v2",
                    "text-embedding-v1",
                    "text-embedding-ada-002",
                ]

            for model_to_try in model_list:
                if not model_to_try:
                    continue
                try:
                    from langchain_openai import OpenAIEmbeddings
                    print(f"  - 尝试 OpenAI API Embedding: {model_to_try} @ {embedding_base_url}")
                    embedding = OpenAIEmbeddings(
                        model=model_to_try,
                        api_key=embedding_api_key,
                        base_url=embedding_base_url,
                        tiktoken_enabled=False,
                        check_embedding_ctx_length=False,
                    )
                    embedding.embed_query("验证")
                    self.embedding_provider = "openai"
                    print(f"  ✅ OpenAI API Embedding 加载成功：{model_to_try}")
                    return embedding
                except Exception as e:
                    error_msg = str(e)
                    # 如果是 404 或模型不支持，跳过该模型
                    if "404" in error_msg or "not found" in error_msg.lower() or "unsupported" in error_msg.lower():
                        print(f"  ⚠️ 模型 {model_to_try} 不可用：{error_msg[:100]}")
                        continue
                    # 其他错误（如 400 参数错误）也跳过
                    print(f"  ⚠️ 模型 {model_to_try} 不可用：{error_msg[:100]}")
                    continue

        print(f"  - 使用 TF-IDF 关键词检索（纯本地，零依赖）")
        self.embedding_provider = "tfidf"
        return TfidfEmbedding()

    def _load_documents(self) -> List[Document]:
        """加载知识文档目录中的所有文档（含 memory 子目录）"""
        data_dir = Path(__file__).parent.parent / "data"
        documents = []

        md_files = glob.glob(str(data_dir / "*.md"))
        # 也加载 memory 子目录中的记忆文件
        memory_dir = data_dir / "memory"
        if memory_dir.exists():
            md_files += glob.glob(str(memory_dir / "*.md"))
        if not md_files:
            print("⚠️  未找到知识文档文件")
            return documents

        for file_path in md_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                filename = os.path.basename(file_path)
                doc = Document(
                    page_content=content,
                    metadata={"source": filename, "file_path": str(file_path)}
                )
                documents.append(doc)
                print(f"  - 加载文档: {filename}")
            except Exception as e:
                print(f"❌ 加载文档失败 {file_path}: {str(e)}")

        return documents

    def _load_memory_documents(self) -> List[Document]:
        """只加载 memory 子目录中的记忆文档"""
        memory_dir = Path(__file__).parent.parent / "data" / "memory"
        documents = []

        if not memory_dir.exists():
            print("⚠️  memory 目录不存在")
            return documents

        md_files = glob.glob(str(memory_dir / "*.md"))
        if not md_files:
            print("⚠️  未找到记忆文件")
            return documents

        for file_path in md_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                filename = os.path.basename(file_path)
                doc = Document(
                    page_content=content,
                    metadata={"source": filename, "file_path": str(file_path)}
                )
                documents.append(doc)
                print(f"  - 加载记忆: {filename}")
            except Exception as e:
                print(f"❌ 加载记忆失败 {file_path}: {str(e)}")

        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成更小的块，便于检索"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.settings.rag_chunk_size),
            chunk_overlap=int(self.settings.rag_chunk_overlap),
            separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", "。", "！", "？", ". ", "! ", "? "],
        )

        chunks = text_splitter.split_documents(documents)
        print(f"  - 文档分割完成: {len(documents)} 个文档 -> {len(chunks)} 个块")
        return chunks

    def _get_qdrant_client(self):
        """创建 QdrantClient 实例"""
        mode = self.settings.qdrant_mode
        if mode == "cloud":
            return QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )
        elif mode == "local":
            qdrant_path = self.settings.qdrant_path
            os.makedirs(qdrant_path, exist_ok=True)
            return QdrantClient(path=qdrant_path)
        else:  # memory 模式
            return QdrantClient(location=":memory:")

    def _get_qdrant_connection_kwargs(self) -> dict:
        """获取 Qdrant 连接参数字典（用于 from_documents 透传）"""
        mode = self.settings.qdrant_mode
        if mode == "cloud":
            return {
                "url": self.settings.qdrant_url,
                "api_key": self.settings.qdrant_api_key,
            }
        elif mode == "local":
            qdrant_path = self.settings.qdrant_path
            os.makedirs(qdrant_path, exist_ok=True)
            return {"path": qdrant_path}
        else:  # memory 模式
            return {"location": ":memory:"}

    def _collection_exists(self, connection_kwargs: dict, collection_name: str) -> bool:
        """检查 Qdrant 集合是否已存在"""
        client = QdrantClient(**connection_kwargs)
        existing = client.get_collections().collections
        return collection_name in [c.name for c in existing]

    def initialize(self) -> bool:
        """初始化向量存储

        加载文档 -> 分割 -> 嵌入 -> 构建 Qdrant 索引（或连接已有索引）
        """
        try:
            print("🔄 初始化 RAG 向量存储 (Qdrant)...")

            documents = self._load_documents()
            if not documents:
                print("⚠️  RAG 跳过: 无知识文档可加载")
                return False

            chunks = self._split_documents(documents)

            if self.embedding_model is None:
                self.embedding_model = self._init_embeddings()

            connection_kwargs = self._get_qdrant_connection_kwargs()
            collection_name = self.settings.qdrant_collection

            # 检查集合是否已存在
            if self._collection_exists(connection_kwargs, collection_name):
                print(f"  - 连接已有 Qdrant 集合: {collection_name}")
                try:
                    # QdrantVectorStore.__init__() 需要 client 实例
                    client = self._get_qdrant_client()
                    self.vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=collection_name,
                        embedding=self.embedding_model,
                    )
                except Exception as conn_err:
                    # 如果连接失败（如向量维度不匹配），删除集合并重建
                    print(f"  ⚠️ 连接已有集合失败: {conn_err}")
                    print(f"  - 删除并重建集合: {collection_name}")
                    client = self._get_qdrant_client()
                    client.delete_collection(collection_name)
                    self.vector_store = QdrantVectorStore.from_documents(
                        documents=chunks,
                        embedding=self.embedding_model,
                        collection_name=collection_name,
                        **connection_kwargs,
                    )
            else:
                print(f"  - 新建 Qdrant 集合: {collection_name}")
                print(f"  - 生成向量索引...")
                # from_documents() 接受原始的连接参数（url/path/location）
                self.vector_store = QdrantVectorStore.from_documents(
                    documents=chunks,
                    embedding=self.embedding_model,
                    collection_name=collection_name,
                    **connection_kwargs,
                )

            self._initialized = True
            provider_name = self.embedding_provider or "unknown"
            mode_name = self.settings.qdrant_mode
            print(f"✅ RAG 向量存储初始化成功 (集合: {collection_name}, 模式: {mode_name}, Embedding: {provider_name})")
            return True

        except Exception as e:
            print(f"❌ RAG 向量存储初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """语义检索相关知识

        Args:
            query: 查询文本（如城市名、偏好等）
            k: 返回的文档块数量

        Returns:
            相关文档块列表
        """
        if not self._initialized or self.vector_store is None:
            print("⚠️  RAG 未初始化，尝试初始化...")
            success = self.initialize()
            if not success:
                return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ RAG 检索失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_as_context(self, query: str, k: int = 5) -> str:
        """检索相关知识并格式化为上下文文本

        Args:
            query: 查询文本
            k: 返回的文档块数量

        Returns:
            格式化后的上下文文本，可直接注入提示词
        """
        docs = self.retrieve(query, k=k)
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"[知识 {i+1}] (来源: {source})\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def add_documents(self, documents: List[Document]) -> int:
        """增量添加文档到已有的 Qdrant 向量存储

        先分割文档，再添加到已有集合中，无需重建索引。

        Args:
            documents: 待添加的文档列表

        Returns:
            成功添加的文档块数量
        """
        if not documents:
            return 0

        # 确保已初始化
        if not self._initialized or self.vector_store is None:
            success = self.initialize()
            if not success:
                print("❌ RAG 未初始化，无法添加文档")
                return 0

        try:
            chunks = self._split_documents(documents)
            # QdrantVectorStore.add_documents() 会直接嵌入并插入到已有集合
            self.vector_store.add_documents(chunks)
            print(f"✅ 已添加 {len(chunks)} 个文档块到 Qdrant 集合")
            return len(chunks)
        except Exception as e:
            print(f"❌ 添加文档到 Qdrant 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0

    def sync_memories_to_qdrant(self) -> int:
        """将 memory 目录中的所有记忆文档同步到 Qdrant

        读取 data/memory/ 下的所有 .md 文件，分割后添加到 Qdrant 向量存储。

        Returns:
            成功同步的文档块数量
        """
        print("🔄 同步记忆到 Qdrant...")

        documents = self._load_memory_documents()
        if not documents:
            print("⚠️  没有记忆文件需要同步")
            return 0

        return self.add_documents(documents)

    def sync_all_to_qdrant(self) -> int:
        """重新构建整个 Qdrant 索引（全量同步）

        重新加载 data/ 下所有文档（含 memory），并重建 Qdrant 集合。
        注意：这会清空原有集合并重建。

        Returns:
            索引的文档块数量
        """
        print("🔄 全量重建 Qdrant 索引...")
        self._initialized = False
        self.vector_store = None
        success = self.initialize()
        if success:
            print("✅ 全量同步完成")
            return len(self.vector_store.client.count(self.settings.qdrant_collection).count)
        return 0

    # ============ BM25 检索 ============

    def _build_bm25_index(self):
        """构建 BM25 关键词索引

        读取 data/ 和 data/memory/ 下所有文档，创建 BM25 倒排索引。
        """
        from rank_bm25 import BM25Okapi

        documents = self._load_documents()
        if not documents:
            print("⚠️  BM25 无文档可索引")
            return False

        chunks = self._split_documents(documents)
        self._bm25_chunks = chunks

        # 分词器：中文按字符、英文按单词
        def _tokenize(text: str) -> List[str]:
            # 中英文混合分词：按非字母数字字符切分，保留中文单字和英文单词
            text_lower = text.lower()
            # 提取英文单词
            words = re.findall(r'[a-z]+', text_lower)
            # 提取中文字符（单字）
            chars = re.findall(r'[\u4e00-\u9fff]', text_lower)
            return words + chars

        self._bm25_tokenizer = _tokenize
        tokenized_corpus = [_tokenize(doc.page_content) for doc in chunks]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        print(f"  ✅ BM25 索引构建完成: {len(chunks)} 个文档块")
        return True

    def retrieve_bm25(self, query: str, k: int = 5) -> List[Document]:
        """BM25 关键词检索

        Args:
            query: 查询文本
            k: 返回的文档块数量

        Returns:
            相关文档块列表
        """
        if self._bm25_index is None:
            self._build_bm25_index()

        if self._bm25_index is None:
            return []

        tokenized_query = self._bm25_tokenizer(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._bm25_chunks[idx]
                results.append(doc)

        return results

    def retrieve_hybrid(self, query: str, k: int = 5,
                        vector_k: int = 10, bm25_k: int = 10,
                        rrf_constant: int = 60) -> List[Document]:
        """混合检索：BM25 + 向量检索，使用 RRF 融合排序

        RRF (Reciprocal Rank Fusion) 公式:
          score(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))

        Args:
            query: 查询文本
            k: 最终返回的文档块数量
            vector_k: 向量检索取 top-N
            bm25_k: BM25 检索取 top-N
            rrf_constant: RRF 常数 k（默认 60）

        Returns:
            混合排序后的文档块列表
        """
        # 1. 向量检索
        vector_results = self.retrieve(query, k=vector_k)

        # 2. BM25 检索
        bm25_results = self.retrieve_bm25(query, k=bm25_k)

        # 3. RRF 融合
        rrf_scores = {}

        for rank, doc in enumerate(vector_results):
            doc_id = doc.metadata.get("source", "") + doc.page_content[:50]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "doc": doc,
                    "score": 0.0,
                }
            rrf_scores[doc_id]["score"] += 1.0 / (rrf_constant + rank + 1)

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("source", "") + doc.page_content[:50]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "doc": doc,
                    "score": 0.0,
                }
            rrf_scores[doc_id]["score"] += 1.0 / (rrf_constant + rank + 1)

        # 4. 按 RRF 分数排序
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        # 5. 去重：相同 source 只保留最高分
        seen_sources = set()
        deduped = []
        for item in sorted_docs:
            source = item["doc"].metadata.get("source", "")
            if source not in seen_sources:
                seen_sources.add(source)
                deduped.append(item["doc"])

        return deduped[:k]

    def retrieve_hybrid_as_context(self, query: str, k: int = 5) -> str:
        """混合检索并格式化为上下文文本

        Args:
            query: 查询文本
            k: 返回的文档块数量

        Returns:
            格式化后的上下文文本
        """
        docs = self.retrieve_hybrid(query, k=k)
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"[知识 {i+1}] (来源: {source})\n{doc.page_content}")

        return "\n\n".join(context_parts)


# 全局 RAG 服务实例
_rag_service: Optional[RAGService] = None


def get_rag_service(force_reinit: bool = False) -> RAGService:
    """获取 RAG 服务实例（单例模式）

    Args:
        force_reinit: 是否强制重新初始化

    Returns:
        RAGService 实例
    """
    global _rag_service

    if _rag_service is None or force_reinit:
        _rag_service = RAGService()
        _rag_service.initialize()

    return _rag_service
