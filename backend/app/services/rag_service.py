"""RAG 检索增强生成服务 - 基于向量检索的旅行知识增强

支持三种 Embedding 方式（按优先级）:
1. HuggingFace 本地模型（默认，无需 API Key，需下载模型）
2. OpenAI 兼容 API（复用 LLM 配置，需 API 支持 Embedding）
3. TF-IDF 关键词检索（纯本地，零依赖，始终可用）
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..config import get_settings


class TfidfEmbedding:
    """基于 TF-IDF 的简单 Embedding（纯本地实现，无需下载任何模型）

    兼容 LangChain Embeddings 接口：embed_documents, embed_query
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
        embeddings = self.vectorizer.fit_transform(texts).toarray().tolist()
        self._fitted = True
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            return [0.0] * self.vectorizer.max_features
        embedding = self.vectorizer.transform([text]).toarray().tolist()[0]
        return embedding

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


class DashScopeEmbedding:
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
    """基于 FAISS 向量数据库的 RAG 检索服务

    加载旅行知识文档，构建向量索引，并提供语义检索能力。
    检索到的知识将被注入到 LLM 提示词中，增强生成质量。
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = None
        self.embedding_provider = None
        self.vector_store: Optional[FAISS] = None
        self._initialized = False

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

        has_api_key = bool(self.settings.openai_api_key)
        is_dashscope = "dashscope" in (self.settings.openai_base_url or "").lower()
        
        # 如果是通义千问 API，先尝试原生 DashScope Embedding
        if is_dashscope and has_api_key:
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
            # 通义千问等 OpenAI 兼容 API 的 Embedding 模型列表
            embedding_models_to_try = [
                model_name,  # 用户配置的模型
                "text-embedding-v2",  # 通义千问常用 Embedding
                "text-embedding-v1",  # 通义千问旧版
                "text-embedding-ada-002",  # OpenAI 官方
            ]
            
            for model_to_try in embedding_models_to_try:
                if not model_to_try:
                    continue
                try:
                    from langchain_openai import OpenAIEmbeddings
                    print(f"  - 尝试 OpenAI API Embedding: {model_to_try}")
                    embedding = OpenAIEmbeddings(
                        model=model_to_try,
                        api_key=self.settings.openai_api_key,
                        base_url=self.settings.openai_base_url,
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
        """加载知识文档目录中的所有文档"""
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

    def initialize(self) -> bool:
        """初始化向量存储

        加载文档 -> 分割 -> 嵌入 -> 构建 FAISS 索引
        """
        try:
            print("🔄 初始化 RAG 向量存储...")

            documents = self._load_documents()
            if not documents:
                print("⚠️  RAG 跳过: 无知识文档可加载")
                return False

            chunks = self._split_documents(documents)

            if self.embedding_model is None:
                self.embedding_model = self._init_embeddings()

            print(f"  - 生成向量索引...")
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
            )

            self._initialized = True
            provider_name = self.embedding_provider or "unknown"
            print(f"✅ RAG 向量存储初始化成功 (共 {len(chunks)} 个文档块, Embedding: {provider_name})")
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


# 全局 RAG 服务实例
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务实例（单例模式）"""
    global _rag_service

    if _rag_service is None:
        _rag_service = RAGService()
        _rag_service.initialize()

    return _rag_service
