"""配置管理模块"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载环境变量
# 加载项目根目录的.env 文件
root_dir = Path(__file__).parent.parent.parent
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()  # 回退到当前目录


class Settings(BaseSettings):
    """应用配置"""

    # 应用基本配置
    app_name: str = "HelloAgents智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS配置 - 使用字符串,在代码中分割
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # 高德地图API配置
    amap_api_key: str = ""

    # Unsplash API配置
    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    # 日志配置
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量

    def get_cors_origins_list(self) -> List[str]:
        """获取CORS origins列表"""
        return [origin.strip() for origin in self.cors_origins.split(',')]

    # ============ LLM配置属性 (使用os.getenv读取环境变量) ============
    
    @property
    def openai_api_key(self) -> str:
        """获取主LLM API密钥"""
        return os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    
    @property
    def openai_base_url(self) -> str:
        """获取主LLM API地址"""
        return os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    
    @property
    def openai_model(self) -> str:
        """获取主LLM模型名称"""
        return os.getenv("LLM_MODEL_ID") or os.getenv("OPENAI_MODEL") or "gpt-4"
    
    @property
    def llm_api_key_2(self) -> str:
        """获取备用LLM 1 API密钥"""
        return os.getenv("LLM_API_KEY_2") or ""
    
    @property
    def llm_base_url_2(self) -> str:
        """获取备用LLM 1 API地址"""
        return os.getenv("LLM_BASE_URL_2") or ""
    
    @property
    def llm_model_2(self) -> str:
        """获取备用LLM 1模型名称"""
        return os.getenv("LLM_MODEL_2") or ""
    
    @property
    def llm_api_key_3(self) -> str:
        """获取备用LLM 2 API密钥"""
        return os.getenv("LLM_API_KEY_3") or ""
    
    @property
    def llm_base_url_3(self) -> str:
        """获取备用LLM 2 API地址"""
        return os.getenv("LLM_BASE_URL_3") or ""
    
    @property
    def llm_model_3(self) -> str:
        """获取备用LLM 2模型名称"""
        return os.getenv("LLM_MODEL_3") or ""
    
    @property
    def llm_timeout(self) -> int:
        """获取LLM请求超时时间(秒)"""
        return int(os.getenv("LLM_TIMEOUT") or "60")
    
    @property
    def llm_max_retries(self) -> int:
        """获取LLM最大重试次数"""
        return int(os.getenv("LLM_MAX_RETRIES") or "3")
    
    @property
    def llm_retry_delay(self) -> int:
        """获取LLM重试延迟(秒)"""
        return int(os.getenv("LLM_RETRY_DELAY") or "2")

    # ============ RAG配置属性 ============

    @property
    def rag_embedding_model(self) -> str:
        """获取RAG Embedding模型名称

        默认使用本地 HuggingFace 模型，无需 API Key。
        可选值: all-MiniLM-L6-v2, text-embedding-v3, text-embedding-ada-002
        """
        return os.getenv("RAG_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"

    @property
    def rag_embedding_provider(self) -> str:
        """获取RAG Embedding提供者: huggingface / openai"""
        return os.getenv("RAG_EMBEDDING_PROVIDER") or "huggingface"

    @property
    def rag_chunk_size(self) -> int:
        """获取RAG文档块大小"""
        return int(os.getenv("RAG_CHUNK_SIZE") or "500")

    @property
    def rag_chunk_overlap(self) -> int:
        """获取RAG文档块重叠大小"""
        return int(os.getenv("RAG_CHUNK_OVERLAP") or "50")

    @property
    def rag_retrieval_k(self) -> int:
        """获取RAG检索返回文档数"""
        return int(os.getenv("RAG_RETRIEVAL_K") or "5")

    @property
    def rag_enabled(self) -> bool:
        """获取RAG是否启用"""
        return os.getenv("RAG_ENABLED", "true").lower() == "true"

    # ============ Qdrant 向量数据库配置 ============

    @property
    def qdrant_mode(self) -> str:
        """获取Qdrant运行模式: local / memory / cloud"""
        return (os.getenv("QDRANT_MODE") or "local").lower()

    @property
    def qdrant_path(self) -> str:
        """获取Qdrant本地持久化路径"""
        return os.getenv("QDRANT_PATH") or "./qdrant_data"

    @property
    def qdrant_collection(self) -> str:
        """获取Qdrant集合名称"""
        return os.getenv("QDRANT_COLLECTION") or "trip_planner"

    @property
    def qdrant_url(self) -> str:
        """获取Qdrant远程服务器地址"""
        return os.getenv("QDRANT_URL") or ""

    @property
    def qdrant_api_key(self) -> str:
        """获取Qdrant远程API Key"""
        return os.getenv("QDRANT_API_KEY") or ""

    # ============ Neo4j 知识图谱配置 ============

    @property
    def neo4j_uri(self) -> str:
        return os.getenv("NEO4J_URI") or ""

    @property
    def neo4j_username(self) -> str:
        return os.getenv("NEO4J_USERNAME") or ""

    @property
    def neo4j_password(self) -> str:
        return os.getenv("NEO4J_PASSWORD") or ""

    @property
    def neo4j_database(self) -> str:
        return os.getenv("NEO4J_DATABASE") or ""

    @property
    def kg_enabled(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_username and self.neo4j_password)


# 创建全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 验证必要的配置
def validate_config():
    """验证配置是否完整"""
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY未配置")

    # 检查LLM配置 - 至少需要一个有效的LLM配置
    llm_configs_valid = 0
    
    # 检查主LLM配置
    if settings.openai_api_key and settings.openai_base_url and settings.openai_model:
        llm_configs_valid += 1
    
    # 检查备用LLM 1配置
    if settings.llm_api_key_2 and settings.llm_base_url_2 and settings.llm_model_2:
        llm_configs_valid += 1
    
    # 检查备用LLM 2配置
    if settings.llm_api_key_3 and settings.llm_base_url_3 and settings.llm_model_3:
        llm_configs_valid += 1

    if llm_configs_valid == 0:
        errors.append("未配置任何有效的LLM服务(需要配置OPENAI_API_KEY或LLM_API_KEY_2/3)")
    elif llm_configs_valid == 1:
        warnings.append(f"仅配置了{llm_configs_valid}个LLM服务,建议配置多个LLM实现故障转移以提高可用性")
    else:
        print(f"✅ 已配置{llm_configs_valid}个LLM服务,支持故障转移")

    if errors:
        error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")

    return True


# 打印配置信息(用于调试)
def print_config():
    """打印当前配置(隐藏敏感信息)"""
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")

    # 检查LLM配置 - 显示所有配置的LLM服务
    print("\nLLM配置(支持故障转移):")
    
    # 主LLM
    print("  主LLM:")
    print(f"    API Key: {'已配置' if settings.openai_api_key else '未配置'}")
    print(f"    Base URL: {settings.openai_base_url}")
    print(f"    Model: {settings.openai_model}")
    
    # 备用LLM 1
    if settings.llm_api_key_2 or settings.llm_base_url_2:
        print("  备用LLM 1:")
        print(f"    API Key: {'已配置' if settings.llm_api_key_2 else '未配置'}")
        print(f"    Base URL: {settings.llm_base_url_2 or '未配置'}")
        print(f"    Model: {settings.llm_model_2 or '未配置'}")
    
    # 备用LLM 2
    if settings.llm_api_key_3 or settings.llm_base_url_3:
        print("  备用LLM 2:")
        print(f"    API Key: {'已配置' if settings.llm_api_key_3 else '未配置'}")
        print(f"    Base URL: {settings.llm_base_url_3 or '未配置'}")
        print(f"    Model: {settings.llm_model_3 or '未配置'}")
    
    # LLM超时配置
    print(f"\nLLM超时配置:")
    print(f"  超时时间: {settings.llm_timeout}秒")
    print(f"  最大重试次数: {settings.llm_max_retries}")
    print(f"  重试延迟: {settings.llm_retry_delay}秒")
    
    
    # RAG配置
    print(f"\nRAG配置:")
    print(f"  RAG启用: {'是' if settings.rag_enabled else '否'}")
    if settings.rag_enabled:
        print(f"  Embedding提供者: {settings.rag_embedding_provider}")
        print(f"  Embedding模型: {settings.rag_embedding_model}")
        print(f"  文档块大小: {settings.rag_chunk_size}")
        print(f"  文档块重叠: {settings.rag_chunk_overlap}")
        print(f"  检索数量: {settings.rag_retrieval_k}")
        print(f"  向量数据库: Qdrant ({settings.qdrant_mode})")
        if settings.qdrant_mode == "local":
            print(f"  持久化路径: {settings.qdrant_path}")
    
    print(f"\n日志级别: {settings.log_level}")

