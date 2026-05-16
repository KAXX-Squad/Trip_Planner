"""LLM 服务模块 - 基于 LangChain 实现，支持多 API 轮询/故障转移"""

import time
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from ..config import get_settings


class LLMFailoverClient:
    """
    支持故障转移的 LLM 客户端
    
    当主 API 超时或失败时，自动切换到备用 API 继续尝试。
    """
    
    def __init__(self):
        """初始化故障转移客户端"""
        self.settings = get_settings()
        self.llm_instances: List[BaseChatModel] = []
        self.llm_names: List[str] = []
        self.current_index = 0
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化所有可用的 LLM 客户端"""
        configs = [
            {
                'name': 'Primary LLM',
                'api_key': self.settings.openai_api_key,
                'base_url': self.settings.openai_base_url,
                'model': self.settings.openai_model,
                'timeout': self.settings.llm_timeout
            },
            {
                'name': 'Backup LLM 1',
                'api_key': self.settings.llm_api_key_2,
                'base_url': self.settings.llm_base_url_2,
                'model': self.settings.llm_model_2,
                'timeout': self.settings.llm_timeout
            },
            {
                'name': 'Backup LLM 2',
                'api_key': self.settings.llm_api_key_3,
                'base_url': self.settings.llm_base_url_3,
                'model': self.settings.llm_model_3,
                'timeout': self.settings.llm_timeout
            }
        ]
        
        # 筛选有效的配置并创建 ChatOpenAI 实例
        for config in configs:
            if config['api_key'] and config['base_url'] and config['model']:
                try:
                    llm = ChatOpenAI(
                        model=config['model'],
                        api_key=config['api_key'],
                        base_url=config['base_url'],
                        timeout=config['timeout'],
                        temperature=0.7
                    )
                    self.llm_instances.append(llm)
                    self.llm_names.append(config['name'])
                    print(f"✅ 已配置 LLM: {config['name']} ({config['model']})")
                except Exception as e:
                    print(f"❌ 初始化 LLM 失败 {config['name']}: {str(e)}")
        
        if not self.llm_instances:
            raise RuntimeError("未找到任何有效的 LLM 配置")
    
    def _get_next_llm(self) -> BaseChatModel:
        """获取下一个 LLM 实例（轮询）"""
        if self.current_index >= len(self.llm_instances):
            self.current_index = 0
        
        llm = self.llm_instances[self.current_index]
        name = self.llm_names[self.current_index]
        self.current_index += 1
        
        return llm, name
    
    def _reset_index(self):
        """重置轮询索引"""
        self.current_index = 0
    
    def invoke(self, messages: list, **kwargs) -> str:
        """
        非流式调用 LLM，返回完整响应，支持故障转移。
        
        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "Hello"}]
            **kwargs: 额外参数
        
        Returns:
            str: 完整响应文本
        
        Raises:
            Exception: 所有 LLM 都失败时抛出
        """
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        errors: List[str] = []
        
        # 转换为 LangChain 消息格式
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        
        # 尝试所有可用的 LLM
        for attempt in range(len(self.llm_instances)):
            llm, name = self._get_next_llm()
            
            try:
                print(f"🔄 正在尝试 {name} (第{attempt + 1}/{len(self.llm_instances)}次尝试)")
                
                # 调用 LLM
                response = llm.invoke(langchain_messages, **kwargs)
                
                print(f"✅ {name} 调用成功")
                self._reset_index()
                return response.content
                
            except Exception as e:
                error_msg = f"❌ {name} 调用失败：{str(e)}"
                print(error_msg)
                errors.append(error_msg)
                
                # 重试延迟
                if attempt < len(self.llm_instances) - 1:
                    delay = self.settings.llm_retry_delay * (attempt + 1)
                    print(f"⏳ 等待 {delay} 秒后尝试下一个 LLM...")
                    time.sleep(delay)
        
        # 所有 LLM 都失败
        self._reset_index()
        error_summary = "\n".join(errors)
        raise Exception(f"所有 LLM 服务均不可用:\n{error_summary}")


def get_llm() -> LLMFailoverClient:
    """获取 LLM 实例"""
    return LLMFailoverClient()
