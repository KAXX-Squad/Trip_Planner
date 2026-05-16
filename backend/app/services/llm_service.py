"""LLM服务模块 - 支持多API轮询/故障转移"""

import time
from typing import List, Dict, Iterator, Optional
from hello_agents import HelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException
from ..config import get_settings


class LLMFailoverClient(HelloAgentsLLM):
    """
    支持故障转移的LLM客户端
    
    继承自HelloAgentsLLM，当主API超时或失败时，自动切换到备用API继续尝试。
    
    使用方式与HelloAgentsLLM完全兼容：
        llm = LLMFailoverClient()
        response = llm.invoke([{"role": "user", "content": "Hello"}])
        # 或流式调用
        for chunk in llm.think([{"role": "user", "content": "Hello"}]):
            print(chunk)
    """
    
    def __init__(self):
        """初始化故障转移客户端"""
        # 不调用父类的__init__，因为我们需要管理多个LLM实例
        self.settings = get_settings()
        self.llm_instances: List[HelloAgentsLLM] = []
        self.llm_names: List[str] = []
        self.current_index = 0
        self._initialize_clients()
        
        # 设置必要的属性以保持接口兼容
        self.temperature = 0.7
        self.max_tokens = None
        self.timeout = self.settings.llm_timeout
        self.provider = "failover"
    
    def _initialize_clients(self):
        """初始化所有可用的LLM客户端"""
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
        
        # 筛选有效的配置并创建HelloAgentsLLM实例
        for config in configs:
            if config['api_key'] and config['base_url'] and config['model']:
                try:
                    llm = HelloAgentsLLM(
                        model=config['model'],
                        api_key=config['api_key'],
                        base_url=config['base_url'],
                        timeout=config['timeout']
                    )
                    self.llm_instances.append(llm)
                    self.llm_names.append(config['name'])
                    print(f"✅ 已配置LLM: {config['name']} ({config['model']})")
                except Exception as e:
                    print(f"❌ 初始化LLM失败 {config['name']}: {str(e)}")
        
        if not self.llm_instances:
            raise RuntimeError("未找到任何有效的LLM配置")
        
        # 设置当前model属性（使用第一个LLM的model）
        self.model = self.llm_instances[0].model
    
    def _get_next_llm(self) -> HelloAgentsLLM:
        """获取下一个LLM实例（轮询）"""
        if self.current_index >= len(self.llm_instances):
            self.current_index = 0
        
        llm = self.llm_instances[self.current_index]
        name = self.llm_names[self.current_index]
        self.current_index += 1
        
        return llm, name
    
    def _reset_index(self):
        """重置轮询索引"""
        self.current_index = 0
    
    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用LLM进行思考，返回流式响应，支持故障转移。
        
        Args:
            messages: 消息列表
            temperature: 温度参数
        
        Yields:
            str: 流式响应的文本片段
        
        Raises:
            HelloAgentsException: 所有LLM都失败时抛出
        """
        errors: List[str] = []
        
        # 尝试所有可用的LLM
        for attempt in range(len(self.llm_instances)):
            llm, name = self._get_next_llm()
            
            try:
                print(f"🔄 正在尝试 {name} (第{attempt + 1}/{len(self.llm_instances)}次尝试)")
                
                # 调用LLM的think方法（流式）
                collected_content = []
                for chunk in llm.think(messages, temperature):
                    collected_content.append(chunk)
                    yield chunk
                
                print(f"✅ {name} 调用成功")
                self._reset_index()  # 成功后重置索引
                # 更新当前model属性
                self.model = llm.model
                return
                
            except Exception as e:
                error_msg = f"❌ {name} 调用失败: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                
                # 重试延迟
                if attempt < len(self.llm_instances) - 1:
                    delay = self.settings.llm_retry_delay * (attempt + 1)
                    print(f"⏳ 等待 {delay} 秒后尝试下一个LLM...")
                    time.sleep(delay)
        
        # 所有LLM都失败
        self._reset_index()
        error_summary = "\n".join(errors)
        raise HelloAgentsException(f"所有LLM服务均不可用:\n{error_summary}")
    
    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应，支持故障转移。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数（temperature, max_tokens等）
        
        Returns:
            str: 完整响应文本
        
        Raises:
            HelloAgentsException: 所有LLM都失败时抛出
        """
        errors: List[str] = []
        
        # 尝试所有可用的LLM
        for attempt in range(len(self.llm_instances)):
            llm, name = self._get_next_llm()
            
            try:
                print(f"🔄 正在尝试 {name} (第{attempt + 1}/{len(self.llm_instances)}次尝试)")
                
                # 调用LLM的invoke方法（非流式）
                response = llm.invoke(messages, **kwargs)
                
                print(f"✅ {name} 调用成功")
                self._reset_index()  # 成功后重置索引
                # 更新当前model属性
                self.model = llm.model
                return response
                
            except Exception as e:
                error_msg = f"❌ {name} 调用失败: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                
                # 重试延迟
                if attempt < len(self.llm_instances) - 1:
                    delay = self.settings.llm_retry_delay * (attempt + 1)
                    print(f"⏳ 等待 {delay} 秒后尝试下一个LLM...")
                    time.sleep(delay)
        
        # 所有LLM都失败
        self._reset_index()
        error_summary = "\n".join(errors)
        raise HelloAgentsException(f"所有LLM服务均不可用:\n{error_summary}")
    
    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式调用LLM的别名方法，与think方法功能相同。
        保持向后兼容性。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
        
        Yields:
            str: 流式响应的文本片段
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
    
    def chat(self, prompt: str, **kwargs) -> str:
        """
        简单的聊天接口（兼容旧版API）
        
        Args:
            prompt: 输入提示
            **kwargs: 额外参数
        
        Returns:
            LLM响应文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.invoke(messages, **kwargs)


# 全局故障转移LLM实例
_llm_failover_instance = None


def get_llm() -> LLMFailoverClient:
    """
    获取支持故障转移的LLM实例(单例模式)
    
    Returns:
        LLMFailoverClient实例（继承自HelloAgentsLLM）
    """
    global _llm_failover_instance
    
    if _llm_failover_instance is None:
        print("🔄 初始化故障转移LLM服务...")
        _llm_failover_instance = LLMFailoverClient()
        print(f"✅ LLM故障转移服务初始化成功")
        print(f"   可用LLM数量: {len(_llm_failover_instance.llm_instances)}")
        for i, name in enumerate(_llm_failover_instance.llm_names):
            print(f"   LLM {i+1}: {name}")
    
    return _llm_failover_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_failover_instance
    _llm_failover_instance = None
