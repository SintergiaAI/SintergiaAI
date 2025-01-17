from typing import Optional, Dict, Literal
from datetime import datetime
import weakref
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from app.controllers.logger_controller import logger

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"

class MultiProviderLLMManager:
    _system_prompt = """
        """
    # Diccionario de instancias por proveedor
    _instances: Dict[LLMProvider, Dict[str, 'weakref.ref']] = {
        provider: {} for provider in LLMProvider
    }
    
    _last_used: Dict[LLMProvider, Dict[str, datetime]] = {
        provider: {} for provider in LLMProvider
    }
    
    @classmethod
    def create_llm(cls, provider: LLMProvider, api_key: str, **kwargs) -> any:
        """
        Factory method para crear instancias de diferentes proveedores
        """
        if provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                api_key=api_key,
                model=kwargs.get('model', 'gpt-4o-mini'),
                temperature=kwargs.get('temperature', 0),
                streaming=True
            )
            
        elif provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                api_key=api_key,
                model=kwargs.get('model', 'claude-3-opus-20240229'),
                temperature=kwargs.get('temperature', 0),
                streaming=True
            )
            
        elif provider == LLMProvider.GROQ:
            return ChatGroq(
                api_key=api_key,
                model=kwargs.get('model', 'mixtral-8x7b-32768'),
                temperature=kwargs.get('temperature', 0),
                streaming=True
            )
            
        raise ValueError(f"Proveedor no soportado: {provider}")

    def __init__(self, provider: LLMProvider, api_key: str, **kwargs):
        """
        Initialize a new LLM instance for a specific provider
        """
        self._llm = self.create_llm(provider, api_key, **kwargs)
        self.provider = provider
    
    @classmethod
    def get_instance(cls, 
                    provider: LLMProvider,
                    instance_name: str, 
                    api_key: Optional[str] = None,
                    **kwargs) -> 'MultiProviderLLMManager':
        """
        Get or create an LLM instance for a specific provider
        """
        provider_instances = cls._instances[provider]
        
        # Verificar si la instancia existe y está activa
        if instance_name in provider_instances:
            instance = provider_instances[instance_name]()
            if instance is not None:
                cls._last_used[provider][instance_name] = datetime.now()
                return instance
        
        # Crear nueva instancia si no existe o está inactiva
        if api_key is None:
            raise ValueError(f"API key is required to create new instance '{instance_name}' for {provider.value}")
        
        new_instance = cls(provider, api_key, **kwargs)
        provider_instances[instance_name] = weakref.ref(new_instance)
        cls._last_used[provider][instance_name] = datetime.now()
        
        logger.info(f"Created new {provider.value} LLM instance: {instance_name}")
        return new_instance
    
    @classmethod
    def remove_instance(cls, provider: LLMProvider, instance_name: str) -> None:
        """
        Remove a specific instance of a provider
        """
        if instance_name in cls._instances[provider]:
            del cls._instances[provider][instance_name]
            cls._last_used[provider].pop(instance_name, None)
            logger.info(f"Removed {provider.value} LLM instance: {instance_name}")
    
    @property
    def llm(self):
        return self._llm
    
    @classmethod
    def get_active_instances(cls) -> Dict[str, list]:
        """
        Get information about all active instances across all providers
        """
        active_instances = {}
        for provider in LLMProvider:
            active_instances[provider.value] = list(cls._instances[provider].keys())
        return active_instances
    