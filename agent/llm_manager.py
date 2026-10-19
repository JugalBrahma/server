from typing import Optional
from config.models import get_rag_llm


class LLMManager:
    """Manages LLM instances using the configured provider (OpenAI / Groq)"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        kwargs = {}
        if model_name:
            kwargs["model"] = model_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        if provider:
            kwargs["provider"] = provider
        self.llm = get_rag_llm(**kwargs)

    
    def get_llm(self):
        return self.llm
