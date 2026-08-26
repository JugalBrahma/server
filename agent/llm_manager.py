from typing import Optional
from config.models import get_rag_llm, get_groq_llm, get_openai_llm


class LLMManager:
    """Manages LLM instances using the configured provider (Groq / OpenAI)"""
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        self.model_name = model_name
        self.temperature = temperature
        kwargs = {}
        if model_name:
            kwargs["model"] = model_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        self.llm = get_rag_llm(**kwargs)

    
    def get_llm(self):
        return self.llm