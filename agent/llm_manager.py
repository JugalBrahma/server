from typing import Optional
from config.models import get_llm, DEFAULT_MODEL


class LLMManager:
    """Manages LLM instances using the configured provider (Groq / OpenAI)"""
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        self.model_name = model_name or DEFAULT_MODEL
        self.temperature = temperature
        self.llm = get_llm(model_name=self.model_name, temperature=self.temperature)
    
    def get_llm(self):
        return self.llm