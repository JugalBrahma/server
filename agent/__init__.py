"""Public classes for the document question-answering agent."""

from .agent import RAGAgent, AgentConfig
from .llm_manager import LLMManager
from .vector_manager import VectorSearchManager

__version__ = "1.0.0"
__all__ = [
    "RAGAgent",
    "AgentConfig",
    "LLMManager",
    "VectorSearchManager",
]
