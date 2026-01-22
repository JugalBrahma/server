"""
Agent Module
- LangGraph-based RAG agent
- Smart decision making for greetings vs document queries
- Tool integration for document retrieval
"""

from .agent import RAGAgent, AgentConfig
from .llm_manager import LLMManager
from .vector_manager import VectorSearchManager

__version__ = "1.0.0"
__all__ = [
    "RAGAgent",
    "AgentConfig",
    "LLMManager", 
    "VectorSearchManager"
]
