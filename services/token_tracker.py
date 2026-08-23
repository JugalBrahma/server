"""
Token Usage & Metadata Tracking for LangSmith / LangChain
Accumulates prompt, completion, and per-node token metrics.
"""

from typing import Any, Dict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class LangSmithTokenTracker(BaseCallbackHandler):
    """Callback handler accumulating token metrics for LangSmith and API responses."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.node_breakdown: Dict[str, int] = {}
        self.current_node: str = "rag_agent"

    def set_current_node(self, node_name: str) -> None:
        if node_name:
            self.current_node = str(node_name).strip().lower()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract exact token usage from LLM response metadata."""
        p_tokens, c_tokens, t_tokens = 0, 0, 0

        # Standard token usage metadata from Gemini / Groq / OpenAI / LangSmith
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage", {})
            if isinstance(usage, dict):
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                t_tokens = usage.get("total_tokens", p_tokens + c_tokens)

        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        self.total_tokens += t_tokens

        node_key = self.current_node or "general"
        self.node_breakdown[node_key] = self.node_breakdown.get(node_key, 0) + t_tokens

    def get_usage(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "node_breakdown": dict(self.node_breakdown),
            "source": "langsmith",
        }
