from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from config.models import get_rag_llm


class LLMManager:
    """Creates the chat model used to answer questions (via OpenRouter)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        kwargs = {}
        if model_name:
            kwargs["model"] = model_name
        if temperature is not None:
            # pyrefly: ignore[unsupported-operation]
            kwargs["temperature"] = temperature
        self.llm = get_rag_llm(**kwargs)

    def get_llm(self) -> BaseChatModel:
        return self.llm
