import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
DEFAULT_OPENAI_MODEL = os.getenv("AGENT_MODEL") or "gpt-4o-mini"
DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


def get_groq_llm(
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
) -> BaseChatModel:
    return ChatGroq(model=model, temperature=temperature, **kwargs)


def get_openai_llm(
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
) -> BaseChatModel:
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def get_rag_llm(
    model: str | None = None,
    temperature: float = 0.3,
    provider: str | None = None,
    **kwargs,
) -> BaseChatModel:
    selected_provider = (provider or DEFAULT_PROVIDER).lower()
    if selected_provider == "groq":
        return get_groq_llm(model=model or DEFAULT_GROQ_MODEL, temperature=temperature, **kwargs)
    return get_openai_llm(model=model or DEFAULT_OPENAI_MODEL, temperature=temperature, **kwargs)
