import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))


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
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.3,
    **kwargs,
) -> BaseChatModel:
    return get_groq_llm(model=model, temperature=temperature, **kwargs)


def get_guardrail_llm(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    return get_groq_llm(model=model, temperature=temperature, **kwargs)


def get_vision_llm(
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.2,
    **kwargs,
) -> BaseChatModel:
    return get_openai_llm(model=model, temperature=temperature, **kwargs)


def get_evaluator_llm(
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    return get_openai_llm(model=model, temperature=temperature, **kwargs)
