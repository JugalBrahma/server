import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

groq_llm = "qwen/qwen3.6-27b"
openai_llm = "gpt-4o-mini"

DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))


# ==========================================================
# 🔌 BASE PROVIDER FUNCTIONS
# ==========================================================

def get_groq_llm(
    model: str = groq_llm,
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
) -> BaseChatModel:
    """Initialize a Groq chat model."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_openai_llm(
    model: str = openai_llm,
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
) -> BaseChatModel:
    """Initialize an OpenAI chat model."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        **kwargs,
    )


# ==========================================================
# 🎯 FEATURE-SPECIFIC LLM FUNCTIONS
# ==========================================================

def get_rag_llm(
    model: str = groq_llm,
    temperature: float = 0.3,
    **kwargs,
) -> BaseChatModel:
    """LLM for RAG document answering and chat synthesis."""
    return get_groq_llm(
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_guardrail_llm(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """Fast, deterministic LLM for guardrails and greeting checks."""
    return get_groq_llm(
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_vision_llm(
    model: str = openai_llm,
    temperature: float = 0.2,
    **kwargs,
) -> BaseChatModel:
    """Multimodal Vision LLM for describing images/charts from PDFs."""
    return get_openai_llm(
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_evaluator_llm(
    model: str = openai_llm,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """LLM for RAG evaluation (Ragas) and quality benchmarking."""
    return get_openai_llm(
        model=model,
        temperature=temperature,
        **kwargs,
    )
