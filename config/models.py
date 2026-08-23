import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()

# ==========================================================
# 🎯 QUICK MODEL CONFIGURATION
# Change your active model here or set AGENT_MODEL in .env!
# ==========================================================
# Examples:
#   Groq:   "qwen/qwen3.6-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"
#   OpenAI: "gpt-4o", "gpt-4o-mini", "o3-mini"
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3.6-27b")
DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> BaseChatModel:
    """
    Factory function to initialize and return an LLM instance.
    Automatically detects the provider (OpenAI, Groq) based on model name.
    
    Args:
        model_name: Name of the model (e.g. 'qwen/qwen3.6-27b', 'gpt-4o-mini').
                    If None, uses DEFAULT_MODEL or AGENT_MODEL from .env.
        temperature: Temperature setting (0.0 to 1.0).
        **kwargs: Additional parameters passed to the model constructor.
    """
    model = model_name or DEFAULT_MODEL
    temp = DEFAULT_TEMPERATURE if temperature is None else temperature
    model_lower = model.lower()

    # Provider: OpenAI
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY is not set in environment.")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temp,
            **kwargs
        )

    # Provider: Groq (Default for Qwen, LLaMA, Mixtral, DeepSeek, etc.)
    else:
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ Warning: GROQ_API_KEY is not set in environment.")
        return ChatGroq(
            model=model,
            temperature=temp,
            groq_api_key=api_key,
            **kwargs
        )

