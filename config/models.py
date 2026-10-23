import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# DEFAULT_OPENAI_MODEL = os.getenv("AGENT_MODEL")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
DEFAULT_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
DEFAULT_RAGAS_EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "deepseek/deepseek-chat")


# def get_groq_llm(
#     model: str = DEFAULT_GROQ_MODEL,
#     temperature: float = DEFAULT_TEMPERATURE,
#     **kwargs,
# ) -> BaseChatModel:
#     return ChatGroq(model=model, temperature=temperature, **kwargs)


# def get_openai_llm(
#     model: str = DEFAULT_OPENAI_MODEL,
#     temperature: float = DEFAULT_TEMPERATURE,
#     **kwargs,
# ) -> BaseChatModel:
#     return ChatOpenAI(model=model, temperature=temperature, **kwargs)
# def get_openai_embeddings(
#     model: str | None = None,
#     dimensions: int | None = None,
#     **kwargs,
# ) -> OpenAIEmbeddings:
#     selected_model = model or DEFAULT_EMBEDDING_MODEL
#     selected_dim = dimensions or DEFAULT_EMBEDDING_DIM
#     return OpenAIEmbeddings(
#         model=selected_model,
#         dimensions=selected_dim,
#         **kwargs,
#     )


def get_openrouter_llm(
    # pyrefly: ignore[bad-function-definition]
    model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
) -> BaseChatModel:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for openrouter provider")

    return ChatOpenRouter(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def get_rag_llm(
    model: str | None = None,
    temperature: float = 0.3,
    provider: str | None = None,
    **kwargs,
) -> BaseChatModel:
    # selected_provider = (provider or DEFAULT_PROVIDER).lower()
    # if selected_provider == "openai":
    #     return get_openai_llm(model=model or DEFAULT_OPENAI_MODEL, temperature=temperature, **kwargs)
    return get_openrouter_llm(
        model=model or DEFAULT_OPENROUTER_MODEL,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------- embeddings



def get_openrouter_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
    **kwargs,
) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for openrouter provider")

    selected_model = model or DEFAULT_EMBEDDING_MODEL
    selected_dim = dimensions or DEFAULT_EMBEDDING_DIM
    return OpenAIEmbeddings(
        model=selected_model,
        dimensions=selected_dim,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        check_embedding_ctx_length=False,
        **kwargs,
    )


def get_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
    provider: str | None = None,
    **kwargs,
) -> OpenAIEmbeddings:
    """Returns the configured embeddings instance (default: OpenRouter)."""
    # selected_provider = (provider or DEFAULT_PROVIDER).lower()
    # if selected_provider == "openai":
    #     return get_openai_embeddings(model=model, dimensions=dimensions, **kwargs)
    return get_openrouter_embeddings(model=model, dimensions=dimensions, **kwargs)


# ---------------------------------------------------------------- ragas evaluation


def get_ragas_evaluator_llm(
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
):
    """Returns the dedicated evaluator LLM for Ragas benchmarks via OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    from ragas.llms import LangchainLLMWrapper

    selected_model = model or DEFAULT_RAGAS_EVAL_MODEL
    llm = ChatOpenAI(
        model=selected_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
    return LangchainLLMWrapper(langchain_llm=llm)
