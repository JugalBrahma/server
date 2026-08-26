import sys
import types
from typing import List

# Compatibility shim for ragas importing deprecated/relocated langchain modules
if "langchain_community.chat_models.vertexai" not in sys.modules:
    v = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI
        v.ChatVertexAI = ChatVertexAI
    except Exception:
        v.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = v

if "langchain_community.llms" not in sys.modules:
    v2 = types.ModuleType("langchain_community.llms")
    try:
        from langchain_google_vertexai import VertexAI
        v2.VertexAI = VertexAI
    except Exception:
        v2.VertexAI = None
    sys.modules["langchain_community.llms"] = v2

from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import Faithfulness

client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=client)
faithfulness_metric = Faithfulness(llm=evaluator_llm)


async def evaluate_faithfulness(
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
) -> float:
    score = await faithfulness_metric.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"Faithfulness Score: {score}")
    return score