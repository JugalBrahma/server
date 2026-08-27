import os
import re
import sys
import types
from pathlib import Path
from typing import List
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")

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
from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall

_faithfulness_metric = None
_context_precision_metric = None
_context_recall_metric = None


def _get_metrics():
    global _faithfulness_metric, _context_precision_metric, _context_recall_metric
    if _faithfulness_metric is None or _context_precision_metric is None or _context_recall_metric is None:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        evaluator_llm = llm_factory("gpt-4o-mini", client=client)
        _faithfulness_metric = Faithfulness(llm=evaluator_llm)
        _context_precision_metric = ContextPrecision(llm=evaluator_llm)
        _context_recall_metric = ContextRecall(llm=evaluator_llm)
    return _faithfulness_metric, _context_precision_metric, _context_recall_metric


def clean_llm_response(text: str) -> str:
    """Extracts only the final response after reasoning blocks (<think>...</think>) and removes doc tags."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<doc>", "").replace("</doc>", "").strip()


async def evaluate_faithfulness(
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
) -> float:
    faithfulness_metric, _, _ = _get_metrics()
    cleaned_response = clean_llm_response(response)
    score = await faithfulness_metric.ascore(
        user_input=user_input,
        response=cleaned_response,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"Faithfulness Score: {score}")
    return score


async def evaluate_context_precision(
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
) -> float:
    _, context_precision_metric, _ = _get_metrics()
    score = await context_precision_metric.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"Context Precision Score: {score}")
    return score


async def evaluate_context_recall(
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
) -> float:
    _, _, context_recall_metric = _get_metrics()
    score = await context_recall_metric.ascore(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )
    print(f"Context Recall Score: {score}")
    return score