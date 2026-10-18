import logging
import re
import sys
import types
import inspect
from importlib import import_module
from pathlib import Path
from typing import Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Ragas 0.3.x still imports Vertex AI from this path, which newer
# langchain-community releases removed in favor of langchain-google-vertexai.
from langchain_google_vertexai import ChatVertexAI, VertexAI

import langchain_community.chat_models as community_chat_models
import langchain_community.llms as community_llms

_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
setattr(_vertexai_module, "ChatVertexAI", ChatVertexAI)
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_module)
setattr(
    community_chat_models,
    "vertexai",
    sys.modules["langchain_community.chat_models.vertexai"],
)
setattr(community_llms, "VertexAI", VertexAI)

from ragas.llms import llm_factory
SingleTurnSample: Any = None
try:
    _metrics_module = import_module("ragas.metrics.collections")
except ImportError:
    _metrics_module = import_module("ragas.metrics")
    SingleTurnSample = getattr(import_module("ragas.dataset_schema"), "SingleTurnSample")

    _LEGACY_METRICS_API = True
else:
    _LEGACY_METRICS_API = False

ContextPrecision = getattr(_metrics_module, "ContextPrecision")
ContextRecall = getattr(_metrics_module, "ContextRecall")
Faithfulness = getattr(_metrics_module, "Faithfulness")

logger = logging.getLogger(__name__)

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")


_faithfulness_metric = None
_context_precision_metric = None
_context_recall_metric = None


def _score_value(score) -> float:
    return getattr(score, "value", score)


async def _score(metric, **sample_fields):
    if _LEGACY_METRICS_API:
        return await metric.single_turn_ascore(SingleTurnSample(**sample_fields))
    return await metric.ascore(**sample_fields)


def _get_metrics():
    global _faithfulness_metric, _context_precision_metric, _context_recall_metric
    if _faithfulness_metric is None:
        llm_factory_kwargs: dict[str, Any] = {}
        if "client" in inspect.signature(llm_factory).parameters:
            llm_factory_kwargs["client"] = AsyncOpenAI()
        evaluator_llm = llm_factory("gpt-4o-mini", **llm_factory_kwargs)
        _faithfulness_metric = Faithfulness(llm=evaluator_llm)
        _context_precision_metric = ContextPrecision(llm=evaluator_llm)
        _context_recall_metric = ContextRecall(llm=evaluator_llm)
    return _faithfulness_metric, _context_precision_metric, _context_recall_metric


def clean_llm_response(text: str) -> str:
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
    cleaned = clean_llm_response(response)
    score = await _score(
        faithfulness_metric,
        user_input=user_input,
        response=cleaned,
        retrieved_contexts=retrieved_contexts,
    )
    logger.info("Faithfulness score: %s", score)
    return _score_value(score)


async def evaluate_context_precision(
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
) -> float:
    _, context_precision_metric, _ = _get_metrics()
    score = await _score(
        context_precision_metric,
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    logger.info("Context precision score: %s", score)
    return _score_value(score)


async def evaluate_context_recall(
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
) -> float:
    _, _, context_recall_metric = _get_metrics()
    score = await _score(
        context_recall_metric,
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )
    logger.info("Context recall score: %s", score)
    return _score_value(score)
