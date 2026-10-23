import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

# Compatibility redirect: ragas 0.3.x imports ChatVertexAI from deprecated langchain_community path
_vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_shim.ChatVertexAI = ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_shim)

from config.models import get_ragas_evaluator_llm
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness

logger = logging.getLogger(__name__)

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")

_faithfulness_metric = None
_context_precision_metric = None
_context_recall_metric = None


def _get_metrics():
    global _faithfulness_metric, _context_precision_metric, _context_recall_metric
    if _faithfulness_metric is None:
        evaluator_llm = get_ragas_evaluator_llm()
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
    sample = SingleTurnSample(
        user_input=user_input,
        response=clean_llm_response(response),
        retrieved_contexts=retrieved_contexts,
    )
    score = await faithfulness_metric.single_turn_ascore(sample)
    val = float(getattr(score, "value", score) or 0.0)
    logger.info("Faithfulness score: %s", val)
    return val


async def evaluate_context_precision(
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
) -> float:
    _, context_precision_metric, _ = _get_metrics()
    sample = SingleTurnSample(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    score = await context_precision_metric.single_turn_ascore(sample)
    val = float(getattr(score, "value", score) or 0.0)
    logger.info("Context precision score: %s", val)
    return val


async def evaluate_context_recall(
    user_input: str,
    reference: str,
    retrieved_contexts: List[str],
) -> float:
    _, _, context_recall_metric = _get_metrics()
    sample = SingleTurnSample(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    score = await context_recall_metric.single_turn_ascore(sample)
    val = float(getattr(score, "value", score) or 0.0)
    logger.info("Context recall score: %s", val)
    return val
