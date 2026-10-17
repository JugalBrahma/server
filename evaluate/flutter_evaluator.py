import asyncio
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate

from evaluate.ragas import (
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_faithfulness,
)

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATASET_NAME = "A Country Doctor QA Dataset"

_client: Client | None = None
_cached_examples: list | None = None


def get_client() -> Client | None:
    global _client
    if _client is None and (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        try:
            _client = Client()
        except Exception as exc:
            logger.warning("Failed to initialize LangSmith client: %s", exc)
    return _client


def get_examples() -> list:
    global _cached_examples
    if _cached_examples is not None:
        return _cached_examples

    client = get_client()
    if not client:
        return []

    try:
        if client.has_dataset(dataset_name=DATASET_NAME):
            _cached_examples = list(client.list_examples(dataset_name=DATASET_NAME))
            return _cached_examples
    except Exception as exc:
        logger.warning("Error fetching evaluation examples: %s", exc)
    return []


def _clean_answer(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<doc>", "").replace("</doc>", "").strip()


def _fetch_contexts(query: str, namespace: str) -> list[str]:
    try:
        from agent.vector_manager import VectorSearchManager

        results = VectorSearchManager(namespace=namespace).search(query=query, k=8)
        return [doc.page_content for doc in results if getattr(doc, "page_content", None)]
    except Exception as exc:
        logger.debug("Context retrieval failed: %s", exc)
        return []


def _get_reference(example) -> str:
    outputs = getattr(example, "outputs", {}) or {}
    return outputs.get("expected_answer") or outputs.get("reference") or outputs.get("answer") or ""


def _eval_faithfulness(run, example) -> dict:
    question = example.inputs.get("question", "")
    ans = run.outputs.get("output", "")
    ctxs = run.outputs.get("retrieved_contexts", [])

    if not ctxs:
        return {"key": "faithfulness", "score": 0.0}

    try:
        score = asyncio.run(evaluate_faithfulness(question, ans, ctxs))
        return {"key": "faithfulness", "score": score}
    except Exception as exc:
        logger.warning("Faithfulness evaluation error: %s", exc)
        return {"key": "faithfulness", "score": 0.0}


def _eval_context_precision(run, example) -> dict:
    question = example.inputs.get("question", "")
    reference = _get_reference(example)
    ctxs = run.outputs.get("retrieved_contexts", [])

    if not ctxs or not reference:
        return {"key": "context_precision", "score": 0.0}

    try:
        score = asyncio.run(evaluate_context_precision(question, reference, ctxs))
        return {"key": "context_precision", "score": score}
    except Exception as exc:
        logger.warning("Context precision evaluation error: %s", exc)
        return {"key": "context_precision", "score": 0.0}


def _eval_context_recall(run, example) -> dict:
    question = example.inputs.get("question", "")
    reference = _get_reference(example)
    ctxs = run.outputs.get("retrieved_contexts", [])

    if not ctxs or not reference:
        return {"key": "context_recall", "score": 0.0}

    try:
        score = asyncio.run(evaluate_context_recall(question, reference, ctxs))
        return {"key": "context_recall", "score": score}
    except Exception as exc:
        logger.warning("Context recall evaluation error: %s", exc)
        return {"key": "context_recall", "score": 0.0}


EVALUATORS = [_eval_faithfulness, _eval_context_precision, _eval_context_recall]


def evaluate_flutter_query(
    query_text: str,
    answer_text: str,
    namespace: str | None = None,
    retrieved_contexts: list[str] | None = None,
):
    client = get_client()
    if not query_text or not client:
        return

    clean_query = query_text.strip().lower()
    matched = next(
        (
            ex for ex in get_examples()
            if (q := ex.inputs.get("question", "").strip().lower())
            and (q in clean_query or clean_query in q)
        ),
        None,
    )
    if not matched:
        return

    try:
        contexts = (
            retrieved_contexts
            if retrieved_contexts is not None
            else (_fetch_contexts(query_text, namespace) if namespace else [])
        )
        answer = _clean_answer(answer_text)

        logger.info("Evaluating benchmark query: '%s'", query_text)
        results = evaluate(
            lambda _: {"output": answer, "retrieved_contexts": contexts},
            data=[matched],
            evaluators=EVALUATORS,
            client=client,
            experiment_prefix="flutter-ui-eval",
            max_concurrency=1,
        )
        logger.info("Evaluation complete: %s", results)
    except Exception as exc:
        logger.warning("Evaluation failed: %s", exc)
