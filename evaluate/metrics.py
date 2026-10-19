"""Shared evaluators for the evaluation scripts."""

import asyncio
import logging

from .ragas import (
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_faithfulness,
)

logger = logging.getLogger(__name__)


def _evaluate(key: str, metric, run, example) -> dict:
    contexts = run.outputs.get("retrieved_contexts", [])
    question = example.inputs.get("question", "")

    if key == "faithfulness":
        if not contexts:
            return {"key": key, "score": 0.0}
        arguments = {
            "user_input": question,
            "response": run.outputs.get("output", ""),
            "retrieved_contexts": contexts,
        }
    else:
        outputs = getattr(example, "outputs", {}) or {}
        reference = (
            outputs.get("expected_answer")
            or outputs.get("reference")
            or outputs.get("answer")
            or ""
        )
        if not contexts or not reference:
            return {"key": key, "score": 0.0}
        arguments = {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        }

    try:
        score = asyncio.run(metric(**arguments))
        return {"key": key, "score": score}
    except Exception as exc:
        logger.warning("%s evaluation failed: %s", key, exc)
        return {"key": key, "score": 0.0}


def faithfulness_evaluator(run, example) -> dict:
    return _evaluate("faithfulness", evaluate_faithfulness, run, example)


def context_precision_evaluator(run, example) -> dict:
    return _evaluate("context_precision", evaluate_context_precision, run, example)


def context_recall_evaluator(run, example) -> dict:
    return _evaluate("context_recall", evaluate_context_recall, run, example)


EVALUATORS = [
    faithfulness_evaluator,
    context_precision_evaluator,
    context_recall_evaluator,
]
