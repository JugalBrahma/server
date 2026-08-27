import os
import re
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client, evaluate
from evaluate.ragas import (
    evaluate_faithfulness,
    evaluate_context_precision,
    evaluate_context_recall,
)

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")

DATASET_NAME = "A Country Doctor QA Dataset"

_client: Client | None = None
_cached_examples = None


def get_client() -> Client | None:
    global _client
    if _client is None and (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        try:
            _client = Client()
        except Exception as e:
            print(f"[FLUTTER EVAL] Client init error: {e}")
    return _client


def get_examples():
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
    except Exception as e:
        print(f"[FLUTTER EVAL] Error fetching examples: {e}")
    return []


def evaluate_flutter_query(
    query_text: str,
    answer_text: str,
    namespace: str | None = None,
    retrieved_contexts: list[str] | None = None,
):
    """
    Checks if a query from Flutter matches a benchmark question.
    If matched, logs an evaluation run with faithfulness, context precision, and context recall scores to LangSmith.
    """
    client = get_client()
    if not query_text or not client:
        return

    clean_query = query_text.strip().lower()
    examples = get_examples()

    matched_example = next(
        (
            ex for ex in examples
            if (q := ex.inputs.get("question", "").strip().lower())
            and (q in clean_query or clean_query in q)
        ),
        None,
    )

    if not matched_example:
        return

    try:
        if "</think>" in answer_text:
            clean_answer = answer_text.split("</think>")[-1]
        else:
            clean_answer = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL)
        clean_answer = clean_answer.replace("<doc>", "").replace("</doc>", "").strip()

        # Retrieve context chunks from Pinecone if not directly supplied
        if retrieved_contexts is None and namespace:
            try:
                from agent.vector_manager import VectorSearchManager
                vm = VectorSearchManager(namespace=namespace)
                search_results = vm.search(query=query_text, k=8)
                retrieved_contexts = [
                    doc.page_content for doc in search_results
                    if hasattr(doc, "page_content") and doc.page_content
                ]
            except Exception as exc:
                print(f"[FLUTTER EVAL] Context retrieval notice: {exc}")
                retrieved_contexts = []

        contexts = retrieved_contexts or []

        def faithfulness_evaluator(run, example) -> dict:
            question = example.inputs.get("question", "")
            ans = run.outputs.get("output", "")
            ctxs = run.outputs.get("retrieved_contexts", [])

            if not ctxs:
                return {"key": "faithfulness", "score": 0.0}

            try:
                score = asyncio.run(
                    evaluate_faithfulness(
                        user_input=question,
                        response=ans,
                        retrieved_contexts=ctxs,
                    )
                )
                return {"key": "faithfulness", "score": score}
            except Exception as e:
                print(f"[FLUTTER EVAL] Faithfulness evaluation error: {e}")
                return {"key": "faithfulness", "score": 0.0}

        def context_precision_evaluator(run, example) -> dict:
            question = example.inputs.get("question", "")
            reference = (
                example.outputs.get("expected_answer")
                or example.outputs.get("reference")
                or example.outputs.get("answer")
                or ""
            )
            ctxs = run.outputs.get("retrieved_contexts", [])

            if not ctxs or not reference:
                return {"key": "context_precision", "score": 0.0}

            try:
                score = asyncio.run(
                    evaluate_context_precision(
                        user_input=question,
                        reference=reference,
                        retrieved_contexts=ctxs,
                    )
                )
                return {"key": "context_precision", "score": score}
            except Exception as e:
                print(f"[FLUTTER EVAL] Context precision evaluation error: {e}")
                return {"key": "context_precision", "score": 0.0}

        def context_recall_evaluator(run, example) -> dict:
            question = example.inputs.get("question", "")
            reference = (
                example.outputs.get("expected_answer")
                or example.outputs.get("reference")
                or example.outputs.get("answer")
                or ""
            )
            ctxs = run.outputs.get("retrieved_contexts", [])

            if not ctxs or not reference:
                return {"key": "context_recall", "score": 0.0}

            try:
                score = asyncio.run(
                    evaluate_context_recall(
                        user_input=question,
                        reference=reference,
                        retrieved_contexts=ctxs,
                    )
                )
                return {"key": "context_recall", "score": score}
            except Exception as e:
                print(f"[FLUTTER EVAL] Context recall evaluation error: {e}")
                return {"key": "context_recall", "score": 0.0}

        print(f"\n[FLUTTER EVAL] Benchmark Question Detected: '{query_text}'")
        results = evaluate(
            lambda _: {"output": clean_answer, "retrieved_contexts": contexts},
            data=[matched_example],
            evaluators=[
                faithfulness_evaluator,
                context_precision_evaluator,
                context_recall_evaluator,
            ],
            client=client,
            experiment_prefix="flutter-ui-eval",
            max_concurrency=1,
        )
        print(f"[FLUTTER EVAL] Logged experiment run to LangSmith: {results}\n")
    except Exception as e:
        print(f"[FLUTTER EVAL] Notice: {e}")

