import logging
import os
import re
import threading
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
from langsmith import Client, evaluate
from evaluate.datasets_data import examples as local_examples

from evaluate.metrics import EVALUATORS

logger = logging.getLogger(__name__)

load_dotenv(SERVER_ROOT / ".env")

DATASET_NAME = os.getenv("DATASET_NAME", "datasets_data")
STATIC_EXPERIMENT_QUESTIONS = (
    "Why was the doctor initially unable to travel to his sick patient?",
    "Where did the doctor unexpectedly find the horses and the groom?",
    "What violent act did the groom commit against the servant girl, Rosa?",
)
_client: Client | None = None
_cached_examples: list | None = None
_examples_lock = threading.Lock()


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

    with _examples_lock:
        if _cached_examples is not None:
            return _cached_examples

        client = get_client()
        if not client:
            return []

        try:
            dataset_id = None
            if not client.has_dataset(dataset_name=DATASET_NAME):
                dataset = client.create_dataset(
                    dataset_name=DATASET_NAME,
                    description="QA Dataset for A Country Doctor",
                )
                dataset_id = dataset.id
                client.create_examples(
                    inputs=[example["inputs"] for example in local_examples],
                    outputs=[example["outputs"] for example in local_examples],
                    dataset_id=dataset.id,
                )

            all_examples = list(client.list_examples(dataset_name=DATASET_NAME))
            remote_questions = {
                _normalize_question(example.inputs.get("question", "")) # type: ignore
                for example in all_examples
            }
            missing_local_examples = [
                example for example in local_examples
                if _normalize_question(example["inputs"].get("question", "")) not in remote_questions
            ]
            if missing_local_examples:
                # Keep the remote dataset aligned with the checked-in source list.
                if all_examples:
                    dataset_id = all_examples[0].dataset_id
                elif dataset_id is None:
                    dataset_id = client.read_dataset(dataset_name=DATASET_NAME).id
                client.create_examples(
                    inputs=[example["inputs"] for example in missing_local_examples],
                    outputs=[example["outputs"] for example in missing_local_examples],
                    dataset_id=dataset_id,
                )
                all_examples = list(client.list_examples(dataset_name=DATASET_NAME))

            # Deduplicate examples by question to avoid redundant iterations.
            seen_questions = set()
            _cached_examples = []
            for example in all_examples:
                question = _normalize_question(example.inputs.get("question", "")) # type: ignore
                if question and question not in seen_questions:
                    seen_questions.add(question)
                    _cached_examples.append(example)
            return _cached_examples
        except Exception as exc:
            logger.warning("Error fetching evaluation examples: %s", exc)
            return []


def _normalize_question(text: str) -> str:
    """Normalize case, whitespace, and punctuation for exact question matching."""
    return " ".join(re.findall(r"\w+", text.casefold()))


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


def evaluate_flutter_query(
    query_text: str,
    namespace: str | None = None,
):
    client = get_client()
    if not query_text or not client:
        return

    if not namespace:
        logger.warning("Cannot run the default experiment without a RAG namespace")
        return

    dataset_examples = get_examples()
    static_questions = {_normalize_question(question) for question in STATIC_EXPERIMENT_QUESTIONS}
    experiment_examples = [
        example for example in dataset_examples
        if _normalize_question(example.inputs.get("question", "")) in static_questions
    ]
    if len(experiment_examples) != len(STATIC_EXPERIMENT_QUESTIONS):
        logger.warning(
            "Default experiment needs %d static questions, but found %d in LangSmith dataset '%s'",
            len(STATIC_EXPERIMENT_QUESTIONS),
            len(experiment_examples),
            DATASET_NAME,
        )
        return

    logger.info(
        "User query '%s' received. Triggering the default %d-question LangSmith experiment.",
        query_text,
        len(experiment_examples),
    )

    try:
        from agent.agent import RAGAgent

        eval_agent = RAGAgent(namespace=namespace)

        def predict(inputs: dict) -> dict:
            question = inputs.get("question", "")
            result = eval_agent.query(user_input=question) # type: ignore
            contexts = result.get("contexts") or _fetch_contexts(question, namespace)
            return {
                "output": _clean_answer(result.get("answer", "")),
                "retrieved_contexts": contexts,
            }

        results = evaluate(
            predict,
            data=experiment_examples,
            evaluators=EVALUATORS,
            client=client,
            experiment_prefix="flutter-default-eval",
            max_concurrency=1,
        )
        logger.info("Evaluation complete: %s", results)
    except Exception as exc:
        logger.exception("Evaluation failed after user query '%s': %s", query_text, exc)
