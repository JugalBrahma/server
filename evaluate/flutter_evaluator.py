import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client, evaluate
from evaluate.datasets_data import examples as local_examples

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")

DATASET_NAME = "A Country Doctor QA Dataset"

# Cache of LangSmith Example objects
_cached_examples = None
_client = None


def get_client() -> Client | None:
    global _client
    if _client is None:
        if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
            try:
                _client = Client()
            except Exception as e:
                print(f"[FLUTTER EVAL] Could not initialize LangSmith Client: {e}")
    return _client


def get_dataset_examples():
    """Gets or loads LangSmith Example objects, creating dataset if needed."""
    global _cached_examples
    if _cached_examples is not None:
        return _cached_examples

    client = get_client()
    if not client:
        return []

    try:
        if not client.has_dataset(dataset_name=DATASET_NAME):
            print(f"[FLUTTER EVAL] Dataset '{DATASET_NAME}' not found in LangSmith. Creating now...")
            dataset = client.create_dataset(
                dataset_name=DATASET_NAME,
                description="QA Dataset for A Country Doctor",
            )
            client.create_examples(
                inputs=[e["inputs"] for e in local_examples],
                outputs=[e["outputs"] for e in local_examples],
                dataset_id=dataset.id,
            )
        
        _cached_examples = list(client.list_examples(dataset_name=DATASET_NAME))
        return _cached_examples
    except Exception as e:
        print(f"[FLUTTER EVAL] Error fetching dataset examples: {e}")
        return []


def evaluate_flutter_query(query_text: str, answer_text: str):
    """
    Checks if a query from Flutter matches a benchmark question.
    If matched, logs an evaluation run / experiment to LangSmith.
    """
    client = get_client()
    if not query_text or not client:
        return

    clean_query = query_text.strip().lower()
    dataset_examples = get_dataset_examples()

    matched_example = None
    for ex in dataset_examples:
        q = ex.inputs.get("question", "").strip().lower()
        if q and (q in clean_query or clean_query in q):
            matched_example = ex
            break

    if not matched_example:
        return

    try:
        clean_answer = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL)
        clean_answer = clean_answer.replace("<doc>", "").replace("</doc>", "").strip()

        print(f"\n[FLUTTER EVAL] Benchmark Question Detected: '{query_text}'")
        results = evaluate(
            lambda inputs: {"output": clean_answer},
            data=[matched_example],
            client=client,
            experiment_prefix="flutter-ui-eval",
            max_concurrency=1,
        )
        print(f"[FLUTTER EVAL] Logged experiment run to LangSmith: {results}\n")
    except Exception as e:
        print(f"[FLUTTER EVAL] Notice: {e}")

