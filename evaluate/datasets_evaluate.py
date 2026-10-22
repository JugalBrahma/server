import argparse
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parent.parent
script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != script_dir]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))
load_dotenv(SERVER_ROOT / ".env")

from langsmith import Client, evaluate  # noqa: E402

from agent.agent import RAGAgent  # noqa: E402
from evaluate.datasets_data import examples  # noqa: E402
from evaluate.metrics import EVALUATORS

logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv("DATASET_NAME", "datasets_data")
DATASET_DESCRIPTION = "QA Dataset for A Country Doctor"
EXPERIMENT_PREFIX = "country-doctor-eval"
DEFAULT_NAMESPACE = os.getenv(
    "EVAL_NAMESPACE",
    "user_gpYcXNxxJ7hzdctAvJm5eAaafKl2_folder_AWMbOvw6jBs2N11YirBJ",
)
TOP_K = 8

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_DOC_TAGS = re.compile(r"</?doc>")


def clean_answer(raw: str) -> str:
    """Drop reasoning blocks and <doc> markers, keeping only the final answer."""
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    else:
        raw = _THINK_BLOCK.sub("", raw)
        raw = raw.split("<think>")[0]  # unclosed think block
    return _DOC_TAGS.sub("", raw).strip()


def ensure_dataset(client: Client) -> None:
    """Create and populate the dataset if missing. Raises on failure."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' found", DATASET_NAME)
        return

    logger.info("Dataset '%s' not found, creating", DATASET_NAME)
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME, description=DATASET_DESCRIPTION
    )
    try:
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_id=dataset.id,
        )
    except Exception:
        client.delete_dataset(dataset_id=dataset.id)  # don't leave an empty dataset
        raise
    logger.info("Uploaded %d examples", len(examples))


def build_predict(rag_agent: RAGAgent):
    def predict(inputs: dict) -> dict:
        question = inputs.get("question", "")
        result = rag_agent.query(user_input=question) # type: ignore

        # Use the actual reranked contexts the agent retrieved
        contexts = result.get("contexts")
        if not contexts:
            docs = rag_agent.config.vector_manager.search(query=question, k=TOP_K)
            contexts = [d.page_content for d in docs if getattr(d, "page_content", None)]

        return {
            "output": clean_answer(result.get("answer", "")),
            "retrieved_contexts": contexts,
        }

    return predict


def run_experiment(namespace: str = DEFAULT_NAMESPACE, dataset_name: str = DATASET_NAME):
    if not namespace:
        raise ValueError("Provide a namespace via --namespace or EVAL_NAMESPACE")

    client = Client()
    ensure_dataset(client)

    logger.info("Starting evaluation for '%s'", dataset_name)
    results = evaluate(
        build_predict(RAGAgent(namespace=namespace)),
        data=dataset_name,
        evaluators=EVALUATORS,
        client=client,
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
    )
    logger.info("Finished: %s", results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--dataset", default=DATASET_NAME)
    args = parser.parse_args()
    run_experiment(namespace=args.namespace, dataset_name=args.dataset)
