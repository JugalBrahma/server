"""Run the RAG evaluation experiment. Run from the server root: python -m evaluate.datasets_evaluate"""
import argparse
import logging
import os

from dotenv import load_dotenv
from langsmith import Client, evaluate

load_dotenv()

from agent.agent import RAGAgent  # noqa: E402
from evaluate.datasets_data import examples  # noqa: E402
from evaluate.metrics import EVALUATORS  # noqa: E402
from evaluate.ragas import clean_llm_response  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv("DATASET_NAME", "datasets_data")
DATASET_DESCRIPTION = "QA Dataset for A Country Doctor"
EXPERIMENT_PREFIX = "country-doctor-eval"
DEFAULT_NAMESPACE = os.getenv("EVAL_NAMESPACE")
TOP_K = 8


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, e.g. true or false.")


def build_predict(rag_agent: RAGAgent, enable_reranker: bool = True):
    def predict(inputs: dict) -> dict:
        question = inputs.get("question", "")
        result = rag_agent.query(user_input=question, enable_reranker=enable_reranker) # type: ignore

        # Use the actual reranked contexts the agent retrieved
        contexts = result.get("contexts")
        if not contexts:
            top_k = rag_agent.deps.rerank_top_k if not enable_reranker else TOP_K
            docs = rag_agent.deps.vector_manager.search(query=question, k=top_k)
            contexts = [d.page_content for d in docs if getattr(d, "page_content", None)]

        return {
            "output": clean_llm_response(result.get("answer", "")),
            "retrieved_contexts": contexts,
        }

    return predict


def run_experiment(
    namespace: str | None = DEFAULT_NAMESPACE,
    dataset_name: str = DATASET_NAME,
    enable_reranker: bool = True,
):
    if not namespace:
        raise ValueError("Provide a namespace via --namespace or EVAL_NAMESPACE")

    client = Client()
    ensure_dataset(client)

    rerank_suffix = "rerank-true" if enable_reranker else "rerank-false"
    experiment_prefix = f"{EXPERIMENT_PREFIX}-{rerank_suffix}"

    logger.info(
        "Starting evaluation for '%s' (reranker=%s, prefix=%s)",
        dataset_name,
        enable_reranker,
        experiment_prefix,
    )
    rag_agent = RAGAgent(namespace=namespace, enable_reranker=enable_reranker)
    results = evaluate(
        build_predict(rag_agent, enable_reranker=enable_reranker),
        data=dataset_name,
        evaluators=EVALUATORS,
        client=client,
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )
    logger.info("Finished: %s", results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on dataset")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--reranker",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable or disable neural reranker: --reranker true or --reranker false (default: true)",
    )
    args = parser.parse_args()
    run_experiment(
        namespace=args.namespace,
        dataset_name=args.dataset,
        enable_reranker=args.reranker,
    )
