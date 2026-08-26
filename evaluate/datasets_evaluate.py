import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parent.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

load_dotenv(SERVER_ROOT / ".env")

from langsmith import Client, evaluate
from agent.agent import RAGAgent
from evaluate.ragas import evaluate_faithfulness
from evaluate.datasets_data import examples

DATASET_NAME = "A Country Doctor QA Dataset"
DEFAULT_NAMESPACE = "user_gpYcXNxxJ7hzdctAvJm5eAaafKl2_folder_AWMbOvw6jBs2N11YirBJ"

client = Client()


def ensure_dataset_exists():
    """Ensures the dataset exists on LangSmith; creates and uploads it if missing."""
    try:
        if not client.has_dataset(dataset_name=DATASET_NAME):
            print(f"[DATASET] Dataset '{DATASET_NAME}' not found in LangSmith. Creating it now...")
            dataset = client.create_dataset(
                dataset_name=DATASET_NAME,
                description="QA Dataset for A Country Doctor",
            )
            client.create_examples(
                inputs=[e["inputs"] for e in examples],
                outputs=[e["outputs"] for e in examples],
                dataset_id=dataset.id,
            )
            print(f"[DATASET] Successfully uploaded {len(examples)} examples to '{DATASET_NAME}'.")
        else:
            print(f"[DATASET] Dataset '{DATASET_NAME}' found in LangSmith.")
    except Exception as e:
        print(f"[DATASET] Notice while checking/creating dataset: {e}")


def run_experiment(namespace: str = DEFAULT_NAMESPACE):
    """Runs LangSmith evaluate on the dataset using Ragas faithfulness evaluator."""
    ensure_dataset_exists()
    
    rag_agent = RAGAgent(namespace=namespace)

    def predict(inputs: dict) -> dict:
        question = inputs.get("question", "")
        # Query RAG agent
        res = rag_agent.query(question)
        answer = res.get("answer", "")

        # Retrieve dynamic contexts from Pinecone
        search_results = rag_agent.config.vector_manager.search(query=question, k=8)
        contexts = [
            doc.page_content for doc in search_results if hasattr(doc, "page_content") and doc.page_content
        ]

        return {
            "output": answer,
            "retrieved_contexts": contexts,
        }

    def faithfulness_evaluator(run, example) -> dict:
        question = example.inputs.get("question", "")
        answer = run.outputs.get("output", "")
        contexts = run.outputs.get("retrieved_contexts", [])

        if not contexts:
            return {"key": "faithfulness", "score": 0.0}

        score = asyncio.run(
            evaluate_faithfulness(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
        )

        return {"key": "faithfulness", "score": score}

    print(f"\n[EVAL] Starting LangSmith evaluation for '{DATASET_NAME}'...")
    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[faithfulness_evaluator],
        client=client,
        experiment_prefix="country-doctor-eval",
        max_concurrency=1,
    )
    print(f"[EVAL] Finished! LangSmith experiment results: {results}")
    return results


if __name__ == "__main__":
    run_experiment()
