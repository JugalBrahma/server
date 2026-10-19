"""Run the ingestion pipeline on one local PDF."""

import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env", override=True)

from rag.rag_pipeline import run_optimized_pipeline  # noqa: E402


def main() -> None:
    pdf_path = os.getenv("TEST_PDF_PATH", "data/rag_for_nlp.pdf")
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        print("Set TEST_PDF_PATH to the PDF you want to process.")
        return

    print(f"Processing {pdf_path}")
    _, chunks = run_optimized_pipeline(pdf_path=pdf_path, use_cache=True)
    print(f"Created {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
