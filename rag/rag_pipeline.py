"""
RAG Pipeline Main Module
- Orchestrates the complete RAG processing pipeline
- Integrates text and image processing modules with vector search
"""

import logging
from typing import Dict, List, Tuple

from openai import OpenAI

from agent.vector_manager import VectorSearchManager
from .ingestion import prepare_chunks
from .text_processing import extract_pdf_elements

logger = logging.getLogger(__name__)

# Backwards compatibility alias
VectorStoreManager = VectorSearchManager


def run_optimized_pipeline(
    pdf_path: str,
    use_cache: bool = True,
    index_name: str = "askadocument",
    namespace: str = "test-2",
) -> Tuple[VectorSearchManager, List[Dict]]:
    """Run the complete RAG ingestion pipeline for a PDF document."""
    logger.info("Starting RAG ingestion pipeline for: %s", pdf_path)

    client = OpenAI()
    vector_manager = VectorSearchManager(index_name=index_name, namespace=namespace)

    # 1. Extract elements
    raw_elements = extract_pdf_elements(pdf_path)
    logger.info("Extracted %d raw elements", len(raw_elements))

    # 2. Prepare text and image chunks
    final_chunks = prepare_chunks(raw_elements, client=client, use_cache=use_cache)
    vector_manager.add_documents(final_chunks)

    logger.info(
        "Pipeline complete: %d total chunks (%d with images, %d text-only)",
        len(final_chunks),
        sum(1 for c in final_chunks if c.get("has_image")),
        sum(1 for c in final_chunks if not c.get("has_image")),
    )

    return vector_manager, final_chunks
