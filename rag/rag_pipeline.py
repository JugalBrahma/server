"""
RAG Pipeline Main Module
- Orchestrates the complete RAG processing pipeline
- Integrates text and image processing modules
- Manages vector store operations
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from .image_processing import (
    compress_image,
    deduplicate_images,
    describe_images_batch,
    get_cached_descriptions,
    usage_stats,
)
from .text_processing import (
    combine_text_and_images,
    extract_pdf_elements,
    process_extracted_elements_smart,
    smart_chunk,
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages Pinecone vector store operations for the pipeline."""

    def __init__(self, index_name: Optional[str] = None, namespace: str = "test-2"):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", os.getenv("AGENT_INDEX_NAME", "askadocument"))
        self.namespace = namespace

        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
        )
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> PineconeVectorStore:
        try:
            return PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace,
            )
        except Exception as e:
            logger.warning("Could not connect to index '%s' (%s). Creating new index.", self.index_name, e)
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
            return PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace,
            )

    def add_documents(self, chunks: List[Dict]) -> None:
        """Add document chunks with cleaned metadata to Pinecone."""
        try:
            texts = [chunk["content"] for chunk in chunks]
            metadatas = []
            unwanted_keys = {"image_base64", "coordinates", "image_payload"}

            for chunk in chunks:
                meta = {}
                raw_meta = chunk.get("metadata") or {}

                for k, v in raw_meta.items():
                    if k in unwanted_keys:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                        meta[k] = v
                    else:
                        meta[f"{k}_json"] = str(v)

                meta["content_type"] = chunk.get("type", "Text")
                if chunk.get("has_image", False):
                    meta["has_image"] = True

                metadatas.append(meta)

            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                namespace=self.namespace,
            )
            logger.info("Added %d documents to namespace '%s'", len(chunks), self.namespace)

        except Exception as e:
            logger.error("Failed to add documents to vector store: %s", e)
            raise

    def search(self, query: str, k: int = 5) -> List[Any]:
        """Search for relevant documents in the namespace."""
        try:
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                namespace=self.namespace,
            )
        except Exception as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return []


def run_optimized_pipeline(
    pdf_path: str,
    use_cache: bool = True,
    index_name: str = "askadocument",
    namespace: str = "test-2",
) -> Tuple[VectorStoreManager, List[Dict]]:
    """Run the complete RAG ingestion pipeline for a PDF document."""
    logger.info("Starting RAG ingestion pipeline for: %s", pdf_path)

    client = OpenAI()
    vector_manager = VectorStoreManager(index_name, namespace)

    # 1. Extract elements
    raw_elements = extract_pdf_elements(pdf_path)
    logger.info("Extracted %d raw elements", len(raw_elements))

    # 2. Filter text & images
    text_chunks, image_chunks = process_extracted_elements_smart(raw_elements)
    logger.info("Processed %d text chunks and %d image elements", len(text_chunks), len(image_chunks))

    # 3. Deduplicate images
    unique_images = deduplicate_images(image_chunks)

    # 4. Compress images
    compressed_images = [
        {**img, "base64": compress_image(img["base64"])}
        for img in unique_images
    ]

    # 5. Generate descriptions
    if use_cache:
        image_descriptions = get_cached_descriptions(compressed_images, client)
    else:
        image_descriptions = describe_images_batch(compressed_images, client)

    # 6. Combine text + image descriptions
    combined_chunks = combine_text_and_images(text_chunks, image_descriptions)

    # 7. Chunk and upsert
    final_chunks = smart_chunk(combined_chunks, chunk_size=1000, overlap=200)
    vector_manager.add_documents(final_chunks)

    logger.info(
        "Pipeline complete: %d total chunks (%d with images, %d text-only)",
        len(final_chunks),
        sum(1 for c in final_chunks if c.get("has_image")),
        sum(1 for c in final_chunks if not c.get("has_image")),
    )

    return vector_manager, final_chunks
