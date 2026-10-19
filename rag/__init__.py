"""PDF extraction, image processing, chunking, and vector indexing."""

from .ingestion import prepare_chunks
from .image_processing import (
    compress_image,
    deduplicate_images,
    describe_images_batch,
    get_cached_descriptions,
)
from .rag_pipeline import (
    VectorSearchManager,
    VectorStoreManager,
    run_optimized_pipeline,
)
from .text_processing import (
    combine_text_and_images,
    extract_pdf_elements,
    process_extracted_elements_smart,
    smart_chunk,
)

__version__ = "1.0.0"

__all__ = [
    # Ingestion and pipeline
    "prepare_chunks",
    "run_optimized_pipeline",
    "VectorSearchManager",
    "VectorStoreManager",
    # Text processing
    "extract_pdf_elements",
    "process_extracted_elements_smart",
    "combine_text_and_images",
    "smart_chunk",
    # Image processing
    "compress_image",
    "deduplicate_images",
    "describe_images_batch",
    "get_cached_descriptions",
]
