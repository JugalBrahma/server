"""
RAG Pipeline Module
- Complete PDF processing pipeline
- Image optimization and batch processing
- Vector store management with Pinecone
- Smart chunking and document preparation
"""

from .rag_pipeline import (
    run_optimized_pipeline,
    VectorStoreManager
)
from .text_processing import (
    extract_pdf_elements,
    process_extracted_elements_smart,
    combine_text_and_images,
    smart_chunk,
    clean_text
)
from .image_processing import (
    is_meaningful_image,
    compress_image,
    deduplicate_images,
    describe_images_batch,
    get_cached_descriptions,
    usage_stats
)

__version__ = "1.0.0"
__all__ = [
    # Main pipeline
    "run_optimized_pipeline",
    "VectorStoreManager",
    
    # Text processing
    "extract_pdf_elements",
    "process_extracted_elements_smart", 
    "combine_text_and_images",
    "smart_chunk",
    "clean_text",
    
    # Image processing
    "is_meaningful_image",
    "compress_image",
    "deduplicate_images",
    "describe_images_batch",
    "get_cached_descriptions",
    "usage_stats"
]
