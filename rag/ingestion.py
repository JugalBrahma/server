"""Prepare extracted PDF content for storage in the vector database."""

from typing import Dict, List

from openai import OpenAI

from .image_processing import (
    compress_image,
    deduplicate_images,
    describe_images_batch,
    get_cached_descriptions,
)
from .text_processing import (
    combine_text_and_images,
    process_extracted_elements_smart,
    smart_chunk,
)


def prepare_chunks(elements: List[Dict], client=None, use_cache: bool = True) -> List[Dict]:
    """Turn extracted PDF elements into text and image chunks."""
    if client is None:
        client = OpenAI()

    text_chunks, image_chunks = process_extracted_elements_smart(elements)
    images = [
        {**image, "base64": compress_image(image["base64"])}
        for image in deduplicate_images(image_chunks)
    ]

    if use_cache:
        descriptions = get_cached_descriptions(images, client)
    else:
        descriptions = describe_images_batch(images, client)

    combined_chunks = combine_text_and_images(text_chunks, descriptions)
    return smart_chunk(combined_chunks, chunk_size=1500, overlap=300)
