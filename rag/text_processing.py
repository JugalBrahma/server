"""
Text Processing Functions for RAG Pipeline
- PDF element extraction and processing
- Smart chunking and text combination
- Content cleaning and normalization
"""

import base64
from collections import defaultdict
import logging
import os
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def extract_pdf_elements(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract all text and image elements from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        extracted_elements: List[Dict[str, Any]] = []
        images_dir = f"{pdf_path}_images"
        os.makedirs(images_dir, exist_ok=True)

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            # pyrefly: ignore[missing-attribute]
            text = page.get_text("text").strip()
            if text:
                extracted_elements.append({
                    "type": "text",
                    "page": page_num,
                    "text": text,
                })

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image.get("image")
                ext = base_image.get("ext", "png")
                if not img_bytes:
                    continue

                image_filename = f"page{page_num}_img{img_index + 1}.{ext}"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(img_bytes)

                extracted_elements.append({
                    "type": "image",
                    "page": page_num,
                    "image_path": image_path,
                })

        doc.close()
        return extracted_elements
    except Exception as e:
        logger.error("PyMuPDF extraction failed for '%s': %s", pdf_path, e)
        return []


def process_extracted_elements_smart(raw_elements: List[Any]) -> Tuple[List[Dict], List[Dict]]:
    """Split text into manageable chunks and prepare image base64 payloads."""
    text_chunks: List[Dict] = []
    image_chunks: List[Dict] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    for element in raw_elements:
        if not isinstance(element, dict):
            continue

        element_type = element.get("type", "")
        page = element.get("page", 1)

        try:
            if element_type == "text":
                text = element.get("text", "").strip()
                if not text:
                    continue
                if text.isdigit() and len(text) <= 4:
                    continue
                if not any(ch.isalpha() for ch in text) and len(text) < 15:
                    continue

                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 10:
                        text_chunks.append({
                            "type": "Text",
                            "content": chunk.lower(),
                            "metadata": {
                                "page": page,
                                "chunk_index": i,
                                "element_id": f"text_{page}_{i}",
                            },
                        })

            elif element_type == "image":
                image_path = element.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode("utf-8")

                    image_chunks.append({
                        "type": "Image",
                        "image_path": image_path,
                        "base64": image_base64,
                        "metadata": {"page": page},
                    })

        except Exception as e:
            logger.warning("Error processing %s element on page %s: %s", element_type, page, e)
            continue

    return text_chunks, image_chunks


def combine_text_and_images(text_chunks: List[Dict], image_descriptions: List[Dict]) -> List[Dict]:
    """Link image descriptions to their corresponding page text chunks."""
    text_map = {}
    page_text_map: Dict[int, List[Dict]] = defaultdict(list)

    for chunk in text_chunks:
        metadata = chunk.get("metadata", {})
        element_id = metadata.get("element_id")
        page = metadata.get("page")
        if element_id:
            text_map[element_id] = chunk
        if page is not None:
            page_text_map[page].append(chunk)

    for page in page_text_map:
        page_text_map[page].sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

    page_assignment_counts: Dict[int, int] = defaultdict(int)
    combined_chunks: List[Dict] = []
    processed_image_indices = set()

    for idx, img in enumerate(image_descriptions):
        metadata = img.get("metadata", {})
        parent_id = metadata.get("parent_id")
        target_chunk = None

        if parent_id and parent_id in text_map:
            target_chunk = text_map[parent_id]
        else:
            page = metadata.get("page")
            if page in page_text_map and page_text_map[page]:
                assign_idx = page_assignment_counts[page] % len(page_text_map[page])
                page_assignment_counts[page] += 1
                target_chunk = page_text_map[page][assign_idx]

        if target_chunk:
            target_chunk["content"] += f"\n\n[Image Context]: {img.get('content', '')}"
            target_chunk["has_image"] = True
            processed_image_indices.add(idx)
        else:
            logger.debug("Could not find text chunk for image idx=%s. Keeping as image-only.", idx)

    for idx, img in enumerate(image_descriptions):
        if idx not in processed_image_indices:
            combined_chunks.append({
                "type": "image_only",
                "content": f"[Image Only]\n{img.get('content', '')}",
                "has_image": True,
                "metadata": img.get("metadata", {}),
            })

    combined_chunks.extend(text_chunks)
    return combined_chunks


def smart_chunk(combined_chunks: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split oversized text chunks while keeping image-enriched chunks together."""
    final_chunks: List[Dict] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    unwanted_keys = {"image_base64", "coordinates", "image_payload", "has_image", "image_mime_type"}

    for chunk in combined_chunks:
        if "metadata" in chunk and isinstance(chunk["metadata"], dict):
            chunk["metadata"] = {k: v for k, v in chunk["metadata"].items() if k not in unwanted_keys}

        content = chunk.get("content", "")
        if chunk.get("has_image") or len(content) < chunk_size:
            if len(content.strip()) > 10:
                final_chunks.append(chunk)
        else:
            for split in splitter.split_text(content):
                if len(split.strip()) > 10:
                    final_chunks.append({**chunk, "content": split})

    return final_chunks
