"""
Image Processing Functions for RAG Pipeline
- Image filtering and optimization
- Batch processing with OpenAI Vision
- Caching and deduplication
"""

import base64
import hashlib
import io
import logging
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Usage stats kept for backward compatibility
usage_stats = {
    "vision_input": 0,
    "vision_output": 0,
}

VISION_MODEL = os.getenv("VISION_IMAGE_MODEL", "gpt-4o-mini")
PHASH_DISTANCE_THRESHOLD = int(os.getenv("VISION_PHASH_THRESHOLD", "6"))
CACHE_VERSION = os.getenv("VISION_CACHE_VERSION", "v3")
DEFAULT_CACHE_FILE = os.getenv(
    "VISION_CACHE_FILE",
    f"image_cache_{VISION_MODEL.replace('/', '_')}_{CACHE_VERSION}.pkl",
)


def _calc_content_hash(base64_str: str) -> str:
    """Hash raw decoded bytes (or raw string if decode fails) for stability."""
    try:
        return hashlib.md5(base64.b64decode(base64_str)).hexdigest()
    except Exception:
        return hashlib.md5(base64_str.encode()).hexdigest()


def _calc_perceptual_hash(base64_str: str) -> str:
    """Compute simple 8x8 average perceptual hash to catch near-duplicates."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
        img = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        return hex(int(bits, 2))[2:].rjust(16, "0")
    except Exception:
        return ""


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def _resolve_cache_path(cache_file: Optional[str]) -> Path:
    return Path(cache_file or DEFAULT_CACHE_FILE)


def _call_model_with_retry(client, messages: List[Dict], max_retries: int = 3):
    """Invoke vision model with basic exponential backoff."""
    delay = 2
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=1000,
            )
        except Exception as err:
            last_error = err
            if attempt == max_retries:
                raise
            logger.warning("Retry %d/%d after error: %s", attempt, max_retries, err)
            time.sleep(delay)
            delay = min(delay * 2, 10)

    if last_error:
        raise last_error
    raise RuntimeError("Model call failed: maximum retries reached.")


def _build_prompt_for_image(img_chunk: Dict, index: int) -> str:
    metadata = img_chunk.get("metadata") or {}
    page = metadata.get("page")
    page_hint = f" (page {page})" if page is not None else ""

    chunk_type = (img_chunk.get("type") or "Image").lower()
    if chunk_type == "table":
        instruction = (
            f"Image {index}{page_hint}: Extract this table into clean Markdown with clear headers, "
            "units, totals, and any footnotes. Summarize trends after the table."
        )
    elif chunk_type in ("chart", "diagram", "graph"):
        instruction = (
            f"Image {index}{page_hint}: Describe this chart/diagram precisely for technical search. "
            "List axes, scales, legends, data relationships, and any annotations or equations."
        )
    else:
        instruction = (
            f"Image {index}{page_hint}: Describe every visible label, printed text, numbers, "
            "sections, UI elements, and spatial relationships. Capture intent and key insights."
        )

    extra = metadata.get("label")
    if extra:
        instruction += f" Context hint: {extra}."

    return instruction + "\nProvide key facts suitable for retrieval."


def _parse_delimited_response(response_text: str, batch_start: int, batch_size: int) -> Dict[int, str]:
    """Parse vision model response using '=== IMAGE N ===' delimiters."""
    parsed: Dict[int, str] = {}
    if not response_text:
        return parsed

    pattern = re.compile(r"=+\s*IMAGE\s+(\d+)\s*=+", re.IGNORECASE)
    matches = list(pattern.finditer(response_text))

    for i, match in enumerate(matches):
        img_num = int(match.group(1))
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(response_text)
        description = response_text[start_pos:end_pos].strip()

        global_idx = batch_start + img_num
        if 1 <= img_num <= batch_size and description:
            parsed[global_idx] = description

    return parsed


def is_meaningful_image(element: Any) -> bool:
    """Filter out tiny logos, decorations, and inline icons."""
    meta = getattr(element, "metadata", None)
    if not meta:
        return True

    min_dimension = 100
    min_area = 120 * 120
    max_aspect_ratio = 6.0

    if hasattr(meta, "coordinates") and meta.coordinates:
        try:
            points = meta.coordinates.points
            width = max(p[0] for p in points) - min(p[0] for p in points)
            height = max(p[1] for p in points) - min(p[1] for p in points)

            if width < min_dimension or height < min_dimension:
                return False

            area = width * height
            if area < min_area:
                return False

            smaller_side = max(1, min(width, height))
            aspect_ratio = max(width, height) / smaller_side
            if aspect_ratio > max_aspect_ratio and area < (200 * 200):
                return False
        except Exception:
            pass

    parent_id = getattr(meta, "parent_id", None)
    if parent_id and str(parent_id).startswith("Text"):
        return False

    label = getattr(meta, "label", "")
    if isinstance(label, str):
        lowered = label.lower()
        for noisy in ("logo", "decoration", "watermark", "background"):
            if noisy in lowered:
                return False

    confidence = getattr(meta, "confidence", 1.0)
    if isinstance(confidence, (int, float)) and confidence < 0.15:
        return False

    return True


def compress_image(base64_str: str, max_size: int = 1024) -> str:
    """Resize and recompress image to JPEG to reduce token usage."""
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.warning("Could not compress image: %s", e)
        return base64_str


def deduplicate_images(image_chunks: List[Dict]) -> List[Dict]:
    """Remove exact and near-duplicate images using MD5 and perceptual hashing."""
    seen_hashes = set()
    seen_perceptual_buckets: Dict[str, List[str]] = {}
    unique_images: List[Dict] = []
    duplicates = 0
    near_duplicates = 0

    for img in image_chunks:
        base64_data = img.get("base64")
        if not base64_data:
            continue

        content_hash = _calc_content_hash(base64_data)
        perceptual_hash = _calc_perceptual_hash(base64_data)

        if content_hash in seen_hashes:
            duplicates += 1
            continue

        if perceptual_hash:
            bucket_key = perceptual_hash[:4]
            bucket = seen_perceptual_buckets.setdefault(bucket_key, [])
            is_near_dup = any(
                _hamming_distance(h, perceptual_hash) <= PHASH_DISTANCE_THRESHOLD
                for h in bucket
            )
            if is_near_dup:
                near_duplicates += 1
                continue
            bucket.append(perceptual_hash)

        seen_hashes.add(content_hash)
        img.setdefault("metadata", {})
        img["metadata"]["content_hash"] = content_hash
        if perceptual_hash:
            img["metadata"]["perceptual_hash"] = perceptual_hash
        unique_images.append(img)

    if duplicates or near_duplicates:
        logger.info(
            "Deduplication: removed %d byte-identical and %d near-duplicate images",
            duplicates,
            near_duplicates,
        )

    return unique_images


def describe_images_batch(image_chunks: List[Dict], client, batch_size: int = 10) -> List[Dict]:
    """Process multiple images in single API calls and parse responses."""
    global usage_stats
    descriptions: List[Dict] = []

    for i in range(0, len(image_chunks), batch_size):
        batch = image_chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            user_content = []
            for idx, img in enumerate(batch):
                global_index = i + idx + 1
                prompt = _build_prompt_for_image(img, global_index)
                user_content.append({"type": "text", "text": prompt})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"},
                })

            user_content.append({
                "type": "text",
                "text": (
                    "\n\nReturn descriptions in this exact format for each image:\n"
                    "=== IMAGE 1 ===\n"
                    "[detailed description here]\n"
                    "=== IMAGE 2 ===\n"
                    "[detailed description here]\n"
                    "(continue for all images)"
                ),
            })

            response = _call_model_with_retry(
                client,
                messages=[{"role": "user", "content": user_content}],
            )

            usage = getattr(response, "usage", None)
            if usage:
                usage_stats["vision_input"] += usage.prompt_tokens
                usage_stats["vision_output"] += usage.completion_tokens
                logger.info(
                    "Batch %d tokens: %d in / %d out",
                    batch_num,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )

            response_text = response.choices[0].message.content.strip()
            parsed_descriptions = _parse_delimited_response(response_text, i, len(batch))

            for idx, img in enumerate(batch):
                global_index = i + idx + 1
                description_text = parsed_descriptions.get(global_index)

                if not description_text:
                    logger.warning("Missing description for image %d, retrying solo", global_index)
                    fallback_desc = describe_single_image(img, client, global_index)
                    if fallback_desc:
                        descriptions.append(fallback_desc)
                    continue

                descriptions.append({
                    "type": "image_description",
                    "original_type": img.get("type", "Image"),
                    "content": description_text,
                    "metadata": img.get("metadata", {}),
                    "index": global_index,
                })

        except Exception as e:
            logger.error("Error processing batch %d: %s. Falling back to single images.", batch_num, e)
            for idx, img in enumerate(batch):
                desc = describe_single_image(img, client, i + idx + 1)
                if desc:
                    descriptions.append(desc)

    return descriptions


def describe_single_image(img_chunk: Dict, client, index: int) -> Optional[Dict]:
    """Fallback handler to describe an individual image."""
    global usage_stats
    try:
        prompt = _build_prompt_for_image(img_chunk, index)
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_chunk['base64']}"},
            },
            {
                "type": "text",
                "text": "Respond ONLY with plain text description followed by bullet key points.",
            },
        ]

        response = _call_model_with_retry(
            client,
            messages=[{"role": "user", "content": user_content}],
        )
        usage = getattr(response, "usage", None)
        if usage:
            usage_stats["vision_input"] += usage.prompt_tokens
            usage_stats["vision_output"] += usage.completion_tokens

        response_text = response.choices[0].message.content.strip()
        return {
            "type": "image_description",
            "original_type": img_chunk.get("type", "Image"),
            "content": response_text,
            "metadata": img_chunk.get("metadata", {}),
            "index": index,
        }
    except Exception as e:
        logger.error("Error describing image %d: %s", index, e)
        return None


def get_cached_descriptions(
    image_chunks: List[Dict],
    client,
    cache_file: Optional[str] = None,
) -> List[Dict]:
    """Lookup or generate image descriptions, caching results to disk."""
    cache = {}
    cache_path = _resolve_cache_path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            logger.info("Loaded vision cache with %d entries", len(cache))
        except Exception as e:
            logger.warning("Could not read vision cache from %s: %s", cache_path, e)
            cache = {}

    new_images = []
    cached_descriptions = []
    cache_hits = 0

    for img in image_chunks:
        img_hash = img.get("metadata", {}).get("content_hash")
        if not img_hash and img.get("base64"):
            img_hash = _calc_content_hash(img["base64"])

        if img_hash and img_hash in cache:
            cached_descriptions.append(cache[img_hash])
            cache_hits += 1
        else:
            new_images.append((img_hash, img))

    logger.info("Vision cache hits: %d, new images: %d", cache_hits, len(new_images))

    if new_images:
        images_to_process = [img for _, img in new_images]
        new_descriptions = describe_images_batch(images_to_process, client)

        for (img_hash, _), desc in zip(new_images, new_descriptions):
            if img_hash:
                cache[img_hash] = desc
            cached_descriptions.append(desc)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            logger.info("Saved vision cache with %d entries to %s", len(cache), cache_path)
        except Exception as e:
            logger.warning("Could not save vision cache to %s: %s", cache_path, e)

    return cached_descriptions
