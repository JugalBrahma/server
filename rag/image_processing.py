"""
Image Processing Functions for RAG Pipeline
- Image filtering and optimization
- Batch processing with OpenAI Vision
- Caching and deduplication
"""

import os
import sys
import hashlib
import base64
import json
import io
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from openai import OpenAI

# Global usage stats
usage_stats = {
    "vision_input": 0,
    "vision_output": 0
}

VISION_MODEL = os.getenv("VISION_IMAGE_MODEL", "gpt-4o-mini")
PHASH_DISTANCE_THRESHOLD = int(os.getenv("VISION_PHASH_THRESHOLD", "6"))
CACHE_VERSION = os.getenv("VISION_CACHE_VERSION", "v3")
DEFAULT_CACHE_FILE = os.getenv(
    "VISION_CACHE_FILE",
    f"image_cache_{VISION_MODEL.replace('/', '_')}_{CACHE_VERSION}.pkl"
)



def _calc_content_hash(base64_str: str) -> str:
    """Hash the original bytes (not the JPEG-compressed string) for stability."""
    try:
        return hashlib.md5(base64.b64decode(base64_str)).hexdigest()
    except Exception:
        return hashlib.md5(base64_str.encode()).hexdigest()


def _calc_perceptual_hash(base64_str: str) -> str:
    """Simple aHash to catch near-duplicates even if recompressed."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
        img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return hex(int(bits, 2))[2:].rjust(16, '0')
    except Exception:
        return ""


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64  # max for 8x8 hash
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count('1')


def _resolve_cache_path(cache_file: Optional[str]) -> Path:
    return Path(cache_file or DEFAULT_CACHE_FILE)


def _call_model_with_retry(client, messages, max_retries: int = 3):
    delay = 2
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=1000
            )
        except Exception as err:
            last_error = err
            if attempt == max_retries:
                raise
            print(f"  ↻ Retry {attempt}/{max_retries} after error: {err}")
            time.sleep(delay)
            delay = min(delay * 2, 10)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Model call failed: maximum retries reached.")



def _build_prompt_for_image(img_chunk: Dict, index: int) -> str:
    metadata = img_chunk.get('metadata', {}) or {}
    page = metadata.get('page')
    page_hint = f" (page {page})" if page is not None else ""

    chunk_type = (img_chunk.get('type') or 'Image').lower()
    if chunk_type == 'table':
        instruction = (
            f"Image {index}{page_hint}: Extract this table into clean Markdown with clear headers, "
            "units, totals, and any footnotes. Summarize trends after the table."
        )
    elif chunk_type in ('chart', 'diagram', 'graph'):
        instruction = (
            f"Image {index}{page_hint}: Describe this chart/diagram precisely for technical search. "
            "List axes, scales, legends, data relationships, and any annotations or equations."
        )
    else:
        instruction = (
            f"Image {index}{page_hint}: Describe every visible label, printed text, numbers, "
            "sections, UI elements, and spatial relationships. Capture intent and key insights."
        )

    extra = metadata.get('label')
    if extra:
        instruction += f" Context hint: {extra}."

    return instruction + "\nProvide key facts suitable for retrieval."  


def _parse_delimited_response(response_text: str, batch_start: int, batch_size: int) -> Dict[int, str]:
    """Parse response using === IMAGE N === delimiters."""
    parsed: Dict[int, str] = {}
    if not response_text:
        return parsed
    
    # Match === IMAGE N === or ===IMAGE N=== or similar variations
    pattern = re.compile(r"=+\s*IMAGE\s+(\d+)\s*=+", re.IGNORECASE)
    matches = list(pattern.finditer(response_text))
    
    for i, match in enumerate(matches):
        img_num = int(match.group(1))
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(response_text)
        description = response_text[start_pos:end_pos].strip()
        
        # Map to global index
        global_idx = batch_start + img_num
        if 1 <= img_num <= batch_size and description:
            parsed[global_idx] = description
    
    return parsed


def is_meaningful_image(element) -> bool:
    """Filter out tiny logos, decorations, and inline icons."""
    meta = element.metadata
    
    min_dimension = 100
    min_area = 120 * 120
    max_aspect_ratio = 6.0
    width = height = None

    # Skip if too small (likely a logo/icon)
    if hasattr(meta, 'coordinates') and meta.coordinates:
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

    # Skip if embedded in text (likely inline icon)
    parent_id = getattr(meta, 'parent_id', None)
    if parent_id and str(parent_id).startswith("Text"):
        return False

    label = getattr(meta, 'label', '')
    if isinstance(label, str):
        lowered = label.lower()
        for noisy in ("logo", "decoration", "watermark", "background"):
            if noisy in lowered:
                return False

    confidence = getattr(meta, 'confidence', 1.0)
    if isinstance(confidence, (int, float)) and confidence < 0.15:
        return False

    return True


def compress_image(base64_str: str, max_size: int = 1024) -> str:
    """Resize and compress images to reduce token usage."""
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to JPEG with optimization
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format='JPEG', quality=85, optimize=True)
        
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"  Warning: Could not compress image: {e}")
        return base64_str


def deduplicate_images(image_chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate images using MD5 hashing."""
    seen_hashes = {}
    seen_perceptual_buckets: Dict[str, List[str]] = {}
    unique_images = []
    duplicates = 0
    near_duplicates = 0

    for img in image_chunks:
        base64_data = img.get('base64')
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
            is_near_duplicate = False
            for existing_hash in bucket:
                if _hamming_distance(existing_hash, perceptual_hash) <= PHASH_DISTANCE_THRESHOLD:
                    near_duplicates += 1
                    is_near_duplicate = True
                    break
            if is_near_duplicate:
                continue
            bucket.append(perceptual_hash)

        seen_hashes[content_hash] = True
        img.setdefault('metadata', {})
        img['metadata']['content_hash'] = content_hash
        if perceptual_hash:
            img['metadata']['perceptual_hash'] = perceptual_hash
        unique_images.append(img)

    if duplicates > 0:
        print(f"  Removed {duplicates} byte-identical duplicate images")
    if near_duplicates > 0:
        print(f"  Removed {near_duplicates} perceptually similar images (hamming <= {PHASH_DISTANCE_THRESHOLD})")

    return unique_images


def describe_images_batch(image_chunks: List[Dict], client, batch_size: int = 10) -> List[Dict]:
    """Process multiple images per API call and parse responses by Image N labels."""
    global usage_stats
    descriptions: List[Dict] = []

    for i in range(0, len(image_chunks), batch_size):
        batch = image_chunks[i : i + batch_size]

        try:
            user_content = []

            for idx, img in enumerate(batch):
                global_index = i + idx + 1
                prompt = _build_prompt_for_image(img, global_index)

                user_content.append({"type": "text", "text": prompt})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
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
                )
            })

            response = _call_model_with_retry(
                client,
                messages=[{"role": "user", "content": user_content}]
            )

            usage = getattr(response, "usage", None)
            if usage:
                usage_stats["vision_input"] += usage.prompt_tokens
                usage_stats["vision_output"] += usage.completion_tokens
                print(
                    f"  [Batch {i//batch_size + 1}] Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out"
                )

            response_text = response.choices[0].message.content.strip()
            parsed_descriptions = _parse_delimited_response(response_text, i, len(batch))

            for idx, img in enumerate(batch):
                global_index = i + idx + 1
                description_text = parsed_descriptions.get(global_index, "No description available")
                if description_text == "No description available":
                    print(f"  ⚠️ Missing description for image {global_index}, retrying solo...")
                    fallback_desc = describe_single_image(img, client, global_index)
                    if fallback_desc:
                        descriptions.append(fallback_desc)
                        continue
                descriptions.append({
                    'type': 'image_description',
                    'original_type': img['type'],
                    'content': description_text,
                    'metadata': img['metadata'],
                    'index': global_index,
                })
                print(f"  ✓ Image {global_index} described: {description_text[:80]}...")

        except Exception as e:
            print(f"✗ Error processing batch {i//batch_size + 1}: {e}")
            for idx, img in enumerate(batch):
                desc = describe_single_image(img, client, i + idx + 1)
                if desc:
                    descriptions.append(desc)

    return descriptions


def describe_single_image(img_chunk: Dict, client, index: int) -> Dict:
    """Fallback for single image processing."""
    global usage_stats
    try:
        prompt = _build_prompt_for_image(img_chunk, index)

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_chunk['base64']}"}
            },
            {
                "type": "text",
                "text": "Respond ONLY with plain text description followed by bullet key points."
            }
        ]

        response = _call_model_with_retry(client, messages=[{"role": "user", "content": user_content}])
        usage = getattr(response, "usage", None)
        
        if usage:
            usage_stats["vision_input"] += usage.prompt_tokens
            usage_stats["vision_output"] += usage.completion_tokens
            print(f"  [Image {index}] Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
        
        response_text = response.choices[0].message.content.strip()
        return {
            'type': 'image_description',
            'original_type': img_chunk['type'],
            'content': response_text,
            'metadata': img_chunk['metadata'],
            'index': index
        }
    except Exception as e:
        print(f"✗ Error at Image {index}: {e}")
        return None



def get_cached_descriptions(image_chunks: List[Dict], client, cache_file: Optional[str] = None) -> List[Dict]:
    """
    Cache image descriptions to avoid reprocessing.
    Only processes new/changed images.
    """
    cache = {}
    cache_path = _resolve_cache_path(cache_file)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Loaded cache with {len(cache)} entries")
    
    new_images = []
    cached_descriptions = []
    cache_hits = 0
    
    for img in image_chunks:
        # Create hash of image
        img_hash = img.get('metadata', {}).get('content_hash')
        if not img_hash:
            img_hash = _calc_content_hash(img['base64'])

        
        if img_hash in cache:
            cached_descriptions.append(cache[img_hash])
            cache_hits += 1
        else:
            new_images.append((img_hash, img))
    
    print(f"  Cache hits: {cache_hits}, New images: {len(new_images)}")
    
    # Only process new images
    if new_images:
        new_descriptions = describe_images_batch([img for _, img in new_images], client)
        
        # Update cache
        for (img_hash, _), desc in zip(new_images, new_descriptions):
            cache[img_hash] = desc
            cached_descriptions.append(desc)
        
        # Save updated cache
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"  Cache updated with {len(new_images)} new entries")
    
    return cached_descriptions
