"""
Image Processing Functions for RAG Pipeline
- Image filtering and optimization
- Batch processing with Gemini Vision
- Caching and deduplication
"""

import sys
import hashlib
import base64
import json
import io
import pickle
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from google import genai

# Global usage stats
usage_stats = {
    "vision_input": 0,
    "vision_output": 0
}

def is_meaningful_image(element) -> bool:
    """Filter out tiny logos, decorations, and inline icons."""
    meta = element.metadata
    
    # Skip if too small (likely a logo/icon)
    if hasattr(meta, 'coordinates') and meta.coordinates:
        try:
            points = meta.coordinates.points
            width = max(p[0] for p in points) - min(p[0] for p in points)
            height = max(p[1] for p in points) - min(p[1] for p in points)
            
            # Skip images smaller than 100x100 pixels
            if width < 100 or height < 100:
                return False
        except:
            pass
    
    # Skip if embedded in text (likely inline icon)
    parent_id = getattr(meta, 'parent_id', None)
    if parent_id and str(parent_id).startswith("Text"):
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
    unique_images = []
    duplicates = 0
    
    for img in image_chunks:
        # Create hash of the image data
        img_hash = hashlib.md5(img['base64'].encode()).hexdigest()
        
        if img_hash not in seen_hashes:
            seen_hashes[img_hash] = True
            unique_images.append(img)
        else:
            duplicates += 1
    
    if duplicates > 0:
        print(f"  Removed {duplicates} duplicate images")
    
    return unique_images


def describe_images_batch(image_chunks: List[Dict], client, batch_size: int = 10) -> List[Dict]:
    """Process multiple images per API call and parse responses by Image N labels."""
    import re

    global usage_stats
    descriptions: List[Dict] = []

    for i in range(0, len(image_chunks), batch_size):
        batch = image_chunks[i : i + batch_size]

        try:
            parts = []

            for idx, img in enumerate(batch):
                item_num = idx + 1  # 1-based within batch
                if img['type'] == 'Table':
                    prompt = (
                        f"Image {item_num}: Extract this table into clean Markdown format with proper headers "
                        "and describe its content.\n"
                    )
                else:
                    prompt = (
                        f"Image {item_num}: Describe this diagram/image in detail for technical search. Include "
                        "all visible labels, text, relationships, and key elements.\n"
                    )

                parts.append({"text": prompt})
                parts.append({
                    "inline_data": {"mime_type": "image/jpeg", "data": img['base64']}
                })

            parts.append({
                "text": (
                    "\n\nFor each image, provide a detailed description. Format your response as:\n"
                    "Image 1: [description]\nImage 2: [description]\n..."
                )
            })

            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[{"role": "user", "parts": parts}],
            )

            usage = response.usage_metadata
            if usage:
                usage_stats["vision_input"] += usage.prompt_token_count
                usage_stats["vision_output"] += usage.candidates_token_count
                print(
                    f"  [Batch {i//batch_size + 1}] Tokens: {usage.prompt_token_count} in / {usage.candidates_token_count} out"
                )

            response_text = response.text.strip()
            pattern = r"Image\s+(\d+):\s*"
            parts_split = re.split(pattern, response_text)

            parsed_descriptions = {}
            for j in range(1, len(parts_split), 2):
                if j + 1 < len(parts_split):
                    img_num = int(parts_split[j])
                    description = parts_split[j + 1].strip()
                    parsed_descriptions[img_num] = description

            for idx, img in enumerate(batch):
                img_num = idx + 1
                description_text = parsed_descriptions.get(img_num, "No description available")
                descriptions.append({
                    'type': 'image_description',
                    'original_type': img['type'],
                    'content': description_text,
                    'metadata': img['metadata'],
                    'index': i + idx + 1,
                })
                print(f"  ✓ Image {i + idx + 1} described: {description_text[:80]}...")

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
        model_id = "gemini-2.0-flash-lite"
        
        prompt = (
            "Extract this table into a Markdown format. Ensure headers are clear." 
            if img_chunk['type'] == "Table" else 
            "Describe this diagram/image in detail for a technical search engine. Include labels and relationships."
        )

        contents = [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_chunk['base64']}}
            ]
        }]

        response = client.models.generate_content(model=model_id, contents=contents)
        usage = response.usage_metadata
        
        if usage:
            usage_stats["vision_input"] += usage.prompt_token_count
            usage_stats["vision_output"] += usage.candidates_token_count
            print(f"  [Image {index}] Tokens: {usage.prompt_token_count} in / {usage.candidates_token_count} out")
        
        return {
            'type': 'image_description',
            'original_type': img_chunk['type'],
            'content': response.text.strip(),
            'metadata': img_chunk['metadata'],
            'index': index
        }
    except Exception as e:
        print(f"✗ Error at Image {index}: {e}")
        return None


def get_cached_descriptions(image_chunks: List[Dict], client, cache_file: str = "image_cache.pkl") -> List[Dict]:
    """
    Cache image descriptions to avoid reprocessing.
    Only processes new/changed images.
    """
    cache = {}
    if Path(cache_file).exists():
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Loaded cache with {len(cache)} entries")
    
    new_images = []
    cached_descriptions = []
    cache_hits = 0
    
    for img in image_chunks:
        # Create hash of image
        img_hash = hashlib.md5(img['base64'].encode()).hexdigest()
        
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
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"  Cache updated with {len(new_images)} new entries")
    
    return cached_descriptions
