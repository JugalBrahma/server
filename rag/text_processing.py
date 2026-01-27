"""
Text Processing Functions for RAG Pipeline
- PDF element extraction and processing
- Smart chunking and text combination
- Content cleaning and optimization
"""

import os
import base64
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Tuple, Optional

def extract_pdf_elements(pdf_path: str) -> List[Any]:
    """Extract all elements from PDF including images and tables using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        extracted_elements: List[Dict[str, Any]] = []
        images_dir = f"{pdf_path}_images"
        os.makedirs(images_dir, exist_ok=True)

        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            if text.strip():
                extracted_elements.append({
                    "type": "text",
                    "page": page_index + 1,
                    "text": text,
                })

            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image.get("image")
                ext = base_image.get("ext", "png")
                if not img_bytes:
                    continue
                image_filename = f"page{page_index + 1}_img{img_index + 1}.{ext}"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, "wb") as image_file:
                    image_file.write(img_bytes)

                extracted_elements.append({
                    "type": "image",
                    "page": page_index + 1,
                    "image_path": image_path,
                })

        doc.close()
        return extracted_elements
    except Exception as e:
        print(f"❌ PyMuPDF extraction failed: {e}")
        return []


def process_extracted_elements_smart(raw_elements: List[Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process elements intelligently from PyMuPDF extraction:
    - Extract text and images
    - Filter out decorative elements
    """
    text_chunks = []
    image_chunks = []
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    for element in raw_elements:
        if not isinstance(element, dict):
            continue
            
        element_type = element.get("type", "")
        
        try:
            # ---------- TEXT ----------
            if element_type == "text":
                text = element.get("text", "").strip()
                if not text:
                    continue
                    
                # Filter out meaningless text
                if text.isdigit() and len(text) <= 4:
                    continue
                    
                has_alpha = any(ch.isalpha() for ch in text)
                if not has_alpha and len(text) < 15:
                    continue
                
                # Split text into chunks
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 10:
                        text_chunks.append({
                            "type": "Text",
                            "content": chunk.lower(),  # Convert to lowercase
                            "metadata": {
                                "page": element.get("page", 1),
                                "chunk_index": i
                            }
                        })
            
            # ---------- IMAGE ----------
            elif element_type == "image":
                image_path = element.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    try:
                        with open(image_path, "rb") as img_file:
                            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    except Exception as img_err:
                        print(f"❌ Could not read image {image_path}: {img_err}")
                        continue

                    print(f"🖼️ Extracted image from page {element.get('page', 1)} -> {image_path}")
                    image_chunks.append({
                        "type": "Image",
                        "image_path": image_path,
                        "base64": image_base64,
                        "metadata": {
                            "page": element.get("page", 1)
                        }
                    })
                    print(f"   ↳ Image chunk #{len(image_chunks)} ready (size: {len(image_base64)} base64 chars)")
        
        except Exception as e:
            print(f"❌ Error processing element {element_type}: {e}")
            continue

    return text_chunks, image_chunks


def combine_text_and_images(text_chunks: List[Dict], image_descriptions: List[Dict]) -> List[Dict]:
    """
    Link images to their parent text using parent_id metadata.
    Preserves context better than proximity-based methods.
    """
    # Create lookup map
    text_map = {c['metadata'].get('element_id'): c for c in text_chunks if c.get('metadata') and c['metadata'].get('element_id')}
    combined_chunks = []
    processed_image_indices = set()

    # Attach images to their direct parents
    for idx, img in enumerate(image_descriptions):
        parent_id = img['metadata'].get('parent_id') if img.get('metadata') else None
        if parent_id and parent_id in text_map:
            parent_text = text_map[parent_id]
            parent_text['content'] += f"\n\n[Image Context]: {img['content']}"
            parent_text['has_image'] = True
            processed_image_indices.add(idx)

    # Handle orphan images
    for idx, img in enumerate(image_descriptions):
        if idx not in processed_image_indices:
            combined_chunks.append({
                'type': 'image_only',
                'content': f"[Image Only]\n{img['content']}",
                'has_image': True,
                'metadata': img.get('metadata', {})
            })

    # Add all text chunks
    for t in text_chunks:
        combined_chunks.append(t)

    return combined_chunks


def smart_chunk(combined_chunks: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Intelligent chunking that preserves image-text relationships.
    Doesn't split chunks that contain image descriptions.
    """
    final_chunks = []
    
    for chunk in combined_chunks:
        # Clean metadata (remove binary data and unwanted fields)
        if 'metadata' in chunk:
            chunk['metadata'] = {
                k: v for k, v in chunk['metadata'].items() 
                if k not in ['image_base64', 'coordinates', 'image_payload', 'has_image', 'image_mime_type']
            }
        
        content_len = len(chunk['content'])
        
        # Keep image-enriched chunks together (don't split)
        # Also keep short chunks intact
        if chunk.get('has_image') or content_len < chunk_size:
            # Only add if meaningful (not tiny fragments)
            if len(chunk['content'].strip()) > 10:
                final_chunks.append(chunk)
        else:
            # Only split pure text chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            splits = splitter.split_text(chunk['content'])
            
            for split in splits:
                if len(split.strip()) > 10:  # Skip tiny fragments
                    final_chunks.append({
                        **chunk,
                        'content': split
                    })
    
    return final_chunks


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    import re
    
    # Remove hyphenated line breaks
    text = re.sub(r"-\s*\n", "", text)
    
    # Replace newlines with space
    text = re.sub(r"\n+", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove non-ASCII garbage (OCR noise)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    
    return text.strip()
