"""
Text Processing Functions for RAG Pipeline
- PDF element extraction and processing
- Smart chunking and text combination
- Content cleaning and optimization
"""

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Tuple, Optional

def extract_pdf_elements(pdf_path: str) -> List[Any]:
    """Extract all elements from PDF including images and tables."""
    try:
        raw_pdf_elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            skip_infer_table_structure=False,
            model_name="yolox"
        )
        return raw_pdf_elements
    except Exception as e:
        print(f"❌ Unstructured partition failed: {e}")
        # Fallback to simple extraction if hi_res fails
        try:
            raw_pdf_elements = partition_pdf(filename=pdf_path, strategy="fast")
            return raw_pdf_elements
        except Exception as fallback_err:
            print(f"❌ Fallback partition also failed: {fallback_err}")
            return []


def process_extracted_elements_smart(raw_elements: List[Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process elements intelligently:
    - Use text extraction for tables when available
    - Only process meaningful images
    - Filter out decorative elements
    """
    from .image_processing import is_meaningful_image
    
    text_chunks = []
    image_chunks = []

    for element in raw_elements:
        element_type = element.__class__.__name__
        
        try:
            # ---------- TABLE ----------
            if element_type == "Table":
                # Tables already extracted as text by unstructured
                # Only use vision if text extraction failed
                if element.text and len(element.text.strip()) > 50:
                    metadata_dict = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}
                    text_chunks.append({
                        "type": "Table",
                        "content": element.text,
                        "metadata": metadata_dict
                    })
                else:
                    # Vision fallback for complex tables
                    image_base64 = getattr(element.metadata, "image_base64", None)
                    if image_base64:
                        metadata_dict = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}
                        image_chunks.append({
                            "type": "Table",
                            "base64": image_base64,
                            "metadata": metadata_dict
                        })

            # ---------- IMAGE ----------
            elif element_type == "Image":
                # Only process meaningful images
                if is_meaningful_image(element):
                    image_base64 = getattr(element.metadata, "image_base64", None)
                    if image_base64:
                        metadata_dict = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}
                        image_chunks.append({
                            "type": "Image",
                            "base64": image_base64,
                            "metadata": metadata_dict
                        })

            # ---------- TEXT ----------
            else:
                if hasattr(element, "text") and element.text.strip():
                    content = element.text.strip()
                    if content.isdigit() and len(content) <= 4:
                        continue

                    has_alpha = any(ch.isalpha() for ch in content)
                    if not has_alpha and len(content) < 15:
                        continue

                    metadata_dict = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}
                    text_chunks.append({
                        "type": element_type,
                        "content": content,
                        "metadata": metadata_dict
                    })
        
        except Exception as e:
            print(f"❌ Error processing element {element_type}: {e}")
            # Skip problematic element but continue processing
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
