"""
Simple PDF Text Extractor
- Basic PDF text extraction utilities
- Clean and normalize text content
"""

import re

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove hyphenated line breaks
    text = re.sub(r"-\s*\n", "", text)
    
    # Replace newlines with space
    text = re.sub(r"\n+", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove non-ASCII garbage (OCR noise)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    
    return text.strip()
