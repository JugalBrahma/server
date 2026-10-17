"""
Configuration Module
- S3 and Firebase configurations
- Environment variables setup
"""

from .s3 import s3, BUCKET_NAME, upload_file_to_s3, download_pdf_from_s3
from .firebase import get_firestore, verify_firebase_token, get_current_user_id
from .langsmith import init_langsmith
from .models import (
    get_groq_llm,
    get_openai_llm,
    get_rag_llm,
    get_guardrail_llm,
    get_vision_llm,
    get_evaluator_llm,
)

__all__ = [
    "s3",
    "BUCKET_NAME", 
    "upload_file_to_s3",
    "download_pdf_from_s3",
    "get_firestore",
    "verify_firebase_token",
    "get_current_user_id",
    "init_langsmith",
    "get_groq_llm",
    "get_openai_llm",
    "get_rag_llm",
    "get_guardrail_llm",
    "get_vision_llm",
    "get_evaluator_llm",
]
