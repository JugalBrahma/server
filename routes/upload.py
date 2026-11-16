import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone

import pymupdf
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from openai import OpenAI
from redis import Redis
from agent.agent import RAGAgent
from agent.vector_manager import folder_key, user_namespace
from config.firebase import get_current_user_id, get_firestore
from config.s3 import upload_file_to_s3
from rag.ingestion import prepare_chunks
from rag.text_processing import extract_pdf_elements

logger = logging.getLogger(__name__)
router = APIRouter()
db = get_firestore()
openai_client = OpenAI()
MAX_PAGES = 50
EXTRACTION_METHOD = "rag_pipeline_s3_connected_with_images"


# ---------- helpers ----------

@contextmanager
def temp_pdf(upload: UploadFile):
    """Stream an upload to disk and always clean it up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(upload.file, tmp)
        path = tmp.name
    try:
        yield path
    finally:
        with suppress(FileNotFoundError):
            os.remove(path)


def now() -> datetime:
    return datetime.now(timezone.utc)


def get_owner_ref(user_id: str, folder_id: str | None):
    """Firestore doc that owns the pdfs subcollection (user or folder)."""
    user_ref = db.collection("users").document(user_id)
    if folder_id:
        return user_ref.collection("folders").document(folder_id)
    return user_ref


def count_pages(path: str) -> int:
    """Read the page count from the file itself, without extracting any content."""
    with pymupdf.open(path) as doc:
        return doc.page_count


def upload_to_s3(path: str, key: str) -> None:
    with open(path, "rb") as f:
        result = upload_file_to_s3(file_obj=f, file_key=key)
    if not result.get("success"):
        raise HTTPException(500, result.get("error") or "S3 upload failed")

# ---------- rate limiting ----------

# Normal (sync) client, because upload_pdf is a plain def and can't await.
rate_redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))


def check_rate_limit(key: str, limit: int, window_seconds: int, message: str) -> None:
    try:
        count = rate_redis.incr(key)
        if count == 1:
            rate_redis.expire(key, window_seconds)
        if count > limit:
            ttl = rate_redis.ttl(key)
            raise HTTPException(status_code=429, detail=f"{message} Try again in {ttl}s.")
    except HTTPException:
        raise
    except Exception:
        logger.warning("Rate limiter unavailable, allowing request")


# ---------- route ----------

@router.post("/upload-pdf")
def upload_pdf(  # plain def -> runs in threadpool, doesn't block the event loop
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    folder_id: str | None = Header(None, convert_underscores=False),
):
    check_rate_limit("ratelimit:upload:global", 20, 60, "Server is busy.")

    pdf_id = str(uuid.uuid4())
    s3_key = f"pdfs/{user_id}/{pdf_id}.pdf"
    file_name = file.filename
    base_response = {"pdf_id": pdf_id, "s3_key": s3_key, "file_name": file_name}

    try:
        with temp_pdf(file) as path:
            
            try:
                page_count = count_pages(path)
            except Exception:
                return {"success": False, "error": "This file is not a valid PDF.", **base_response}

            if page_count > MAX_PAGES:
                return {
                    "success": False,
                    "error": f"PDF too large: {page_count} pages. Maximum allowed is {MAX_PAGES} pages.",
                    "page_count": page_count,
                    "limit": MAX_PAGES,
                    **base_response,
                }

            elements = extract_pdf_elements(path)
            upload_to_s3(path, s3_key)

        chunks = prepare_chunks(elements, client=openai_client)
        namespace = user_namespace(user_id)

        added = RAGAgent(namespace=namespace).add_documents(
            chunks, s3_key=s3_key, file_name=file_name, folder_id=folder_id
        )
        if not added:
            raise HTTPException(502, "Vectorization failed")

        image_count = sum(1 for c in chunks if c.get("has_image"))

        metadata = {
            **base_response,
            "user_id": user_id,
            "extraction_method": EXTRACTION_METHOD,
            "text_chunks": len(chunks) - image_count,
            "image_chunks": image_count,
            "total_elements": len(chunks),
            "page_count": page_count,
            "page_limit": MAX_PAGES,
            "s3_connected": True,
            "vectorization": {
                "success": True,
                "namespace": namespace,
                "folder_id": folder_key(folder_id),
                "chunks_vectorized": len(chunks),
                "vectorization_method": "pinecone",
            },
            "uploaded_at": now(),
        }

        owner_ref = get_owner_ref(user_id, folder_id)
        owner_ref.set({"updated_at": now()}, merge=True)
        owner_ref.collection("pdfs").document(pdf_id).set(metadata)

        logger.info("PDF %s uploaded for user %s", pdf_id, user_id)
        return {
            **base_response,
            "extraction_method": EXTRACTION_METHOD,
            "text_length": len(chunks),
            "message": "PDF uploaded successfully",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process PDF upload")
        return {"success": False, "error": f"Failed to process PDF: {exc}", **base_response}
