import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from openai import OpenAI

from agent.agent import RAGAgent
from config.firebase import get_current_user_id, get_firestore
from config.s3 import upload_file_to_s3
from rag.image_processing import (
    compress_image,
    deduplicate_images,
    get_cached_descriptions,
)
from rag.text_processing import (
    combine_text_and_images,
    extract_pdf_elements,
    process_extracted_elements_smart,
    smart_chunk,
)

logger = logging.getLogger(__name__)
router = APIRouter()
db = get_firestore()
openai_client = OpenAI()

MAX_PAGES = 50
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
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


def get_namespace(user_id: str, folder_id: str | None) -> str:
    base = f"user_{user_id}"
    return f"{base}_folder_{folder_id}" if folder_id else base


def get_owner_ref(user_id: str, folder_id: str | None):
    """Firestore doc that owns the pdfs subcollection (user or folder)."""
    user_ref = db.collection("users").document(user_id)
    if folder_id:
        return user_ref.collection("folders").document(folder_id)
    return user_ref


def count_pages(elements) -> int:
    pages = {e["page"] for e in elements if isinstance(e, dict) and "page" in e}
    return len(pages) or 1


def build_chunks(elements) -> list[dict]:
    text_chunks, image_chunks = process_extracted_elements_smart(elements)

    if image_chunks:
        images = [
            {**img, "base64": compress_image(img["base64"])}
            for img in deduplicate_images(image_chunks)
        ]
        descriptions = get_cached_descriptions(images, openai_client)
        text_chunks = combine_text_and_images(text_chunks, descriptions)

    return smart_chunk(text_chunks, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


def upload_to_s3(path: str, key: str) -> None:
    with open(path, "rb") as f:
        result = upload_file_to_s3(file_obj=f, file_key=key)
    if not result.get("success"):
        raise HTTPException(500, result.get("error") or "S3 upload failed")


# ---------- route ----------

@router.post("/upload-pdf")
def upload_pdf(  # plain def -> runs in threadpool, doesn't block the event loop
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    folder_id: str | None = Header(None, convert_underscores=False),
):
    pdf_id = str(uuid.uuid4())
    s3_key = f"pdfs/{user_id}/{pdf_id}.pdf"
    file_name = file.filename
    base_response = {"pdf_id": pdf_id, "s3_key": s3_key, "file_name": file_name}

    try:
        with temp_pdf(file) as path:
            elements = extract_pdf_elements(path)
            page_count = count_pages(elements)

            if page_count > MAX_PAGES:
                return {
                    "success": False,
                    "error": f"PDF too large: {page_count} pages. Maximum allowed is {MAX_PAGES} pages.",
                    "page_count": page_count,
                    "limit": MAX_PAGES,
                    **base_response,
                }

            upload_to_s3(path, s3_key)

        chunks = build_chunks(elements)
        namespace = get_namespace(user_id, folder_id)

        if not RAGAgent(namespace=namespace).add_documents(chunks, s3_key=s3_key):
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
