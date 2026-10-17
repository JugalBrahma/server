import logging
from fastapi import APIRouter, Depends, Header, HTTPException

from agent.vector_manager import VectorSearchManager
from config.firebase import get_current_user_id, get_firestore
from config.s3 import BUCKET_NAME, s3

logger = logging.getLogger(__name__)

router = APIRouter()
db = get_firestore()


@router.delete("/pdf/{pdf_id}")
async def delete_pdf(
    pdf_id: str,
    user_id: str = Depends(get_current_user_id),
    folder_id: str = Header(None, convert_underscores=False),
):
    # Locate the PDF document in Firestore
    if folder_id:
        doc_ref = (
            db.collection("users")
            .document(user_id)
            .collection("folders")
            .document(folder_id)
            .collection("pdfs")
            .document(pdf_id)
        )
    else:
        doc_ref = (
            db.collection("users")
            .document(user_id)
            .collection("pdfs")
            .document(pdf_id)
        )

    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="PDF not found")

    data = doc.to_dict() or {}
    s3_key = data.get("s3_key")
    if not s3_key:
        raise HTTPException(status_code=400, detail="Missing s3_key for PDF")

    # Delete the PDF file from S3
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info("Deleted S3 object: %s", s3_key)
    except Exception as exc:
        logger.exception("Failed to delete S3 object %s: %s", s3_key, exc)
        raise HTTPException(status_code=500, detail=f"S3 delete failed: {str(exc)}")

    # Delete the document from Firestore
    doc_ref.delete()
    logger.info("Deleted Firestore document for PDF: %s", pdf_id)

    # Delete vector embeddings from Pinecone
    try:
        namespace = f"user_{user_id}_folder_{folder_id}" if folder_id else f"user_{user_id}"
        vector_manager = VectorSearchManager(index_name="askadocument", namespace=namespace)
        vector_manager.delete_documents_by_s3_key(s3_key)
        logger.info("Deleted vectors for s3_key %s in namespace %s", s3_key, namespace)
    except Exception as exc:
        logger.warning("Failed to delete vectors from Pinecone: %s", exc)

    return {"message": "PDF deleted successfully"}
