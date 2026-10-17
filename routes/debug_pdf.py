import logging
import os
import tempfile

from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import APIRouter, HTTPException, Query

from config.s3 import BUCKET_NAME, s3
from rag.text_processing import extract_pdf_elements, process_extracted_elements_smart

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/extract-pdf")
def extract_pdf_text(s3_key: str = Query(...)):
    """Extract and analyze text and image elements from an S3-stored PDF."""
    try:
        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            pdf_bytes = obj["Body"].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                raise HTTPException(status_code=404, detail=f"PDF file not found in S3: {s3_key}")
            raise HTTPException(
                status_code=500,
                detail=f"S3 error: {e.response.get('Error', {}).get('Message')}",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            elements = extract_pdf_elements(tmp_path)
            text_chunks, image_chunks = process_extracted_elements_smart(elements)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return {
            "method": "rag_pipeline",
            "text_chunks": len(text_chunks),
            "image_chunks": len(image_chunks),
            "total_elements": len(elements),
        }

    except HTTPException:
        raise
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not configured properly")
    except Exception as e:
        logger.exception("Failed to extract PDF from %s: %s", s3_key, e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
