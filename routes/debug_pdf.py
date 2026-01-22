from fastapi import APIRouter, Query, HTTPException
import os
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError
from config.s3 import s3, BUCKET_NAME
from rag.text_processing import extract_pdf_elements, process_extracted_elements_smart

router = APIRouter()

@router.get("/extract-pdf")
def extract_pdf_text(s3_key: str = Query(...)):
    try:
        # Check if S3 key exists first
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise HTTPException(
                    status_code=404, 
                    detail=f"PDF file not found in S3: {s3_key}"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"S3 error: {e.response['Error']['Message']}"
                )
        
        # Get the PDF object
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        pdf_bytes = obj["Body"].read()

        # Save to temp file and process with RAG
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        try:
            elements = extract_pdf_elements(tmp_path)
            text_chunks, image_chunks = process_extracted_elements_smart(elements)
            
            result = {
                "method": "rag_pipeline",
                "text_chunks": len(text_chunks),
                "image_chunks": len(image_chunks),
                "total_elements": len(elements)
            }
        finally:
            os.unlink(tmp_path)  # Cleanup temp file

        return {
            "method": result["method"],
            "text_chunks": result["text_chunks"],
            "image_chunks": result["image_chunks"],
            "total_elements": result["total_elements"]
        }

    except NoCredentialsError:
        raise HTTPException(
            status_code=500, 
            detail="AWS credentials not configured properly"
        )
    except ClientError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"S3 error: {e.response['Error']['Message']}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )
