import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables with validation
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("AWS_S3_BUCKET")

# Validate environment variables
if not all([aws_access_key_id, aws_secret_access_key, aws_region, bucket_name]):
    raise ValueError("Missing AWS environment variables. Please check your .env file.")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

BUCKET_NAME = bucket_name

def upload_file_to_s3(file_obj, file_key):
    """Upload file to S3 and return URL"""
    try:
        s3.upload_fileobj(
            file_obj,
            BUCKET_NAME,
            file_key,
            ExtraArgs={
                "ContentType": "application/pdf"
            }
        )
        
        # Generate public URL
        file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_key}"
        
        return {
            "success": True,
            "file_key": file_key,
            "url": file_url
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def download_pdf_from_s3(bucket: str, key: str):
    """Download PDF from S3 and return as bytes buffer"""
    try:
        import io
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise Exception(f"Failed to download from S3: {str(e)}")

def extract_text_from_pdf(pdf_buffer):
    """Extract text from PDF buffer"""
    try:
        import PyPDF2
        from PyPDF2 import PdfReader
        
        reader = PdfReader(pdf_buffer)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        return text
    except Exception as e:
        raise Exception(f"Failed to extract PDF text: {str(e)}")
