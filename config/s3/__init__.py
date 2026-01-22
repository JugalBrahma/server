"""
S3 Configuration Module
- AWS S3 client setup and configuration
- Environment variables validation
"""

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

# Import functions from s3_client.py
from .s3_client import upload_file_to_s3, download_pdf_from_s3

def get_s3_client():
    """Get the S3 client instance"""
    return s3

def get_bucket_name():
    """Get the bucket name"""
    return BUCKET_NAME
