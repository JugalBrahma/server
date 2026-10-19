"""S3 client and helpers used by the application."""

from .s3_client import BUCKET_NAME, s3, upload_file_to_s3


def get_s3_client():
    """Get the S3 client instance"""
    return s3


def get_bucket_name():
    """Get the bucket name"""
    return BUCKET_NAME
