"""S3 client and helpers used by the application."""

from .s3_client import BUCKET_NAME, s3, upload_file_to_s3

__all__ = ["BUCKET_NAME", "s3", "upload_file_to_s3"]
