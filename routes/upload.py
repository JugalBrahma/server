import uuid
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Depends
from datetime import datetime
from config.firebase import get_current_user_id
from config.s3 import upload_file_to_s3
from config.firebase import get_firestore
from rag.text_processing import (
    extract_pdf_elements,
    process_extracted_elements_smart,
    combine_text_and_images,
    smart_chunk,
)
from rag.image_processing import (
    deduplicate_images,
    compress_image,
    get_cached_descriptions,
    usage_stats,
)
from openai import OpenAI
from agent.agent import RAGAgent

router = APIRouter()

db = get_firestore()


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    folder_id: str = Header(None, convert_underscores=False)
):
    """
    Upload PDF to S3 and store metadata in Firestore
    """
    
    # 2️⃣ Save file temporarily - Store uploaded file on server temporarily
    pdf_id = str(uuid.uuid4())  # Generate unique ID for this PDF
    s3_key = f"pdfs/{user_id}/{pdf_id}.pdf"  # S3 path where file will be stored

    original_filename = file.filename  # Keep original filename for display

    # Create temporary file on server
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()  # Read uploaded file content
        tmp.write(contents)  # Write to temporary file
        temp_path = tmp.name  # Get temporary file path

    try:
        # 3️⃣ Upload to S3 - Store the actual PDF file in cloud storage
        # This uploads the file to S3 at the path: pdfs/{user_id}/{pdf_id}.pdf
        with open(temp_path, "rb") as f:
            upload_result = upload_file_to_s3(
                file_obj=f,           # The PDF file content
                file_key=s3_key       # S3 path: "pdfs/user123/abc-123.pdf"
            )

        # Check if S3 upload was successful
        if not upload_result.get("success"):
            raise HTTPException(status_code=500, detail=upload_result.get("error") or "S3 upload failed")

        # 4️⃣ Download from S3 and extract text (S3-Connected RAG)
        print(f"Downloading from S3: {s3_key}")
        from config.s3 import s3, BUCKET_NAME
        
        try:
            # Download the uploaded file from S3
            s3_response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            pdf_bytes = s3_response["Body"].read()
            
            # Save downloaded bytes to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as download_tmp:
                download_tmp.write(pdf_bytes)
                download_tmp_path = download_tmp.name
            
            try:
                # Extract text from S3-downloaded file using RAG pipeline
                print(f"Extracting text from S3 file: {download_tmp_path}")
                elements = extract_pdf_elements(download_tmp_path)
                text_chunks, image_chunks = process_extracted_elements_smart(elements)
                
                # Helper function to safely get page number from metadata
                def get_page_number(elem):
                    if not hasattr(elem, 'metadata'):
                        return 1
                    metadata = elem.metadata
                    
                    # Try attribute access first (for objects like Title)
                    if hasattr(metadata, 'page_number'):
                        return metadata.page_number
                    
                    # Try dictionary access (for dict-like metadata)
                    if isinstance(metadata, dict):
                        return metadata.get('page_number', 1)
                    
                    # Try to_dict() method for ElementMetadata objects
                    if hasattr(metadata, 'to_dict'):
                        metadata_dict = metadata.to_dict()
                        return metadata_dict.get('page_number', 1)
                    
                    return 1
                
                # Check page limit
                page_count = len(set([get_page_number(elem) for elem in elements]))
                if page_count > 50:
                    print(f"⚠️ PDF has {page_count} pages (limit: 50)")
                    extraction = {
                        "method": "page_limit_exceeded",
                        "page_count": page_count,
                        "limit": 50,
                        "s3_connected": True,
                        "vectorization": {"success": False, "error": f"PDF too large: {page_count} pages. Maximum allowed is 50 pages."}
                    }
                else:
                    # FULL RAG PIPELINE (text + image processing)
                    print(f"\n[Pipeline] Found {len(text_chunks)} text chunks, {len(image_chunks)} images")
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    combined_chunks = text_chunks

                    if len(image_chunks) > 0:
                        print("\n[Pipeline] Deduplicating images...")
                        unique_images = deduplicate_images(image_chunks)
                        print(f"  ✓ Kept {len(unique_images)} unique images")

                        print("\n[Pipeline] Compressing images...")
                        compressed_images = [
                            {**img, "base64": compress_image(img["base64"])}
                            for img in unique_images
                        ]
                        print(f"  ✓ Compressed {len(compressed_images)} images")

                        print("\n[Pipeline] Describing images with OpenAI Vision...")
                        image_descriptions = get_cached_descriptions(compressed_images, client)
                        print(f"  ✓ Got {len(image_descriptions)} descriptions")
                        total_vision_tokens = usage_stats["vision_input"] + usage_stats["vision_output"]
                        print(
                            f"    Vision tokens used: {total_vision_tokens:,} (in: {usage_stats['vision_input']:,}, out: {usage_stats['vision_output']:,})"
                        )

                        print("\n[Pipeline] Combining text + images...")
                        combined_chunks = combine_text_and_images(text_chunks, image_descriptions)
                        print(
                            f"  ✓ Created {len(combined_chunks)} combined chunks (includes {len(image_descriptions)} image descriptions)"
                        )
                    else:
                        print("\n[Pipeline] ⚠️ No images detected, proceeding with text-only chunks")

                    print("\n[Pipeline] Smart chunking...")
                    final_chunks = smart_chunk(combined_chunks, chunk_size=1000, overlap=200)
                    print(f"  ✓ Split into {len(final_chunks)} final chunks")

                    # VECTORIZE AND STORE IN PINECONE
                    print(f"\nVectorizing {len(final_chunks)} chunks...")
                    try:
                        # Create namespace per folder (separate embeddings per folder)
                        if folder_id:
                            namespace = f"user_{user_id}_folder_{folder_id}"
                        else:
                            namespace = f"user_{user_id}"
                        
                        # Initialize RAG agent with folder-specific namespace
                        rag_agent = RAGAgent(namespace=namespace)
                        
                        # Add enriched chunks to Pinecone vector database
                        success = rag_agent.add_documents(final_chunks, s3_key=s3_key)
                        
                        if success:
                            print(f"✅ Successfully vectorized {len(final_chunks)} chunks in namespace: {namespace}")
                            vectorization_result = {
                                "success": True,
                                "namespace": namespace,
                                "chunks_vectorized": len(final_chunks),
                                "vectorization_method": "pinecone"
                            }
                        else:
                            print(f"❌ Failed to vectorize {len(final_chunks)} chunks")
                            vectorization_result = {
                                "success": False,
                                "error": "Vectorization failed - check logs",
                                "namespace": namespace,
                                "chunks_vectorized": 0
                            }
                            
                    except Exception as vector_error:
                        print(f"❌ Vectorization failed: {vector_error}")
                        vectorization_result = {
                            "success": False,
                            "error": str(vector_error),
                            "namespace": namespace,
                            "chunks_vectorized": 0
                        }
                    
                    # Create extraction result for successful processing
                    extraction = {
                        "method": "rag_pipeline_s3_connected_with_images",
                        "text_chunks": sum(1 for c in final_chunks if not c.get("has_image")),
                        "image_chunks": sum(1 for c in final_chunks if c.get("has_image")),
                        "total_elements": len(final_chunks),
                        "s3_key": s3_key,
                        "s3_connected": True,
                        "vectorization": {
                            **vectorization_result,
                            "vision_tokens": usage_stats["vision_input"] + usage_stats["vision_output"],
                        },
                    }
                
                # Store metadata regardless of page limit (so user knows what happened)
                print(f"S3-Connected extraction result: {extraction}")
                
                # Check if page limit was exceeded and return error
                if extraction.get("method") == "page_limit_exceeded":
                    # Return error response for large PDF
                    return {
                        "success": False,
                        "error": f"PDF too large: {extraction['page_count']} pages. Maximum allowed is 50 pages.",
                        "page_count": extraction['page_count'],
                        "limit": 50,
                        "pdf_id": pdf_id,
                        "s3_key": s3_key,
                        "file_name": original_filename
                    }
            
            except Exception as e:
                print(f"❌ Error processing S3 PDF: {e}")
                return {
                    "success": False,
                    "error": f"Failed to process PDF: {str(e)}",
                    "pdf_id": pdf_id,
                    "s3_key": s3_key,
                    "file_name": original_filename
                }
        
        finally:
            # Cleanup download temp file
            os.unlink(download_tmp_path)
        
        # Store metadata in Firestore (Database)
        # This stores information ABOUT the file, not the file itself
        # Firestore acts as an index/directory for all files
        metadata = {
            "pdf_id": pdf_id,                    # Unique identifier for this PDF
            "file_name": original_filename,      # Original filename from user
            "s3_key": s3_key,                    # Path to file in S3 storage
            "user_id": user_id,                  # Who owns this file
            "extraction_method": extraction.get("method"),  # How text was extracted
            "text_chunks": extraction.get("text_chunks", 0),    # Number of text chunks
            "image_chunks": extraction.get("image_chunks", 0),  # Number of image chunks
            "total_elements": extraction.get("total_elements", 0),  # Total elements extracted
            "page_count": extraction.get("page_count", 0),  # Number of pages
            "page_limit": extraction.get("limit", 50),  # Page limit applied
            "s3_connected": extraction.get("s3_connected", False),  # S3 connection status
            "vectorization": extraction.get("vectorization", {}),  # Vectorization results
            "uploaded_at": datetime.utcnow(),    # When file was uploaded
        }

        print(f"Saving metadata for user: {user_id}, folder_id: {folder_id}, pdf_id: {pdf_id}")
        try:
            # Save to folder subcollection if folder_id provided, otherwise save to user root
            if folder_id:
                # FOLDER STRUCTURE: Store PDF inside a specific folder
                # Firestore path: users/{user_id}/folders/{folder_id}/pdfs/{pdf_id}
                print("Storing in folder structure...")
                
                # Update folder's last modified timestamp
                db.collection("users").document(user_id).collection("folders").document(folder_id).set(
                    {
                        "folder_id": folder_id,
                        "updated_at": datetime.utcnow(),
                    },
                    merge=True,
                )
                
                # Store PDF metadata inside the folder
                db.collection("users").document(user_id).collection("folders").document(folder_id).collection("pdfs").document(pdf_id).set(metadata)
                print(f"✅ PDF metadata saved to folder {folder_id}")
            else:
                # ROOT STRUCTURE: Store PDF directly under user (no folder)
                # Firestore path: users/{user_id}/pdfs/{pdf_id}
                print("Storing in user root (no folder)...")
                
                # Update user's last modified timestamp
                db.collection("users").document(user_id).set(
                    {
                        "user_id": user_id,
                        "updated_at": datetime.utcnow(),
                    },
                    merge=True,
                )
                
                # Store PDF metadata at user root level
                db.collection("users").document(user_id).collection("pdfs").document(pdf_id).set(metadata)
                print(f"✅ PDF metadata saved to user root")
            print("🎉 Metadata saved successfully")
        except Exception as e:
            print(f"Firestore save failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Firestore save failed: {str(e)}")
    
    finally:
        # Clean up temporary file - Remove file from server storage
        # This is important to free up server disk space
        os.remove(temp_path)
        print(f"🧹 Temporary file cleaned up: {temp_path}")

    # Return success response to Flutter app
    return {
        "pdf_id": pdf_id,           # Unique identifier for the uploaded PDF
        "s3_key": s3_key,           # Path where file is stored in S3
        "file_name": original_filename,  # Original filename from user
        "extraction_method": extraction.get("method"),  # How text was extracted
        "text_length": extraction.get("length"),  # Length of extracted text
        "message": "PDF uploaded successfully"
    }
