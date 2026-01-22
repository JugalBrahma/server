from fastapi import APIRouter, Header, HTTPException, Depends
from config.firebase import get_current_user_id
from config.firebase import get_firestore
from config.s3 import s3, BUCKET_NAME
from agent.agent import RAGAgent

router = APIRouter()
db = get_firestore()


@router.delete("/pdf/{pdf_id}")
async def delete_pdf(
    pdf_id: str, 
    user_id: str = Depends(get_current_user_id),
    folder_id: str = Header(None, convert_underscores=False)
):
    # 2️⃣ Get PDF metadata - check folder first, then user root
    if folder_id:
        # Look in folder subcollection
        doc_ref = db.collection("users") \
            .document(user_id) \
            .collection("folders") \
            .document(folder_id) \
            .collection("pdfs") \
            .document(pdf_id)
    else:
        # Look in user root (fallback for old uploads)
        doc_ref = db.collection("users") \
            .document(user_id) \
            .collection("pdfs") \
            .document(pdf_id)

    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="PDF not found")

    data = doc.to_dict()
    s3_key = data["s3_key"]

    print(f"Deleting PDF: {pdf_id}, s3_key: {s3_key}, folder_id: {folder_id}")

    # 3️⃣ Delete from S3
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        print(f"Deleted S3 object: {s3_key}")
    except Exception as e:
        print(f"Failed to delete S3 object {s3_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3 delete failed: {str(e)}")

    # 4️⃣ Delete Firestore metadata
    doc_ref.delete()
    print(f"Deleted Firestore PDF: {pdf_id}")

    # 🆕 5️⃣ Delete from Pinecone vector store
    try:
        # Create namespace for this FOLDER (separate memory per folder)
        if folder_id:
            # Use folder-specific namespace for folder isolation
            namespace = f"user_{user_id}_folder_{folder_id}"
        else:
            # Fallback to user-wide namespace if no folder specified
            namespace = f"user_{user_id}"
        
        # Initialize RAG agent with folder-specific namespace
        rag_agent = RAGAgent(namespace=namespace)
        
        # Get Pinecone index directly
        from agent.vector_manager import VectorSearchManager
        vector_manager = VectorSearchManager(index_name="askadocument", namespace=namespace)
        
        # Delete only vectors with matching s3_key in metadata
        try:
            # Use Pinecone's delete with metadata filter
            index = vector_manager.pc.Index("askadocument")
            
            # Delete vectors where s3_key metadata matches the PDF's s3_key
            delete_result = index.delete(
                namespace=namespace,
                filter={
                    "s3_key": s3_key  # Match only this PDF's s3_key
                }
            )
            
            print(f"🗑️ Deleted vectors for s3_key: {s3_key}")
            print(f"📊 Delete result: {delete_result}")
            print(f"🎯 Targeted deletion from namespace: {namespace}")
            
        except Exception as delete_error:
            print(f"⚠️ Could not delete specific vectors: {delete_error}")
            # Fallback: try to get all vectors and filter manually
            try:
                # Get all vectors in namespace (limited)
                stats = index.describe_index_stats()
                namespace_stats = stats.get('namespaces', {}).get(namespace, {})
                vector_count = namespace_stats.get('vectorCount', 0)
                
                if vector_count > 0:
                    print(f"📊 Found {vector_count} vectors in namespace {namespace}")
                    # For now, delete entire namespace as fallback
                    index.delete(namespace=namespace, delete_all=True)
                    print(f"🔄 Fallback: Deleted entire namespace due to filter limitations")
                else:
                    print(f"ℹ️ No vectors found in namespace {namespace}")
                    
            except Exception as fallback_error:
                print(f"❌ Fallback deletion also failed: {fallback_error}")
            
    except Exception as e:
        print(f"❌ Failed to delete from Pinecone: {e}")
        # Continue anyway - S3 and Firestore deletion already done
        
    return {"message": "PDF deleted successfully"}
