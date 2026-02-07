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
        # Use folder-specific namespace when available (matches upload behavior)
        if folder_id:
            namespace = f"user_{user_id}_folder_{folder_id}"
        else:
            namespace = f"user_{user_id}"
        
        # Initialize vector manager with the correct namespace
        from agent.vector_manager import VectorSearchManager
        vector_manager = VectorSearchManager(index_name="askadocument", namespace=namespace)
        
        # Delete vectors by s3_key (this properly handles the metadata filter)
        deleted = vector_manager.delete_documents_by_s3_key(s3_key)
        
        if deleted:
            print(f"🗑️ Deleted vectors for s3_key: {s3_key} from namespace: {namespace}")
        else:
            print(f"⚠️ No vectors found for s3_key: {s3_key} in namespace: {namespace}")
            
    except Exception as e:
        print(f"❌ Failed to delete from Pinecone: {e}")
        # Continue anyway - S3 and Firestore deletion already done
        
    return {"message": "PDF deleted successfully"}
