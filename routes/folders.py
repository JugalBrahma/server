from fastapi import APIRouter, Header, HTTPException, Depends
from config.firebase import get_current_user_id, get_firestore
from agent.vector_manager import VectorSearchManager

router = APIRouter()
db = get_firestore()

@router.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id)
):
    try:
        folder_ref = (
            db.collection("users")
            .document(user_id)
            .collection("folders")
            .document(folder_id)
        )

        folder_doc = folder_ref.get()
        if not folder_doc.exists:
            raise HTTPException(status_code=404, detail="Folder not found")

        # 2️⃣ Check if folder has PDFs (prevent deletion if PDFs exist)
        print(f"Checking folder {folder_id} for PDFs before deletion...")
        pdfs = folder_ref.collection("pdfs").stream()
        pdf_count = 0
        pdf_ids = []
        for pdf in pdfs:
            pdf_count += 1
            pdf_ids.append(pdf.id)

        print(f"Found {pdf_count} PDFs in folder {folder_id}: {pdf_ids}")

        if pdf_count > 0:
            error_msg = (
                f"Cannot delete folder with {pdf_count} PDF(s). "
                "Please delete all PDFs from the upload section first."
            )
            print(f"BLOCKING deletion: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg,
            )

        print(f"Folder {folder_id} has no PDFs, proceeding with deletion")

        # 3️⃣ Delete messages from main_chat then delete the chat document
        delete_count = 0
        
        # Get the main_chat document reference
        main_chat_ref = folder_ref.collection("chats").document("main_chat")
        
        print(f"  Fetching messages from main_chat for folder {folder_id}...")
        messages = main_chat_ref.collection("messages").stream()
        msg_count = 0
        for msg in messages:
            try:
                msg.reference.delete()
                delete_count += 1
                msg_count += 1
            except Exception as msg_err:
                print(f"    ERROR deleting message {msg.id}: {msg_err}")
                raise
        
        print(f"  Deleted {msg_count} messages from main_chat")
        
        # Delete the main_chat document
        try:
            main_chat_ref.delete()
            delete_count += 1
            print(f"  Deleted main_chat document")
        except Exception as chat_err:
            print(f"  ERROR deleting main_chat: {chat_err}")
            raise
        
        print(f"Deleted {delete_count} total docs")

        # 4️⃣ Delete Pinecone namespace for this folder
        namespace = f"user_{user_id}_folder_{folder_id}"
        print(f"  Deleting Pinecone namespace: {namespace}")
        try:
            vector_manager = VectorSearchManager(namespace=namespace)
            vector_manager.delete_namespace()
            print(f"  Deleted Pinecone namespace: {namespace}")
        except Exception as vector_err:
            print(f"  WARNING: Failed to delete Pinecone namespace: {vector_err}")
            # Don't fail the folder deletion if Pinecone cleanup fails

        # 5️⃣ Delete folder doc
        try:
            folder_ref.delete()
            print(f"Successfully deleted folder {folder_id} and all its data")
        except Exception as folder_err:
            print(f"  ERROR deleting folder {folder_id}: {folder_err}")
            raise

        return {"status": "deleted", "folder_id": folder_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")
