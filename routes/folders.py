import logging
from fastapi import APIRouter, Depends, HTTPException
from config.firebase import get_current_user_id, get_firestore
from agent.vector_manager import VectorSearchManager

logger = logging.getLogger(__name__)

router = APIRouter()
db = get_firestore()


@router.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
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

        # Block deletion if PDFs still exist in the folder
        pdfs = list(folder_ref.collection("pdfs").stream())
        if pdfs:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete folder with PDF(s). Please delete all PDFs first.",
            )

        # Remove chat messages and the main_chat doc
        main_chat_ref = folder_ref.collection("chats").document("main_chat")
        for msg in main_chat_ref.collection("messages").stream():
            msg.reference.delete()
        main_chat_ref.delete()

        # Clean up vector search namespace
        namespace = f"user_{user_id}_folder_{folder_id}"
        try:
            VectorSearchManager(namespace=namespace).delete_namespace()
        except Exception as exc:
            logger.warning("Failed to delete Pinecone namespace %s: %s", namespace, exc)

        folder_ref.delete()
        logger.info("Deleted folder %s for user %s", folder_id, user_id)

        return {"status": "deleted", "folder_id": folder_id}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error deleting folder %s: %s", folder_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete folder")
