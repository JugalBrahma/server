from fastapi import APIRouter, Header, HTTPException, Depends
from config.firebase import get_current_user_id, get_firestore

router = APIRouter()
db = get_firestore()

@router.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id)
):

    folder_ref = (
        db.collection("users")
        .document(user_id)
        .collection("folders")
        .document(folder_id)
    )

    if not folder_ref.get().exists:
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

    # 3️⃣ Delete chats + messages
    chats = folder_ref.collection("chats").stream()
    for chat in chats:
        messages = chat.reference.collection("messages").stream()
        for msg in messages:
            msg.reference.delete()
        chat.reference.delete()

    # 4️⃣ Delete folder doc
    folder_ref.delete()

    return {"status": "deleted"}
