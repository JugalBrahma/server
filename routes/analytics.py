from fastapi import APIRouter, Depends, HTTPException
from firebase_admin import firestore
from config.firebase import get_current_user_id, get_firestore

router = APIRouter()
db = get_firestore()

@router.get("/analytics/{folder_id}")
async def get_folder_analytics(
    folder_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get analytics for a specific folder including:
    - Total number of chat messages
    - Number of MCQ questions asked
    - Number of times data was retrieved from documents
    """
    try:
        # Reference to the messages collection
        messages_ref = (
            db.collection("users")
            .document(user_id)
            .collection("folders")
            .document(folder_id)
            .collection("chats")
            .document("main_chat")
            .collection("messages")
        )
        
        # Get all messages
        messages = messages_ref.stream()
        
        # Initialize counters
        total_messages = 0
        mcq_count = 0
        doc_retrieval_count = 0
        
        # Count metrics
        for message in messages:
            data = message.to_dict()
            if data.get("role") == "error":
                continue
            total_messages += 1
            
            # Debug logging
            is_user = data.get('isUser', False)
            mcq_is_true = data.get('mcqIsTrue', False) or data.get('mcq_is_true', False)
            has_mcqs = 'mcqs' in data and data['mcqs']
            
            print(f"Message: isUser={is_user}, mcqIsTrue={mcq_is_true}, has_mcqs={has_mcqs}")
            print(f"  All data keys: {list(data.keys())}")
            
            # Check if MCQ was asked (assistant message with mcqIsTrue)
            if not is_user and mcq_is_true:
                mcq_count += 1
                print(f"  -> MCQ counted! Total: {mcq_count}")
            elif not is_user and has_mcqs:
                print(f"  -> Has MCQs but mcqIsTrue is false!")
            
            # Count document references based on stored fields
            if not is_user:
                if data.get('from_documents') is True:
                    doc_retrieval_count += int(data.get('doc_reference_count', 0)) or 1
        
        return {
            "folder_id": folder_id,
            "total_messages": total_messages,
            "mcq_count": mcq_count,
            "doc_retrieval_count": doc_retrieval_count,
            "user_messages": total_messages // 2,  # Approximate
            "assistant_messages": total_messages // 2,  # Approximate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")
