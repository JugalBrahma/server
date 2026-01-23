from fastapi import APIRouter, Header, HTTPException, Request, Depends
from pydantic import BaseModel
from agent.agent import RAGAgent
from config.firebase import get_current_user_id
import json

class Query(BaseModel):
    query: str
    folder_id: str = None  # Optional folder ID for namespace

class MemoryAction(BaseModel):
    action: str  # "clear" or "summary"

router = APIRouter()

@router.post("/agent")
async def agent(
    query: Query, 
    user_id: str = Depends(get_current_user_id),
    request: Request = None
):
    """Chat with RAG agent"""
    try:
        # DEBUG: Log raw request body
        if request:
            body = await request.body()
            print(f"DEBUG: Raw request body: {body.decode()}")
        
        # DEBUG: Log the received folder_id
        print(f"DEBUG: folder_id='{query.folder_id}', type={type(query.folder_id)}")
        
        # 2️⃣ Create single namespace per user (not per folder)
        # Use user ID or email as namespace
        namespace = f"user_{user_id}"
        print(f"DEBUG: Using user namespace: {namespace}")
        
        # 3️⃣ Initialize agent with folder-specific namespace
        rag_agent = RAGAgent(namespace=namespace)
        
        # 4️⃣ Query agent (convert user query to lowercase)
        ai = rag_agent.query(query.query.lower())
        print(f"Agent response from namespace {namespace}: {ai}")
        return {"response": ai, "namespace": namespace}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/memory")
async def manage_memory(
    action: MemoryAction, 
    user_id: str = Depends(get_current_user_id),
    folder_id: str = Header(None, convert_underscores=False)
):
    """Manage conversation memory"""
    try:
        # 2️⃣ Get user's agent namespace
        if folder_id:
            namespace = f"user_{user_id}_folder_{folder_id}"
        else:
            namespace = f"user_{user_id}"
            
        rag_agent = RAGAgent(namespace=namespace)
        
        # 3️⃣ Perform memory action
        if action.action == "clear":
            rag_agent.clear_memory()
            return {"message": "Memory cleared successfully"}
        elif action.action == "summary":
            summary = rag_agent.get_memory_summary()
            return {"summary": summary}
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'clear' or 'summary'")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Memory management error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
