from fastapi import APIRouter, Header, HTTPException, Request, Depends
from pydantic import BaseModel
from agent.agent import RAGAgent
from config.firebase import get_current_user_id
import json
import re

class Query(BaseModel):
    query: str
    folder_id: str = None  # Optional folder ID for namespace

class MemoryAction(BaseModel):
    action: str  # "clear" or "summary"

router = APIRouter()

# Cache for RAGAgent instances: {namespace: rag_agent}
_agent_cache: dict[str, RAGAgent] = {}

def get_cached_agent(namespace: str) -> RAGAgent:
    """Get or create RAGAgent from cache."""
    if namespace not in _agent_cache:
        print(f"Creating new RAGAgent for namespace: {namespace}")
        _agent_cache[namespace] = RAGAgent(namespace=namespace)
    else:
        print(f"Using cached RAGAgent for namespace: {namespace}")
    return _agent_cache[namespace]

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
        
        # 2️⃣ Create namespace per folder (separate embeddings per folder)
        if query.folder_id:
            namespace = f"user_{user_id}_folder_{query.folder_id}"
        else:
            namespace = f"user_{user_id}"
        print(f"DEBUG: Using namespace: {namespace}")
        
        # 3️⃣ Get cached agent (or create if not exists)
        rag_agent = get_cached_agent(namespace)
        
        # 4️⃣ Query agent (convert user query to lowercase)
        ai = rag_agent.query(query.query.lower())
        print(f"Agent response from namespace {namespace}: {ai}")
        
        # Check if the response contains document markers and count them
        response_text = ai.get("answer", "")
        has_documents = "<doc>" in response_text and "</doc>" in response_text
        
        # Count document references
        doc_count = len(re.findall(r'<doc>.*?</doc>', response_text))
        
        return {
            "response": ai, 
            "namespace": namespace,
            "from_documents": has_documents,
            "doc_reference_count": doc_count
        }
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
            
        rag_agent = get_cached_agent(namespace)
        
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
