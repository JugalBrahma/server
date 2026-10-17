from fastapi import APIRouter, Header, Request
from pydantic import BaseModel
import json

class TestQuery(BaseModel):
    query: str
    folder_id: str = None

router = APIRouter()

@router.post("/test-request")
async def test_request(query: TestQuery, authorization: str = Header(...), request: Request = None):
    """Test endpoint to debug request parsing"""
    try:
        # Log raw request body
        if request:
            body = await request.body()
            print(f"TEST - Raw request body: {body.decode()}")
        
        # Log parsed query object
        print(f"TEST - Parsed query: '{query.query}'")
        print(f"TEST - Parsed folder_id: '{query.folder_id}' (type: {type(query.folder_id)})")
        
        return {
            "received_query": query.query,
            "received_folder_id": query.folder_id,
            "folder_id_is_none": query.folder_id is None,
            "success": True
        }
    except Exception as e:
        print(f"TEST - Error: {e}")
        return {"success": False, "error": str(e)}
