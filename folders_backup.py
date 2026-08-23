"""
Backup of Original Folders Module
"""

from fastapi import APIRouter, Header, HTTPException
from firebase_admin import auth
from config.firebase import get_firestore
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Initialize API Router
router = APIRouter()
db = get_firestore()

class FolderCreateRequest(BaseModel):
    folder_name: str

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Folders API",
        "endpoints": {
            "folders": "/folders",
            "test": "/test"
        }
    }

@router.get("/test")
async def test_route():
    """Simple test route to verify the server is working"""
    return {
        "message": "Folders API is working!",
        "status": "healthy",
        "router_working": True
    }

@router.get("/folders")
async def get_folders(authorization: str = Header(...)):
    """Get all folders for the authenticated user"""
    try:
        token = authorization.split(" ")[1]
        user = auth.verify_id_token(token)
        user_id = user["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        folders_ref = db.collection("users").document(user_id).collection("folders")
        folders = folders_ref.stream()
        
        folder_list = []
        for folder in folders:
            folder_data = folder.to_dict()
            folder_data["id"] = folder.id
            folder_list.append(folder_data)
        
        return {"folders": folder_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get folders: {str(e)}")

@router.post("/folders")
async def create_folder(folder_request: FolderCreateRequest, authorization: str = Header(...)):
    """Create a new folder"""
    try:
        token = authorization.split(" ")[1]
        user = auth.verify_id_token(token)
        user_id = user["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        from datetime import datetime
        
        folder_ref = db.collection("users").document(user_id).collection("folders").document()
        folder_data = {
            "name": folder_request.folder_name,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "user_id": user_id
        }
        
        folder_ref.set(folder_data)
        
        return {
            "id": folder_ref.id,
            "name": folder_request.folder_name,
            "created_at": folder_data["created_at"].isoformat(),
            "updated_at": folder_data["updated_at"].isoformat(),
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {str(e)}")
