"""
Firebase Configuration Module
- Firebase Admin SDK initialization
- Firestore database setup
- Authentication utilities
"""

# Import functions from database.py
from .database import get_firestore, db
from .firebase_auth import verify_firebase_token
from .auth import get_current_user_id

def get_firebase_auth():
    """Get Firebase auth instance"""
    import firebase_admin
    from firebase_admin import auth
    return auth

def get_firebase_app():
    """Get Firebase app instance"""
    import firebase_admin
    return firebase_admin.get_app()
