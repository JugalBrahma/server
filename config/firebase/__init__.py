"""
Firebase Configuration Module
- Firestore database instance
- Authentication utilities
"""

from .database import get_firestore, db
from .auth import get_current_user_id

__all__ = ["get_firestore", "db", "get_current_user_id"]
