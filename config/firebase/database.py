import os
import sys
from pathlib import Path
from firebase_admin import credentials, firestore
import firebase_admin

# -------------------------------
# 🔐 Firebase Initialization (ONCE)
# -------------------------------
if not firebase_admin._apps:
    # For Railway deployment, use environment variable or default path
    firebase_sdk_path = os.getenv("FIREBASE_ADMIN_SDK_PATH", "firebase-adminsdk.json")
    cred = credentials.Certificate(firebase_sdk_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def get_firestore():
    """Get Firestore database instance"""
    return db
