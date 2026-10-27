import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

# -------------------------------
# Firebase Initialization (ONCE)
# -------------------------------
if not firebase_admin._apps:
    # For Railway deployment or local dev, resolve relative to server root
    default_path = Path(__file__).resolve().parent.parent.parent / "firebase-adminsdk.json"
    firebase_sdk_path = os.getenv("FIREBASE_ADMIN_SDK_PATH") or str(default_path)
    if not os.path.exists(firebase_sdk_path) and os.path.exists("firebase-adminsdk.json"):
        firebase_sdk_path = "firebase-adminsdk.json"
    cred = credentials.Certificate(firebase_sdk_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def get_firestore():
    """Get Firestore database instance"""
    return db
