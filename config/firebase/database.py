import os
import sys
from pathlib import Path
from firebase_admin import credentials, firestore
import firebase_admin

# Add parent directory to path for firebase-adminsdk.json
sys.path.append(str(Path(__file__).parent.parent.parent))

# -------------------------------
# 🔐 Firebase Initialization (ONCE)
# -------------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(
        r"C:\Flutter Projects\1-Flutter Projects\ProductionReady\Ragtest\backendtest\server\firebase-adminsdk.json"
    )
    firebase_admin.initialize_app(cred)

db = firestore.client()

def get_firestore():
    """Get Firestore database instance"""
    return db
