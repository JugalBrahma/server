from firebase_admin import auth
from .database import get_firestore


def verify_firebase_token(token: str) -> dict:
    # Ensure Firebase Admin is initialized (handled in services.database)
    get_firestore()
    return auth.verify_id_token(token)
