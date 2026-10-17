import os
import sys
from pinecone import Pinecone
from firebase_admin import credentials, firestore
from config.firebase import get_firestore

def cleanup_pinecone_and_firestore():
    """Delete all namespaces from Pinecone and clear Firestore"""
    
    print("=" * 60)
    print("CLEANUP: Pinecone + Firestore")
    print("=" * 60)
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found")
        return
    
    pc = Pinecone(api_key=api_key)
    
    try:
        # Connect to askadocument index
        index_name = "askadocument"
        index = pc.Index(index_name)
        
        # Get current namespaces
        print("\n[1/3] Getting current namespaces...")
        stats = pc.describe_index(index_name)
        current_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        
        print(f"✓ Found {len(current_namespaces)} namespaces: {current_namespaces}")
        
        # Delete all namespaces
        if current_namespaces:
            print(f"\n[2/3] Deleting {len(current_namespaces)} namespaces...")
            for namespace in current_namespaces:
                try:
                    index.delete_namespace(namespace)
                    print(f"✓ Deleted namespace: {namespace}")
                except Exception as e:
                    print(f"❌ Failed to delete {namespace}: {e}")
        else:
            print("\n✓ No namespaces to delete")
        
        # Verify deletion
        print("\n[3/3] Verifying deletion...")
        updated_stats = pc.describe_index(index_name)
        remaining_namespaces = list(updated_stats.namespaces.keys()) if updated_stats.namespaces else []
        
        if not remaining_namespaces:
            print("✓ All namespaces deleted successfully!")
        else:
            print(f"⚠️ Remaining namespaces: {remaining_namespaces}")
            
    except Exception as e:
        print(f"❌ Pinecone cleanup failed: {e}")
    
    # Clear Firestore
    try:
        print("\n[4/4] Clearing Firestore...")
        db = get_firestore()
        
        # Get all users
        users = db.collection('users').stream()
        deleted_users = 0
        deleted_folders = 0
        deleted_pdfs = 0
        
        for user_doc in users:
            user_id = user_doc.id
            print(f"\n  Processing user: {user_id}")
            
            # Delete all folders for this user
            folders = db.collection('users').document(user_id).collection('folders').stream()
            for folder_doc in folders:
                folder_id = folder_doc.id
                print(f"    Deleting folder: {folder_id}")
                
                # Delete all PDFs in folder
                pdfs = db.collection('users').document(user_id).collection('folders').document(folder_id).collection('pdfs').stream()
                for pdf_doc in pdfs:
                    pdf_doc.reference.delete()
                    deleted_pdfs += 1
                
                # Delete all chats in folder
                chats = db.collection('users').document(user_id).collection('folders').document(folder_id).collection('chats').stream()
                for chat_doc in chats:
                    # Delete all messages in chat
                    messages = chat_doc.reference.collection('messages').stream()
                    for msg_doc in messages:
                        msg_doc.reference.delete()
                    
                    chat_doc.reference.delete()
                    deleted_folders += 1
                
                # Delete the folder itself
                folder_doc.reference.delete()
            
            # Delete the user document
            user_doc.reference.delete()
            deleted_users += 1
        
        print(f"\n✓ Firestore cleanup complete!")
        print(f"  Users deleted: {deleted_users}")
        print(f"  Folders deleted: {deleted_folders}")
        print(f"  PDFs deleted: {deleted_pdfs}")
        
    except Exception as e:
        print(f"❌ Firestore cleanup failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    cleanup_pinecone_and_firestore()
