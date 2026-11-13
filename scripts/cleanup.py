"""Wipe every Pinecone namespace and all Firestore data. Run: python -m scripts.cleanup"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone

from config.firebase import get_firestore

load_dotenv()

def cleanup_pinecone_and_firestore():
    """Delete all namespaces from Pinecone and clear Firestore"""
    
    print("=" * 60)
    print("CLEANUP: Pinecone + Firestore")
    print("=" * 60)
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not found")
        return
    
    pc = Pinecone(api_key=api_key)
    
    try:
        # Connect to askadocument index
        index_name = os.getenv("PINECONE_INDEX_NAME", "askadocument")
        index = pc.Index(index_name)
        
        # Get current namespaces
        print("\n[1/3] Getting current namespaces...")
        stats = index.describe_index_stats()
        current_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        
        print(f"Found {len(current_namespaces)} namespaces: {current_namespaces}")
        
        # Delete all namespaces
        if current_namespaces:
            print(f"\n[2/3] Deleting {len(current_namespaces)} namespaces...")
            for namespace in current_namespaces:
                try:
                    index.delete_namespace(namespace=namespace)
                    print(f"Deleted namespace: {namespace}")
                except Exception as e:
                    print(f"Failed to delete {namespace}: {e}")
        else:
            print("\nNo namespaces to delete")
        
        # Verify deletion
        print("\n[3/3] Verifying deletion...")
        updated_stats = index.describe_index_stats()
        remaining_namespaces = list(updated_stats.namespaces.keys()) if updated_stats.namespaces else []
        
        if not remaining_namespaces:
            print("All namespaces deleted successfully!")
        else:
            print(f"Remaining namespaces: {remaining_namespaces}")
            
    except Exception as e:
        print(f"Pinecone cleanup failed: {e}")
    
    # Clear Firestore
    try:
        print("\n[4/4] Clearing Firestore...")
        db = get_firestore()
        
        # list_documents() also returns docs that have no fields but do have
        # subcollections, which stream() skips. recursive_delete() removes a doc
        # together with every subcollection under it (folders, PDFs, chats, messages).
        deleted_users = 0
        deleted_docs = 0
        for user_ref in db.collection('users').list_documents():
            print(f"\n  Deleting user {user_ref.id} and everything under it")
            deleted_docs += db.recursive_delete(user_ref)
            deleted_users += 1

        print(f"\nFirestore cleanup complete!")
        print(f"  Users deleted: {deleted_users}")
        print(f"  Documents deleted (folders, PDFs, chats, messages, users): {deleted_docs}")
        
    except Exception as e:
        print(f"Firestore cleanup failed: {e}")

    # Clear BM25 statistics: pointers in Firestore, counts in S3
    try:
        print("\n[5/5] Clearing BM25 statistics...")
        from agent.bm25_store import POINTER_COLLECTION, S3_PREFIX, default_backend

        pointers = 0
        for doc in get_firestore().collection(POINTER_COLLECTION).stream():
            doc.reference.delete()
            pointers += 1
        default_backend().delete_prefix(f"{S3_PREFIX}/")
        print(f"  BM25 pointers deleted: {pointers}; S3 prefix '{S3_PREFIX}/' cleared")
    except Exception as e:
        print(f"BM25 cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE!")
    print("=" * 60)

def main() -> None:
    index_name = os.getenv("PINECONE_INDEX_NAME", "askadocument")
    print(
        f"This deletes ALL namespaces in Pinecone index '{index_name}', ALL Firestore users, "
        "folders and PDFs, and ALL BM25 statistics (Firestore + S3)."
    )
    if input(f"Type the index name '{index_name}' to continue: ").strip() != index_name:
        print("Aborted.")
        return
    cleanup_pinecone_and_firestore()


if __name__ == "__main__":
    main()
