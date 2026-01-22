#!/usr/bin/env python3
"""
Test script to verify RAG retrieval functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.vector_manager import VectorSearchManager

def test_retrieval():
    """Test document retrieval from the user's namespace"""
    
    # Initialize with the same namespace as the user
    namespace = "user_bCaPXdNf4mgkg5SGkc02513sbnP2"
    vector_manager = VectorSearchManager(namespace=namespace)
    
    print("=" * 60)
    print("TESTING RETRIEVAL FUNCTIONALITY")
    print("=" * 60)
    
    # Get stats first
    stats = vector_manager.get_stats()
    print(f"Vector Store Stats: {stats}")
    
    # Test queries
    test_queries = [
        "what is rag",
        "rag",
        "what",
        "document content"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 40)
        
        results = vector_manager.search(query=query, k=5)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n📄 Result {i}:")
                print(f"   Content: {result.page_content[:200]}...")
                if hasattr(result, 'metadata'):
                    print(f"   Metadata: {result.metadata}")
        else:
            print("❌ No results found")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_retrieval()
