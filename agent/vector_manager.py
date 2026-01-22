"""
Vector Search Manager Module
- Manages all vector database operations
- Handles document retrieval and similarity search
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from google import genai
from google.genai import types

# Suppress verbose logging
logging.getLogger("google").setLevel(logging.WARNING)

class VectorSearchManager:
    """Manages vector database operations and document retrieval"""
    
    def __init__(self, index_name: str = "askadocument", namespace: str = "test-2"):
        """
        Initialize Vector Search Manager
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Namespace for the index
        """
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embeddings
        self.embeddings = self._init_embeddings()
        
        # Initialize vector store
        self.vector_store = self._init_vector_store()

        # Hybrid search configuration
        self.hybrid_alpha = float(os.getenv("PINECONE_HYBRID_ALPHA", 0.5))
        self.hybrid_requested = os.getenv("ENABLE_PINECONE_HYBRID", "true").lower() == "true"
        self.hybrid_enabled = self.hybrid_requested and self._check_sparse_support()

        self.bm25_storage_dir: Optional[Path] = None
        self.bm25_state_path: Optional[Path] = None
        self.bm25_corpus_path: Optional[Path] = None
        self.bm25_encoder: Optional[BM25Encoder] = None
        self.retriever: Optional[PineconeHybridSearchRetriever] = None

        if self.hybrid_enabled:
            self.bm25_storage_dir = Path(os.getenv(
                "BM25_STORAGE_DIR",
                Path(__file__).resolve().parent.parent.parent / "bm25_cache"
            ))
            self.bm25_storage_dir.mkdir(parents=True, exist_ok=True)
            safe_namespace = self.namespace.replace("/", "_")
            self.bm25_state_path = self.bm25_storage_dir / f"{safe_namespace}_bm25.json"
            self.bm25_corpus_path = self.bm25_storage_dir / f"{safe_namespace}_corpus.json"
            self.bm25_encoder = self._load_bm25_encoder()
            self.retriever = self._init_hybrid_retriever()
        else:
            reason = "disabled by user setting" if not self.hybrid_requested else "index metric must be dotproduct"
            print(f"ℹ️ Hybrid search disabled ({reason}). Falling back to dense embeddings only.")
    
    def _init_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Initialize Google embeddings"""
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            config=types.EmbedContentConfig(output_dimensionality=3072)  # Match index dimension
        )
    
    def _init_vector_store(self) -> PineconeVectorStore:
        """Initialize or connect to existing Pinecone index"""
        try:
            # Try to connect to existing index
            vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace
            )
            return vector_store
        except Exception as e:
            print(f"⚠️ Could not connect to existing index: {e}")
            
            # Create new index if it doesn't exist
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # Match existing index dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"✅ Created new index: {self.index_name}")
                
                # Connect to the new index
                vector_store = PineconeVectorStore.from_existing_index(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=self.namespace
                )
                return vector_store
            except Exception as create_error:
                raise Exception(f"Failed to create or connect to index: {create_error}")

    def _check_sparse_support(self) -> bool:
        """Return True if Pinecone index allows sparse vectors (dotproduct metric)."""
        try:
            description = self.pc.describe_index(name=self.index_name)
            metric = (description.get("metric") or description.get("spec", {}).get("metric") or "").lower()
            if metric != "dotproduct":
                print(f"⚠️ Pinecone index '{self.index_name}' uses metric '{metric or 'unknown'}', which cannot accept sparse vectors.")
                return False
            return True
        except Exception as e:
            print(f"⚠️ Unable to verify Pinecone sparse support: {e}. Defaulting to dense search.")
            return False

    def _load_bm25_encoder(self) -> BM25Encoder:
        """Load BM25 encoder state for this namespace if available"""
        encoder = BM25Encoder().default()
        if self.bm25_state_path and self.bm25_state_path.exists():
            try:
                encoder = encoder.load(str(self.bm25_state_path))
                print(f"📁 Loaded BM25 encoder for namespace {self.namespace}")
            except Exception as e:
                print(f"⚠️ Failed to load BM25 encoder state: {e}. Reinitializing with defaults.")
        return encoder

    def _init_hybrid_retriever(self) -> Optional[PineconeHybridSearchRetriever]:
        """Create Pinecone hybrid search retriever if BM25 encoder is ready"""
        try:
            retriever = PineconeHybridSearchRetriever(
                embeddings=self.embeddings,
                sparse_encoder=self.bm25_encoder,
                index=self.index,
                alpha=self.hybrid_alpha,
                namespace=self.namespace,
                text_key="text"
            )
            return retriever
        except Exception as e:
            print(f"⚠️ Hybrid retriever initialization failed: {e}. Falling back to dense search.")
            return None

    def _update_bm25_encoder(self, texts: List[str]):
        """Update BM25 encoder with new texts and persist state"""
        if not texts or not self.hybrid_enabled:
            return

        corpus: List[str] = []
        if self.bm25_corpus_path and self.bm25_corpus_path.exists():
            try:
                corpus = json.loads(self.bm25_corpus_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ BM25 corpus file corrupted, resetting it.")
                corpus = []

        corpus.extend(texts)
        if self.bm25_corpus_path:
            self.bm25_corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

        try:
            encoder = BM25Encoder().default()
            encoder.fit(corpus)
            if self.bm25_state_path:
                encoder.dump(str(self.bm25_state_path))
            self.bm25_encoder = encoder
            self.retriever = self._init_hybrid_retriever()
            print(f"📦 BM25 encoder updated with {len(corpus)} documents for {self.namespace}")
        except Exception as e:
            print(f"⚠️ Failed to update BM25 encoder: {e}")

    def search(self, query: str, k: int = 2) -> List[Any]:
        """
        Search for similar documents
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        try:
            print(f"🔍 Starting search for query: '{query}'")
            print(f"🔍 Using k={k} results")
            print(f"🔍 Index: {self.index_name}")
            print(f"🔍 Namespace: {self.namespace}")
            print(f"🔍 Embedding model: {self.embeddings.model}")
            print(f"🔍 Vector store type: {type(self.vector_store)}")
            
            if self.hybrid_enabled and self.retriever:
                results = self.retriever.invoke(query)
            else:
                if not self.hybrid_enabled:
                    print("ℹ️ Hybrid search not available; using dense similarity search.")
                results = self.vector_store.similarity_search(query, k=k)
            
            print(f"🔍 Search completed. Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"🔍 Result {i+1}: {result.page_content[:100]}...")
                if hasattr(result, 'metadata'):
                    print(f"🔍 Metadata {i+1}: {result.metadata}")
            
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Full error: {repr(e)}")
            return []
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, s3_key: Optional[str] = None) -> bool:
        """
        Add documents to vector store
        
        Args:
            documents: List of document texts OR list of document dictionaries
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔄 Adding {len(documents)} documents to namespace: {self.namespace}")
            print(f"📄 Document types: {[type(doc).__name__ for doc in documents[:3]]}")
            
            # Log Pinecone operation details
            print(f"🌲 Index: {self.index_name}")
            print(f"🌲 Namespace: {self.namespace}")
            print(f"🌲 Embedding model: {self.embeddings.model}")
            print(f"🌲 Embedding dimension: 3072")  # Fixed dimension
            
            # Handle both string and dictionary formats
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    content = doc.strip()
                    if content.isdigit() and len(content) <= 4:
                        continue
                    if len(content) < 10:
                        continue
                    texts.append(content)
                    # Add metadata for string documents
                    metadatas.append({
                        "doc_id": f"doc_{i}",
                        "source": "user_upload",
                        "content_type": "text",
                        "has_image": False
                    })
                elif isinstance(doc, dict) and 'content' in doc:
                    content = str(doc['content']).strip()
                    if content.isdigit() and len(content) <= 4:
                        continue
                    if len(content) < 10:
                        continue
                    texts.append(content)
                    
                    # Get provided metadata and clean it for Pinecone
                    metadata = doc.get('metadata', {
                        "doc_id": f"doc_{i}",
                        "source": "user_upload",
                        "content_type": "text",
                        "has_image": False
                    })
                    
                    # Clean metadata - remove complex objects that Pinecone can't handle
                    cleaned_metadata = {}
                    
                    # Keep ONLY essential metadata to stay under 40KB limit
                    essential_keys = ['page_number', 'content_type', 'languages']
                    
                    for key, value in metadata.items():
                        if key in essential_keys:
                            if isinstance(value, (str, int, float, bool)):
                                # Keep simple types
                                cleaned_metadata[key] = value
                            elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                                # Keep lists of strings (like languages)
                                cleaned_metadata[key] = value[:10]  # Limit list size
                            else:
                                # Convert other types to string (limit length)
                                cleaned_metadata[key] = str(value)[:100]  # Limit string length
                    
                    # Add minimal required fields
                    cleaned_metadata.update({
                        "doc_id": f"doc_{i}",
                        "s3_key": s3_key if s3_key else "user_upload",
                        "content_type": doc.get('type', 'text'),
                        "has_image": False
                    })
                    
                    # Check metadata size (Pinecone limit is 40KB per vector)
                    import json
                    metadata_size = len(json.dumps(cleaned_metadata).encode('utf-8'))
                    if metadata_size > 35000:  # Leave some buffer
                        print(f"⚠️ Metadata too large ({metadata_size} bytes), further reducing...")
                        # Keep only absolute essentials
                        cleaned_metadata = {
                            "doc_id": f"doc_{i}",
                            "s3_key": s3_key if s3_key else "user_upload",
                            "content_type": "text"
                        }
                        final_size = len(json.dumps(cleaned_metadata).encode('utf-8'))
                        print(f"✅ Reduced metadata size to {final_size} bytes")
                    
                    metadatas.append(cleaned_metadata)
                else:
                    print(f"⚠️ Unexpected document format: {type(doc)} - {doc}")
                    continue
            
            print(f"📝 Processing {len(texts)} text chunks with {len(metadatas)} metadata entries")
            print(f"🧹 Cleaned metadata sample: {metadatas[0] if metadatas else 'None'}")
            
            self._update_bm25_encoder(texts)

            if self.hybrid_enabled and self.retriever:
                result = self.retriever.add_texts(texts=texts, metadatas=metadatas, namespace=self.namespace)
            else:
                if not self.hybrid_enabled and self.hybrid_requested:
                    print("ℹ️ Hybrid ingestion skipped because current Pinecone index does not support sparse vectors.")
                result = self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            print(f"📊 Pinecone result: {result}")
            print(f"✅ Successfully added {len(texts)} documents to namespace: {self.namespace}")
            return True
        except Exception as e:
            print(f"❌ Failed to add documents to namespace {self.namespace}")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Error message: {str(e)}")
            print(f"❌ Full error: {repr(e)}")
            print(f"❌ Error args: {e.args}")
            print(f"❌ Pinecone operation: {type(self.vector_store).__name__}")
            print(f"❌ Vector store type: {type(self.vector_store)}")
            return False
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from vector store by ID
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🗑️ Deleting {len(doc_ids)} documents from namespace: {self.namespace}")
            result = self.vector_store.delete(ids=doc_ids)
            print(f"📊 Delete result: {result}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "namespace": self.namespace,
                "dimension": 3072,
                "metric": "cosine",
                "vector_count": stats.get("total_vector_count", 0),
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            return {"error": str(e)}

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Initialize Vector Search Manager
    vector_manager = VectorSearchManager()
    
    # Get stats
    print("🔍 Vector Search Manager Stats:")
    stats = vector_manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    results = vector_manager.search("machine learning", k=2)
    print(f"📄 Found {len(results)} documents")
