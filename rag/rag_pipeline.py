"""
RAG Pipeline Main Module
- Orchestrates the complete RAG processing pipeline
- Integrates text and image processing modules
- Manages vector store operations
"""

from typing import List, Dict, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os

# Import modular components
from .text_processing import (
    extract_pdf_elements,
    process_extracted_elements_smart,
    combine_text_and_images,
    smart_chunk
)
from .image_processing import (
    compress_image,
    deduplicate_images,
    get_cached_descriptions,
    usage_stats
)

# ==================== VECTOR STORE MANAGEMENT ====================
class VectorStoreManager:
    """Manages Pinecone vector store operations"""
    
    def __init__(self, index_name: str = "askadocument", namespace: str = "test-2"):
        self.index_name = index_name
        self.namespace = namespace
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY is not set in environment.")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072,
            api_key=api_key
        )
        self.vector_store = self._init_vector_store()

    
    def _init_vector_store(self):
        """Initialize or connect to existing Pinecone index"""
        try:
            # Try to connect to existing index
            vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace
            )
            # print(f"✅ Connected to existing index: {self.index_name}")  # Removed for speed
            return vector_store
        except Exception as e:
            print(f"⚠️ Could not connect to existing index: {e}")
            
            # Create new index if it doesn't exist
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"✅ Created new index: {self.index_name}")
                
                return PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=self.namespace
                )
            except Exception as create_error:
                print(f"❌ Failed to create index: {create_error}")
                raise
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to vector store"""
        try:
            # Prepare documents and metadata
            texts = [chunk["content"] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                # Clean metadata for Pinecone compatibility
                meta = {}
                for k, v in chunk["metadata"].items():
                    # Skip binary/complex data
                    if k in ["image_base64", "coordinates", "image_payload"]:
                        continue
                    
                    # Handle different data types
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif isinstance(v, list):
                        if all(isinstance(item, str) for item in v):
                            meta[k] = v
                        else:
                            meta[f"{k}_json"] = str(v)
                    elif isinstance(v, dict):
                        meta[f"{k}_json"] = str(v)
                
                # Add content type
                meta["content_type"] = chunk["type"]
                
                # Add has_image flag
                if chunk.get("has_image", False):
                    meta["has_image"] = True
                
                metadatas.append(meta)
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                namespace=self.namespace
            )
            
            print(f"✅ Added {len(chunks)} documents to vector store")
            
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Any]:
        """Search for relevant documents"""
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                namespace=self.namespace
            )
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

# ==================== MAIN PIPELINE ====================
def run_optimized_pipeline(pdf_path: str, use_cache: bool = True, index_name: str = "askadocument", namespace: str = "test-2") -> Tuple[VectorStoreManager, List[Dict]]:
    """
    Run the complete optimized RAG pipeline.
    
    Args:
        pdf_path: Path to PDF file
        use_cache: Whether to use image description cache
        index_name: Pinecone index name
        namespace: Namespace for multi-tenancy
    
    Returns:
        Tuple of (vector_store_manager, final_chunks)
    """
    print("=" * 60)
    print("OPTIMIZED RAG PIPELINE")
    print("=" * 60)
    
    # Initialize clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vector_manager = VectorStoreManager(index_name, namespace)
    
    print("\n[1/7] Extracting PDF...")
    raw_elements = extract_pdf_elements(pdf_path)
    print(f"✓ Extracted {len(raw_elements)} elements")
    
    print("\n[2/7] Processing elements (smart filtering)...")
    text_chunks, image_chunks = process_extracted_elements_smart(raw_elements)
    print(f"✓ Found {len(text_chunks)} text chunks, {len(image_chunks)} images/tables")
    
    print("\n[3/7] Deduplicating images...")
    unique_images = deduplicate_images(image_chunks)
    print(f"✓ Kept {len(unique_images)} unique images")
    
    print("\n[4/7] Compressing images...")
    compressed_images = [
        {**img, 'base64': compress_image(img['base64'])}
        for img in unique_images
    ]
    print(f"✓ Compressed {len(compressed_images)} images")
    
    print("\n[5/7] Describing images (batched + cached)...")
    if use_cache:
        image_descriptions = get_cached_descriptions(compressed_images, client)
    else:
        from .image_processing import describe_images_batch
        image_descriptions = describe_images_batch(compressed_images, client)
    print(f"✓ Got {len(image_descriptions)} descriptions")
    
    print("\n[6/7] Combining text + images...")
    combined_chunks = combine_text_and_images(text_chunks, image_descriptions)
    print(f"✓ Created {len(combined_chunks)} combined chunks")
    
    print("\n[7/7] Smart chunking...")
    final_chunks = smart_chunk(combined_chunks, chunk_size=1000, overlap=200)
    print(f"✓ Split into {len(final_chunks)} final chunks")
    
    print("\n[8/8] Storing in vector database...")
    vector_manager.add_documents(final_chunks)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {len(final_chunks)}")
    print(f"  With images: {sum(1 for c in final_chunks if c.get('has_image'))}")
    print(f"  Text only: {sum(1 for c in final_chunks if not c.get('has_image'))}")
    
    print(f"\nToken Usage:")
    print(f"  Vision input: {usage_stats['vision_input']:,}")
    print(f"  Vision output: {usage_stats['vision_output']:,}")
    print(f"  Total: {usage_stats['vision_input'] + usage_stats['vision_output']:,}")
    
    return vector_manager, final_chunks
