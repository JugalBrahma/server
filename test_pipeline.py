import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.text_processing import extract_pdf_elements, process_extracted_elements_smart
from rag.image_processing import deduplicate_images, compress_image, get_cached_descriptions, describe_images_batch, combine_text_and_images, smart_chunk
from rag.vector_store import store_in_pineconedb
from utils.usage_stats import usage_stats

def run_optimized_pipeline(pdf_path="data/rag_for_nlp.pdf", use_cache=True):
    """
    Run the complete optimized RAG pipeline.
    
    Optimizations:
    - Batch image processing (80-90% cost reduction)
    - Image deduplication and filtering
    - Compression to reduce token usage
    - Caching to avoid reprocessing
    - Smart chunking to preserve context
    """
    print("=" * 60)
    print("OPTIMIZED RAG PIPELINE")
    print("=" * 60)
    
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
        image_descriptions = get_cached_descriptions(compressed_images)
    else:
        image_descriptions = describe_images_batch(compressed_images)
    print(f"✓ Got {len(image_descriptions)} descriptions")
    
    print("\n[6/7] Combining text + images...")
    combined_chunks = combine_text_and_images(text_chunks, image_descriptions)
    print(f"✓ Created {len(combined_chunks)} combined chunks")
    
    print("\n[7/7] Smart chunking...")
    final_chunks = smart_chunk(combined_chunks, chunk_size=1000, overlap=200)
    print(f"✓ Split into {len(final_chunks)} final chunks")

    print("\n[8/7] Storing in Pinecone...")
    vector_store = store_in_pineconedb(final_chunks)
    
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
    
    # Model info
    try:
        from google import genai
        client = genai.Client()
        model_info = client.models.get(model="gemini-2.0-flash-lite")
        print(f"\nModel Limits:")
        print(f"  Input: {model_info.input_token_limit:,}")
        print(f"  Output: {model_info.output_token_limit:,}")
    except:
        pass
    
    return vector_store, final_chunks


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Run the optimized pipeline
    vector_store, chunks = run_optimized_pipeline(
        pdf_path="data/rag_for_nlp.pdf",
        use_cache=True  # Set to False to force reprocessing
    )
    
    print("\n✅ Ready for querying!")
