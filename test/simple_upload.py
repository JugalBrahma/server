
from typing import List, Any
from unstructured.partition.pdf import partition_pdf
from fastapi import APIRouter
from fastapi import File, UploadFile

router = APIRouter()

@router.post("/extract-pdf")
def extract_pdf_elements(upload_file: UploadFile=File(...)) -> List[Any]:
    """Extract all elements from PDF including images and tables."""
    
    # Save uploaded file temporarily
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(upload_file.file.read())
        tmp_path = tmp.name
    
    try:
        # Use the temp file path
        raw_pdf_elements = partition_pdf(
            filename=tmp_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            skip_infer_table_structure=False,
            model_name="yolox"
        )
        
        if raw_pdf_elements:
            print("done")
            return [{"status": "success", "count": len(raw_pdf_elements)}]
        else:
            print("No elements found in the PDF.")
            return [{"status": "empty", "count": 0}]
            
    except Exception as e:
        print(f"Error: {e}")
        return [{"status": "error", "message": str(e)}]
        
    finally:
        # Cleanup temp file
        os.unlink(tmp_path)