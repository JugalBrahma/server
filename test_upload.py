import requests
import json

def test_upload():
    """Test upload endpoint with a sample PDF"""
    
    # Test data
    test_url = "http://10.0.2.2:8000/upload-pdf"
    
    # Create a simple test file
    test_content = b"%PDF-1.4\n%a1%b0%a1%a1\n1 0 obj\n<<\n/Length 44\nstream\nx%c5%89\nendstream\nendobj\n"
    
    # Prepare headers (you'll need to replace with actual token)
    headers = {
        "Authorization": "Bearer YOUR_FIREBASE_TOKEN",
        "folder_id": "test_folder_123"
    }
    
    files = {
        "file": ("test.pdf", test_content, "application/pdf")
    }
    
    try:
        print("🧪 Testing upload endpoint...")
        response = requests.post(test_url, headers=headers, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"PDF ID: {data.get('pdf_id')}")
            print(f"S3 Key: {data.get('s3_key')}")
            print(f"Vectorization: {data.get('extraction', {}).get('vectorization', {})}")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_upload()
