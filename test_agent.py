import requests
import json

def test_agent():
    """Test agent endpoint with a sample query"""
    
    # Test data
    test_url = "http://10.0.2.2:8000/agent"
    
    # Test query
    test_query = {
        "query": "What is machine learning?",
        "folder_id": "ctQaumavt40JDm4TeZ3K"  # Your folder ID
    }
    
    # Prepare headers (you'll need to replace with actual token)
    headers = {
        "Authorization": "Bearer YOUR_FIREBASE_TOKEN",
        "Content-Type": "application/json"
    }
    
    try:
        print("🧪 Testing agent endpoint...")
        response = requests.post(test_url, headers=headers, json=test_query)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Agent query successful!")
            print(f"Response: {data.get('response')}")
            print(f"Namespace: {data.get('namespace')}")
        else:
            print(f"❌ Agent query failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_agent()
