import requests
import json

def test_agent_memory():
    """Test agent memory functionality"""
    
    base_url = "http://10.0.2.2:8000"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_FIREBASE_TOKEN"  # Add your token here
    }
    
    print("🧠 Testing Agent Memory System")
    print("=" * 50)
    
    # Test 1: First question
    print("\n1️⃣ First Question:")
    response1 = requests.post(f"{base_url}/agent", headers=headers, json={"query": "What is machine learning?"})
    print(f"Q: What is machine learning?")
    print(f"A: {response1.json().get('response', 'No response')}")
    
    # Test 2: Follow-up question (should remember context)
    print("\n2️⃣ Follow-up Question:")
    response2 = requests.post(f"{base_url}/agent", headers=headers, json={"query": "Can you explain more about what you just mentioned?"})
    print(f"Q: Can you explain more about what you just mentioned?")
    print(f"A: {response2.json().get('response', 'No response')}")
    
    # Test 3: Memory summary
    print("\n3️⃣ Memory Summary:")
    summary_response = requests.post(f"{base_url}/agent/memory", headers=headers, json={"action": "summary"})
    print(f"Summary: {summary_response.json().get('summary', 'No summary')}")
    
    # Test 4: Clear memory
    print("\n4️⃣ Clear Memory:")
    clear_response = requests.post(f"{base_url}/agent/memory", headers=headers, json={"action": "clear"})
    print(f"Clear: {clear_response.json().get('message', 'No message')}")
    
    # Test 5: Question after memory clear (should not remember previous context)
    print("\n5️⃣ Question After Memory Clear:")
    response3 = requests.post(f"{base_url}/agent", headers=headers, json={"query": "What did we just talk about?"})
    print(f"Q: What did we just talk about?")
    print(f"A: {response3.json().get('response', 'No response')}")

if __name__ == "__main__":
    test_agent_memory()
