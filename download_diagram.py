import zlib
import base64
import urllib.request

mermaid_code = """flowchart TD
    Client[Flutter Client App]
    
    subgraph FastAPI Backend
        API[FastAPI Main Application]
        UploadRoute[Upload Flow\\nroutes/upload.py]
        AgentRoute[Agent Flow\\nroutes/agent_fixed.py]
        RAG[RAG Pipeline\\nrag/text_processing.py & image_processing.py]
        LangGraph[LangGraph RAG Agent\\nagent/agent.py]
    end
    
    subgraph Storage & Cloud
        S3[(AWS S3\\nPDF Storage)]
        Firestore[(Firebase Firestore\\nFile Metadata)]
        Pinecone[(Pinecone\\nVector Database)]
    end
    
    subgraph External APIs
        GeminiVision[Gemini Vision API\\nImage Description]
        GeminiLLM[Gemini LLM\\nResponse Generation]
    end

    %% Upload Flow
    Client -- "1. Upload PDF" --> API
    API -- "Route to" --> UploadRoute
    UploadRoute -- "2. Save File" --> S3
    UploadRoute -- "3. Trigger Processing" --> RAG
    RAG -- "4. Describe Images" --> GeminiVision
    RAG -- "5. Embed & Store" --> Pinecone
    UploadRoute -- "6. Save Metadata" --> Firestore
    UploadRoute -- "7. Return Success" --> Client
    
    %% Query Flow
    Client -- "A. Ask Question" --> API
    API -- "Route to" --> AgentRoute
    AgentRoute -- "B. Forward Query" --> LangGraph
    LangGraph -- "C. Retrieve Docs" --> Pinecone
    Pinecone -- "D. Return Chunks" --> LangGraph
    LangGraph -- "E. Prompt Generation" --> GeminiLLM
    GeminiLLM -- "F. Generated Answer" --> LangGraph
    LangGraph -- "G. Return to Router" --> AgentRoute
    AgentRoute -- "H. Return JSON Response" --> Client

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#000;
    classDef client fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef backend fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000;
    classDef storage fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef external fill:#ffe0b2,stroke:#f57c00,stroke-width:2px,color:#000;
    
    class Client client;
    class API,UploadRoute,AgentRoute,RAG,LangGraph backend;
    class S3,Firestore,Pinecone storage;
    class GeminiVision,GeminiLLM external;"""

# Compress and encode for Kroki API
compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')

# Kroki URL for Mermaid to PNG
url = f"https://kroki.io/mermaid/png/{encoded}"

print("Downloading diagram image...")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as response:
        with open("backend_architecture.png", "wb") as f:
            f.write(response.read())
    print("Success! Downloaded as backend_architecture.png")
except Exception as e:
    print(f"Error: {e}")
