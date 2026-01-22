from fastapi import FastAPI
import os

app = FastAPI(title="RAG Backend API")

@app.get("/")
async def root():
    return {"message": "RAG Backend API is running", "port": os.environ.get("PORT", "not set")}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}")
    uvicorn.run("main_simple:app", host="0.0.0.0", port=port)
