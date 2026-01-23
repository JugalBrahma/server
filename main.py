from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.debug_pdf import router as debug_router
from routes.delete_pdf import router as delete_router
from routes.folders import router as folders_router
from routes.agent_fixed import router as agent_router
from routes.analytics import router as analytics_router

app = FastAPI(title="RAG Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(folders_router, prefix="/api")
app.include_router(upload_router)
app.include_router(delete_router)
app.include_router(agent_router)
app.include_router(analytics_router, prefix="/api")
app.include_router(debug_router, prefix="/debug")

@app.get("/")
async def root():
    return {"message": "RAG Backend API is running"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)