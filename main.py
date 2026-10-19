import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(override=True)

# Configure tracing after loading the environment and before loading the routes.
import config.langsmith  # noqa: E402

from routes.analytics import router as analytics_router
from routes.chat import router as chat_router
from routes.debug_pdf import router as debug_pdf_router
from routes.delete_pdf import router as delete_pdf_router
from routes.folders import router as folders_router
from routes.upload import router as upload_router

app = FastAPI(title="RAG Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

app.include_router(folders_router, prefix="/api")
app.include_router(upload_router)
app.include_router(delete_pdf_router)
# Keep the /agent path because existing clients use it.
app.include_router(chat_router)
app.include_router(analytics_router, prefix="/api")
app.include_router(debug_pdf_router, prefix="/debug")


@app.get("/")
async def root():
    return {"message": "RAG Backend API is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Keep local Flutter development on the same port as ApiConfig, and bind
    # beyond loopback so physical devices on the LAN can reach the API.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
