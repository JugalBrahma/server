import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import httpx
from redis.asyncio import Redis

load_dotenv(override=True)

# Configure tracing after loading the environment and before loading the routes.
import config.langsmith  # noqa: E402

import routes.chat as chat
from routes.analytics import router as analytics_router
from routes.chat import router as chat_router
from routes.debug_pdf import router as debug_pdf_router
from routes.delete_pdf import router as delete_pdf_router
from routes.folders import router as folders_router
from routes.upload import router as upload_router
from agent.reranker import DocumentReranker

app = FastAPI(title="RAG Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log which field made a request fail with 422. Values are left out so tokens never reach the log."""
    problems = [
        f"{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg')}"
        for err in exc.errors()
    ]
    logging.getLogger(__name__).warning(
        "422 on %s %s: %s", request.method, request.url.path, "; ".join(problems)
    )
    return await request_validation_exception_handler(request, exc)

app.include_router(folders_router, prefix="/api")
app.include_router(upload_router)
app.include_router(delete_pdf_router)
# Keep the /agent path because existing clients use it.
app.include_router(chat_router)
app.include_router(analytics_router, prefix="/api")
if os.getenv("ENABLE_DEBUG_ROUTES", "false").lower() == "true":
    app.include_router(debug_pdf_router, prefix="/debug")


@app.on_event("startup")
def warm_reranker() -> None:
    """Avoid loading or downloading the FlashRank model during a user query."""
    if os.getenv("ENABLE_RERANKER", "true").lower() != "true":
        return
    try:
        DocumentReranker().warm()
        logging.getLogger(__name__).info("FlashRank reranker warmed successfully")
    except Exception:
        # Keep the API available; each answer exposes a reranker fallback if it occurs.
        logging.getLogger(__name__).exception("FlashRank warm-up failed")

@app.on_event("startup")
async def startup_event():
    chat.redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    chat.http_client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown_event():
    if chat.redis is not None:
        await chat.redis.aclose()
    if chat.http_client is not None:
        await chat.http_client.aclose()


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
