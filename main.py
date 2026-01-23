from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib
import logging

app = FastAPI(title="RAG Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

def include_router_safe(module_path: str, router_name: str = "router", prefix: str | None = None) -> None:
    try:
        module = importlib.import_module(module_path)
        router = getattr(module, router_name)
        if prefix:
            app.include_router(router, prefix=prefix)
        else:
            app.include_router(router)
        logging.info("Loaded router: %s", module_path)
    except Exception as exc:
        logging.exception("Failed to load router %s: %s", module_path, exc)

# include_router_safe("routes.folders", prefix="/api")
# include_router_safe("routes.upload")  # Temporarily disabled
# include_router_safe("routes.delete_pdf")
include_router_safe("routes.agent_fixed")
# include_router_safe("routes.analytics", prefix="/api")
# include_router_safe("routes.debug_pdf", prefix="/debug")

@app.get("/")
async def root():
    return {"message": "RAG Backend API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)