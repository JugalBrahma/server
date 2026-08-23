"""
LangSmith Tracing Configuration for Fusion Chat
- Loads environment variables from .env
- Configures global tracing flags for LangSmith and LangChain
- Verifies and logs LangSmith connection status
"""

import os
import logging
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ----------------------------------------------------
# 🔍 LangSmith Global Tracing Configuration
# ----------------------------------------------------
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "fusion-chat")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Backwards compatibility / LangChain V2 support
os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"]
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

logger = logging.getLogger("langsmith")

def init_langsmith():
    """Initializes and logs LangSmith tracing status."""
    project = os.environ.get("LANGSMITH_PROJECT", "fusion-chat")
    tracing_enabled = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
    api_key_present = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))

    if tracing_enabled:
        if api_key_present:
            logger.info("🔍 LangSmith Tracing ENABLED for project: '%s'", project)
        else:
            logger.warning("⚠️ LangSmith Tracing is enabled for project '%s', but LANGSMITH_API_KEY is not set in .env.", project)
    else:
        logger.info("LangSmith Tracing is disabled.")

init_langsmith()
