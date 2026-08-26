import sys
import types

# Compatibility shim for ragas importing deprecated/relocated langchain modules
if "langchain_community.chat_models.vertexai" not in sys.modules:
    v = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI
        v.ChatVertexAI = ChatVertexAI
    except Exception:
        v.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = v

if "langchain_community.llms" not in sys.modules:
    v2 = types.ModuleType("langchain_community.llms")
    try:
        from langchain_google_vertexai import VertexAI
        v2.VertexAI = VertexAI
    except Exception:
        v2.VertexAI = None
    sys.modules["langchain_community.llms"] = v2

from .flutter_evaluator import evaluate_flutter_query

__all__ = [
    "evaluate_flutter_query",
]
