import logging
import re
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from agent.agent import RAGAgent
from config.firebase import get_current_user_id
from evaluate.flutter_evaluator import evaluate_flutter_query

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_cache: dict[str, RAGAgent] = {}


class Query(BaseModel):
    query: str
    folder_id: Optional[str] = None


def get_agent_for_namespace(namespace: str) -> RAGAgent:
    if namespace not in _agent_cache:
        logger.info("Initializing RAGAgent for namespace: %s", namespace)
        _agent_cache[namespace] = RAGAgent(namespace=namespace)
    return _agent_cache[namespace]


@router.post("/agent")
async def chat_with_agent(
    query: Query,
    user_id: str = Depends(get_current_user_id),
):
    try:
        namespace = f"user_{user_id}_folder_{query.folder_id}" if query.folder_id else f"user_{user_id}"
        agent = get_agent_for_namespace(namespace)

        ai_response = agent.query(
            query.query.strip().lower(),
            folder_id=query.folder_id,
            user_id=user_id,
        )

        answer = ai_response.get("answer", "")
        doc_refs = re.findall(r"<doc>.*?</doc>", answer)
        has_documents = bool(doc_refs)

        threading.Thread(
            target=evaluate_flutter_query,
            args=(query.query, answer),
            kwargs={"namespace": namespace},
            daemon=True,
        ).start()

        return {
            "response": ai_response,
            "namespace": namespace,
            "from_documents": has_documents,
            "doc_reference_count": len(doc_refs),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error processing agent query: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")
