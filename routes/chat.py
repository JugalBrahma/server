import json
import logging
import re
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.agent import RAGAgent
from agent.vector_manager import user_namespace
from config.firebase import get_current_user_id
from routes.upload import get_owner_ref

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_cache: dict[str, RAGAgent] = {}


class ChatRequest(BaseModel):
    query: str
    folder_id: str | None = None
    enable_reranker: bool | None = None


def get_agent_for_namespace(namespace: str) -> RAGAgent:
    if namespace not in _agent_cache:
        logger.info("Initializing RAGAgent for namespace: %s", namespace)
        _agent_cache[namespace] = RAGAgent(namespace=namespace)
    return _agent_cache[namespace]


def fill_source_file_names(sources: list[dict], user_id: str, folder_id: str | None) -> None:
    """Look up the uploaded file name for sources whose chunks were indexed without one."""
    for source in sources:
        if source.get("file_name"):
            continue
        # s3_key is "pdfs/{user_id}/{pdf_id}.pdf" and pdf_id is the Firestore doc id
        pdf_id = PurePosixPath(source.get("s3_key", "")).stem
        file_name = None
        try:
            if pdf_id:
                snapshot = get_owner_ref(user_id, folder_id).collection("pdfs").document(pdf_id).get()
                file_name = (snapshot.to_dict() or {}).get("file_name") if snapshot.exists else None
        except Exception as exc:
            logger.warning("Failed to look up file name for %s: %s", source.get("s3_key"), exc)
        source["file_name"] = file_name or "Unknown document"


@router.post("/agent")
async def chat_with_agent(
    chat_request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    try:
        folder_id = chat_request.folder_id
        # One namespace per user; the agent filters retrieval to folder_id.
        namespace = user_namespace(user_id)
        agent = get_agent_for_namespace(namespace)

        # pyrefly: ignore[missing-argument]
        ai_response = agent.query(
            user_input=chat_request.query.strip().lower(),
            folder_id=folder_id,
            user_id=user_id,
            enable_reranker=chat_request.enable_reranker,
        )

        fill_source_file_names(ai_response.get("sources", []), user_id, folder_id)

        answer = ai_response.get("answer", "")
        doc_refs = re.findall(r"<doc>.*?</doc>", answer)
        has_documents = bool(doc_refs)

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


@router.post("/agent/stream")
def chat_with_agent_stream(
    chat_request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Like /agent, but sends the answer piece by piece (Server-Sent Events)."""
    folder_id = chat_request.folder_id
    agent = get_agent_for_namespace(user_namespace(user_id))

    def send_events():
        try:
            for event in agent.query_stream(
                user_input=chat_request.query.strip().lower(),
                folder_id=folder_id,
                user_id=user_id,
                enable_reranker=chat_request.enable_reranker,
            ):
                if event["type"] == "done":
                    fill_source_file_names(event["sources"], user_id, folder_id)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming query failed")
            error = {"type": "error", "message": "Something went wrong while answering."}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        send_events(),
        media_type="text/event-stream",
        # Tell proxies not to hold the pieces back until the end
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
