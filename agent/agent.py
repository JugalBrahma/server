import os
import logging
from collections import defaultdict
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from .guardrail import check_greeting_guardrail
from .llm_manager import LLMManager
from .vector_manager import VectorSearchManager

logger = logging.getLogger(__name__)


class AgentConfig:
    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: str = "test-2",
        model_name: Optional[str] = None,
    ):
        index = index_name or os.getenv("PINECONE_INDEX_NAME") or os.getenv("AGENT_INDEX_NAME", "askadocument")
        self.llm_manager = LLMManager(model_name=model_name)
        self.llm = self.llm_manager.get_llm()
        self.vector_manager = VectorSearchManager(index_name=index, namespace=namespace)


class RAGAgent:
    """Document QA agent backed by Pinecone vector retrieval and LLM synthesis."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: str = "test-2",
        model_name: Optional[str] = None,
    ):
        self.config = AgentConfig(index_name=index_name, namespace=namespace, model_name=model_name)
        self.namespace = namespace

    @traceable(run_type="retriever", name="Pinecone Document Retrieval")
    def retrieve_documents(self, query: str) -> str:
        try:
            logger.info("Retrieving chunks for query '%s' in namespace '%s'", query, self.namespace)
            results = self.config.vector_manager.search(query=query, k=8)
            if not results:
                return ""

            results_by_doc = defaultdict(list)
            for doc in results:
                metadata = getattr(doc, "metadata", {}) or {}
                doc_id = metadata.get("doc_id") or metadata.get("s3_key", "unknown_doc")
                s3_key = metadata.get("s3_key", "unknown_source")
                namespace_hint = metadata.get("namespace", self.namespace)
                results_by_doc[(doc_id, s3_key, namespace_hint)].append({
                    "content": doc.page_content,
                    "metadata": metadata,
                    "content_type": metadata.get("content_type", "unknown"),
                })

            formatted = []
            for i, ((doc_id, s3_key, _), chunks) in enumerate(results_by_doc.items(), 1):
                chunks.sort(key=lambda c: c["metadata"].get("page_number", 0))
                combined_text = "\n\n".join(c["content"] for c in chunks)

                pages = [c["metadata"].get("page_number") for c in chunks if c["metadata"].get("page_number") is not None]
                if pages:
                    first_page, last_page = pages[0], pages[-1]
                    page_span = f"pages {first_page}–{last_page}" if first_page != last_page else f"page {first_page}"
                else:
                    page_span = "unknown pages"

                content_type = chunks[0]["content_type"]
                header = (
                    f"[Document {i}]\n"
                    f"ID: {doc_id}\n"
                    f"Source: {s3_key}\n"
                    f"Type: {content_type}\n"
                    f"Pages: {page_span}\n"
                )
                formatted.append(f"{header}Content:\n{combined_text}\n" + "=" * 50 + "\n")

            return "\n".join(formatted)
        except Exception:
            logger.exception("Retrieval error for namespace '%s'", self.namespace)
            return ""

    @traceable(run_type="chain", name="RAG Agent Query")
    def query(
        self,
        user_input: str,
        folder_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        try:
            run_tree = get_current_run_tree()
            if run_tree:
                if folder_id:
                    run_tree.extra.setdefault("metadata", {})["folder_id"] = folder_id
                    tag = f"folder_{folder_id}"
                    if tag not in run_tree.tags:
                        run_tree.tags.append(tag)
                if user_id:
                    run_tree.extra.setdefault("metadata", {})["user_id"] = user_id
                    tag = f"user_{user_id}"
                    if tag not in run_tree.tags:
                        run_tree.tags.append(tag)
        except Exception as exc:
            logger.warning("Failed to attach trace metadata: %s", exc)

        is_greeting, greeting_reply = check_greeting_guardrail(user_input)
        if is_greeting and greeting_reply:
            logger.info("Greeting intercepted for query: '%s'", user_input)
            return {"answer": greeting_reply}

        retrieved_docs = self.retrieve_documents(user_input)

        if retrieved_docs:
            system_prompt = (
                "You are an intelligent document assistant. "
                "Answer the question based strictly on the retrieved document context below. "
                "Wrap all facts, quotes, figures, and direct information from the documents in <doc> and </doc> tags. "
                "If the context does not contain the answer, politely state that the information was not found in the documents."
            )
            user_prompt = f'User Question: "{user_input}"\n\nRetrieved Documents:\n{retrieved_docs}\n\nAnswer:'
        else:
            system_prompt = (
                "You are a helpful assistant. If the user greets you, reply politely. "
                "If they ask a question about specific documents, inform them that no matching documents were found in this folder."
            )
            user_prompt = user_input

        response = self.config.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )

        return {"answer": content}

    def add_documents(self, documents, metadatas=None, s3_key=None):
        return self.config.vector_manager.add_documents(documents, metadatas, s3_key)

    def delete_documents(self, doc_ids: List[str]) -> bool:
        return self.config.vector_manager.delete_documents(doc_ids)
