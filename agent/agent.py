import os
import logging
from collections import defaultdict
from typing import Any, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from .guardrail import check_greeting_guardrail
from .llm_manager import LLMManager
from .reranker import DocumentReranker, RerankResult
from .vector_manager import VectorSearchManager

logger = logging.getLogger(__name__)


class AgentConfig:
    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: str = "test-2",
        model_name: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ):
        index = index_name or os.getenv("PINECONE_INDEX_NAME") or os.getenv("AGENT_INDEX_NAME", "askadocument")
        self.llm_manager = LLMManager(model_name=model_name)
        self.llm = self.llm_manager.get_llm()
        self.vector_manager = VectorSearchManager(index_name=index, namespace=namespace)
        if enable_reranker is not None:
            self.enable_reranker = enable_reranker
        else:
            self.enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        self.reranker = DocumentReranker() if self.enable_reranker else None
        self.fetch_k = int(os.getenv("RETRIEVAL_FETCH_K", "15"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "8"))


class RAGAgent:
    """Document QA agent backed by Pinecone vector retrieval, neural reranking, and LLM synthesis."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: str = "test-2",
        model_name: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ):
        self.config = AgentConfig(
            index_name=index_name,
            namespace=namespace,
            model_name=model_name,
            enable_reranker=enable_reranker,
        )
        self.namespace = namespace

    @traceable(run_type="retriever", name="Document Retrieval & Reranking")
    def retrieve_documents(
        self,
        query: str,
        enable_reranker: Optional[bool] = None,
    ) -> Tuple[str, List[Any], RerankResult]:
        """
        Retrieves matching document chunks from Pinecone vector store,
        optionally reranks them using neural cross-attention (FlashRank),
        and formats them for the LLM.
        """
        try:
            logger.info("Retrieving chunks for query '%s' in namespace '%s'", query, self.namespace)
            reranker = (
                self.config.reranker
                if enable_reranker is None
                else (self.config.reranker if enable_reranker else None)
            )

            # Step 1: Query vector database for top matching candidate chunks
            fetch_k = self.config.fetch_k if reranker else self.config.rerank_top_k
            results = self.config.vector_manager.search(query=query, k=fetch_k)
            if not results:
                return "", [], RerankResult([], applied=False, fallback_used=False)

            # Step 2: Neural cross-encoder reranking
            if reranker:
                rerank_result = reranker.rerank_with_status(
                    query=query,
                    documents=results,
                    top_k=self.config.rerank_top_k,
                )
                results = rerank_result.documents
            else:
                results = results[:self.config.rerank_top_k]
                rerank_result = RerankResult(results, applied=False, fallback_used=False)

            # Step 3: Group chunks by document identity (doc_id, s3_key, namespace)
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

            # Step 4: Sort chunks by page number and format each document into a labeled text block
            formatted = []
            for i, ((doc_id, s3_key, _), chunks) in enumerate(results_by_doc.items(), 1):
                # Order chunks by page number
                chunks.sort(key=lambda c: c["metadata"].get("page_number", 0))
                combined_text = "\n\n".join(c["content"] for c in chunks)

                # Determine the page range (e.g., 'pages 2–4' or 'page 2')
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

            return "\n".join(formatted), results, rerank_result
        except Exception:
            logger.exception("Retrieval error for namespace '%s'", self.namespace)
            return "", [], RerankResult([], applied=False, fallback_used=False)

    @traceable(run_type="chain", name="RAG Agent Query")
    def query(
        self,
        user_input: str,
        folder_id: Optional[str] = None,
        user_id: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ) -> dict:
        """
        Executes the end-to-end RAG workflow for a user query.

        Steps:
        1. Attach LangSmith tracing metadata and tags (folder_id, user_id).
        2. Fast-path check: intercept greetings with conversational replies.
        3. Retrieve and neural-rerank relevant document context via retrieve_documents().
        4. Construct system & human prompts (grounded QA if docs exist, fallback if not).
        5. Invoke the LLM and return the generated answer along with retrieved contexts.
        """
        # Step 1: Attach LangSmith tracing metadata and tags
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

        # Step 2: Check greeting guardrail (fast-path for greetings)
        is_greeting, greeting_reply = check_greeting_guardrail(user_input)
        reranker_is_active = (
            self.config.reranker is not None
            if enable_reranker is None
            else (self.config.reranker is not None and enable_reranker)
        )

        if is_greeting and greeting_reply:
            logger.info("Greeting intercepted for query: '%s'", user_input)
            return {
                "answer": greeting_reply,
                "contexts": [],
                "retrieval": {
                    "reranker_enabled": reranker_is_active,
                    "reranker_applied": False,
                    "reranker_fallback_used": False,
                },
            }

        # Step 3: Retrieve and rerank relevant document chunks from Pinecone
        retrieved_docs, raw_chunks, rerank_result = self.retrieve_documents(
            user_input,
            enable_reranker=enable_reranker,
        )

        # Step 4: Build prompts (grounded QA if documents found, fallback if not)
        if retrieved_docs:
            system_prompt = (
                "You are an intelligent document assistant. "
                "Answer the question based strictly on the retrieved document context below. "
                "Pay close attention to chronological and temporal qualifiers in questions (such as 'initially', 'first examination', 'before', 'later'). "
                "Note that an initial assessment or finding that a patient is healthy or has sluggish circulation counts as the initial diagnosis—do not confuse an initial assessment with any later re-examination or subsequent discovery of wounds/illness. "
                "Wrap only the exact facts, quotes, figures, and direct information from the documents in <doc> and </doc> tags. Use normal plain text for your own connecting words, and do not wrap entire sentences in <doc> tags. "
                "If the context does not contain the answer, politely state that the information was not found in the documents."
            )
            user_prompt = f'User Question: "{user_input}"\n\nRetrieved Documents:\n{retrieved_docs}\n\nAnswer:'
        else:
            system_prompt = (
                "You are a helpful assistant. If the user greets you, reply politely. "
                "If they ask a question about specific documents, inform them that no matching documents were found in this folder."
            )
            user_prompt = user_input

        # Step 5: Invoke the LLM and extract the response
        response = self.config.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": content,
            "contexts": [getattr(c, "page_content", str(c)) for c in raw_chunks],
            "retrieval": {
                "reranker_enabled": reranker_is_active,
                "reranker_applied": rerank_result.applied,
                "reranker_fallback_used": rerank_result.fallback_used,
            },
        }

    def add_documents(self, documents, metadatas=None, s3_key=None):
        return self.config.vector_manager.add_documents(documents, metadatas, s3_key)

    def delete_documents(self, doc_ids: List[str]) -> bool:
        return self.config.vector_manager.delete_documents(doc_ids)
