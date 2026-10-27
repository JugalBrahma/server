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
from .vector_manager import DEFAULT_NAMESPACE, VectorSearchManager

logger = logging.getLogger(__name__)


class AgentDependencies:
    """The LLM, vector store and reranker an agent works with."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
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
        namespace: str = DEFAULT_NAMESPACE,
        model_name: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ):
        self.deps = AgentDependencies(
            index_name=index_name,
            namespace=namespace,
            model_name=model_name,
            enable_reranker=enable_reranker,
        )
        self.namespace = namespace

    def _active_reranker(self, enable_reranker: Optional[bool]) -> Optional[DocumentReranker]:
        """The reranker to use for one call: a per-call override wins over the configured default."""
        if enable_reranker is False:
            return None
        return self.deps.reranker

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
            reranker = self._active_reranker(enable_reranker)

            # Step 1: Query vector database for top matching candidate chunks
            fetch_k = self.deps.fetch_k if reranker else self.deps.rerank_top_k
            results = self.deps.vector_manager.search(query=query, k=fetch_k)
            if not results:
                return "", [], RerankResult([], applied=False, fallback_used=False)

            # Step 2: Neural cross-encoder reranking
            if reranker:
                rerank_result = reranker.rerank_with_status(
                    query=query,
                    documents=results,
                    top_k=self.deps.rerank_top_k,
                )
                results = rerank_result.documents
            else:
                results = results[:self.deps.rerank_top_k]
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

    @staticmethod
    def _attach_trace_metadata(folder_id: Optional[str], user_id: Optional[str]) -> None:
        """Tag the current LangSmith run with the folder and user, if tracing is active."""
        try:
            run_tree = get_current_run_tree()
            if not run_tree:
                return
            for key, value in (("folder_id", folder_id), ("user_id", user_id)):
                if not value:
                    continue
                run_tree.extra.setdefault("metadata", {})[key] = value
                tag = f"{key.removesuffix('_id')}_{value}"
                if tag not in run_tree.tags:
                    run_tree.tags.append(tag)
        except Exception as exc:
            logger.warning("Failed to attach trace metadata: %s", exc)

    @staticmethod
    def _build_prompts(user_input: str, retrieved_docs: str) -> Tuple[str, str]:
        """Grounded QA prompts if documents were found, a fallback prompt if not."""
        if not retrieved_docs:
            system_prompt = (
                "You are a helpful assistant. If the user greets you, reply politely. "
                "If they ask a question about specific documents, inform them that no matching documents were found in this folder."
            )
            return system_prompt, user_input

        system_prompt = (
            "You are an intelligent document assistant. "
            "Answer the question based strictly on the retrieved document context below. "
            "Pay close attention to chronological and temporal qualifiers in questions (such as 'initially', 'first', 'before', 'later'). "
            "Wrap only the exact facts, quotes, figures, and direct information from the documents in <doc> and </doc> tags. Use normal plain text for your own connecting words, and do not wrap entire sentences in <doc> tags. "
            "If the context does not contain the answer, politely state that the information was not found in the documents."
        )
        user_prompt = f'User Question: "{user_input}"\n\nRetrieved Documents:\n{retrieved_docs}\n\nAnswer:'
        return system_prompt, user_prompt

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
        self._attach_trace_metadata(folder_id, user_id)

        # Step 2: Check greeting guardrail (fast-path for greetings)
        is_greeting, greeting_reply = check_greeting_guardrail(user_input)
        reranker_is_active = self._active_reranker(enable_reranker) is not None

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

        system_prompt, user_prompt = self._build_prompts(user_input, retrieved_docs)

        # Step 5: Invoke the LLM and extract the response
        response = self.deps.llm.invoke([
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
        return self.deps.vector_manager.add_documents(documents, metadatas, s3_key)

    def delete_documents(self, doc_ids: List[str]) -> bool:
        return self.deps.vector_manager.delete_documents(doc_ids)
