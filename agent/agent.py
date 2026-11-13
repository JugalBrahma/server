import os
import logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from .citations import resolve_citations
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
        self.fetch_k = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))


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
        folder_id: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ) -> Tuple[str, List[Any], RerankResult, Dict[int, Dict[str, Any]]]:
        """
        Retrieves matching document chunks from Pinecone vector store,
        optionally reranks them using neural cross-attention (FlashRank),
        and formats them for the LLM with a numbered ID per chunk.

        Also returns the metadata of each chunk keyed by its ID, so citations
        the LLM makes can be resolved to files and pages.
        """
        try:
            logger.info(
                "Retrieving chunks for query '%s' in namespace '%s', folder '%s'",
                query, self.namespace, folder_id or "root",
            )
            reranker = self._active_reranker(enable_reranker)

            # Step 1: Query this folder only for top matching candidate chunks
            fetch_k = self.deps.fetch_k if reranker else self.deps.rerank_top_k
            results = self.deps.vector_manager.search(query=query, folder_id=folder_id, k=fetch_k)
            if not results:
                return "", [], RerankResult([], applied=False, fallback_used=False), {}

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

            # Step 3: Group chunks by source file
            results_by_doc = defaultdict(list)
            for doc in results:
                metadata = getattr(doc, "metadata", {}) or {}
                s3_key = metadata.get("s3_key", "unknown_source")
                results_by_doc[s3_key].append({
                    "content": doc.page_content,
                    "metadata": metadata,
                    "content_type": metadata.get("content_type", "unknown"),
                })

            # Step 4: Sort chunks by page number and label each chunk with a numbered ID.
            # Page numbers stay out of the prompt: they are added from metadata via the cited IDs.
            formatted = []
            chunk_refs: Dict[int, Dict[str, Any]] = {}
            for i, (s3_key, chunks) in enumerate(results_by_doc.items(), 1):
                chunks.sort(key=lambda c: c["metadata"].get("page_number", 0))
                passages = []
                for chunk in chunks:
                    chunk_id = len(chunk_refs) + 1
                    chunk_refs[chunk_id] = chunk["metadata"]
                    content_type = chunk["content_type"]
                    label = f"[{chunk_id}]" if content_type in ("text", "unknown") else f"[{chunk_id}] ({content_type})"
                    passages.append(f"{label}\n{chunk['content']}")

                file_name = chunks[0]["metadata"].get("file_name") or s3_key
                header = f"[Document {i}]\nSource: {file_name}\n"
                formatted.append(f"{header}Passages:\n" + "\n\n".join(passages) + "\n" + "=" * 50 + "\n")

            return "\n".join(formatted), results, rerank_result, chunk_refs
        except Exception:
            logger.exception("Retrieval error for namespace '%s'", self.namespace)
            return "", [], RerankResult([], applied=False, fallback_used=False), {}

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
            "Highlight key facts by wrapping them in <doc> and </doc> tags. Wrap only short exact phrases taken from the documents, "
            "such as a name, number, date, place or brief quote of a few words. Never wrap a whole sentence or a whole clause; "
            "most of each sentence must stay in plain text. "
            "Example: 'The bridge opened in <doc>1932</doc> and cost <doc>$4.2 million</doc> [2].' "
            "Each passage is labeled with a numbered ID such as [1]. After each statement that uses a passage, cite its ID in square brackets, "
            "for example: 'He planted roses <doc>in his yard</doc> [2].' Cite several IDs as [1, 3]. Only cite IDs that appear in the context. "
            "Do not mention page numbers or file names yourself; sources are added to your answer automatically. "
            "If the context does not contain the answer, politely state that the information was not found in the documents, without citing anything. "
            "Be concise: answer the question directly and stop. Do not add closing offers, follow-up questions or lines such as "
            "'let me know if you meant something else'."
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
        6. Resolve the chunk IDs the LLM cited into sources (file and pages) from chunk metadata.
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
                "sources": [],
                "retrieval": {
                    "reranker_enabled": reranker_is_active,
                    "reranker_applied": False,
                    "reranker_fallback_used": False,
                },
            }

        # Step 3: Retrieve and rerank relevant document chunks from Pinecone
        retrieved_docs, raw_chunks, rerank_result, chunk_refs = self.retrieve_documents(
            user_input,
            folder_id=folder_id,
            enable_reranker=enable_reranker,
        )

        system_prompt, user_prompt = self._build_prompts(user_input, retrieved_docs)

        # Step 5: Invoke the LLM and extract the response
        response = self.deps.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)

        # Step 6: Map cited chunk IDs to files and pages; unknown IDs are dropped
        answer, sources = resolve_citations(content, chunk_refs)

        return {
            "answer": answer,
            "contexts": [getattr(c, "page_content", str(c)) for c in raw_chunks],
            "sources": sources,
            "retrieval": {
                "reranker_enabled": reranker_is_active,
                "reranker_applied": rerank_result.applied,
                "reranker_fallback_used": rerank_result.fallback_used,
            },
        }

    @traceable(run_type="chain", name="RAG Agent Query")
    def query_stream(
        self,
        user_input: str,
        folder_id: Optional[str] = None,
        user_id: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ) -> Iterator[dict]:
        """
        Same steps as query(), but sends the answer in pieces while the LLM writes it.

        Yields:
            {"type": "token", "text": "..."}                     one piece of the answer
            {"type": "done", "answer": "...", "sources": [...]}  the final, cleaned answer
        """
        self._attach_trace_metadata(folder_id, user_id)

        # 1. Greetings get the fixed reply, with no search and no LLM call
        is_greeting, greeting_reply = check_greeting_guardrail(user_input)
        if is_greeting and greeting_reply:
            yield {"type": "done", "answer": greeting_reply, "sources": []}
            return

        # 2. Find and rerank the passages (same as query())
        retrieved_docs, _, _, chunk_refs = self.retrieve_documents(
            user_input,
            folder_id=folder_id,
            enable_reranker=enable_reranker,
        )
        system_prompt, user_prompt = self._build_prompts(user_input, retrieved_docs)

        # 3. Ask the LLM, passing each piece on as soon as it arrives
        full_answer = ""
        for piece in self.deps.llm.stream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]):
            text = piece.content if isinstance(piece.content, str) else ""
            if text:
                full_answer += text
                yield {"type": "token", "text": text}

        # 4. Turn the [1], [2] markers into sources (same as query())
        answer, sources = resolve_citations(full_answer, chunk_refs)
        yield {"type": "done", "answer": answer, "sources": sources}

    def add_documents(self, documents, metadatas=None, s3_key=None, file_name=None, folder_id=None):
        return self.deps.vector_manager.add_documents(documents, metadatas, s3_key, file_name, folder_id)

    def delete_documents(self, doc_ids: List[str]) -> bool:
        return self.deps.vector_manager.delete_documents(doc_ids)
