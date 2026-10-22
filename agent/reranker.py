import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    """Documents selected by the reranker and whether FlashRank was applied."""

    documents: List[Any]
    applied: bool
    fallback_used: bool


class DocumentReranker:
    """Lightweight neural cross-encoder reranker powered by FlashRank."""

    # Agents are cached per namespace, so an instance attribute would load one
    # ONNX session for every user/folder. Keep sessions process-wide instead.
    _rankers: ClassVar[Dict[Tuple[str, str], Ranker]] = {}
    _rankers_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = (
            model_name
            or os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
        )
        self.cache_dir = cache_dir or os.getenv("RERANKER_CACHE_DIR", "/tmp/flashrank")

    @property
    def ranker(self) -> Ranker:
        key = (self.model_name, self.cache_dir)
        ranker = self._rankers.get(key)
        if ranker is None:
            with self._rankers_lock:
                ranker = self._rankers.get(key)
                if ranker is None:
                    logger.info("Initializing FlashRank reranker with model '%s'", self.model_name)
                    ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
                    self._rankers[key] = ranker
        return ranker

    def warm(self) -> None:
        """Load the shared model before it is needed by a user request."""
        _ = self.ranker

    def rerank(self, query: str, documents: List[Any], top_k: int = 5) -> List[Any]:
        """Backwards-compatible shorthand that returns only selected documents."""
        return self.rerank_with_status(query, documents, top_k).documents

    def rerank_with_status(
        self,
        query: str,
        documents: List[Any],
        top_k: int = 5,
    ) -> RerankResult:
        """Rerank candidates and explicitly report whether the model was used."""
        if not documents:
            return RerankResult([], applied=False, fallback_used=False)
        if len(documents) <= 1:
            return RerankResult(documents[:top_k], applied=False, fallback_used=False)
        if top_k <= 0:
            logger.warning("Reranking skipped because top_k must be positive; got %d", top_k)
            return RerankResult([], applied=False, fallback_used=True)

        try:
            passages = []
            for idx, doc in enumerate(documents):
                text = (
                    getattr(doc, "page_content", "")
                    or (doc.get("content", "") if isinstance(doc, dict) else str(doc))
                )
                passages.append({"id": idx, "text": text})

            rerank_request = RerankRequest(query=query, passages=passages)
            ranked = self.ranker.rerank(rerank_request)

            reranked_docs = []
            for item in ranked[:top_k]:
                original_doc = documents[item["id"]]
                if hasattr(original_doc, "metadata") and isinstance(original_doc.metadata, dict):
                    original_doc.metadata["rerank_score"] = float(item.get("score", 0.0))
                reranked_docs.append(original_doc)

            logger.info("Reranked %d candidates down to %d", len(documents), len(reranked_docs))
            return RerankResult(reranked_docs, applied=True, fallback_used=False)
        except Exception as exc:
            logger.exception("Reranking failed; using original candidate order: %s", exc)
            return RerankResult(documents[:top_k], applied=False, fallback_used=True)
