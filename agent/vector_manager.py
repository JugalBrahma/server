import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_pinecone import PineconeVectorStore
from config.models import DEFAULT_EMBEDDING_DIM, get_embeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

logger = logging.getLogger(__name__)

DENSE_DEFAULT_K = 2
MIN_CHUNK_CHARS = 10
MAX_METADATA_BYTES = 35_000  # Pinecone limit is 40KB
MAX_LIST_METADATA = 10
KEPT_METADATA_KEYS = ("page_number", "page", "pages", "languages")


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class VectorConfig:
    index_name: str
    api_key: str
    region: str
    hybrid_alpha: float
    hybrid_requested: bool
    bm25_dir: Path

    @classmethod
    def from_env(cls, index_name: Optional[str] = None) -> "VectorConfig":
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        return cls(
            index_name=index_name
            or os.getenv("PINECONE_INDEX_NAME", os.getenv("AGENT_INDEX_NAME", "askadocument")),
            api_key=api_key,
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            hybrid_alpha=float(os.getenv("PINECONE_HYBRID_ALPHA", 0.5)),
            hybrid_requested=os.getenv("ENABLE_PINECONE_HYBRID", "true").lower() == "true",
            bm25_dir=Path(os.getenv("BM25_STORAGE_DIR", "bm25_cache")),
        )


# ---------------------------------------------------------- BM25 storage

class Bm25Store:
    """On-disk BM25 state for one namespace: corpus, source mapping, fitted encoder."""

    def __init__(self, directory: Path, namespace: str):
        directory.mkdir(parents=True, exist_ok=True)
        stem = namespace.replace("/", "_")
        self._state_path = directory / f"{stem}_bm25.json"
        self._corpus_path = directory / f"{stem}_corpus.json"
        self._mapping_path = directory / f"{stem}_mapping.json"
        self._lock = threading.Lock()
        self.encoder = self._load_encoder()

    # -- public

    def add(self, texts: List[str], source: str) -> None:
        with self._lock:
            corpus = self._read(self._corpus_path, [])
            mapping = self._read(self._mapping_path, {})
            start = len(corpus)
            corpus.extend(texts)
            mapping.update({str(start + i): source for i in range(len(texts))})
            self._save(corpus, mapping)
            self._refit(corpus)

    def remove_source(self, source: str) -> None:
        with self._lock:
            if not self._mapping_path.exists():
                return
            corpus = self._read(self._corpus_path, [])
            mapping = self._read(self._mapping_path, {})

            kept = [
                (text, mapping.get(str(i)))
                for i, text in enumerate(corpus)
                if mapping.get(str(i)) != source
            ]
            if not kept:
                self._clear_unlocked()
                return
            self._save(
                [t for t, _ in kept],
                {str(i): s for i, (_, s) in enumerate(kept)},
            )
            self._refit([t for t, _ in kept])

    def clear(self) -> None:
        with self._lock:
            self._clear_unlocked()

    # -- internals

    def _load_encoder(self) -> BM25Encoder:
        encoder = BM25Encoder().default()
        if self._state_path.exists():
            try:
                return encoder.load(str(self._state_path))
            except Exception as e:
                logger.warning("Failed to load BM25 encoder state: %s", e)
        return encoder

    @staticmethod
    def _read(path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _save(self, corpus: List[str], mapping: Dict[str, str]) -> None:
        self._corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
        self._mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    def _refit(self, corpus: List[str]) -> None:
        try:
            encoder = BM25Encoder().default()
            encoder.fit(corpus)
            encoder.dump(str(self._state_path))
            self.encoder = encoder
        except Exception as e:
            logger.error("Failed to update BM25 encoder: %s", e)

    def _clear_unlocked(self) -> None:
        for path in (self._state_path, self._corpus_path, self._mapping_path):
            path.unlink(missing_ok=True)
        self.encoder = BM25Encoder().default()


# --------------------------------------------------- chunk preparation

def _unpack(doc: Any, fallback_meta: Optional[Dict]) -> Tuple[str, Dict, Dict]:
    """Return (content, raw_metadata, doc_fields) for a str or dict chunk."""
    if isinstance(doc, str):
        return doc.strip(), fallback_meta or {}, {}
    if isinstance(doc, dict) and "content" in doc:
        return str(doc["content"]).strip(), doc.get("metadata") or {}, doc
    return "", {}, {}


def _is_noise(content: str) -> bool:
    if len(content) < MIN_CHUNK_CHARS:
        return True
    return content.isdigit() and len(content) <= 4


def _safe_metadata(raw: Dict) -> Dict:
    out: Dict[str, Any] = {}
    for key in KEPT_METADATA_KEYS:
        if key not in raw:
            continue
        val = raw[key]
        if isinstance(val, (str, int, float, bool)):
            out[key] = val
        elif isinstance(val, list):
            out[key] = [str(x) for x in val[:MAX_LIST_METADATA]]
    return out


def _build_metadata(i: int, s3_key: Optional[str], fields: Dict, raw: Dict) -> Dict:
    base = {"doc_id": f"doc_{i}", "s3_key": s3_key or "user_upload"}
    metadata = {
        **base,
        "content_type": fields.get("type", "text"),
        "has_image": bool(fields.get("has_image", False)),
        **_safe_metadata(raw),
    }
    if len(json.dumps(metadata).encode("utf-8")) > MAX_METADATA_BYTES:
        return {**base, "content_type": "text"}
    return metadata


def prepare_chunks(
    documents: List[Any],
    metadatas: Optional[List[Dict]] = None,
    s3_key: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """Filter noise and build clean (texts, metadatas) ready for indexing."""
    texts, cleaned = [], []
    for i, doc in enumerate(documents):
        fallback = metadatas[i] if metadatas and i < len(metadatas) else None
        content, raw, fields = _unpack(doc, fallback)
        if not content or _is_noise(content):
            continue
        texts.append(content)
        cleaned.append(_build_metadata(i, s3_key, fields, raw))
    return texts, cleaned


# --------------------------------------------------------------- manager

class VectorSearchManager:
    """Vector search over Pinecone, with optional BM25 hybrid retrieval."""

    def __init__(self, index_name: Optional[str] = None, namespace: str = "test-2"):
        self.config = VectorConfig.from_env(index_name)
        self.index_name = self.config.index_name
        self.namespace = namespace

        self.pc = Pinecone(api_key=self.config.api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
        self.metric = self._read_metric()

        self.embeddings = get_embeddings()
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name, embedding=self.embeddings, namespace=namespace
        )

        self.hybrid_enabled = self.config.hybrid_requested and self.metric == "dotproduct"
        self.bm25: Optional[Bm25Store] = None
        self.retriever: Optional[PineconeHybridSearchRetriever] = None
        if self.hybrid_enabled:
            self.bm25 = Bm25Store(self.config.bm25_dir, namespace)
            self._refresh_retriever()
        else:
            logger.info("Hybrid search disabled. Using dense embeddings.")

    @property
    def bm25_encoder(self) -> Optional[BM25Encoder]:  # backwards-compatible accessor
        return self.bm25.encoder if self.bm25 else None

    # -- setup

    def _ensure_index(self) -> None:
        if self.pc.has_index(self.index_name):
            return
        # Hybrid search needs dotproduct; dense-only can use cosine.
        metric = "dotproduct" if self.config.hybrid_requested else "cosine"
        logger.warning("Index '%s' not found. Creating (%s).", self.index_name, metric)
        self.pc.create_index(
            name=self.index_name,
            dimension=DEFAULT_EMBEDDING_DIM,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=self.config.region),
        )

    def _read_metric(self) -> str:
        try:
            return (self.pc.describe_index(name=self.index_name).metric or "").lower()
        except Exception as e:
            logger.warning("Could not read Pinecone index metric: %s", e)
            return ""

    def _refresh_retriever(self) -> None:
        """(Re)build the hybrid retriever so it picks up the latest BM25 encoder."""
        if not self.bm25:
            return
        try:
            self.retriever = PineconeHybridSearchRetriever(
                embeddings=self.embeddings,
                sparse_encoder=self.bm25.encoder,
                index=self.index,
                alpha=self.config.hybrid_alpha,
                namespace=self.namespace,
                text_key="text",
            )
        except Exception as e:
            logger.warning("Hybrid retriever initialization failed: %s", e)
            self.retriever = None

    # -- search

    def search(self, query: str, k: Optional[int] = None) -> List[Any]:
        """Hybrid search if available, otherwise dense. `k` defaults per mode."""
        try:
            if self.retriever:
                retriever = (
                    self.retriever
                    if k is None
                    else self.retriever.model_copy(update={"top_k": k})
                )
                return retriever.invoke(query)
            return self.vector_store.similarity_search(query, k=k or DENSE_DEFAULT_K)
        except Exception as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return []

    # -- write

    def add_documents(
        self,
        documents: List[Any],
        metadatas: Optional[List[Dict]] = None,
        s3_key: Optional[str] = None,
    ) -> bool:
        try:
            texts, cleaned = prepare_chunks(documents, metadatas, s3_key)
            if not texts:
                logger.warning("No valid text chunks to index.")
                return False

            if self.bm25:
                self.bm25.add(texts, s3_key or "unknown")
                self._refresh_retriever()  # must see the refit encoder before upserting

            if self.retriever:
                self.retriever.add_texts(texts=texts, metadatas=cleaned, namespace=self.namespace)
            else:
                self.vector_store.add_texts(texts=texts, metadatas=cleaned)

            logger.info("Added %d documents to namespace '%s'", len(texts), self.namespace)
            return True
        except Exception as e:
            logger.error("Failed to add documents to namespace '%s': %s", self.namespace, e)
            return False

    # -- delete

    def delete_documents(self, doc_ids: List[str]) -> bool:
        try:
            self.vector_store.delete(ids=doc_ids)
            return True
        except Exception as e:
            logger.error("Failed to delete documents: %s", e)
            return False

    def delete_documents_by_s3_key(self, s3_key: str) -> bool:
        try:
            self.index.delete(filter={"s3_key": s3_key}, namespace=self.namespace)
            if self.bm25:
                self.bm25.remove_source(s3_key)
                self._refresh_retriever()
            return True
        except Exception as e:
            logger.error("Failed to delete documents for %s: %s", s3_key, e)
            return False

    def delete_namespace(self) -> bool:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            if self.bm25:
                self.bm25.clear()
                self._refresh_retriever()
            return True
        except Exception as e:
            logger.error("Failed to delete namespace %s: %s", self.namespace, e)
            return False

    # -- info

    def get_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "namespace": self.namespace,
                "dimension": DEFAULT_EMBEDDING_DIM,
                "metric": self.metric,
                "vector_count": stats.get("total_vector_count", 0),
                "namespaces": stats.get("namespaces", {}),
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Stats:", VectorSearchManager().get_stats())