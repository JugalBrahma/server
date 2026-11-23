import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_pinecone import PineconeVectorStore
from config.models import DEFAULT_EMBEDDING_DIM, get_embeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from .bm25_store import Bm25Store

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "test-2"
DENSE_DEFAULT_K = 2
MIN_CHUNK_CHARS = 10
MAX_METADATA_BYTES = 35_000  # Pinecone limit is 40KB
MAX_LIST_METADATA = 10
KEPT_METADATA_KEYS = ("page_number", "page", "pages", "languages")

# Each user has one namespace. Every vector stores the folder it belongs to, and
# every search filters on it. PDFs uploaded outside any folder use ROOT_FOLDER,
# because a Pinecone filter cannot match a missing field.
ROOT_FOLDER = "root"


def user_namespace(user_id: str) -> str:
    return f"user_{user_id}"


def folder_key(folder_id: Optional[str]) -> str:
    return folder_id or ROOT_FOLDER


def folder_filter(folder_id: Optional[str]) -> Dict[str, Any]:
    return {"folder_id": {"$eq": folder_key(folder_id)}}


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class VectorConfig:
    index_name: str
    api_key: str
    region: str
    hybrid_alpha: float
    hybrid_requested: bool

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
        )


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


def _build_metadata(
    i: int,
    s3_key: Optional[str],
    fields: Dict,
    raw: Dict,
    file_name: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> Dict:
    # folder_id lives in `base` so it survives the oversized-metadata fallback below.
    base = {
        "doc_id": f"doc_{i}",
        "s3_key": s3_key or "user_upload",
        "folder_id": folder_key(folder_id),
    }
    if file_name:
        base["file_name"] = file_name
    metadata = {
        **base,
        "content_type": fields.get("type", "text"),
        "has_image": bool(fields.get("has_image", False)),
        **_safe_metadata(raw),
    }
    if len(json.dumps(metadata).encode("utf-8")) > MAX_METADATA_BYTES:
        return {**base, "content_type": "text"}
    return metadata


def normalize_for_index(
    documents: List[Any],
    metadatas: Optional[List[Dict]] = None,
    s3_key: Optional[str] = None,
    file_name: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """Filter noise and build clean (texts, metadatas) ready for indexing."""
    texts, cleaned = [], []
    for i, doc in enumerate(documents):
        fallback = metadatas[i] if metadatas and i < len(metadatas) else None
        content, raw, fields = _unpack(doc, fallback)
        if not content or _is_noise(content):
            continue
        texts.append(content)
        cleaned.append(_build_metadata(i, s3_key, fields, raw, file_name, folder_id))
    return texts, cleaned


# --------------------------------------------------------------- manager

class VectorSearchManager:
    """Vector search over Pinecone, with optional BM25 hybrid retrieval."""

    def __init__(self, index_name: Optional[str] = None, namespace: str = DEFAULT_NAMESPACE):
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
        # BM25 statistics live in S3 + Firestore and are shared by every instance.
        self.bm25: Optional[Bm25Store] = Bm25Store(namespace) if self.hybrid_enabled else None
        if not self.hybrid_enabled:
            logger.info("Hybrid search disabled. Using dense embeddings.")

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

    def _hybrid_retriever(self, encoder: BM25Encoder, k: Optional[int] = None) -> PineconeHybridSearchRetriever:
        """A hybrid retriever using the given (current) BM25 encoder."""
        retriever = PineconeHybridSearchRetriever(
            embeddings=self.embeddings,
            sparse_encoder=encoder,
            index=self.index,
            alpha=self.config.hybrid_alpha,
            namespace=self.namespace,
            text_key="text",
        )
        return retriever if k is None else retriever.model_copy(update={"top_k": k})

    def _current_encoder(self) -> Optional[BM25Encoder]:
        """The namespace's BM25 encoder; None means use dense search."""
        if not self.bm25:
            return None
        try:
            return self.bm25.encoder
        except Exception as e:
            logger.warning("BM25 state unavailable for %s, using dense search: %s", self.namespace, e)
            return None

    # -- search

    def search(self, query: str, folder_id: Optional[str], k: Optional[int] = None) -> List[Any]:
        """Search one folder of this namespace: hybrid if available, otherwise dense.

        `folder_id` is required (None means the root folder) so a search can never
        return chunks from the user's other folders. `k` defaults per mode.
        """
        flt = folder_filter(folder_id)
        try:
            encoder = self._current_encoder()
            if encoder:
                return self._hybrid_retriever(encoder, k).invoke(query, filter=flt)
            return self.vector_store.similarity_search(query, k=k or DENSE_DEFAULT_K, filter=flt)
        except Exception as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return []

    # -- write

    def add_documents(
        self,
        documents: List[Any],
        metadatas: Optional[List[Dict]] = None,
        s3_key: Optional[str] = None,
        file_name: Optional[str] = None,
        folder_id: Optional[str] = None,
    ) -> bool:
        try:
            texts, cleaned = normalize_for_index(documents, metadatas, s3_key, file_name, folder_id)
            if not texts:
                logger.warning("No valid text chunks to index.")
                return False

            # Count this PDF into the BM25 totals first, so its sparse vectors are
            # encoded with statistics that include it.
            encoder = self.bm25.add(texts, s3_key or "unknown") if self.bm25 else None

            if encoder:
                self._hybrid_retriever(encoder).add_texts(texts=texts, metadatas=cleaned, namespace=self.namespace)
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
            return True
        except Exception as e:
            logger.error("Failed to delete documents for %s: %s", s3_key, e)
            return False

    def delete_documents_by_folder(self, folder_id: Optional[str]) -> bool:
        """Delete every vector of one folder; the user's other folders are untouched.

        BM25 counts are not touched here: they are subtracted per PDF in
        delete_documents_by_s3_key, and a folder can only be deleted once empty.
        """
        try:
            self.index.delete(filter=folder_filter(folder_id), namespace=self.namespace)
            return True
        except Exception as e:
            logger.error("Failed to delete folder %s in %s: %s", folder_key(folder_id), self.namespace, e)
            return False

    def delete_namespace(self) -> bool:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            if self.bm25:
                self.bm25.clear()
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
