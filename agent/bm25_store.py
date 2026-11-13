"""BM25 statistics per namespace, kept in S3 with a version pointer in Firestore.

BM25 needs three numbers, and all three are sums over chunks:
  doc_freq  for each token, the number of chunks that contain it
  n_docs    the number of chunks
  sum_len   the total number of tokens (avgdl = sum_len / n_docs)

So each PDF's contribution is stored on its own, added to the namespace totals on
upload and subtracted on delete. Nothing is refit and no chunk text is stored.

Layout:
  S3         bm25/{namespace}/pdfs/{sha1(source)}.json   one PDF's stats
  S3         bm25/{namespace}/totals/{uuid}.json         namespace totals (never modified)
  Firestore  bm25_state/{namespace}                      {totals_key, version, n_docs, sum_len}

An update writes a new totals object, then moves the Firestore pointer to it with
a conditional write (only if the pointer doc is unchanged since it was read: one
round trip, unlike a transaction's three). If someone else moved it first, the
update retries on top of their totals, so concurrent uploads for the same user
never overwrite each other.

Readers cache each namespace's encoder in memory and re-read the pointer at most
every BM25_REFRESH_SECONDS, so every instance picks up uploads made on another one.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple

from pinecone_text.sparse import BM25Encoder

logger = logging.getLogger(__name__)

POINTER_COLLECTION = "bm25_state"
S3_PREFIX = "bm25"
MAX_UPDATE_ATTEMPTS = 8
REFRESH_SECONDS = float(os.getenv("BM25_REFRESH_SECONDS", "5"))
CACHE_SIZE = int(os.getenv("BM25_CACHE_SIZE", "256"))

# Only used to tokenize; never fitted. A plain BM25Encoder() downloads nothing.
_tokenizer = BM25Encoder()

# Runs independent S3 calls side by side.
_io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bm25-io")


# ------------------------------------------------------------------ statistics

@dataclass
class Bm25Stats:
    doc_freq: Dict[int, int] = field(default_factory=dict)
    n_docs: int = 0
    sum_len: int = 0

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "Bm25Stats":
        """Count exactly what BM25Encoder.fit() counts, for these texts only."""
        doc_freq: Counter = Counter()
        n_docs = sum_len = 0
        for text in texts:
            indices, tf = _tokenizer._tf(text)
            if not indices:
                continue
            n_docs += 1
            sum_len += sum(tf)
            doc_freq.update(indices)
        return cls(dict(doc_freq), n_docs, sum_len)

    def combine(self, other: "Bm25Stats", sign: int) -> "Bm25Stats":
        """Add (sign=1) or subtract (sign=-1) another set of stats."""
        doc_freq = dict(self.doc_freq)
        for token, count in other.doc_freq.items():
            value = doc_freq.get(token, 0) + sign * count
            if value > 0:
                doc_freq[token] = value
            else:
                doc_freq.pop(token, None)
        return Bm25Stats(
            doc_freq,
            max(0, self.n_docs + sign * other.n_docs),
            max(0, self.sum_len + sign * other.sum_len),
        )

    def to_encoder(self) -> Optional[BM25Encoder]:
        """A ready-to-use encoder, or None when there is nothing to score against."""
        if self.n_docs <= 0:
            return None
        encoder = BM25Encoder()
        encoder.doc_freq = dict(self.doc_freq)
        encoder.n_docs = self.n_docs
        encoder.avgdl = self.sum_len / self.n_docs
        return encoder

    def to_json(self) -> bytes:
        return json.dumps({
            "doc_freq": {str(k): v for k, v in self.doc_freq.items()},
            "n_docs": self.n_docs,
            "sum_len": self.sum_len,
        }).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "Bm25Stats":
        raw = json.loads(data)
        return cls(
            {int(k): int(v) for k, v in raw.get("doc_freq", {}).items()},
            int(raw.get("n_docs", 0)),
            int(raw.get("sum_len", 0)),
        )


# --------------------------------------------------------------------- storage

class Bm25Backend(Protocol):
    def read_pointer(self, namespace: str) -> Tuple[Optional[str], Any]:
        """(totals_key, token); token identifies the pointer version that was read (None = no pointer)."""
        ...

    def swap_pointer(self, namespace: str, token: Any, new: str, stats: Bm25Stats) -> bool:
        """Point to `new` only if the pointer is still at `token`. False if someone else moved it."""
        ...

    def delete_pointer(self, namespace: str) -> None: ...
    def read_object(self, key: str) -> Optional[bytes]: ...
    def write_object(self, key: str, data: bytes) -> None: ...
    def delete_object(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...


class S3FirestoreBackend:
    """Big data (token counts) in S3; the small version pointer in Firestore."""

    def __init__(self):
        from google.api_core import exceptions as gexc
        from google.cloud import firestore as gcf

        from config.firebase import get_firestore
        from config.s3 import BUCKET_NAME, s3

        self._gcf = gcf
        self._conflicts = (gexc.Conflict, gexc.FailedPrecondition, gexc.NotFound)
        self._db = get_firestore()
        self._s3 = s3
        self._bucket = BUCKET_NAME

    def _ref(self, namespace: str):
        return self._db.collection(POINTER_COLLECTION).document(namespace)

    def read_pointer(self, namespace: str) -> Tuple[Optional[str], Any]:
        snap = self._ref(namespace).get()
        if not snap.exists:
            return None, None
        data = snap.to_dict() or {}
        return data.get("totals_key"), (snap.update_time, int(data.get("version", 0)))

    def swap_pointer(self, namespace: str, token: Any, new: str, stats: Bm25Stats) -> bool:
        ref = self._ref(namespace)
        data = {
            "totals_key": new,
            "n_docs": stats.n_docs,
            "sum_len": stats.sum_len,
            "unique_tokens": len(stats.doc_freq),
            "updated_at": self._gcf.SERVER_TIMESTAMP,
        }
        try:
            if token is None:
                ref.create({**data, "version": 1})  # fails if another writer created it first
            else:
                update_time, version = token
                ref.update(
                    {**data, "version": version + 1},
                    option=self._db.write_option(last_update_time=update_time),  # fails if changed since read
                )
            return True
        except self._conflicts:
            return False

    def delete_pointer(self, namespace: str) -> None:
        self._ref(namespace).delete()

    def read_object(self, key: str) -> Optional[bytes]:
        try:
            return self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None

    def write_object(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType="application/json")

    def delete_object(self, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                self._s3.delete_objects(Bucket=self._bucket, Delete={"Objects": keys})


_default_backend: Optional[Bm25Backend] = None
_default_backend_lock = threading.Lock()


def default_backend() -> Bm25Backend:
    global _default_backend
    with _default_backend_lock:
        if _default_backend is None:
            _default_backend = S3FirestoreBackend()
        return _default_backend


# ----------------------------------------------------------------------- store

@dataclass
class _CacheEntry:
    totals_key: Optional[str]
    encoder: Optional[BM25Encoder]
    checked_at: float


class Bm25Store:
    """BM25 state of one namespace. Instances share one process-wide encoder cache."""

    _cache: "OrderedDict[Tuple[int, str], _CacheEntry]" = OrderedDict()
    _cache_lock = threading.Lock()

    def __init__(self, namespace: str, backend: Optional[Bm25Backend] = None):
        self.namespace = namespace
        self._backend = backend or default_backend()
        self._cache_key = (id(self._backend), namespace)

    # -- keys

    def _totals_key(self) -> str:
        return f"{S3_PREFIX}/{self.namespace}/totals/{uuid.uuid4().hex}.json"

    def _source_key(self, source: str) -> str:
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        return f"{S3_PREFIX}/{self.namespace}/pdfs/{digest}.json"

    # -- read

    @property
    def encoder(self) -> Optional[BM25Encoder]:
        """The current encoder (None if the namespace has no chunks yet)."""
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(self._cache_key)
            if entry and now - entry.checked_at < REFRESH_SECONDS:
                self._cache.move_to_end(self._cache_key)
                return entry.encoder

        totals_key, _ = self._backend.read_pointer(self.namespace)
        if entry and entry.totals_key == totals_key:
            self._remember(totals_key, entry.encoder)
            return entry.encoder

        totals_key, stats = self._load_totals(totals_key)
        encoder = stats.to_encoder()
        self._remember(totals_key, encoder)
        return encoder

    def _load_totals(self, totals_key: Optional[str]) -> Tuple[Optional[str], Bm25Stats]:
        """Read the totals the pointer names; re-read the pointer if a writer just replaced them."""
        for _ in range(3):
            if totals_key is None:
                return None, Bm25Stats()
            data = self._backend.read_object(totals_key)
            if data is not None:
                return totals_key, Bm25Stats.from_json(data)
            latest, _ = self._backend.read_pointer(self.namespace)
            if latest == totals_key:
                break
            totals_key = latest
        logger.error("BM25 totals %s for %s are missing; treating as empty", totals_key, self.namespace)
        return totals_key, Bm25Stats()

    def _remember(self, totals_key: Optional[str], encoder: Optional[BM25Encoder]) -> None:
        with self._cache_lock:
            self._cache[self._cache_key] = _CacheEntry(totals_key, encoder, time.monotonic())
            self._cache.move_to_end(self._cache_key)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)

    # -- write

    def add(self, texts: Iterable[str], source: str) -> Optional[BM25Encoder]:
        """Count one PDF's chunks into the totals. Returns the updated encoder."""
        delta = Bm25Stats.from_texts(texts)
        source_key, source_data = self._source_key(source), delta.to_json()
        totals, encoder = self._update(
            delta, sign=1, also=lambda: self._backend.write_object(source_key, source_data)
        )
        logger.info(
            "BM25 %s: +%d chunks from %s (now %d chunks, %d tokens)",
            self.namespace, delta.n_docs, source, totals.n_docs, len(totals.doc_freq),
        )
        return encoder

    def remove_source(self, source: str) -> None:
        """Subtract one PDF's counts from the totals."""
        key = self._source_key(source)
        data = self._backend.read_object(key)
        if data is None:
            logger.warning("BM25 %s: no stats stored for %s; nothing to subtract", self.namespace, source)
            return
        totals, _ = self._update(
            Bm25Stats.from_json(data), sign=-1, also=lambda: self._backend.delete_object(key)
        )
        logger.info("BM25 %s: removed %s (now %d chunks)", self.namespace, source, totals.n_docs)

    def clear(self) -> None:
        self._backend.delete_pointer(self.namespace)
        self._backend.delete_prefix(f"{S3_PREFIX}/{self.namespace}/")
        with self._cache_lock:
            self._cache.pop(self._cache_key, None)

    def _update(
        self, delta: Bm25Stats, sign: int, also: Callable[[], None]
    ) -> Tuple[Bm25Stats, Optional[BM25Encoder]]:
        """Apply a change to the totals with optimistic concurrency, retrying on conflict.

        `also` (saving or deleting the PDF's own stats file) runs only once the change
        is committed, in parallel with removing the replaced totals object.
        """
        for attempt in range(MAX_UPDATE_ATTEMPTS):
            pointer_key, token = self._backend.read_pointer(self.namespace)
            base_key, base = self._load_totals(pointer_key)
            totals = base.combine(delta, sign)
            new_key = self._totals_key()
            self._backend.write_object(new_key, totals.to_json())

            # If _load_totals had to follow a newer pointer, `token` is stale and this fails.
            if self._backend.swap_pointer(self.namespace, token, new_key, totals):
                cleanup = _io_pool.submit(self._backend.delete_object, base_key) if base_key else None
                also()
                if cleanup:
                    try:
                        cleanup.result()
                    except Exception as exc:  # an orphan object is harmless
                        logger.warning("Could not delete old BM25 totals %s: %s", base_key, exc)
                encoder = totals.to_encoder()
                self._remember(new_key, encoder)
                return totals, encoder

            # Another writer moved the pointer first: discard ours and redo on top of theirs.
            self._backend.delete_object(new_key)
            time.sleep(random.uniform(0.05, 0.2) * (attempt + 1))

        raise RuntimeError(f"BM25 update for {self.namespace} kept conflicting; gave up")
