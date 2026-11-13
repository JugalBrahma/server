import threading

from pinecone_text.sparse import BM25Encoder

import agent.bm25_store as bm25_store
from agent.bm25_store import Bm25Stats, Bm25Store

PDF_A = ["the cat sat on the mat", "the dog chased the cat"]
PDF_B = ["photosynthesis converts light into energy", "plants use light and water"]
PDF_C = ["the cat climbed a tall tree", "energy flows through the food chain"]


class FakeBackend:
    """In-memory S3 + Firestore with the same compare-and-swap pointer semantics."""

    def __init__(self, share: "FakeBackend | None" = None):
        # A backend sharing another's storage acts like a second server: same data,
        # separate in-memory encoder cache.
        self.objects = share.objects if share else {}
        self.pointers = share.pointers if share else {}
        self.lock = share.lock if share else threading.Lock()

    def read_pointer(self, namespace):
        # The token is the pointer itself: every swap writes a fresh, unique key.
        key = self.pointers.get(namespace)
        return key, key

    def swap_pointer(self, namespace, token, new, stats):
        with self.lock:
            if self.pointers.get(namespace) != token:
                return False
            self.pointers[namespace] = new
            return True

    def delete_pointer(self, namespace):
        self.pointers.pop(namespace, None)

    def read_object(self, key):
        return self.objects.get(key)

    def write_object(self, key, data):
        self.objects[key] = data

    def delete_object(self, key):
        self.objects.pop(key, None)

    def delete_prefix(self, prefix):
        for key in [k for k in self.objects if k.startswith(prefix)]:
            del self.objects[key]


def fitted(corpus):
    return BM25Encoder().fit(corpus)


def assert_same_encoder(actual, expected):
    assert actual.doc_freq == expected.doc_freq
    assert actual.n_docs == expected.n_docs
    assert abs(actual.avgdl - expected.avgdl) < 1e-9
    for query in ["cat energy", "photosynthesis", "tall tree light"]:
        assert actual.encode_queries(query) == expected.encode_queries(query)
    for doc in PDF_A + PDF_C:
        assert actual.encode_documents(doc) == expected.encode_documents(doc)


def test_incremental_adds_match_a_full_refit():
    store = Bm25Store("user_t1", backend=FakeBackend())
    store.add(PDF_A, "a.pdf")
    store.add(PDF_B, "b.pdf")
    encoder = store.add(PDF_C, "c.pdf")
    assert_same_encoder(encoder, fitted(PDF_A + PDF_B + PDF_C))


def test_delete_subtracts_exactly_one_pdf():
    store = Bm25Store("user_t2", backend=FakeBackend())
    for texts, source in [(PDF_A, "a.pdf"), (PDF_B, "b.pdf"), (PDF_C, "c.pdf")]:
        store.add(texts, source)
    store.remove_source("b.pdf")
    assert_same_encoder(store.encoder, fitted(PDF_A + PDF_C))


def test_deleting_everything_leaves_no_encoder_and_no_counts():
    backend = FakeBackend()
    store = Bm25Store("user_t3", backend=backend)
    store.add(PDF_A, "a.pdf")
    store.remove_source("a.pdf")
    assert store.encoder is None
    totals = Bm25Stats.from_json(backend.objects[backend.pointers["user_t3"]])
    assert totals.n_docs == 0 and totals.doc_freq == {}


def test_empty_namespace_has_no_encoder():
    assert Bm25Store("user_empty", backend=FakeBackend()).encoder is None


def test_no_chunk_text_is_stored():
    backend = FakeBackend()
    Bm25Store("user_t4", backend=backend).add(PDF_B, "b.pdf")
    stored = b"".join(backend.objects.values())
    assert b"photosynthesis" not in stored and b"light" not in stored


def test_other_server_sees_new_uploads():
    saved = bm25_store.REFRESH_SECONDS
    bm25_store.REFRESH_SECONDS = 0  # check the pointer on every read
    try:
        server_1 = FakeBackend()
        server_2 = FakeBackend(share=server_1)
        writer = Bm25Store("user_t5", backend=server_1)
        reader = Bm25Store("user_t5", backend=server_2)
        writer.add(PDF_A, "a.pdf")
        assert reader.encoder.n_docs == 2
        writer.add(PDF_B, "b.pdf")
        assert reader.encoder.n_docs == 4
    finally:
        bm25_store.REFRESH_SECONDS = saved


def test_chat_on_the_same_server_sees_uploads_immediately():
    # The old code kept a stale encoder in the cached chat agent until restart.
    backend = FakeBackend()
    chat = Bm25Store("user_t9", backend=backend)
    upload = Bm25Store("user_t9", backend=backend)
    upload.add(PDF_A, "a.pdf")
    assert chat.encoder.n_docs == 2
    upload.add(PDF_B, "b.pdf")
    assert chat.encoder.n_docs == 4


def test_concurrent_uploads_lose_nothing():
    backend = FakeBackend()
    pdfs = [[f"chunk {i} about topic{i} and shared words", f"more text for pdf{i}"] for i in range(12)]

    def upload(i):
        Bm25Store("user_t6", backend=backend).add(pdfs[i], f"{i}.pdf")

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(len(pdfs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    totals = Bm25Stats.from_json(backend.objects[backend.pointers["user_t6"]])
    expected = fitted([text for pdf in pdfs for text in pdf])
    assert totals.n_docs == expected.n_docs
    assert totals.doc_freq == expected.doc_freq
    # Losers clean up after themselves: one totals object plus one stats file per PDF.
    assert len(backend.objects) == 1 + len(pdfs)


def test_never_downloads_the_default_params():
    original = BM25Encoder.default

    def fail():
        raise AssertionError("BM25Encoder.default() downloads 65 MB and must not be called")

    BM25Encoder.default = staticmethod(fail)
    try:
        store = Bm25Store("user_t7", backend=FakeBackend())
        store.add(PDF_A, "a.pdf")
        store.remove_source("a.pdf")
        store.encoder
    finally:
        BM25Encoder.default = original


def test_clear_removes_all_state():
    backend = FakeBackend()
    store = Bm25Store("user_t8", backend=backend)
    store.add(PDF_A, "a.pdf")
    Bm25Store("user_other", backend=backend).add(PDF_B, "b.pdf")
    store.clear()
    assert store.encoder is None
    assert "user_t8" not in backend.pointers
    assert all("user_t8" not in key for key in backend.objects)
    assert "user_other" in backend.pointers
