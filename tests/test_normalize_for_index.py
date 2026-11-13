from agent.vector_manager import ROOT_FOLDER, folder_filter, normalize_for_index, user_namespace


def test_every_chunk_records_its_folder():
    _, metadatas = normalize_for_index(["a real sentence here"], folder_id="f1")
    assert metadatas[0]["folder_id"] == "f1"


def test_chunks_without_a_folder_go_to_root():
    _, metadatas = normalize_for_index(["a real sentence here"])
    assert metadatas[0]["folder_id"] == ROOT_FOLDER


def test_folder_survives_oversized_metadata_fallback():
    chunk = {"content": "a real sentence here", "metadata": {"languages": ["x" * 5000] * 10}}
    _, metadatas = normalize_for_index([chunk], folder_id="f1")
    assert metadatas[0]["folder_id"] == "f1"


def test_filter_and_namespace():
    assert folder_filter("f1") == {"folder_id": {"$eq": "f1"}}
    assert folder_filter(None) == {"folder_id": {"$eq": ROOT_FOLDER}}
    assert user_namespace("u1") == "user_u1"


def test_drops_noise_and_keeps_text():
    texts, metadatas = normalize_for_index(["a real sentence here", "12", "tiny", ""], s3_key="k.pdf")
    assert texts == ["a real sentence here"]
    assert metadatas[0]["s3_key"] == "k.pdf"
    assert metadatas[0]["content_type"] == "text"


def test_dict_chunks_keep_type_and_safe_metadata():
    chunk = {"content": "chunk with an image", "type": "image", "has_image": True,
             "metadata": {"page_number": 3, "unknown_key": "dropped"}}
    _, metadatas = normalize_for_index([chunk])
    assert metadatas[0]["content_type"] == "image"
    assert metadatas[0]["has_image"] is True
    assert metadatas[0]["page_number"] == 3
    assert "unknown_key" not in metadatas[0]
