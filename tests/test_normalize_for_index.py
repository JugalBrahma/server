from agent.vector_manager import normalize_for_index


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
