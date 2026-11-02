from agent.citations import chunk_pages, resolve_citations

REFS = {
    1: {"s3_key": "a.pdf", "file_name": "Report.pdf", "page_number": 3},
    2: {"s3_key": "a.pdf", "file_name": "Report.pdf", "pages": ["4", "5"]},
    3: {"s3_key": "b.pdf", "page_number": 7},
}


def test_only_cited_pages_are_listed():
    answer, sources = resolve_citations("He planted roses <doc>in his yard</doc> [2].", REFS)
    assert answer == "He planted roses <doc>in his yard</doc>."
    assert sources == [{"file_name": "Report.pdf", "s3_key": "a.pdf", "pages": [4, 5]}]


def test_unknown_ids_are_dropped_and_stripped():
    answer, sources = resolve_citations("A fact [9]. Another [1, 9].", REFS)
    assert answer == "A fact. Another."
    assert sources == [{"file_name": "Report.pdf", "s3_key": "a.pdf", "pages": [3]}]


def test_groups_by_file_in_citation_order():
    _, sources = resolve_citations("X [3]. Y [1][2]. Z [1,3].", REFS)
    assert [s["s3_key"] for s in sources] == ["b.pdf", "a.pdf"]
    assert sources[0]["file_name"] is None
    assert sources[1]["pages"] == [3, 4, 5]


def test_no_citations():
    assert resolve_citations("Not found in the documents.", REFS) == ("Not found in the documents.", [])


def test_chunk_pages_ignores_missing_or_bad_values():
    assert chunk_pages({}) == []
    assert chunk_pages({"pages": ["2", "x"]}) == [2]
