from evaluate.ragas import clean_llm_response


def test_strips_think_block_and_doc_tags():
    raw = "<think>reasoning</think>The horse <doc>died</doc>."
    assert clean_llm_response(raw) == "The horse died."


def test_empty_input():
    assert clean_llm_response("") == ""
