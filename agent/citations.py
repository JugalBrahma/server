import re
from typing import Any, Dict, List, Tuple

# Matches citation markers such as [2] or [1, 3], plus the whitespace before them.
CITATION_PATTERN = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


def chunk_pages(metadata: Dict[str, Any]) -> List[int]:
    """Page numbers a chunk spans, read from its index metadata."""
    raw = metadata.get("pages") or metadata.get("page_number") or metadata.get("page")
    values = raw if isinstance(raw, list) else [raw]
    pages = []
    for value in values:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return pages


def resolve_citations(answer: str, chunk_refs: Dict[int, Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Turn the chunk IDs the LLM cited into sources taken from chunk metadata.

    Returns the answer with all citation markers removed, and one source per cited
    file (in order of first citation) listing only the pages of the cited chunks.
    IDs that were not in the prompt are dropped.
    """
    sources: Dict[str, Dict[str, Any]] = {}
    for match in CITATION_PATTERN.finditer(answer):
        for chunk_id in (int(n) for n in match.group(1).split(",")):
            metadata = chunk_refs.get(chunk_id)
            if metadata is None:
                continue
            s3_key = metadata.get("s3_key", "unknown_source")
            source = sources.setdefault(s3_key, {
                "file_name": metadata.get("file_name"),
                "s3_key": s3_key,
                "pages": set(),
            })
            source["pages"].update(chunk_pages(metadata))

    clean_answer = CITATION_PATTERN.sub("", answer).strip()
    return clean_answer, [{**s, "pages": sorted(s["pages"])} for s in sources.values()]
