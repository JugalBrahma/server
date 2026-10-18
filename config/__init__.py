"""Configuration package with lazy exports.

Importing a focused module such as ``config.models`` should not initialize
unrelated services (for example, S3) or require their credentials.
"""

from importlib import import_module


_EXPORTS = {
    "s3": (".s3", "s3"),
    "BUCKET_NAME": (".s3", "BUCKET_NAME"),
    "upload_file_to_s3": (".s3", "upload_file_to_s3"),
    "download_pdf_from_s3": (".s3", "download_pdf_from_s3"),
    "get_firestore": (".firebase", "get_firestore"),
    "verify_firebase_token": (".firebase", "verify_firebase_token"),
    "get_current_user_id": (".firebase", "get_current_user_id"),
    "init_langsmith": (".langsmith", "init_langsmith"),
    "get_groq_llm": (".models", "get_groq_llm"),
    "get_openai_llm": (".models", "get_openai_llm"),
    "get_rag_llm": (".models", "get_rag_llm"),
    "get_guardrail_llm": (".models", "get_guardrail_llm"),
    "get_vision_llm": (".models", "get_vision_llm"),
    "get_evaluator_llm": (".models", "get_evaluator_llm"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a config export only when a caller requests it."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
