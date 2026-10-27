# Clean Code Review

A review of the RAG server (~3,400 lines, 51 tracked files). The codebase is small and mostly readable. The items below are duplication, leftover code and readability issues worth fixing.

## What already reads well

- `routes/upload.py`: small helpers, one route, comments that explain *why*.
- `agent/vector_manager.py`: sensible split into `Bm25Store`, helper functions and `VectorSearchManager`.
- `rag/`: clear flow of extract, process, describe images, chunk, then index.
- `services/token_tracker.py`, `agent/guardrail.py`, `agent/llm_manager.py`: short and focused.

## 1. Duplicated code

- [x] **Merge the two eval runners.** `evaluate/flutter_evaluator.py` and `evaluate/datasets_evaluate.py` both create the dataset, sync examples, strip `<think>`/`<doc>` tags and run the same evaluators. Keep one module.
- [x] **Keep one answer-cleaning helper.** `clean_answer` (`datasets_evaluate.py`), `_clean_answer` (`flutter_evaluator.py`) and `clean_llm_response` (`ragas.py`) do the same job.
- [x] **Rename one `prepare_chunks`.** One is in `rag/ingestion.py`, the other in `agent/vector_manager.py:184`, with different signatures. For example, rename the vector-manager one to `normalize_for_index`.

## 2. Dead or leftover code

- [x] **Delete the commented-out code in `config/models.py`** (the Groq and OpenAI functions and their imports). Git keeps the history.
- [x] **Check whether `evaluate_flutter_query` is used.** Only `evaluate/__init__.py` references it. If the Flutter app doesn't call it, remove the module.
- [x] **Remove the `__main__` block in `agent/vector_manager.py:370`.** `scripts/check_retrieval.py` already prints stats.
- [x] **Replace hard-coded values.**
  - The default namespace `"test-2"` appears in `agent`, `rag_pipeline` and `VectorSearchManager`. Define it once.
  - `datasets_evaluate.py` has a real-looking user and folder ID as its default namespace. Require it from the environment.

## 3. Too many small helper files

- [x] **`scripts/`**: four manual smoke tests (`check_agent_endpoint`, `check_upload_endpoint`, `check_retrieval`, `run_pipeline`), while `tests/` and `test/` are empty. Turn them into pytest tests in `tests/`, or keep one or two. Remove the empty duplicate folder.
- [x] **Move root `cleanup.py` into `scripts/`.** It deletes every namespace in the Pinecone index and has a hard-coded index name. Add a confirmation prompt.
- [x] **`routes/debug_pdf.py`**: a debug endpoint mounted in production with no auth seen. Gate it behind an env flag or remove it.

## 4. Readability

- [x] **Split `RAGAgent.query()`** (`agent/agent.py`). It handles tracing setup, the greeting check, retrieval, prompt building and the LLM call. Extract `_attach_trace_metadata()` and `_build_prompts()`.
- [x] **Remove the eval-specific rule from the system prompt.** The "initial diagnosis / sluggish circulation" text is tailored to the Country Doctor test story. Keep the prompt generic.
- [x] **Compute the reranker toggle once.** The `enable_reranker is None` expression is duplicated in `retrieve_documents` and `query`. Use a helper.
- [x] **Rework `AgentConfig`.** It builds the LLM, vector manager and reranker in `__init__`. Either rename it to reflect that (for example `AgentDependencies`) or build those objects in `RAGAgent`.
- [x] **Drop the `sys.path` hacks and `# noqa: E402`** in `evaluate/` and `scripts/`. Run them as modules, for example `python -m evaluate.datasets_evaluate`.
- [x] **Trim `__init__.py` re-exports** (`rag/__init__.py`, `services/__init__.py`) to what is actually imported from the package.
- [x] **Clean up `requirements.txt`.** `langchain-groq` appears unused. `langchain-google-vertexai` is only used for a ragas compatibility shim. Pin versions. _(`langchain-groq` removed; pinning versions skipped by choice.)_

## 5. Housekeeping

- [x] **Keep secrets and caches out of the Docker image.** `firebase-adminsdk.json`, `image_cache_gpt-4o-mini_v3.pkl` and `bm25_cache/` are untracked (correct for git). Confirm `.dockerignore` excludes them.

## Suggested order

1. Merge the eval runners and the cleaning helpers.
2. Delete dead and commented code.
3. Rename the duplicate `prepare_chunks`.
4. Split `RAGAgent.query()` and clean up the prompt.
5. Consolidate `scripts/` into real tests.

Steps 1 to 3 remove roughly 200 lines without changing behavior.
