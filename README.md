# RAG Backend

A FastAPI service for uploading PDFs, searching document content, and answering questions with an LLM.

## Project folders

- `agent/` contains the question-answering agent and vector search manager.
- `config/` contains service and model configuration.
- `evaluate/` contains evaluation datasets and metrics.
- `rag/` contains PDF extraction, image handling, and chunk preparation.
- `routes/` contains the FastAPI endpoints.
- `scripts/` contains manual scripts for trying the API and ingestion pipeline.
- `services/` contains shared service helpers.

## Run locally

Install the packages from `requirements.txt`, configure the required environment variables, then run:

```sh
uvicorn main:app --reload
```

Use `.env.example` as a starting point for local configuration.

Run manual scripts from the project root with Python's module syntax, for example:

```sh
python -m scripts.run_pipeline
python -m scripts.check_agent_endpoint
python -m scripts.check_upload_endpoint
python -m scripts.check_retrieval
```

The API scripts read `API_URL` and `FIREBASE_TOKEN` from the environment. The upload script also needs `PDF_PATH`, and the retrieval script needs `RETRIEVAL_NAMESPACE`.

## Documentation

- [LangSmith setup](docs/LANGSMITH_SETUP_GUIDE.md)
