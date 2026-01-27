# Railway Deployment Guide

This project already contains a working `Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Railway automatically injects the `PORT` environment variable, so no extra start command is required. Follow the steps below to deploy the FastAPI server to Railway.

---

## 1. Prerequisites

1. Install the Railway CLI (optional but recommended):
   ```bash
   npm i -g @railway/cli
   railway login
   ```
2. Make sure your repository is pushed to GitHub (Railway deploys from Git or directly from CLI).

---

## 2. Required Environment Variables

Set these in the Railway dashboard (Project → Variables) **before** deploying:

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini API key for embeddings + LLM |
| `PINECONE_API_KEY` | Pinecone serverless index access |
| `PINECONE_ENVIRONMENT` *(if used)* | Only required for legacy Pinecone setups |
| `FIREBASE_PROJECT_ID` | Firestore connection |
| `FIREBASE_CLIENT_EMAIL` | Service account email |
| `FIREBASE_PRIVATE_KEY` | Multiline JSON key (wrap with `"` or use Railway Secret files) |
| `FIREBASE_DATABASE_URL` | Firestore RTDB URL if needed |
| `BM25_STORAGE_DIR` *(optional)* | Override BM25 cache path if you mount a volume |
| `ENABLE_PINECONE_HYBRID` | `true`/`false` to toggle hybrid search |
| `PINECONE_HYBRID_ALPHA` | Blend ratio for hybrid search |
| Anything else from `.env` | (e.g., analytics keys, S3 creds) |

> Tip: copy the local `.env` entries (without quotes) into Railway’s Variables panel. For multiline secrets (Firebase key), use Railway’s “Raw Text” mode.

---

## 3. Deploy Steps

1. **Create project**: In Railway dashboard → “New Project” → “Deploy from GitHub” (or `railway up`).
2. **Select repo**: Choose this repository and the `server/` directory as the build context.
3. **Set root**: In the Railway service settings, set the “Root Directory” to `server` so that `requirements.txt`, `Procfile`, and `main.py` are discovered.
4. **Python environment**:
   - Railway auto-detects `requirements.txt` and installs dependencies.
   - Ensure `python` build image is selected (default for requirements).
5. **Start command**: Leave blank (Railway will use the `Procfile`). If you need to override, set it to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
6. **Deploy**: Trigger a deploy. Watch the logs to confirm the app boots and listens on the assigned port.

---

## 4. Verifying Deployment

After the build succeeds, Railway shows a service URL like `https://fusion-chat-production.up.railway.app`. Test it:

```bash
curl https://<railway-url>/health
```

You should get `{"status":"healthy"}` from `routes.debug_pdf` defaults (or 404 if that route isn’t mounted). Update your Flutter `ApiConfig.baseUrl` with this URL.

---

## 5. Tips

- **Memory**: Choose a plan with ≥1 GB RAM; hybrid Pinecone + Gemini embeddings exceed 512 MB.
- **BM25 cache**: By default it writes to `server/bm25_cache/`. For Railway, mount a persistent volume or set `BM25_STORAGE_DIR=/data/bm25_cache` and attach a storage volume.
- **Logs**: Use Railway’s “Logs” tab or `railway logs` CLI to monitor errors.
- **Secrets rotation**: If you regenerate API keys, update Railway variables immediately.

This guide keeps the repo ready for Railway while staying compatible with other platforms.
