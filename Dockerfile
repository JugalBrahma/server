FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV RERANKER_MODEL=ms-marco-MiniLM-L-12-v2 \
    RERANKER_CACHE_DIR=/opt/flashrank
RUN python -c "import os; from flashrank import Ranker; Ranker(model_name=os.environ['RERANKER_MODEL'], cache_dir=os.environ['RERANKER_CACHE_DIR'])"
COPY . .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --loop asyncio"]
