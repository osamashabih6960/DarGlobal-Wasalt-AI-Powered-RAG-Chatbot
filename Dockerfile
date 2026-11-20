FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/.cache PORT=7860
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the vector index at image build time (needs data/pages.jsonl committed)
RUN python ingest.py && chmod -R a+rwX /app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
