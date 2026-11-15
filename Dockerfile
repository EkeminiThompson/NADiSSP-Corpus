# Multi-stage image for the NADIP inference service (Table 3.15b). CPU only.
FROM python:3.12-slim AS builder
WORKDIR /w
COPY requirements-serve.txt .
RUN pip wheel --no-cache-dir -r requirements-serve.txt -w /wheels

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 NADIP_MODEL=/app/models/nadip.onnx NADIP_META=/app/data/processed/corpus_meta.json \
    NADIP_FEEDBACK_BUFFER=/app/buffer/feedback_buffer.jsonl
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY src/__init__.py src/config.py ./src/
COPY src/data ./src/data
COPY src/serving ./src/serving
# model + metadata are mounted or copied after running scripts/deployment_sim.py and scripts/prepare_corpus.py
COPY models ./models
COPY data/processed/corpus_meta.json ./data/processed/corpus_meta.json
RUN useradd -m nadip && mkdir -p /app/buffer && chown -R nadip /app
USER nadip
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "--factory", "src.serving.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
