# ====================== STAGE 1: Builder ======================
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download NLTK data needed by summarization/sentence splitting
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/root/.local/share/nltk_data')"

# ====================== STAGE 2: Runtime ======================
FROM python:3.12-slim-bookworm

# Non-root user (UID 1000 for K8s compatibility)
RUN useradd -m -u 1000 flaskuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/flaskuser/.local

# Copy NLTK data
COPY --from=builder /root/.local/share/nltk_data /home/flaskuser/.local/share/nltk_data

ENV PATH=/home/flaskuser/.local/bin:$PATH \
    PYTHONPATH=/app:/app/app/engine \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NLTK_DATA=/home/flaskuser/.local/share/nltk_data

# Copy application code + engine
COPY --chown=flaskuser:flaskuser app.py .
COPY --chown=flaskuser:flaskuser gunicorn.conf.py .
COPY --chown=flaskuser:flaskuser app/ ./app/

# Models directory (mount externally or bake in)
# docker run -v /path/to/models:/app/models ...
RUN mkdir -p /app/models && chown flaskuser:flaskuser /app/models

USER flaskuser
EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5006/health')" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
