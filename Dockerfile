# =============================================================================
# Dockerfile — XplagiaX AI Detection Microservice
#
# SERVER DEPLOYMENT WORKFLOW:
#   Before running docker build, place model weights at:
#     app/engine/modernbert.bin             (~570 MB)
#     app/engine/Model_groups_3class_seed12/ (~570 MB)
#     app/engine/Model_groups_3class_seed22/ (~570 MB)
#
#   Then build + run:
#     docker build -t xplagiax .
#     docker run -p 5006:5006 xplagiax
# =============================================================================

# ====================== STAGE 1: Builder ======================
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app

# System build deps (gcc needed by some Python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy requirements and install all Python packages in a SINGLE pip call.
# NOTE: torch>=2.2.2,<2.3 in requirements.txt is the local macOS pin.
#       On Python 3.12 Linux (this container) we override torch to >=2.4.0
#       which is compatible with numpy<2.0 and transformers<5.0.
#       We pass the overrides AFTER -r so they win over the file constraints.
COPY requirements.txt .
RUN pip install --user --no-cache-dir --prefer-binary \
    -r requirements.txt \
    "torch>=2.4.0" \
    "numpy>=1.26.4,<2.0" \
    "transformers>=4.48.2,<5.0" \
    "spacy>=3.7.0"

# ── Pre-download NLTK data (punkt_tab used by sentence splitting) ────────────
RUN python -c "\
import nltk; \
nltk.download('punkt_tab', download_dir='/root/nltk_data'); \
nltk.download('averaged_perceptron_tagger_eng', download_dir='/root/nltk_data')"

# ── Pre-download spaCy model (en_core_web_sm for HallucinationProfiler) ─────
RUN python -m spacy download en_core_web_sm

# ── Pre-cache HuggingFace tokenizer + config for ModernBERT-base ─────────────
# detector_final.py uses local_files_only=True so the HF cache MUST exist
# inside the container. This downloads only the tokenizer/config files (~5 MB),
# NOT the model weights (those come from app/engine/*.bin via COPY below).
RUN python -c "\
from transformers import AutoConfig, AutoTokenizer; \
AutoConfig.from_pretrained('answerdotai/ModernBERT-base', num_labels=41); \
AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base'); \
print('HF tokenizer/config cached OK')"


# ====================== STAGE 2: Runtime ======================
FROM python:3.12-slim-bookworm

# Non-root user (UID 1000 for K8s / rootless podman compatibility)
RUN useradd -m -u 1000 flaskuser

WORKDIR /app

# ── Python packages from builder ─────────────────────────────────────────────
COPY --from=builder /root/.local /home/flaskuser/.local

# ── NLTK data ────────────────────────────────────────────────────────────────
COPY --from=builder --chown=flaskuser:flaskuser \
    /root/nltk_data /home/flaskuser/nltk_data

# ── HuggingFace tokenizer/config cache ───────────────────────────────────────
# CRITICAL: without this, local_files_only=True in detector_final.py fails.
COPY --from=builder --chown=flaskuser:flaskuser \
    /root/.cache/huggingface /home/flaskuser/.cache/huggingface

# ── Environment ──────────────────────────────────────────────────────────────
# Thread counts prevent torch/numpy spawning N threads per gunicorn worker.
ENV PATH=/home/flaskuser/.local/bin:$PATH \
    PYTHONPATH=/app:/app/app/engine \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NLTK_DATA=/home/flaskuser/nltk_data \
    HF_HOME=/home/flaskuser/.cache/huggingface

# ── Application code ─────────────────────────────────────────────────────────
# app/engine/ contains the pre-placed model weights (*.bin, Model_groups_*)
# Make sure those files are present on the host BEFORE running docker build.
COPY --chown=flaskuser:flaskuser app.py .
COPY --chown=flaskuser:flaskuser gunicorn.conf.py .
COPY --chown=flaskuser:flaskuser app/ ./app/

# ── Runtime dirs ─────────────────────────────────────────────────────────────
RUN mkdir -p /app/models && chown flaskuser:flaskuser /app/models

USER flaskuser
EXPOSE 5006

# start-period=90s: ModernBERT models (~1.8 GB total) take 30-60 s to load.
# Increase if your server has slow disk / no SSD.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5006/health')" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
