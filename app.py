"""
app.py — Application entry point.

Gunicorn calls create_app() via --preload.  All heavy model loading
happens at *module import time* (inside app/plugins/) so Linux CoW
shares the memory pages across forked workers.

Usage:
    gunicorn --preload -c gunicorn.conf.py "app:create_app()"
"""

import os

# ── Freeze thread counts BEFORE any model import ──────────────────
# Prevents torch/numpy from spawning N threads per worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass  # torch not installed — lightweight mode

from app import create_app  # noqa: E402  (after env setup)

# Dev-only runner — production uses gunicorn
if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5006, debug=True)
