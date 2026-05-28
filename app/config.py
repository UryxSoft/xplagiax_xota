"""
app/config.py — All configuration via environment variables.

Twelve-Factor App compliant.  Every setting has a sensible default
so the service starts with zero env vars for local development.
"""

import os


class Config:
    # ── Flask ──────────────────────────────────────────────────────
    SECRET_KEY =  "edw-32fdx-34f421-m56e"
    DEBUG =  "1"

    # ── Limits ─────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # ── Caching (Flask-Caching) ────────────────────────────────────
    CACHE_TYPE =  "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 300

    # ── Compression (Flask-Compress) ───────────────────────────────
    COMPRESS_ALGORITHM = "gzip"
    COMPRESS_MIN_SIZE = 256

    # ── Celery (optional, for heavy async plugins) ─────────────────
    REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", f"{REDIS_URL}/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", f"{REDIS_URL}/1")

    # ── Plugin settings ────────────────────────────────────────────
    PLUGIN_DIR = "app/plugins"
    PLUGIN_TIMEOUT = 30  # seconds per plugin

    # ── API Security ───────────────────────────────────────────────
    API_KEY = os.environ.get("API_KEY", "")  # empty = auth disabled

    # ── Antiplagio / Citation validator ────────────────────────────
    CROSSREF_EMAIL = os.environ.get("CROSSREF_EMAIL", "antiplagio@example.com")

    # ── Logging ────────────────────────────────────────────────────
    LOG_LEVEL = "INFO"
