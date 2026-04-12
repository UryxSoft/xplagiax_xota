"""
app/config.py — All configuration via environment variables.

Twelve-Factor App compliant.  Every setting has a sensible default
so the service starts with zero env vars for local development.
"""

import os


class Config:
    # ── Flask ──────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    # ── Limits ─────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_MB", "16")) * 1024 * 1024

    # ── Caching (Flask-Caching) ────────────────────────────────────
    CACHE_TYPE = os.getenv("CACHE_TYPE", "SimpleCache")
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", "300"))

    # ── Compression (Flask-Compress) ───────────────────────────────
    COMPRESS_ALGORITHM = "gzip"
    COMPRESS_MIN_SIZE = 256

    # ── Celery (optional, for heavy async plugins) ─────────────────
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # ── Plugin settings ────────────────────────────────────────────
    PLUGIN_DIR = os.getenv("PLUGIN_DIR", "app/plugins")
    PLUGIN_TIMEOUT = int(os.getenv("PLUGIN_TIMEOUT", "30"))  # seconds per plugin

    # ── Logging ────────────────────────────────────────────────────
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
