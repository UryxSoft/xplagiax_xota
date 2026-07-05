"""
app/config.py — All configuration via environment variables.

Twelve-Factor App compliant.  Every setting has a sensible default
so the service starts with zero env vars for local development.
"""

import os
import secrets


def redis_url() -> str:
    """
    REDIS_URL with REDIS_PASSWORD injected when set (Fix #2).

    The docker-compose redis service may run with --requirepass $REDIS_PASSWORD,
    but REDIS_URL defaults to `redis://redis:6379` with no credentials. Every
    Redis call (Celery broker enqueue, rate-limiter, cache) would then fail with
    NOAUTH and retry/hang — making the "instant" async endpoint block. This
    builds an authenticated URL so producer and limiter connect cleanly.
    """
    url = os.environ.get("REDIS_URL", "redis://redis:6379")
    password = os.environ.get("REDIS_PASSWORD", "")
    # Only inject if a password exists and the URL has no credentials yet.
    if password and "@" not in url.split("//", 1)[-1]:
        scheme, sep, rest = url.partition("//")
        url = f"{scheme}{sep}:{password}@{rest}"
    return url


def _require_secret_key() -> str:
    """Return SECRET_KEY from env, generating a secure fallback for local dev only."""
    key = os.environ.get("SECRET_KEY", "")
    if key:
        return key
    if os.environ.get("FLASK_ENV", "development") == "production":
        raise RuntimeError(
            "SECRET_KEY environment variable is required in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    # Local development only — ephemeral key, never persisted
    return secrets.token_hex(32)


class Config:
    # ── Flask ──────────────────────────────────────────────────────
    SECRET_KEY = _require_secret_key()
    DEBUG = os.environ.get("DEBUG", "0") == "1"

    # ── Limits ─────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024

    # ── Caching (Flask-Caching) ────────────────────────────────────
    # RedisCache is shared across all gunicorn workers; SimpleCache is per-process
    # and misses 50%+ of requests in multi-worker deployments.
    CACHE_TYPE = "RedisCache" if os.environ.get("REDIS_URL") else "SimpleCache"
    CACHE_REDIS_URL = redis_url()
    CACHE_DEFAULT_TIMEOUT = 300
    # Explicit pool cap — prevents runaway connections under burst traffic.
    REDIS_MAX_CONNECTIONS = int(os.environ.get("REDIS_MAX_CONNECTIONS", "10"))
    # max_connections is a RedisCache pool option — SimpleCache (used when no
    # REDIS_URL) rejects it and would crash create_app() on flask-caching >= 2.3.
    # Only pass it when actually using Redis.
    CACHE_OPTIONS = (
        {"max_connections": REDIS_MAX_CONNECTIONS}
        if os.environ.get("REDIS_URL") else {}
    )

    # ── Compression (Flask-Compress) ───────────────────────────────
    COMPRESS_ALGORITHM = "gzip"
    COMPRESS_MIN_SIZE = 256

    # ── Celery (optional, for heavy async plugins) ─────────────────
    # redis_url() injects REDIS_PASSWORD so the broker enqueue authenticates
    # instead of hanging on NOAUTH (Fix #2/#3).
    REDIS_URL = redis_url()
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
