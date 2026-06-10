"""
app/__init__.py — Application factory + extension wiring.
"""

import os
import uuid

from flask import Flask, g, request
from flask_caching import Cache
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app.config import redis_url

cache = Cache()
compress = Compress()

# DT-05: Rate limiter — uses Redis when REDIS_URL is set, memory otherwise.
# memory:// is per-process (not shared across workers) — acceptable for dev;
# production should set REDIS_URL so limits are enforced cluster-wide.
#
# Fix #2: redis_url() injects REDIS_PASSWORD, and swallow_errors=True means a
# Redis hiccup degrades the limiter gracefully instead of hanging or 500-ing
# every request — including the "instant" /analyze_document_async enqueue.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url() if os.environ.get("REDIS_URL") else "memory://",
    storage_options={
        "max_connections": int(os.environ.get("REDIS_MAX_CONNECTIONS", "10")),
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
    },
    default_limits=["500/hour"],
    strategy="fixed-window",
    swallow_errors=True,
)


def create_app() -> Flask:
    """
    Application factory.

    Called ONCE by gunicorn --preload, then each forked worker inherits
    the fully-initialised app object via CoW.
    """
    app = Flask(__name__)

    # ── Load config ────────────────────────────────────────────────
    from app.config import Config
    app.config.from_object(Config)

    # ── Extensions ─────────────────────────────────────────────────
    cache.init_app(app)
    compress.init_app(app)
    limiter.init_app(app)

    # ── Auto-discover & register plugins ───────────────────────────
    from app.plugin_registry import registry
    registry.discover()
    app.config["PLUGIN_REGISTRY"] = registry

    # ── Blueprints ─────────────────────────────────────────────────
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    from app.antiplagio.flask_routes import register_antiplagio_routes
    register_antiplagio_routes(app)

    # ── DT-13: Request correlation ID ─────────────────────────────
    # Propagates caller-supplied X-Request-ID or generates an 8-char UUID.
    # Returned in the response header so clients can correlate log lines.
    @app.before_request
    def _set_request_id() -> None:
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]

    @app.after_request
    def _add_request_id_header(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        return response

    # ── S-02: Block DEBUG in production (Werkzeug RCE risk) ───────
    if app.config.get("DEBUG") and os.environ.get("FLASK_ENV", "development") == "production":
        raise RuntimeError(
            "DEBUG=True is forbidden in production (S-02: Werkzeug RCE). "
            "Unset the DEBUG environment variable."
        )

    # ── S-05: Require API_KEY in production ────────────────────────
    _is_prod = os.environ.get("FLASK_ENV", "development") == "production"
    if _is_prod and not app.config.get("API_KEY"):
        raise RuntimeError(
            "API_KEY must be set in production (S-05: all endpoints publicly "
            "accessible without it). Set the API_KEY environment variable."
        )
    elif not app.config.get("API_KEY"):
        app.logger.warning("API_KEY is not set — all endpoints are publicly accessible")

    # ── S-12: Warn on example.com CROSSREF_EMAIL ──────────────────
    if app.config.get("CROSSREF_EMAIL", "").endswith("@example.com"):
        app.logger.warning(
            "CROSSREF_EMAIL uses example.com — CrossRef may throttle or block "
            "requests. Set CROSSREF_EMAIL to a real institutional address."
        )

    # ── Startup log ────────────────────────────────────────────────
    app.logger.info(
        "TextAnalyzer ready — %d plugins: %s",
        len(registry),
        ", ".join(registry.list_plugins()),
    )

    return app
