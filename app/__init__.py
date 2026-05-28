"""
app/__init__.py — Application factory + extension wiring.
"""

from flask import Flask
from flask_caching import Cache
from flask_compress import Compress

cache = Cache()
compress = Compress()


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

    # ── Auto-discover & register plugins ───────────────────────────
    from app.plugin_registry import registry
    registry.discover()
    app.config["PLUGIN_REGISTRY"] = registry

    # ── Blueprints ─────────────────────────────────────────────────
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    from app.antiplagio.flask_routes import register_antiplagio_routes
    register_antiplagio_routes(app)

    # ── Startup log ────────────────────────────────────────────────
    app.logger.info(
        "TextAnalyzer ready — %d plugins: %s",
        len(registry),
        ", ".join(registry.list_plugins()),
    )

    return app
