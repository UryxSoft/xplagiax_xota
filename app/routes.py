"""
app/routes.py — API blueprint.

Endpoints
---------
POST /analyze      Main analysis endpoint — runs requested plugins.
GET  /health       Liveness probe (always 200 if process is alive).
GET  /ready        Readiness probe (200 only if plugins are loaded).
GET  /plugins      List available plugins with descriptions.
"""

from __future__ import annotations

import time
import logging
from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# ═══════════════════════════════════════════════════════════════════
# POST /analyze
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Run requested plugins on the submitted text.

    Request JSON:
        {
            "text": "...",
            "plugins": ["sentiment", "keyphrases", ...]
        }

    Response JSON:
        {
            "status": "ok",
            "word_count": 1234,
            "plugins_requested": [...],
            "results": {
                "sentiment": {"status": "ok", "data": {...}, "elapsed_ms": 12.3},
                ...
            },
            "total_elapsed_ms": 45.6
        }
    """
    t0 = time.perf_counter()

    # ── Validate request ──────────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = payload.get("text", "")
    plugins_requested = payload.get("plugins", [])

    if not text or not isinstance(text, str):
        return jsonify({"error": "'text' field is required and must be a non-empty string"}), 400

    if not plugins_requested or not isinstance(plugins_requested, list):
        return jsonify({"error": "'plugins' field is required and must be a non-empty list"}), 400

    # Sanitise plugin names
    plugins_requested = [str(p).strip().lower() for p in plugins_requested]

    # ── Run plugins ───────────────────────────────────────────────
    registry = current_app.config["PLUGIN_REGISTRY"]
    timeout = current_app.config.get("PLUGIN_TIMEOUT", 30)

    results = registry.run(plugins_requested, text, timeout=timeout)

    elapsed = time.perf_counter() - t0

    logger.info(
        "Analyzed %d words with %d plugins in %.1fms",
        len(text.split()), len(plugins_requested), elapsed * 1000,
    )

    return jsonify({
        "status": "ok",
        "word_count": len(text.split()),
        "plugins_requested": plugins_requested,
        "results": results,
        "total_elapsed_ms": round(elapsed * 1000, 1),
    })


# ═══════════════════════════════════════════════════════════════════
# GET /health — Kubernetes liveness probe
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/health", methods=["GET"])
def health():
    """Always 200 if the process is alive."""
    return jsonify({"status": "healthy"}), 200


# ═══════════════════════════════════════════════════════════════════
# GET /ready — Kubernetes readiness probe
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/ready", methods=["GET"])
def ready():
    """200 only if plugins are loaded and ready to serve."""
    registry = current_app.config.get("PLUGIN_REGISTRY")
    if registry is None or len(registry) == 0:
        return jsonify({
            "status": "not_ready",
            "reason": "No plugins loaded",
        }), 503

    return jsonify({
        "status": "ready",
        "plugins_loaded": len(registry),
        "plugins": registry.list_plugins(),
    }), 200


# ═══════════════════════════════════════════════════════════════════
# GET /plugins — Plugin catalogue
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/plugins", methods=["GET"])
def list_plugins():
    """Return all registered plugins with descriptions."""
    registry = current_app.config["PLUGIN_REGISTRY"]
    return jsonify({
        "count": len(registry),
        "plugins": registry.list_plugins_with_info(),
    })


# ═══════════════════════════════════════════════════════════════════
# GET /report/<path> — Serve generated HTML forensic reports
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/report/<path:filename>", methods=["GET"])
def serve_report(filename):
    """Serve a generated HTML forensic report from /tmp."""
    import os
    filepath = os.path.join("/tmp", os.path.basename(filename))
    if not os.path.isfile(filepath):
        return jsonify({"error": "Report not found"}), 404

    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()

    from flask import Response
    return Response(html, mimetype="text/html")
