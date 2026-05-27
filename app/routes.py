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
import functools
from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = current_app.config.get("API_KEY", "")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════
# POST /analyze
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze", methods=["POST"])
@require_api_key
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
# POST /analyze_document
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze_document", methods=["POST"])
@require_api_key
def analyze_document():
    """
    Analyze a long document with dynamic plugins AND per-segment breakdown.

    Runs any requested plugins via the registry (same as /analyze), then
    additionally runs the per-paragraph HybridSegmentAnalyzer and merges
    the segment scores into the ai_detection result (if requested).

    Request JSON:
        {
            "text": "...",                          # required
            "plugins": ["ai_detection", "..."]      # optional — defaults to ["ai_detection"]
        }

    Response JSON:
        {
            "status": "ok",
            "word_count": 320,
            "total_elapsed_ms": 6268.1,
            "plugins_requested": ["ai_detection"],
            "results": {
                "ai_detection": {
                    "status": "ok",
                    "elapsed_ms": 6000.0,
                    "data": {
                        ...global scores...,
                        "segments": [
                            {
                                "segment_id": 1,
                                "text": "paragraph text...",
                                "dominant_label": "AI",
                                "score": 90.67,
                                "forensic_analysis": {}
                            }
                        ]
                    }
                }
            }
        }
    """
    t0 = time.perf_counter()

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = payload.get("text", "")
    if not text or not isinstance(text, str):
        return jsonify({"error": "'text' field is required and must be a non-empty string"}), 400

    # Optional plugins list — mirrors /analyze; defaults to ai_detection
    plugins_requested = payload.get("plugins", ["ai_detection"])
    if not isinstance(plugins_requested, list) or not plugins_requested:
        return jsonify({"error": "'plugins' must be a non-empty list"}), 400
    plugins_requested = [str(p).strip().lower() for p in plugins_requested]

    registry = current_app.config["PLUGIN_REGISTRY"]
    timeout = current_app.config.get("PLUGIN_TIMEOUT", 30)

    # ── 1. Run all requested plugins (same engine as /analyze) ────
    results = registry.run(plugins_requested, text, timeout=timeout)

    # ── 2. Per-segment analysis via analyze_long_document ────────
    max_tokens = int(payload.get("max_tokens", 150))
    doc_result = {}
    try:
        import app.engine  # noqa — ensures sys.path is set
        from detector_final import analyze_long_document
        doc_result = analyze_long_document(text, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning("analyze_long_document failed: %s", exc)

    segments = doc_result.get("segments", [])

    # ── 3. Enrich ai_detection result with segments + overall summary ──
    if "ai_detection" in results and doc_result:
        ai_result = results["ai_detection"]
        if ai_result.get("status") == "ok" and isinstance(ai_result.get("data"), dict):
            ai_result["data"]["segments"] = segments
            
            summary = doc_result.get("overall_summary", {})
            ai_result["data"]["overall_summary"] = summary
            
            if summary:
                ai_result["data"]["human_percentage"] = summary.get("total_human_percentage", 50)
                ai_result["data"]["ai_percentage"] = summary.get("total_ai_percentage", 50)
                ai_result["data"]["confidence"] = max(summary.get("total_human_percentage", 50), summary.get("total_ai_percentage", 50))
                ai_result["data"]["prediction"] = summary.get("overall_prediction", "Unknown")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Document analyzed: %d words, plugins=%s, %d segments in %.1fms",
        len(text.split()), plugins_requested, len(segments), elapsed * 1000,
    )

    return jsonify({
        "status": "ok",
        "word_count": len(text.split()),
        "plugins_requested": plugins_requested,
        "results": results,
        "total_elapsed_ms": round(elapsed * 1000, 1),
    })


# ═══════════════════════════════════════════════════════════════════
# ASYNC ENDPOINTS (Celery)
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze_document_async", methods=["POST"])
@require_api_key
def analyze_document_async():
    """
    Enqueue the document analysis task and return immediately.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = payload.get("text", "")
    if not text or not isinstance(text, str):
        return jsonify({"error": "'text' field is required and must be a non-empty string"}), 400

    # Optional plugins
    plugins_requested = payload.get("plugins", ["ai_detection"])
    if not isinstance(plugins_requested, list) or not plugins_requested:
        return jsonify({"error": "'plugins' must be a non-empty list"}), 400
    
    payload["plugins"] = [str(p).strip().lower() for p in plugins_requested]

    try:
        from app.tasks import analyze_document_task
        task = analyze_document_task.delay(payload)
        return jsonify({"status": "accepted", "task_id": task.id}), 202
    except Exception as e:
        logger.error(f"Error enqueueing task: {e}")
        return jsonify({"error": "Failed to enqueue task"}), 500


@api_bp.route("/analyze_status/<task_id>", methods=["GET"])
@require_api_key
def analyze_status(task_id):
    """
    Check the status of an async analysis task.
    """
    from app.celery_app import celery
    task = celery.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'status': 'pending',
            'state': task.state
        }
    elif task.state != 'FAILURE':
        response = {
            'status': 'processing' if task.state != 'SUCCESS' else 'ok',
            'state': task.state
        }
        if task.state == 'SUCCESS':
            # task.info contains the returned dict from the task
            # The task returns {"status": "ok", "results": ...}
            response.update(task.info)
    else:
        # something went wrong in the background job
        response = {
            'status': 'error',
            'state': task.state,
            'error': str(task.info)  # exception raised
        }
    return jsonify(response)


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
