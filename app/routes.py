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

import hashlib
import hmac
import json
import os
import time
import logging
import functools
from typing import Any, Dict
from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from app import cache, limiter
from app.plugin_registry import adaptive_timeout

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

_MAX_TEXT_CHARS = 500_000  # ~125 K words; prevents OOM on oversized payloads
_ANALYSIS_CACHE_TTL = 3600  # 1 hour — analysis of identical text+plugins reuses result

# Ceiling for plugin timeouts on the SYNCHRONOUS endpoints. Long documents
# should go through /analyze_document_async; this cap keeps sync requests
# from outliving typical client/proxy timeouts (gunicorn default: 120 s).
_SYNC_TIMEOUT_CAP = int(os.getenv("SYNC_PLUGIN_TIMEOUT_CAP", "100"))

# Async (Celery) budget: soft limit scales with document size up to 1 hour.
_ASYNC_SOFT_LIMIT_CAP = int(os.getenv("CELERY_SOFT_TIME_LIMIT_CAP", "3600"))


def _plugin_timeout(word_count: int, cap: int) -> int:
    """Document-size-aware per-plugin timeout (see plugin_registry.adaptive_timeout)."""
    cfg = current_app.config
    return adaptive_timeout(
        word_count,
        base=cfg.get("PLUGIN_TIMEOUT", 30),
        per_kwords=cfg.get("PLUGIN_TIMEOUT_PER_KWORDS", 15.0),
        cap=cap,
    )

# [C-16 FIX] Cache key is namespaced by model version so a model/pipeline update
# invalidates stale results instead of serving them for up to an hour. Bump
# MODEL_VERSION (env or constant) whenever weights, thresholds, or fusion change.
_MODEL_VERSION = os.getenv("MODEL_VERSION", "2026.07")


def _analysis_cache_key(text: str, plugins: list) -> str:
    """Deterministic cache key: sha256(model_version + text + sorted plugin list)."""
    content = _MODEL_VERSION + "\x00" + text + "\x00" + ",".join(sorted(plugins))
    return "analysis:" + hashlib.sha256(content.encode()).hexdigest()


def _drift_warning():
    """'model_drift_detected' when the drift monitor flagged degradation, else None.

    Computed per-response (not cached) so a degradation that starts AFTER a result
    was cached still reaches clients. Fail-open: monitoring must never 500 the API.
    """
    try:
        from app.engine.drift_monitor import get_drift_monitor
        return "model_drift_detected" if get_drift_monitor().is_degraded() else None
    except Exception:
        return None


def _merge_segment_results(results: Dict[str, Any], doc_result: Dict[str, Any]) -> None:
    """Enrich the ai_detection entry in *results* with per-segment data from *doc_result*.

    Mutates *results* in-place. Shared by the sync endpoint and the Celery task
    to avoid divergence from copy-pasted logic.
    """
    if "ai_detection" not in results or not doc_result:
        return
    ai_result = results["ai_detection"]
    if ai_result.get("status") != "ok" or not isinstance(ai_result.get("data"), dict):
        return

    ai_result["data"]["segments"] = doc_result.get("segments", [])
    summary = doc_result.get("overall_summary", {})
    ai_result["data"]["overall_summary"] = summary
    if summary:
        ai_result["data"]["human_percentage"] = summary.get("total_human_percentage", 50)
        ai_result["data"]["ai_percentage"] = summary.get("total_ai_percentage", 50)
        ai_result["data"]["confidence"] = max(
            summary.get("total_human_percentage", 50),
            summary.get("total_ai_percentage", 50),
        )
        ai_result["data"]["prediction"] = summary.get("overall_prediction", "Unknown")
        ai_result["data"]["detected_model"] = summary.get("detected_model")


def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = current_app.config.get("API_KEY", "")
        if api_key and not hmac.compare_digest(
            request.headers.get("X-API-Key", ""), api_key
        ):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════
# POST /analyze
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze", methods=["POST"])
@require_api_key
@limiter.limit("60/minute")
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

    if len(text) > _MAX_TEXT_CHARS:
        return jsonify({"error": f"Text too large. Maximum {_MAX_TEXT_CHARS} characters."}), 413

    if not plugins_requested or not isinstance(plugins_requested, list):
        return jsonify({"error": "'plugins' field is required and must be a non-empty list"}), 400

    # Sanitise plugin names
    plugins_requested = [str(p).strip().lower() for p in plugins_requested]

    # ── Result cache check ────────────────────────────────────────
    cache_key = _analysis_cache_key(text, plugins_requested)
    cached = cache.get(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        cached["warning"] = _drift_warning()
        return jsonify(cached)

    # ── Run plugins ───────────────────────────────────────────────
    registry = current_app.config["PLUGIN_REGISTRY"]

    word_count = len(text.split())  # pre-compute once before analysis
    timeout = _plugin_timeout(word_count, cap=_SYNC_TIMEOUT_CAP)
    results = registry.run(plugins_requested, text, timeout=timeout)

    elapsed = time.perf_counter() - t0

    logger.info(
        "Analyzed %d words with %d plugins in %.1fms",
        word_count, len(plugins_requested), elapsed * 1000,
    )

    response_data = {
        "status": "ok",
        "word_count": word_count,
        "plugins_requested": plugins_requested,
        "results": results,
        "model_version": _MODEL_VERSION,
        "warning": _drift_warning(),
        "total_elapsed_ms": round(elapsed * 1000, 1),
    }
    cache.set(cache_key, response_data, timeout=_ANALYSIS_CACHE_TTL)
    return jsonify(response_data)


# ═══════════════════════════════════════════════════════════════════
# POST /analyze_document
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze_document", methods=["POST"])
@require_api_key
@limiter.limit("10/minute")
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

    if len(text) > _MAX_TEXT_CHARS:
        return jsonify({"error": f"Text too large. Maximum {_MAX_TEXT_CHARS} characters."}), 413

    # Optional plugins list — mirrors /analyze; defaults to ai_detection
    plugins_requested = payload.get("plugins", ["ai_detection"])
    if not isinstance(plugins_requested, list) or not plugins_requested:
        return jsonify({"error": "'plugins' must be a non-empty list"}), 400
    plugins_requested = [str(p).strip().lower() for p in plugins_requested]

    registry = current_app.config["PLUGIN_REGISTRY"]
    timeout = _plugin_timeout(len(text.split()), cap=_SYNC_TIMEOUT_CAP)

    # ── 1. Run all requested plugins (same engine as /analyze) ────
    results = registry.run(plugins_requested, text, timeout=timeout)

    # ── 2. Per-segment analysis — skip if ai_detection already has segments ──
    doc_result = {}
    ai_data = (results.get("ai_detection") or {}).get("data") or {}
    if not ai_data.get("segments"):
        try:
            from app.engine.detector_final import analyze_fast
            doc_result = analyze_fast(text)  # P-05: single-pass, adaptive tokens, cached
        except Exception as exc:
            logger.warning("analyze_fast failed: %s", exc)

    segments = doc_result.get("segments", [])

    # ── 3. Enrich ai_detection result with segments + overall summary ──
    _merge_segment_results(results, doc_result)

    elapsed = time.perf_counter() - t0
    word_count = len(text.split())
    logger.info(
        "Document analyzed: %d words, plugins=%s, %d segments in %.1fms",
        word_count, plugins_requested, len(segments), elapsed * 1000,
    )

    return jsonify({
        "status": "ok",
        "word_count": word_count,
        "plugins_requested": plugins_requested,
        "results": results,
        "model_version": _MODEL_VERSION,
        "warning": _drift_warning(),
        "total_elapsed_ms": round(elapsed * 1000, 1),
    })


# ═══════════════════════════════════════════════════════════════════
# ASYNC ENDPOINTS (Celery)
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze_document_async", methods=["POST"])
@require_api_key
@limiter.limit("20/minute")
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

    # Fix #5: cap payload size like the other endpoints. Without this an oversized
    # text slows the JSON parse and bloats the Redis broker payload on enqueue,
    # turning the "instant" 202 endpoint slow (and risking worker OOM later).
    if len(text) > _MAX_TEXT_CHARS:
        return jsonify({"error": f"Text too large. Maximum {_MAX_TEXT_CHARS} characters."}), 413

    # Optional plugins
    plugins_requested = payload.get("plugins", ["ai_detection"])
    if not isinstance(plugins_requested, list) or not plugins_requested:
        return jsonify({"error": "'plugins' must be a non-empty list"}), 400
    
    payload["plugins"] = [str(p).strip().lower() for p in plugins_requested]

    try:
        from app.tasks import analyze_document_task
        # Scale the Celery time limits with document size: the static 240/300 s
        # decorator defaults are sized for papers, not theses. A 125 K-word
        # document on CPU needs a proportionally larger budget; the hard limit
        # trails the soft one so the task can still return a clean timeout error.
        word_count = len(text.split())
        soft_limit = _plugin_timeout(word_count, cap=_ASYNC_SOFT_LIMIT_CAP)
        soft_limit = max(soft_limit, 240)  # never below the decorator default
        task = analyze_document_task.apply_async(
            args=[payload],
            soft_time_limit=soft_limit,
            time_limit=soft_limit + 60,
        )
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
    elif task.state == 'SUCCESS':
        # [Fase-2 M-15/C-08] Normalized contract: `state` is always the Celery state,
        # `status` is always the ANALYSIS outcome. A task that completed but returned
        # an internal error (e.g. soft-timeout payload) reports status="error" with
        # state="SUCCESS" — coherent instead of the old ok/error mix.
        info = task.info if isinstance(task.info, dict) else {}
        response = {'state': task.state, **info}
        response['status'] = info.get('status', 'ok')
    elif task.state != 'FAILURE':
        response = {
            'status': 'processing',
            'state': task.state
        }
    else:
        # something went wrong in the background job
        response = {
            'status': 'error',
            'state': task.state,
            'error': str(task.info)  # exception raised
        }
    return jsonify(response)


# ═══════════════════════════════════════════════════════════════════
# POST /analyze_stream — SSE: results delivered as each plugin finishes
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/analyze_stream", methods=["POST"])
@require_api_key
@limiter.limit("30/minute")
def analyze_stream():
    """
    Server-Sent Events endpoint.  Results are streamed as each plugin completes
    instead of waiting for the slowest one.

    Events (text/event-stream):
        {"type": "init",   "word_count": N, "plugins": [...]}
        {"type": "result", "plugin": "ai_detection", "result": {...}}
        {"type": "done"}

    Client usage:
        const es = new EventSource('/analyze_stream', {method: 'POST', ...});
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = payload.get("text", "")
    if not text or not isinstance(text, str):
        return jsonify({"error": "'text' field is required and must be a non-empty string"}), 400

    if len(text) > _MAX_TEXT_CHARS:
        return jsonify({"error": f"Text too large. Maximum {_MAX_TEXT_CHARS} characters."}), 413

    plugins_requested = payload.get("plugins", [])
    if not plugins_requested or not isinstance(plugins_requested, list):
        return jsonify({"error": "'plugins' field is required and must be a non-empty list"}), 400
    plugins_requested = [str(p).strip().lower() for p in plugins_requested]

    registry = current_app.config["PLUGIN_REGISTRY"]
    word_count = len(text.split())
    timeout = _plugin_timeout(word_count, cap=_SYNC_TIMEOUT_CAP)

    def _generate():
        yield f"data: {json.dumps({'type': 'init', 'word_count': word_count, 'plugins': plugins_requested})}\n\n"
        for pname, result in registry.run_stream(plugins_requested, text, timeout=timeout):
            yield f"data: {json.dumps({'type': 'result', 'plugin': pname, 'result': result})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════
# GET /health — Kubernetes liveness probe
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/health", methods=["GET"])
def health():
    """
    Liveness probe.  503 if Redis is required but unreachable.
    Redis is checked only when REDIS_URL is set (production mode).
    """
    redis_url = current_app.config.get("REDIS_URL", "")
    if redis_url and not redis_url.startswith("memory://"):
        try:
            import redis as _redis
            _r = _redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
            _r.ping()
        except Exception as exc:
            logger.warning("Health check: Redis unreachable — %s", exc)
            return jsonify({"status": "degraded", "reason": "Redis unreachable"}), 503

    return jsonify({"status": "healthy"}), 200


# ═══════════════════════════════════════════════════════════════════
# GET /ready — Kubernetes readiness probe
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/ready", methods=["GET"])
def ready():
    """
    Readiness probe.

    [C-09/C-10/C-11 FIX] Beyond "are plugins registered?", verify their heavy
    backends actually loaded:
      - 503 if no plugins, OR if any *core* engine (the AI detector) failed to load.
      - 200 "ready_degraded" if a non-core plugin's backend is down (service still
        usable, but the client is told which signals are missing).
    The previous version returned 200 whenever len(registry)>0, masking a detector
    that silently failed to load.
    """
    registry = current_app.config.get("PLUGIN_REGISTRY")
    if registry is None or len(registry) == 0:
        return jsonify({
            "status": "not_ready",
            "reason": "No plugins loaded",
        }), 503

    health = registry.health_report()
    degraded = sorted(n for n, ok in health.items() if not ok)
    core_down = registry.core_unhealthy()

    if core_down:
        return jsonify({
            "status": "not_ready",
            "reason": f"Core engine(s) failed to load: {', '.join(sorted(core_down))}",
            "engine_status": health,
        }), 503

    body = {
        "status": "ready_degraded" if degraded else "ready",
        "plugins_loaded": len(registry),
        "plugins": registry.list_plugins(),
        "engine_status": health,
    }
    if degraded:
        body["degraded_plugins"] = degraded
    return jsonify(body), 200


# ═══════════════════════════════════════════════════════════════════
# GET /api/drift-status — Model drift monitor (anti-enshittification)
# ═══════════════════════════════════════════════════════════════════

@api_bp.route("/api/drift-status", methods=["GET"])
def drift_status():
    """
    Current model-quality status from the drift monitor.

    Reports rolling confidence statistics, class balance, and recent alerts so
    monitoring systems detect ensemble degradation (e.g. a new LLM family the
    models were never trained on) BEFORE users lose trust in the verdicts.

    Response JSON:
        {
            "status": "healthy" | "degraded" | "no_data",
            "samples_total": 1234,
            "window_samples": 100,
            "mean_confidence": 0.91,
            "baseline_confidence": 0.93,
            "ai_share": 0.46,
            "recent_alerts": [...],
            "model": {"version": "...", "weights": [...], "fallbacks_used": [...]}
        }

    Unauthenticated by design (like /health and /ready) — it exposes aggregate
    quality metrics only, never analyzed text.
    """
    body = {"status": "unavailable"}
    try:
        from app.engine.drift_monitor import get_drift_monitor
        body = get_drift_monitor().get_status()
    except Exception as exc:
        logger.warning("drift-status: monitor unavailable — %s", exc)

    try:
        from app.engine.detector_final import get_model_info
        body["model"] = get_model_info()
    except Exception:
        body["model"] = {"version": _MODEL_VERSION}

    return jsonify(body), 200


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
    """Serve a generated HTML forensic report."""
    import os
    from flask import Response
    from app.plugins.full_analysis import _REPORT_DIR
    basename = os.path.basename(filename)
    # S-08: Only serve files with the expected prefix — blocks serving arbitrary HTML
    if not basename.startswith("forensic_"):
        return jsonify({"error": "Report not found"}), 404
    filepath = os.path.join(_REPORT_DIR, basename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "Report not found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    return Response(html, mimetype="text/html", headers={
        "Content-Security-Policy": (
            "default-src 'none'; style-src 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self'"
        ),
        "X-Content-Type-Options": "nosniff",
    })
