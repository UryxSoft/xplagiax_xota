"""
app/tasks.py — Background tasks for Celery.
"""
import gc
import time
import logging
from flask import current_app
from app.celery_app import celery
from app.routes import _merge_segment_results

logger = logging.getLogger(__name__)

_B64_KEYS = frozenset({"heatmap_b64", "confidence_chart_b64", "comparison_chart_b64"})


def _strip_base64(results: dict) -> dict:
    """
    Remove base64-encoded chart images from plugin results before Redis serialization.
    Each chart is ~300 KB–1 MB of base64 text; keeping them in Redis for up to 1 hour
    wastes significant memory per task. The HTML report file in /tmp already contains
    the charts — the client should fetch /report/<id> for the visual output.
    """
    for plugin_result in results.values():
        if not isinstance(plugin_result, dict):
            continue
        data = plugin_result.get("data")
        if isinstance(data, dict):
            for key in _B64_KEYS:
                if key in data:
                    data[key] = None
    return results


@celery.task(
    bind=True,
    name="analyze_document_task",
    # Hard kill after 5 min — prevents zombie tasks from blocking the worker
    time_limit=300,
    # Soft warning at 4 min — task can catch SoftTimeLimitExceeded and clean up
    soft_time_limit=240,
)
def analyze_document_task(self, payload):
    """
    Executes the heavy document analysis.
    payload contains: text, plugins, max_tokens

    Memory management:
    - gc.collect() is called after each run to free Python-managed objects.
    - torch cache is cleared to prevent GPU/CPU tensor accumulation.
    - time_limit=300s kills runaway tasks before they OOM the worker.
    - base64 chart images are stripped before Redis serialization.
    """
    t0 = time.perf_counter()

    text = payload.get("text", "")
    plugins_requested = payload.get("plugins", ["ai_detection"])

    registry = current_app.config["PLUGIN_REGISTRY"]
    timeout = current_app.config.get("PLUGIN_TIMEOUT", 120)

    try:
        # 1. Run standard plugins
        results = registry.run(plugins_requested, text, timeout=timeout)

        # 2. Obtain per-segment breakdown.
        # Reuse segments already computed by the ai_detection plugin when possible —
        # avoids running the full batch inference a second time for the same text.
        doc_result = {}
        ai_plugin_data = (results.get("ai_detection") or {})
        if ai_plugin_data.get("status") == "ok":
            ai_data = ai_plugin_data.get("data", {})
            segs = ai_data.get("segments", [])
            if segs:
                doc_result = {
                    "segments": segs,
                    "overall_summary": {
                        "total_human_percentage": ai_data.get("human_percentage", 50),
                        "total_ai_percentage": ai_data.get("ai_percentage", 50),
                        "overall_prediction": ai_data.get("prediction", "Unknown"),
                    },
                }

        if not doc_result:
            # Fallback: ai_detection was not requested or returned no segments.
            # analyze_fast uses adaptive max_tokens — no argument needed.
            try:
                from app.engine.detector_final import analyze_fast as _seg_fn
                doc_result = _seg_fn(text)
            except Exception as exc:
                logger.warning("analyze_fast failed in task: %s", exc)

        segments = doc_result.get("segments", [])

        # 3. Merge results
        _merge_segment_results(results, doc_result)

        elapsed = time.perf_counter() - t0

        return {
            "status": "ok",
            "word_count": len(text.split()),
            "plugins_requested": plugins_requested,
            "results": _strip_base64(results),
            "total_elapsed_ms": round(elapsed * 1000, 1),
        }

    finally:
        # ── Post-analysis memory cleanup ──────────────────────────────────────
        # Runs after EVERY task completion or failure — prevents memory collapse
        # across long-running workers that process hundreds of documents.

        # 1. Free Python-managed objects — only pays off for large texts
        # (GC full sweep on small docs adds latency without significant benefit)
        if len(text) > 10_000:
            collected = gc.collect()
            logger.debug("gc.collect() freed %d objects after task", collected)

        # 2. Clear PyTorch CUDA cache (no-op on CPU-only, safe to always call)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass  # torch not available or no CUDA — safe to ignore

