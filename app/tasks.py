"""
app/tasks.py — Background tasks for Celery.
"""
import gc
import os
import time
import logging
from celery.exceptions import SoftTimeLimitExceeded
from flask import current_app
from app.celery_app import celery
from app.plugin_registry import adaptive_timeout
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
    # DEFAULT limits, sized for paper-length texts. The /analyze_document_async
    # route overrides both per enqueue (apply_async soft_time_limit/time_limit)
    # scaling them with the document's word count, so thesis-sized inputs get a
    # proportional budget instead of being hard-killed at 5 minutes.
    # Hard kill — prevents zombie tasks from blocking the worker.
    time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "300")),
    # Soft warning — task catches SoftTimeLimitExceeded and returns a clean error.
    soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "240")),
)
def analyze_document_task(self, payload):
    """
    Executes the heavy document analysis.
    payload contains: text, plugins, max_tokens

    Memory management:
    - gc.collect() is called after each run to free Python-managed objects.
    - torch cache is cleared to prevent GPU/CPU tensor accumulation.
    - time_limit (default 300 s, scaled with word count at enqueue) kills
      runaway tasks before they OOM the worker.
    - base64 chart images are stripped before Redis serialization.
    """
    t0 = time.perf_counter()

    text = payload.get("text", "")
    plugins_requested = payload.get("plugins", ["ai_detection"])

    # Mark as STARTED immediately so clients can distinguish "queued" from
    # "processing". Without this, the task stays PENDING until it finishes.
    self.update_state(state='STARTED', meta={'plugins': plugins_requested})

    registry = current_app.config["PLUGIN_REGISTRY"]

    # Per-plugin timeout scaled to document size, bounded 30 s under this run's
    # own soft time limit so the plugin future expires (clean per-plugin error)
    # before Celery raises SoftTimeLimitExceeded for the whole task.
    _hard, _soft = (self.request.timelimit or (None, None))
    _soft = _soft or int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "240"))
    timeout = adaptive_timeout(
        len(text.split()),
        base=max(current_app.config.get("PLUGIN_TIMEOUT", 30), 120),
        per_kwords=current_app.config.get("PLUGIN_TIMEOUT_PER_KWORDS", 15.0),
        cap=max(60, _soft - 30),
    )

    try:
        # 1. Run standard plugins. async_mode=True stamps the per-thread execution
        # context (exec_context) so async-only signals (M-7 reference check) activate.
        results = registry.run(plugins_requested, text, timeout=timeout, async_mode=True)
    except SoftTimeLimitExceeded:
        logger.warning("Task %s exceeded soft time limit — returning timeout error", self.request.id)
        return {
            "status": "error",
            "error": "Analysis timed out. Try with a shorter text.",
            "plugins_requested": plugins_requested,
            "total_elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as exc:
        logger.error("Task %s failed: %s", self.request.id, exc, exc_info=True)
        raise
    else:

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

