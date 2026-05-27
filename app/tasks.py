"""
app/tasks.py — Background tasks for Celery.
"""
import gc
import time
import logging
from flask import current_app
from app.celery_app import celery

logger = logging.getLogger(__name__)

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
    """
    t0 = time.perf_counter()

    text = payload.get("text", "")
    plugins_requested = payload.get("plugins", ["ai_detection"])
    max_tokens = int(payload.get("max_tokens", 150))

    registry = current_app.config["PLUGIN_REGISTRY"]
    timeout = current_app.config.get("PLUGIN_TIMEOUT", 120)

    try:
        # 1. Run standard plugins
        results = registry.run(plugins_requested, text, timeout=timeout)

        # 2. Run heavy segmentation
        doc_result = {}
        try:
            import app.engine
            from detector_final import analyze_long_document
            doc_result = analyze_long_document(text, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("analyze_long_document failed in task: %s", exc)

        segments = doc_result.get("segments", [])

        # 3. Merge results
        if "ai_detection" in results and doc_result:
            ai_result = results["ai_detection"]
            if ai_result.get("status") == "ok" and isinstance(ai_result.get("data"), dict):
                ai_result["data"]["segments"] = segments
                ai_result["data"]["overall_summary"] = doc_result.get("overall_summary", {})

        elapsed = time.perf_counter() - t0

        return {
            "status": "ok",
            "word_count": len(text.split()),
            "plugins_requested": plugins_requested,
            "results": results,
            "total_elapsed_ms": round(elapsed * 1000, 1),
        }

    finally:
        # ── Post-analysis memory cleanup ──────────────────────────────────────
        # Runs after EVERY task completion or failure — prevents memory collapse
        # across long-running workers that process hundreds of documents.

        # 1. Free Python-managed objects (circular refs, dead tensors, etc.)
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

