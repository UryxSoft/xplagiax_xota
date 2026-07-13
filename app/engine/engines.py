"""
app/engine/engines.py — Shared lazy singletons for the analysis engines.

[C1 FIX] Before this module existed every plugin instantiated its own engine at
import time AND PluginOrchestrator._init_plugins() built a second copy of each
one for the full_analysis pipeline. Engines like StylometricProfiler (spaCy
pipeline) and PerplexityProfiler (n-gram dict + optional GPT-2) are expensive,
so the service paid the construction cost — and the resident memory — twice.

Every consumer (thin plugins in app/plugins/ and the orchestrator) now obtains
engine instances exclusively through the get_*() factories below, so each engine
is constructed exactly once per process and shared via CoW across forked workers.

Import either as `from app.engine.engines import get_stylometric` (app code) or
`from engines import get_stylometric` (engine-internal bare import) — the alias
finder installed by app/engine/__init__.py guarantees both names resolve to this
same module object.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTANCES: Dict[str, Any] = {}


def _get(key: str, factory: Callable[[], Any]) -> Any:
    """Double-checked lazy construction of the singleton named *key*."""
    inst = _INSTANCES.get(key)
    if inst is None:
        with _LOCK:
            inst = _INSTANCES.get(key)
            if inst is None:
                inst = factory()
                _INSTANCES[key] = inst
                logger.info("Engine singleton created: %s", key)
    return inst


# ── Factories ──────────────────────────────────────────────────────
# Imports stay inside each factory so a missing optional dependency only
# breaks the engine that needs it (callers already catch and degrade).

def get_stylometric() -> Any:
    from stylometric_profiler import StylometricProfiler
    return _get("stylometric", StylometricProfiler)


def get_hallucination_profiler() -> Any:
    from hallucination_profile import HallucinationProfiler
    return _get("hallucination_profiler", HallucinationProfiler)


def get_hallucination_classifier() -> Any:
    from hallucination_profile import HallucinationRiskClassifier
    return _get("hallucination_classifier", HallucinationRiskClassifier)


def get_reasoning_profiler() -> Any:
    from reasoning_profiler import ReasoningProfiler
    return _get("reasoning_profiler", ReasoningProfiler)


def get_reasoning_classifier() -> Any:
    from forensic_reports import ReasoningRiskClassifier
    return _get("reasoning_classifier", ReasoningRiskClassifier)


def get_perplexity_profiler() -> Any:
    """Configured from env (PERPLEXITY_DICT_PATH, PERPLEXITY_TIER2) — the same
    source both the plugin and full_analysis previously read, so the single
    shared instance behaves identically for every consumer."""
    def _build():
        from perplexity_profiler import PerplexityProfiler
        return PerplexityProfiler(
            ngram_dict_path=os.getenv("PERPLEXITY_DICT_PATH"),
            enable_tier2=os.getenv("PERPLEXITY_TIER2", "1") == "1",
        )
    return _get("perplexity_profiler", _build)


def get_perplexity_classifier() -> Any:
    from perplexity_profiler import PerplexityRiskClassifier
    return _get("perplexity_classifier", PerplexityRiskClassifier)


def get_hybrid_analyzer() -> Any:
    """HybridSegmentAnalyzer wired to the ensemble's batch classifier ([C3])."""
    def _build():
        from hybrid_segment_detector import HybridSegmentAnalyzer
        from detector_final import classify_segment, classify_batch
        return HybridSegmentAnalyzer(
            classify_fn=classify_segment,
            classify_batch_fn=classify_batch,
        )
    return _get("hybrid_analyzer", _build)


def get_discourse_analyzer() -> Any:
    from discourse_analyzer import DiscourseAnalyzer
    return _get("discourse_analyzer", DiscourseAnalyzer)


def get_semantic_analyzer() -> Any:
    from semantic_consistency import SemanticConsistencyAnalyzer
    return _get("semantic_analyzer", SemanticConsistencyAnalyzer)


def get_watermark_decoder(device: Optional[str] = None) -> Any:
    """Keyed by device so an explicit device override is honoured; the default
    auto-detect instance is what both the plugin and orchestrator share."""
    def _build():
        import torch
        from watermark_decoder import WatermarkDecoder
        dev = torch.device(device) if device else None
        return WatermarkDecoder(device=dev)
    return _get(f"watermark_decoder:{device or 'auto'}", _build)
