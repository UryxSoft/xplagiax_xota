"""
app/plugins/hallucination_check.py — AI fabrication risk detection.

Wraps HallucinationProfiler and Classifier to detect internal inconsistencies,
unsubstantiated claims, and patterns of fabrication common in LLMs.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_profiler = None
_classifier = None
_available = False

try:
    from app.engine.hallucination_profile import HallucinationProfiler, HallucinationRiskClassifier
    _profiler = HallucinationProfiler()
    _classifier = HallucinationRiskClassifier()
    _available = True
    logger.info("HallucinationProfiler + Classifier loaded")
except Exception as exc:
    logger.warning("HallucinationProfiler not available: %s", exc)


class HallucinationCheckPlugin(BasePlugin):

    def name(self) -> str:
        return "hallucination_check"

    def description(self) -> str:
        return (
            "Detect AI fabrication risk: internal inconsistencies, factual drift, "
            "and patterns typical of hallucinations."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "HallucinationProfiler not loaded."}

        stats = _profiler.compute_stats(text)
        if _classifier:
            analysis = _classifier.classify(stats)
            analysis["feature_values"] = {
                k: stats[k] for k in stats
                if isinstance(stats[k], (int, float))
            }
            return analysis

        return stats
