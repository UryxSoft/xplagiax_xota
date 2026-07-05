"""
app/plugins/reasoning_check.py — Reasoning-model detection.

Wraps ReasoningProfiler to detect Chain-of-Thought (CoT) markers, causal density,
backtracking, and other signals typical of reasoning models (o1, DeepSeek-R1).
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_profiler = None
_classifier = None
_available = False

try:
    from app.engine.reasoning_profiler import ReasoningProfiler
    from app.engine.forensic_reports import ReasoningRiskClassifier
    _profiler = ReasoningProfiler()
    _classifier = ReasoningRiskClassifier()
    _available = True
    logger.info("ReasoningProfiler + Classifier loaded")
except Exception as exc:
    logger.warning("ReasoningProfiler not available: %s", exc)


class ReasoningCheckPlugin(BasePlugin):

    def name(self) -> str:
        return "reasoning_check"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Detect reasoning-model signals (o1, DeepSeek-R1): Chain-of-Thought "
            "markers, backtracking, causal density, and structural reasoning."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "ReasoningProfiler not loaded."}

        vec = _profiler.vectorize(text)
        feat_names = _profiler.feature_names()
        return _classifier.classify(vec, feat_names)
