"""
app/plugins/stylometric_analysis.py — Writing style fingerprinting.

Wraps StylometricProfiler to extract sentence structure, vocabulary richness,
punctuation patterns, and linguistic markers like burstiness.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_profiler = None
_available = False

try:
    # [C1] Shared singleton — same instance the orchestrator (full_analysis) uses.
    from app.engine.engines import get_stylometric
    _profiler = get_stylometric()
    _available = True
    logger.info("StylometricProfiler loaded")
except Exception as exc:
    logger.warning("StylometricProfiler not available: %s", exc)


class StylometricAnalysisPlugin(BasePlugin):

    def name(self) -> str:
        return "stylometric_analysis"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Analyze writing style fingerprint: sentence structure, vocabulary "
            "diversity, burstiness, and punctuation patterns."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "StylometricProfiler not loaded."}

        return _profiler.compute_stats(text)
