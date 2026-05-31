"""
app/plugins/segment_analysis.py — Per-paragraph AI/Human heatmap.

Wraps HybridSegmentAnalyzer with classify_segment as the injected
classification function.  Shows WHERE in the document AI was used.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_analyzer = None
_available = False

try:
    from app.engine.hybrid_segment_detector import HybridSegmentAnalyzer
    from app.engine.detector_final import classify_segment
    _analyzer = HybridSegmentAnalyzer(classify_fn=classify_segment)
    _available = True
    logger.info("HybridSegmentAnalyzer loaded")
except Exception as exc:
    logger.warning("HybridSegmentAnalyzer not available: %s", exc)


class SegmentAnalysisPlugin(BasePlugin):

    def name(self) -> str:
        return "segment_analysis"

    def description(self) -> str:
        return (
            "Per-paragraph AI/Human heatmap via sliding-window ModernBERT. "
            "Shows which paragraphs are AI-generated vs human-written."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "HybridSegmentAnalyzer not loaded."}

        result = _analyzer.analyze(text)
        return result.to_dict()
