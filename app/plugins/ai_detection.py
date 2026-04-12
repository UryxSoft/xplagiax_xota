"""
app/plugins/ai_detection.py — Quick AI vs Human classification.

Wraps detector_final.classify_text() for fast binary detection
without running the full forensic pipeline.
~2 seconds on CPU, ~0.3 seconds on GPU.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_classify_text = None
_available = False

try:
    import app.engine  # noqa
    from detector_final import classify_text
    _classify_text = classify_text
    _available = True
    logger.info("ModernBERT 4-model ensemble loaded for AI detection")
except Exception as exc:
    logger.warning("detector_final not available: %s", exc)


class AIDetectionPlugin(BasePlugin):

    def name(self) -> str:
        return "ai_detection"

    def description(self) -> str:
        return "Quick AI vs Human binary classification (4-model ModernBERT ensemble)."

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "ModernBERT models not loaded. Check model paths."}

        _, _, det = _classify_text(text)

        return {
            "prediction": det.prediction,
            "confidence": det.confidence,
            "human_percentage": det.human_percentage,
            "ai_percentage": det.ai_percentage,
            "detected_model": det.detected_model,
            "uncertainty_zone": det.uncertainty_zone,
            "raw_scores": {
                "human": det.raw_scores.get("human", 0),
                "ai": det.raw_scores.get("ai", 0),
            },
        }
