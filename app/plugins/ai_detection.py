"""
app/plugins/ai_detection.py — Quick AI vs Human classification.

Uses detector_final.analyze_fast for adaptive-chunk inference:
single tokenization pass, BATCH_SIZE=12, max_tokens auto-scaled by word count.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_analyze_text = None
_available = False

try:
    import app.engine  # noqa
    from detector_final import analyze_fast
    _analyze_text = analyze_fast
    _available = True
    logger.info("ModernBERT ensemble loaded for AI detection (analyze_fast)")
except Exception as exc:
    logger.warning("detector_final not available: %s", exc)


class AIDetectionPlugin(BasePlugin):

    def name(self) -> str:
        return "ai_detection"

    def description(self) -> str:
        return "Quick AI vs Human binary classification with semantic segmentation (ModernBERT ensemble)."

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "ModernBERT models not loaded. Check model paths."}

        doc_result = _analyze_text(text)
        
        if "error" in doc_result:
            return {"error": doc_result["error"]}

        summary = doc_result.get("overall_summary", {})
        
        prediction = summary.get("overall_prediction", "Unknown")
        human_pct = summary.get("total_human_percentage", 50)
        ai_pct = summary.get("total_ai_percentage", 50)

        return {
            "prediction": prediction,
            "confidence": max(human_pct, ai_pct),
            "human_percentage": human_pct,
            "ai_percentage": ai_pct,
            "detected_model": None,  # No extraemos el modelo específico en este modo rápido
            "uncertainty_zone": False,
            "raw_scores": {
                "human": human_pct,
                "ai": ai_pct,
            },
            "segments": doc_result.get("segments", [])
        }
