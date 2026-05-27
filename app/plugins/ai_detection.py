"""
app/plugins/ai_detection.py — Quick AI vs Human classification.

Wraps detector_final.analyze_long_documentsd_ for fast binary detection
with semantic segmentation and global percentage calculation.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_analyze_text = None
_available = False

try:
    import app.engine  # noqa
    from detector_final import analyze_long_documentsd_
    _analyze_text = analyze_long_documentsd_
    _available = True
    logger.info("ModernBERT 4-model ensemble loaded for AI detection (Segmented)")
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

        # Ejecutamos el análisis segmentado que ya tienes en detector_final.py
        # max_tokens=150 asegura fragmentación en párrafos u oraciones lógicas.
        doc_result = _analyze_text(text, max_tokens=150)
        
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
