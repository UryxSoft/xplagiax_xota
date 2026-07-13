"""
app/plugins/ai_detection.py — Quick AI vs Human classification.

Uses detector_final.analyze_fast for adaptive-chunk inference:
single tokenization pass, BATCH_SIZE=12, max_tokens auto-scaled by word count.

Every response carries an explicit `uncertainty` block (margin between classes +
ensemble seed disagreement) and the `model_version`, so clients see HOW sure the
ensemble is instead of a falsely crisp verdict. Each prediction is also recorded
in the drift monitor (anti-enshittification safeguard — see /api/drift-status).
"""

import logging
import os
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

# Bump MODEL_VERSION (env) whenever weights, thresholds, or fusion change —
# same constant routes.py uses to namespace its response cache.
_MODEL_VERSION = os.getenv("MODEL_VERSION", "2026.06")

# Uncertainty thresholds — mirror classify_text_aggregate() in detector_final.
_UNCERTAIN_MARGIN_PCT = 15.0      # |human% − ai%| below this → uncertain
_UNCERTAIN_DISAGREEMENT_PCT = 12.0  # per-seed AI-prob std above this → uncertain

_analyze_text = None
_available = False

try:
    from app.engine.detector_final import analyze_fast
    _analyze_text = analyze_fast
    _available = True
    logger.info("ModernBERT ensemble loaded for AI detection (analyze_fast)")
except Exception as exc:
    logger.warning("detector_final not available: %s", exc)

_drift_monitor = None
try:
    from app.engine.drift_monitor import get_drift_monitor
    _drift_monitor = get_drift_monitor()
except Exception as exc:  # monitoring is optional — never block detection
    logger.warning("Drift monitor unavailable: %s", exc)


class AIDetectionPlugin(BasePlugin):

    def name(self) -> str:
        return "ai_detection"

    def is_core(self) -> bool:
        # Primary AI detector — /ready must fail if this engine did not load.
        return True

    def health(self) -> bool:
        return _available

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
        disagreement = float(summary.get("ensemble_disagreement", 0.0))

        margin = abs(ai_pct - human_pct)
        in_uncertain_zone = (
            margin < _UNCERTAIN_MARGIN_PCT
            or disagreement >= _UNCERTAIN_DISAGREEMENT_PCT
        )

        # Anti-enshittification: record every prediction so quality degradation
        # (confidence drift, class collapse) is detected before users notice it.
        warning = None
        if _drift_monitor is not None:
            _drift_monitor.record_prediction(
                confidence=max(human_pct, ai_pct) / 100.0,
                prediction=prediction,
                text_len=len(text),
            )
            if _drift_monitor.is_degraded():
                warning = "model_drift_detected"

        return {
            "prediction": prediction,
            "confidence": max(human_pct, ai_pct),
            "human_percentage": human_pct,
            "ai_percentage": ai_pct,
            "detected_model": summary.get("detected_model"),
            "uncertainty_zone": in_uncertain_zone,
            "uncertainty": {
                # Distance between the two classes, in percentage points.
                "margin_pct": round(margin, 2),
                # Std of the per-seed AI probability across the 3 ModernBERT
                # seeds — high = the models disagree (out-of-distribution text).
                "ensemble_std_pct": round(disagreement, 2),
                "in_uncertain_zone": in_uncertain_zone,
            },
            "model_version": _MODEL_VERSION,
            "warning": warning,
            "raw_scores": {
                "human": human_pct,
                "ai": ai_pct,
            },
            "segments": doc_result.get("segments", [])
        }
