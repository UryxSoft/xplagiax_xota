"""
app/plugins/full_analysis.py — Complete XplagiaX forensic pipeline.

Wraps PluginOrchestrator.run() to execute all enabled analyses:
    ModernBERT detection → Stylometric → Hallucination → Reasoning →
    Perplexity → Hybrid Segment → Reference Validation → Watermark →
    Forensic Report Generation

Returns the full forensic report as JSON + optionally HTML.
"""

import glob
import logging
import os
import tempfile
import time
from typing import Any, Dict


def _cleanup_old_reports(directory: str, prefix: str, max_age_seconds: int) -> None:
    """Delete report files older than max_age_seconds to prevent disk exhaustion."""
    cutoff = time.time() - max_age_seconds
    for path in glob.glob(os.path.join(directory, f"{prefix}*.html")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

# ── Module-level engine loading (shared via CoW) ──────────────────
_orchestrator = None
_PluginConfig = None
_available = False

try:
    import app.engine  # noqa — triggers sys.path setup
    from plugin_orchestrator import PluginOrchestrator, PluginConfig

    _PluginConfig = PluginConfig
    _orchestrator = PluginOrchestrator(PluginConfig(
        enable_stylometric=True,
        enable_hallucination=True,
        enable_reasoning=True,
        enable_perplexity=True,
        enable_hybrid_segment=True,
        enable_reference_check=os.getenv("ENABLE_REFERENCE_CHECK", "0") == "1",
        reference_network=os.getenv("REFERENCE_NETWORK", "1") == "1",
        enable_watermark=os.getenv("ENABLE_WATERMARK", "0") == "1",
        enable_forensic_report=True,
        forensic_output_format="html",
        perplexity_dict_path=os.getenv("PERPLEXITY_DICT_PATH"),
        perplexity_tier2=os.getenv("PERPLEXITY_TIER2", "1") == "1",
    ))
    _available = True
    logger.info("XplagiaX PluginOrchestrator loaded — plugins: %s",
                ", ".join(_orchestrator.active_plugins()))
except Exception as exc:
    logger.warning("XplagiaX engine not available: %s", exc)


class FullAnalysisPlugin(BasePlugin):

    def name(self) -> str:
        return "full_analysis"

    def description(self) -> str:
        return (
            "Complete XplagiaX forensic pipeline: ModernBERT detection, "
            "stylometric, hallucination, reasoning, perplexity, segment heatmap, "
            "citation validation, watermark, and forensic report generation."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available or _orchestrator is None:
            return {
                "error": "XplagiaX engine not loaded. Ensure ModernBERT "
                         "models are present and all dependencies installed.",
                "required_files": [
                    "detector_final.py (+ 4 ModernBERT model weights)",
                    "stylometric_profiler.py",
                    "hallucination_profile.py",
                    "reasoning_profiler.py",
                    "perplexity_profiler.py",
                    "hybrid_segment_detector.py",
                    "reference_validator.py",
                    "forensic_reports.py",
                ],
            }

        # Run full pipeline
        result = _orchestrator.run(text)

        # Extract detection result
        det = result.get("detection_result")
        aa = result.get("additional_analyses", {})
        fr = result.get("forensic_report")

        response: Dict[str, Any] = {
            "detection": {
                "prediction": det.prediction if det else "Unknown",
                "confidence": det.confidence if det else 0.0,
                "human_percentage": det.human_percentage if det else 50.0,
                "ai_percentage": det.ai_percentage if det else 50.0,
                "detected_model": det.detected_model if det else None,
                "uncertainty_zone": det.uncertainty_zone if det else True,
            },
        }

        # Add forensic report data
        if fr is not None:
            response["forensic"] = {
                "report_id": fr.report_id,
                "verdict": fr.verdict,
                "confidence": fr.confidence,
                "word_count": fr.word_count,
                "scores": {
                    "neural": fr.neural_score,
                    "statistical": fr.statistical_score,
                    "stylometric": fr.stylometric_score,
                    "reasoning": fr.reasoning_score,
                    "watermark": fr.watermark_score,
                },
                "executive_summary": fr.executive_summary,
                "evidence_count": len(fr.evidence_points),
            }

            # Generate HTML report to temp file and include path
            try:
                _cleanup_old_reports("/tmp", prefix="forensic_", max_age_seconds=3600)
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".html", prefix="forensic_",
                    dir="/tmp", delete=False,
                )
                _orchestrator._forensic_generator.export_html(fr, tmp.name)
                response["forensic"]["html_report_path"] = tmp.name
            except Exception as exc:
                logger.warning("HTML report generation failed: %s", exc)

        # Add plugin summaries
        if "perplexity" in aa:
            response["perplexity"] = {
                "ai_score": aa["perplexity"].get("ai_score", 0),
                "risk_level": aa["perplexity"].get("risk_level", "N/A"),
                "tier": aa["perplexity"].get("tier", "tier1"),
            }

        if "hybrid_segment" in aa:
            hs = aa["hybrid_segment"]
            response["segment_analysis"] = {
                "classification": hs.get("classification", "N/A"),
                "risk_level": hs.get("risk_level", "N/A"),
                "global_ai_score": hs.get("global_ai_score", 0),
                "total_paragraphs": hs.get("total_paragraphs", 0),
                "breakpoint_count": hs.get("breakpoint_count", 0),
                "paragraph_scores": hs.get("paragraph_scores", []),
            }

        if "reference_check" in aa:
            rc = aa["reference_check"]
            response["citations"] = {
                "ai_score": rc.get("ai_score", 0),
                "risk_level": rc.get("risk_level", "N/A"),
                "total_references": rc.get("total_references", 0),
            }

        # Plain-text summary
        response["summary"] = _orchestrator.summary(result)

        return response
