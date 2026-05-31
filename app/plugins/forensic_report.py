"""
app/plugins/forensic_report.py — HTML forensic report generation.

Wraps ForensicReportGenerator to produce the full HTML forensic report.
Useful when you want the visual report without re-running detection
(pass pre-computed results via the API).
"""

import logging
import os
import tempfile
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_generator_class = None
_available = False

try:
    from app.engine.forensic_reports import ForensicReportGenerator
    _generator_class = ForensicReportGenerator
    _available = True
    logger.info("ForensicReportGenerator loaded (v3.9)")
except Exception as exc:
    logger.warning("ForensicReportGenerator not available: %s", exc)


class ForensicReportPlugin(BasePlugin):

    def name(self) -> str:
        return "forensic_report"

    def description(self) -> str:
        return (
            "Generate a full HTML forensic report with executive summary, "
            "heatmaps, evidence, segment analysis, and actionable steps."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "ForensicReportGenerator not loaded."}

        # Use full_analysis plugin's orchestrator if available
        try:
            from app.plugins.full_analysis import _orchestrator, _available as fa_avail
            if fa_avail and _orchestrator is not None:
                result = _orchestrator.run(text)
                fr = result.get("forensic_report")
                if fr:
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".html", prefix="forensic_",
                        dir="/tmp", delete=False,
                    )
                    _orchestrator._forensic_generator.export_html(fr, tmp.name)

                    return {
                        "report_id": fr.report_id,
                        "verdict": fr.verdict,
                        "confidence": fr.confidence,
                        "html_path": tmp.name,
                        "word_count": fr.word_count,
                        "evidence_count": len(fr.evidence_points),
                    }
        except Exception as exc:
            logger.warning("Full pipeline failed, using standalone: %s", exc)

        return {"error": "Full pipeline required for report generation."}
