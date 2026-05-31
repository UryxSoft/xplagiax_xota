"""
app/plugins/zone_classifier.py — Citation zone detection and style analysis.

Wraps CitationDetector to classify text zones (direct quotes, paraphrases,
original), detect citation style (APA/MLA/IEEE/Chicago/Vancouver/Harvard),
and report coverage + consistency metrics.

No network calls. No ML models. Pure regex + optional spaCy.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_detector = None
_available = False

try:
    from app.antiplagio.citation.detector import CitationDetector, ZoneType
    _detector = CitationDetector()
    _available = True
    logger.info("CitationDetector loaded (zone_classifier ready)")
except Exception as exc:
    logger.warning("CitationDetector not available: %s", exc)


class ZoneClassifierPlugin(BasePlugin):

    def name(self) -> str:
        return "zone_classifier"

    def description(self) -> str:
        return (
            "Classifies text zones (direct quotes, paraphrases, original content) "
            "and detects citation style (APA/MLA/IEEE/Chicago/Vancouver/Harvard). "
            "Reports orphan citations, style consistency, and citation coverage."
        )

    def warmup(self) -> None:
        # _detector is already instantiated at module level (CoW-friendly).
        pass

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "CitationDetector not available."}
        if not text or len(text.strip()) < 30:
            return {
                "dominant_style": "Unknown",
                "style_consistency": 0.0,
                "citation_coverage": 100.0,
                "total_inline_citations": 0,
                "total_bibliography": 0,
                "orphan_citations": 0,
                "uncited_bibliography": 0,
                "zones": [],
                "inline_citations": [],
                "bibliography": [],
                "issues": {"orphan_citations": [], "uncited_bibliography": []},
            }

        result = _detector.analyze(text)

        zones = [
            {
                "type": z.zone_type.value,
                "text_preview": z.text[:200],
                "start_pos": z.start_pos,
                "end_pos": z.end_pos,
                "has_citation": z.has_valid_citation,
                "citation_count": len(z.citation_markers),
                "plagiarism_risk": round(z.plagiarism_risk, 2),
            }
            for z in result.zones
            if z.zone_type != ZoneType.BIBLIOGRAPHY
        ]

        inline_citations = [
            {
                "text": c.raw_text,
                "style": c.style.value,
                "author": c.author,
                "year": c.year,
                "page": c.page,
                "number": c.number,
                "confidence": round(c.confidence, 2),
            }
            for c in result.inline_citations
        ]

        bibliography = [
            {
                "key": e.key,
                "style": e.style.value,
                "authors": e.authors[:3],
                "year": e.year,
                "title": e.title,
                "doi": e.doi,
                "url": e.url,
            }
            for e in result.bibliography[:50]
        ]

        return {
            "dominant_style": result.dominant_style.value,
            "style_consistency": round(result.style_consistency * 100, 1),
            "citation_coverage": round(result.citation_coverage * 100, 1),
            "total_inline_citations": len(result.inline_citations),
            "total_bibliography": len(result.bibliography),
            "orphan_citations": len(result.orphan_citations),
            "uncited_bibliography": len(result.uncited_bibliography),
            "zones": zones,
            "inline_citations": inline_citations,
            "bibliography": bibliography,
            "issues": {
                "orphan_citations": [
                    {"text": c.raw_text, "style": c.style.value}
                    for c in result.orphan_citations[:10]
                ],
                "uncited_bibliography": [
                    {"key": e.key, "authors": e.authors[:2], "year": e.year}
                    for e in result.uncited_bibliography[:10]
                ],
            }
        }
