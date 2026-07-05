"""
app/plugins/discourse_structure.py — Argumentative-structure uniformity plugin.

Thin HTTP wrapper around app.engine.discourse_analyzer.DiscourseAnalyzer. Measures how
*templated* the discourse structure is (even paragraphs, formal connectives, enumeration
and closing scaffolding) — a model-agnostic structural prior that survives paraphrasing.
HIGH uniformity is common in LLM text but also in disciplined human writing, so it is a
prior, not a verdict. See the engine module for the full rationale.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_analyzer = None
_available = False
try:
    from app.engine.discourse_analyzer import DiscourseAnalyzer
    _analyzer = DiscourseAnalyzer()
    _available = True
    logger.info("DiscourseAnalyzer loaded for discourse_structure")
except Exception as exc:  # noqa: BLE001 — degrade gracefully
    logger.warning("discourse_structure unavailable: %s", exc)


class DiscourseStructurePlugin(BasePlugin):

    def name(self) -> str:
        return "discourse_structure"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Argumentative-structure uniformity: detects templated discourse (even "
            "paragraphs, formal connectives, enumeration/closing scaffolding). Model-agnostic "
            "structural prior that survives paraphrasing — high = LLM-like, not a verdict."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available or _analyzer is None:
            return {"error": "DiscourseAnalyzer not loaded."}
        return _analyzer.analyze(text)
