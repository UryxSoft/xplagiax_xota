"""
app/plugins/citation_check.py — Citation existence verification.

Wraps ReferenceValidator to check if cited references actually exist
in CrossRef, Semantic Scholar, and OpenAlex databases.

Requires network access (set REFERENCE_NETWORK=1).
"""

import logging
import os
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_validator = None
_classifier = None
_available = False

try:
    from app.engine.reference_validator import ReferenceValidator, ReferenceRiskClassifier
    _validator = ReferenceValidator(
        enable_network=os.getenv("REFERENCE_NETWORK", "1") == "1",
    )
    _classifier = ReferenceRiskClassifier()
    _available = True
    logger.info("ReferenceValidator loaded (network=%s)",
                os.getenv("REFERENCE_NETWORK", "1"))
except Exception as exc:
    logger.warning("ReferenceValidator not available: %s", exc)


class CitationCheckPlugin(BasePlugin):

    def name(self) -> str:
        return "citation_check"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Verify citations against CrossRef, Semantic Scholar, and OpenAlex. "
            "Detects fabricated, chimeric, and ornamental references."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "ReferenceValidator not loaded."}

        stats = _validator.compute_stats(text)
        analysis = _classifier.classify(stats)
        # "references" kept as backward-compat alias; points to same list object
        analysis["references"] = analysis["validation_results"]
        analysis["feature_values"] = {
            k: v for k, v in stats.items()
            if isinstance(v, (int, float))
        }
        return analysis
