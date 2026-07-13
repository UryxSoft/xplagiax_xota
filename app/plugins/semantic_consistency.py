"""
app/plugins/semantic_consistency.py — Internal-contradiction detection plugin.

Thin HTTP wrapper around app.engine.semantic_consistency.SemanticConsistencyAnalyzer.
Flags places where the document contradicts itself (negation-polarity flips or numeric
mismatches on the same subject; optional NLI tier via SEMANTIC_NLI=1). Self-contradiction
is a coherence failure common in LLM text and is model-agnostic, but humans contradict
themselves too — it is forensic evidence to verify, not a standalone verdict.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_analyzer = None
_available = False
try:
    # [C1] Shared singleton — same instance the orchestrator (full_analysis) uses.
    from app.engine.engines import get_semantic_analyzer
    _analyzer = get_semantic_analyzer()
    _available = True
    logger.info("SemanticConsistencyAnalyzer loaded for semantic_consistency")
except Exception as exc:  # noqa: BLE001 — degrade gracefully
    logger.warning("semantic_consistency unavailable: %s", exc)


class SemanticConsistencyPlugin(BasePlugin):

    def name(self) -> str:
        return "semantic_consistency"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Internal-contradiction detection: flags self-contradicting sentence pairs "
            "(negation flips, numeric mismatches; optional NLI). Model-agnostic coherence "
            "signal — evidence to verify, not a standalone AI verdict."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available or _analyzer is None:
            return {"error": "SemanticConsistencyAnalyzer not loaded."}
        return _analyzer.analyze(text)
