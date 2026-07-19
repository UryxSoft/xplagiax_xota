"""
app/plugins/perplexity_check.py — Text predictability analysis.

Wraps PerplexityProfiler for n-gram proxy perplexity (Tier 1, CPU)
and optionally GPT-2 token-level perplexity (Tier 2, GPU).
"""

import logging
import os
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_profiler = None
_classifier = None
_available = False

try:
    # [C1] Shared singletons — get_perplexity_profiler() reads the same env vars
    # (PERPLEXITY_DICT_PATH, PERPLEXITY_TIER2) this module previously read itself.
    from app.engine.engines import get_perplexity_profiler, get_perplexity_classifier
    _profiler = get_perplexity_profiler()
    _classifier = get_perplexity_classifier()
    _available = True

    logger.info("PerplexityProfiler loaded (%s)", _profiler.tier)
except Exception as exc:
    logger.warning("PerplexityProfiler not available: %s", exc)


class PerplexityCheckPlugin(BasePlugin):

    def name(self) -> str:
        return "perplexity_check"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Text predictability analysis — measures how 'predictable' the text "
            "is to a language model. AI text tends to be highly predictable."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "PerplexityProfiler not loaded."}

        stats = _profiler.compute_stats(text)
        analysis = _classifier.classify(stats)
        analysis["window_ppls"] = stats.get("window_ppls", [])
        analysis["tokens_analysed"] = stats.get("tokens_analysed", 0)
        analysis["feature_values"] = {
            k: v for k, v in stats.items()
            if isinstance(v, (int, float))
        }
        return analysis
