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
    import app.engine  # noqa
    from perplexity_profiler import PerplexityProfiler, PerplexityRiskClassifier
    _profiler = PerplexityProfiler(
        ngram_dict_path=os.getenv("PERPLEXITY_DICT_PATH"),
        enable_tier2=os.getenv("PERPLEXITY_TIER2", "1") == "1",
    )
    _classifier = PerplexityRiskClassifier()
    _available = True

    tier = "tier2" if (_profiler._gpt2 is not None
                       and _profiler._gpt2._available) else "tier1"
    logger.info("PerplexityProfiler loaded (%s)", tier)
except Exception as exc:
    logger.warning("PerplexityProfiler not available: %s", exc)


class PerplexityCheckPlugin(BasePlugin):

    def name(self) -> str:
        return "perplexity_check"

    def description(self) -> str:
        return (
            "Text predictability analysis — measures how 'predictable' the text "
            "is to a language model. AI text tends to be highly predictable."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "PerplexityProfiler not loaded."}

        stats = _profiler.compute_stats(text)

        if _classifier:
            analysis = _classifier.classify(stats)
            analysis["window_ppls"] = stats.get("window_ppls", [])
            analysis["tokens_analysed"] = stats.get("tokens_analysed", 0)
            analysis["feature_values"] = {
                k: stats[k] for k in stats
                if isinstance(stats[k], (int, float))
            }
            return analysis

        return stats
