"""
app/plugins/author_signature.py — Intra-document authorship consistency.

Activates the (previously dormant) authorship analysis inside StylometricProfiler
(see AUDITORIA_CIENTIFICA.md §13 — the pipeline only ever called compute_stats()).

Mode: NO reference author profile is required. The document is split into chunks; a
descriptive stylometric vector is extracted per chunk via StylometricProfiler.compute_stats,
and the dispersion of those vectors is measured in standardised (z-score) space. Tight
clustering ⇒ a single coherent author; an outlier chunk ⇒ possible multiple authors, quoted
material, or an AI-spliced section — a *localization* signal, not a verdict.

Why not build_profile()/compare(): that path z-normalises the combined profile by its OWN
corpus mean, leaving the profile vector ≈ 0, which makes the cosine self-comparison degenerate
(collapses to 0 — itself an audit finding). We therefore use the robust compute_stats features
and a Euclidean z-dispersion measure, which correctly returns full consistency for identical
chunks. Framed as authorship-consistency (not "AI"), it is an anti-false-positive aid: a
uniformly-styled formal/academic document scores HIGH consistency.
"""

import logging
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_profiler = None
_available = False

# Preferred implementation: authorship embeddings (LUAR — opt-in via
# ENABLE_AUTHOR_EMBEDDING=1, see docs/sota/D_AUTHOR_SIGNATURE.md). Falls back
# to the stylometric z-dispersion implementation below when not enabled.
_embedding_engine = None
try:
    from app.engine import author_embedding as _embedding_engine
except Exception as exc:  # noqa: BLE001
    logger.debug("author_embedding module not importable: %s", exc)

try:
    # [C1] Shared singleton — same StylometricProfiler as stylometric_analysis
    # and the orchestrator (previously a third private instance lived here).
    from app.engine.engines import get_stylometric
    from app.engine.authorship_consistency import compute_authorship_consistency
    _profiler = get_stylometric()
    _available = True
    logger.info("StylometricProfiler loaded for author_signature")
except Exception as exc:  # noqa: BLE001 — degrade gracefully
    logger.warning("author_signature unavailable: %s", exc)


class AuthorSignaturePlugin(BasePlugin):

    def name(self) -> str:
        return "author_signature"

    def health(self) -> bool:
        if _embedding_engine is not None and _embedding_engine.is_available():
            return True
        return _available

    def description(self) -> str:
        return (
            "Intra-document authorship-consistency profile (authorship embeddings when "
            "ENABLE_AUTHOR_EMBEDDING=1, else StylometricProfiler dispersion). High "
            "consistency = single coherent author; outlier chunks flag possible "
            "mixed/AI-spliced sections. Localization aid, not a verdict."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if _embedding_engine is not None and _embedding_engine.is_available():
            try:
                result = _embedding_engine.analyze_document(text)
                if result.get("status") == "ok":
                    return result
                logger.warning("author embedding returned %s — falling back",
                               result.get("status"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("author embedding failed (%s) — falling back", exc)
        if not _available or _profiler is None:
            return {"error": "StylometricProfiler not loaded."}
        return compute_authorship_consistency(_profiler, text)
