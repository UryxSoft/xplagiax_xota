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

try:
    from app.engine.stylometric_profiler import StylometricProfiler
    from app.engine.authorship_consistency import compute_authorship_consistency
    _profiler = StylometricProfiler()
    _available = True
    logger.info("StylometricProfiler loaded for author_signature")
except Exception as exc:  # noqa: BLE001 — degrade gracefully
    logger.warning("author_signature unavailable: %s", exc)


class AuthorSignaturePlugin(BasePlugin):

    def name(self) -> str:
        return "author_signature"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Intra-document authorship-consistency profile (reuses StylometricProfiler "
            "feature extraction). High consistency = single coherent author; outlier chunks "
            "flag possible mixed/AI-spliced sections. Localization aid, not a verdict."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available or _profiler is None:
            return {"error": "StylometricProfiler not loaded."}
        return compute_authorship_consistency(_profiler, text)
