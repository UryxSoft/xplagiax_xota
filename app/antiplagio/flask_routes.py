"""
Citation Analysis Flask Routes
================================
Blueprint /api/v2/ with citation detection and bibliography validation endpoints.
PlagiarismEngine excluded — see scorer.py for future Qdrant/SerpAPI integration.

Usage:
    from app.antiplagio.flask_routes import register_antiplagio_routes
    register_antiplagio_routes(app)
"""

import asyncio
import os
from functools import wraps
from flask import Blueprint, request, jsonify, current_app

from .citation.detector import CitationDetector, CitationStyle, ZoneType
from .citation.validator import CitationValidator, ValidationStatus, _shared_cache


antiplagio_bp = Blueprint("antiplagio", __name__, url_prefix="/api/v2")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def async_route(f):
    """Decorator to use async handlers in Flask (compatible with gevent workers)."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    return wrapper


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@antiplagio_bp.route("/citations/detect", methods=["POST"])
def detect_citations():
    """
    Citation detection only (no plagiarism scoring, no external API calls).
    Fast — pure regex + optional spaCy.

    Body: { "text": "..." }

    Returns zone classification, inline citations, bibliography, style metrics.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Campo 'text' requerido"}), 400

    detector = CitationDetector()
    result = detector.analyze(data["text"])

    return jsonify({
        "dominant_style": result.dominant_style.value,
        "style_consistency": round(result.style_consistency * 100, 1),
        "citation_coverage": round(result.citation_coverage * 100, 1),
        "inline_citations": [
            {
                "text": c.raw_text,
                "style": c.style.value,
                "author": c.author,
                "year": c.year,
                "page": c.page,
                "number": c.number,
                "confidence": round(c.confidence * 100, 1),
                "position": {"start": c.start_pos, "end": c.end_pos}
            }
            for c in result.inline_citations
        ],
        "bibliography": [
            {
                "key": e.key,
                "style": e.style.value,
                "authors": e.authors,
                "year": e.year,
                "title": e.title,
                "doi": e.doi,
                "url": e.url,
            }
            for e in result.bibliography
        ],
        "zones": [
            {
                "type": z.zone_type.value,
                "text_preview": z.text[:150],
                "has_citation": z.has_valid_citation,
                "citation_count": len(z.citation_markers),
                "plagiarism_risk": round(z.plagiarism_risk, 2),
            }
            for z in result.zones
            if z.zone_type != ZoneType.BIBLIOGRAPHY
        ],
        "issues": {
            "orphan_citations": len(result.orphan_citations),
            "uncited_bibliography": len(result.uncited_bibliography),
        }
    }), 200


@antiplagio_bp.route("/citations/validate", methods=["POST"])
@async_route
async def validate_citations():
    """
    Validates bibliographic references against CrossRef, OpenAlex, Semantic Scholar.
    Requires network access and aiohttp.

    Body: { "text": "..." }  or  { "bibliography": ["ref1", "ref2", ...] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON requerido"}), 400

    if "text" in data:
        detector = CitationDetector()
        analysis = detector.analyze(data["text"])
        bibliography = analysis.bibliography
    elif "bibliography" in data:
        detector = CitationDetector()
        raw_text = "Referencias\n" + "\n".join(data["bibliography"])
        bibliography = detector._parse_bibliography(raw_text)
    else:
        return jsonify({"error": "Se requiere 'text' o 'bibliography'"}), 400

    if not bibliography:
        return jsonify({
            "message": "No bibliographic entries found",
            "results": []
        }), 200

    validator = CitationValidator(
        crossref_email=current_app.config.get("CROSSREF_EMAIL", "antiplagio@example.com"),
        cache=_shared_cache,
    )
    validation_results = await validator.validate_all(bibliography)

    def _get(key, attr, default=None):
        r = validation_results.get(key)
        return getattr(r, attr, default) if r else default

    return jsonify({
        "total": len(bibliography),
        "valid": sum(1 for v in validation_results.values() if v.status == ValidationStatus.VALID),
        "partial": sum(1 for v in validation_results.values() if v.status == ValidationStatus.PARTIAL),
        "not_found": sum(1 for v in validation_results.values() if v.status == ValidationStatus.NOT_FOUND),
        "unverifiable": sum(1 for v in validation_results.values() if v.status == ValidationStatus.UNVERIFIABLE),
        "results": [
            {
                "key": entry.key,
                "raw": entry.raw_text[:200],
                "validation": {
                    "status": (_get(entry.key, "status", ValidationStatus.ERROR)).value
                        if _get(entry.key, "status") else "not_validated",
                    "confidence": round((_get(entry.key, "confidence") or 0) * 100, 1),
                    "source_api": _get(entry.key, "source_api"),
                    "found_title": _get(entry.key, "found_title"),
                    "found_doi": _get(entry.key, "found_doi"),
                    "discrepancies": _get(entry.key, "discrepancies", []),
                }
            }
            for entry in bibliography
        ]
    }), 200


# ─────────────────────────────────────────────
# Registration factory
# ─────────────────────────────────────────────

def register_antiplagio_routes(app, config: dict = None):
    """
    Registers the citation blueprint on the Flask app.

    Args:
        app    - Flask instance
        config - optional dict with:
                   CROSSREF_EMAIL (default: env var CROSSREF_EMAIL)

    In app/__init__.py:
        from app.antiplagio.flask_routes import register_antiplagio_routes
        register_antiplagio_routes(app)
    """
    config = config or {}
    app.config["CROSSREF_EMAIL"] = config.get(
        "CROSSREF_EMAIL",
        os.environ.get("CROSSREF_EMAIL", "antiplagio@example.com")
    )
    app.register_blueprint(antiplagio_bp)
    app.logger.info("Antiplagio citation routes registered at /api/v2/")
