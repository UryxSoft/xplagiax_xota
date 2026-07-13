"""
app/engine/_nlp.py — Shared spaCy pipeline singleton.

[C7 FIX] Both stylometric_profiler.py and hallucination_profile.py used to call
`spacy.load("en_core_web_sm")` at module import, loading TWO independent copies of
the same pipeline into memory. This module loads it once and hands the same object
to every consumer, halving the spaCy memory footprint with no behavioural change
(identical model, identical parses).

Usage (drop-in for the old module-level pattern):

    from app.engine._nlp import get_nlp, spacy_available
    _NLP = get_nlp()                 # spaCy Language object, or None
    _SPACY_AVAILABLE = spacy_available()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "en_core_web_sm"

_NLP = None
_LOADED = False
_AVAILABLE = False


def _ensure_loaded() -> None:
    global _NLP, _LOADED, _AVAILABLE
    if _LOADED:
        return
    _LOADED = True
    try:
        import spacy as _spacy
        _NLP = _spacy.load(_MODEL_NAME)
        _AVAILABLE = True
        logger.debug("Shared spaCy pipeline '%s' loaded once.", _MODEL_NAME)
    except (ImportError, OSError) as exc:
        _NLP = None
        _AVAILABLE = False
        logger.debug("spaCy unavailable (%s) — regex fallbacks active.", exc)


def get_nlp():
    """Return the shared spaCy Language object, or None if unavailable."""
    _ensure_loaded()
    return _NLP


def spacy_available() -> bool:
    """True if the shared spaCy pipeline loaded successfully."""
    _ensure_loaded()
    return _AVAILABLE
