"""
lang_detect.py — Dependency-free stopword-ratio language identification.
========================================================================

[Fase-2 M-5] The Tier-1 signals (discourse markers, negation cues, CoT scaffolding
regexes) use ENGLISH lexicons. On non-English documents those features are structurally
unreliable, so the fusion gates them out — which requires knowing the language.

This is a coarse O(n) detector over the four languages the product actually serves
(en/es/fr/pt). It counts hits against small high-frequency function-word sets and
returns the best ratio. Function words are the most frequent and most language-specific
tokens, so ~50 per language is enough for document-level identification; short or
ambiguous texts fall back to "en" with low confidence (preserves current behavior).

No external deps, no model, safe to call per request.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_WORD_RE = re.compile(r"[a-záéíóúüñàâçèêëîïôùûœãõ']+", re.IGNORECASE)

_STOPWORDS: Dict[str, frozenset] = {
    "en": frozenset("""
        the of and to in is that it for on with as was are be this at by from or
        an but not they his her which you all we there their has have had one more
        when who will would can could about into than then them these some
    """.split()),
    "es": frozenset("""
        el la de que y en los se del las un por con no una su para es al lo como
        más pero sus le ya o este sí porque esta entre cuando muy sin sobre también
        me hasta hay donde quien desde todo nos durante todos uno les
    """.split()),
    "fr": frozenset("""
        le de la et les des en un du une que est pour qui dans par plus pas au sur
        ne se ce il sont avec ou son au aux cette ses mais comme tout nous leur
        bien être elle deux même ces
    """.split()),
    "pt": frozenset("""
        de a o que e do da em um para é com não uma os no se na por mais as dos
        como mas foi ao ele das tem à seu sua ou ser quando muito há nos já está
        eu também só pelo pela até isso
    """.split()),
}

_MIN_WORDS = 20          # below this, identification is unreliable → default "en"
_MIN_RATIO = 0.08        # winning ratio below this → inconclusive → default "en"
_SAMPLE_CHARS = 20_000   # ratios stabilise long before this; keeps the scan O(1)-ish


def detect_language(text: str) -> Dict[str, Any]:
    """
    Identify the dominant language of *text*.

    Returns {"lang": "en"|"es"|"fr"|"pt", "confidence": float, "method": "stopword_ratio"}.
    Defaults to "en" (with confidence 0.0) on short/inconclusive input so that existing
    English-pipeline behavior is preserved unless there is real evidence otherwise.
    """
    words = [w.lower() for w in _WORD_RE.findall(text[:_SAMPLE_CHARS])]
    if len(words) < _MIN_WORDS:
        return {"lang": "en", "confidence": 0.0, "method": "stopword_ratio",
                "note": f"<{_MIN_WORDS} words — defaulted to en"}

    n = len(words)
    ratios = {
        lang: sum(1 for w in words if w in sw) / n
        for lang, sw in _STOPWORDS.items()
    }
    best = max(ratios, key=ratios.get)
    if ratios[best] < _MIN_RATIO:
        return {"lang": "en", "confidence": round(ratios[best], 4),
                "method": "stopword_ratio", "note": "inconclusive — defaulted to en"}
    return {"lang": best, "confidence": round(ratios[best], 4),
            "method": "stopword_ratio", "ratios": {k: round(v, 4) for k, v in ratios.items()}}
