"""
Fase-2 unit tests — language gate input (M-5) and strong-contradiction split (M-6).
Pure Python, no ML model loading.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "engine"))

from lang_detect import detect_language  # noqa: E402
from semantic_consistency import SemanticConsistencyAnalyzer  # noqa: E402


# ── lang_detect ─────────────────────────────────────────────────────────────

_ES = (
    "El sistema de detección analiza el texto y produce una probabilidad. "
    "Los resultados no deben interpretarse como una prueba absoluta, sino como "
    "un nivel de evidencia que el usuario puede revisar con las señales mostradas. "
    "Cuando la evidencia es débil o contradictoria, el sistema debe abstenerse."
)

_EN = (
    "The detection system analyses the text and produces a probability. "
    "The results should not be interpreted as absolute proof, but as a level of "
    "evidence that the user can review with the displayed signals. "
    "When the evidence is weak or contradictory, the system must abstain."
)


def test_detects_spanish():
    assert detect_language(_ES)["lang"] == "es"


def test_detects_english():
    assert detect_language(_EN)["lang"] == "en"


def test_short_text_defaults_to_english():
    out = detect_language("Hola mundo.")
    assert out["lang"] == "en"
    assert out["confidence"] == 0.0


# ── semantic_consistency: strong vs weak contradictions ─────────────────────

def test_numeric_mismatch_is_strong():
    text = (
        "The survey covered 120 participants from the northern region last year. "
        "Data collection followed the standard protocol throughout the study. "
        "The survey covered 85 participants from the northern region last year. "
        "All responses were anonymised before the statistical analysis began."
    )
    out = SemanticConsistencyAnalyzer().analyze(text)
    assert out["status"] == "ok"
    assert out["strong_contradiction_count"] >= 1
    assert out["strong_contradiction_ratio"] > 0


def test_negation_flip_is_weak_only():
    text = (
        "The model is limited to structured input formats in every case. "
        "Results were consistent across the evaluated configurations overall. "
        "The model is not limited to structured input formats in every case. "
        "Further evaluation would clarify the remaining edge conditions found."
    )
    out = SemanticConsistencyAnalyzer().analyze(text)
    assert out["status"] == "ok"
    # Negation flips are report-only: counted, but never marked strong.
    assert out["contradiction_count"] >= 1
    assert out["strong_contradiction_count"] == 0
    assert out["strong_contradiction_ratio"] == 0.0
