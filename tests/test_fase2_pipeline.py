"""
Fase 2 — pipeline-level unit tests (no ML model loading).

Covers:
  - M-22: HybridSegmentAnalyzer paragraph mode (default) vs legacy sliding windows.
  - M-11: exec_context cooperative deadline.
  - M-18: language-aware discourse markers (es) and semantic negation cues.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "engine"))


# ── M-22: hybrid paragraph mode ─────────────────────────────────────────────

def _long_paragraph(n_words: int) -> str:
    return " ".join(f"word{i % 50}" for i in range(n_words))


class _FakeBatch:
    """Returns alternating AI/human scores, one per requested window."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        out = []
        for _ in texts:
            out.append((10.0, 90.0) if self.calls % 2 == 0 else (90.0, 10.0))
            self.calls += 1
        return out


def test_hybrid_paragraph_mode_one_window_per_paragraph(monkeypatch):
    monkeypatch.delenv("HYBRID_WINDOWS", raising=False)
    from hybrid_segment_detector import HybridSegmentAnalyzer

    fake = _FakeBatch()
    analyzer = HybridSegmentAnalyzer(classify_fn=lambda t: (50.0, 50.0),
                                     classify_batch_fn=fake)
    text = _long_paragraph(120) + "\n\n" + _long_paragraph(120)
    result = analyzer.analyze(text)

    assert result.total_paragraphs == 2
    assert result.total_windows == 2          # paragraph mode: 1 window per paragraph
    assert len(result.paragraph_scores) == 2
    # Alternating fake scores → one AI zone, one HUMAN zone → 1 breakpoint.
    assert len(result.breakpoints) == 1


def test_hybrid_legacy_windows_mode_still_available(monkeypatch):
    monkeypatch.setenv("HYBRID_WINDOWS", "1")
    from hybrid_segment_detector import HybridSegmentAnalyzer

    analyzer = HybridSegmentAnalyzer(classify_fn=lambda t: (50.0, 50.0),
                                     classify_batch_fn=lambda ts: [(50.0, 50.0)] * len(ts))
    text = _long_paragraph(400)               # single paragraph, 400 words
    result = analyzer.analyze(text)

    assert result.total_paragraphs == 1
    assert result.total_windows > 1           # sliding windows re-cover the text


# ── M-11: cooperative deadline ──────────────────────────────────────────────

def test_deadline_checkpoint_raises_after_expiry():
    from exec_context import (
        set_context, clear_context, check_deadline, is_async,
        PluginDeadlineExceeded,
    )
    set_context(deadline=time.monotonic() - 1.0, async_mode=True)
    try:
        assert is_async() is True
        with pytest.raises(PluginDeadlineExceeded):
            check_deadline()
    finally:
        clear_context()
    check_deadline()                          # no context → never raises
    assert is_async() is False


# ── M-18: multilingual lexicons ─────────────────────────────────────────────

_ES_TEXT = (
    "En primer lugar, el estudio presenta los datos generales del problema. "
    "Sin embargo, los resultados muestran una tendencia clara en la población. "
    "Además, el análisis estadístico confirma la hipótesis planteada por el equipo. "
    "Por lo tanto, se puede afirmar que el método propuesto funciona correctamente. "
    "En conclusión, la evidencia respalda las afirmaciones del presente trabajo."
)


def test_discourse_spanish_markers_detected():
    from discourse_analyzer import DiscourseAnalyzer
    out = DiscourseAnalyzer().analyze(_ES_TEXT)
    assert out["status"] == "ok"
    assert out["language"] == "es"
    assert out["features"]["connective_density"] > 0
    assert out["features"]["conclusion_marker"] == 1.0


def test_semantic_spanish_negation_is_weak_only():
    from semantic_consistency import SemanticConsistencyAnalyzer
    text = (
        "El sistema procesa los documentos largos con precisión alta siempre. "
        "El sistema no procesa los documentos largos con precisión alta nunca. "
        "Los autores describen un método nuevo para medir resultados generales. "
        "La medición entrega valores estables durante todas las pruebas realizadas."
    )
    out = SemanticConsistencyAnalyzer().analyze(text)
    assert out["status"] == "ok"
    assert out["language"] == "es"
    # Negation flip found (report-only) but no STRONG contradiction.
    assert out["contradiction_count"] >= 1
    assert out["strong_contradiction_count"] == 0
