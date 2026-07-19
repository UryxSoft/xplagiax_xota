"""
Precision tests against a gold corpus (anti-enshittification safeguard).

Two layers:

1. Smoke layer (always attempted, auto-skips without model weights):
   determinism, output shape, and calibration sanity of analyze_fast().

2. Rolling-corpus layer (opt-in): point ROLLING_CORPUS_DIR at a directory
       <dir>/human/*.txt    verified human-written texts
       <dir>/ai/*.txt       verified AI-generated texts
   and the suite asserts accuracy >= MIN_CORPUS_ACCURACY (default 0.80).
   Refresh the corpus weekly with texts from NEW LLM families — that is what
   catches model aging before users do. scripts/retrain_pipeline.py consumes
   the same layout.

Run: pytest tests/test_precision_corpus.py -m precision
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.precision

_ENGINE_DIR = Path(__file__).resolve().parent.parent / "app" / "engine"
_WEIGHTS_PRESENT = (_ENGINE_DIR / "modernbert.bin").exists()

_needs_models = pytest.mark.skipif(
    not _WEIGHTS_PRESENT,
    reason="ModernBERT weights not present (app/engine/modernbert.bin)",
)


@pytest.fixture(scope="module")
def analyze_fast():
    if not _WEIGHTS_PRESENT:
        pytest.skip("ModernBERT weights not present")
    from app.engine.detector_final import analyze_fast as fn
    return fn


_HUMAN_SMOKE = (
    "Ayer se me quemó el arroz otra vez. Mi abuela decía que el secreto era "
    "no destaparlo nunca, pero yo soy incapaz de resistirme; levanto la tapa, "
    "lo revuelvo, y claro, pasa lo que pasa. Igual nos lo comimos viendo el "
    "partido, medio pegado y todo, y nadie se quejó demasiado."
)


@_needs_models
def test_determinism_same_text_same_verdict(analyze_fast):
    """The same text must produce the identical verdict on repeated calls —
    a detector that fluctuates on identical input cannot be trusted at all."""
    r1 = analyze_fast(_HUMAN_SMOKE)
    r2 = analyze_fast(_HUMAN_SMOKE)
    assert r1["overall_summary"] == r2["overall_summary"]


@_needs_models
def test_output_shape_and_calibration_bounds(analyze_fast):
    result = analyze_fast(_HUMAN_SMOKE)
    summary = result["overall_summary"]
    human, ai = summary["total_human_percentage"], summary["total_ai_percentage"]
    assert 0 <= human <= 100 and 0 <= ai <= 100
    assert summary["overall_prediction"] in ("Human", "AI")
    assert summary["ensemble_disagreement"] >= 0
    assert result["segments"], "at least one segment expected"


# ── Rolling corpus accuracy gate ───────────────────────────────────

def _load_rolling_corpus():
    corpus_dir = os.getenv("ROLLING_CORPUS_DIR", "")
    if not corpus_dir:
        pytest.skip("ROLLING_CORPUS_DIR not set — rolling precision gate disabled")
    root = Path(corpus_dir)
    samples = []
    for label, sub in (("Human", "human"), ("AI", "ai")):
        for path in sorted((root / sub).glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                samples.append((text, label, path.name))
    if len(samples) < 10:
        pytest.skip(f"rolling corpus too small ({len(samples)} samples, need >= 10)")
    return samples


@_needs_models
def test_rolling_corpus_accuracy(analyze_fast):
    """Accuracy on the labeled rolling corpus must stay above the floor.

    When this fails, the ensemble has drifted: check /api/drift-status,
    identify which files were misclassified (listed in the assertion message),
    and run scripts/retrain_pipeline.py.
    """
    samples = _load_rolling_corpus()
    floor = float(os.getenv("MIN_CORPUS_ACCURACY", "0.80"))

    misses = []
    for text, label, name in samples:
        summary = analyze_fast(text).get("overall_summary", {})
        pred = summary.get("overall_prediction", "Unknown")
        if pred != label:
            misses.append(f"{name}: expected {label}, got {pred}")

    accuracy = 1.0 - len(misses) / len(samples)
    assert accuracy >= floor, (
        f"accuracy {accuracy:.2%} below floor {floor:.2%} "
        f"({len(misses)}/{len(samples)} misclassified):\n" + "\n".join(misses)
    )
