"""
Fase 2 — unit tests for the late-fusion + calibration scaffolding.

Pure synthetic data — no ML model loading. Validates:
  - FusionFeatureBuilder assembles a fixed-dim vector and extracts plugin features.
  - FusionClassifier defaults to a transparent neural pass-through (calibrated=False).
  - FusionClassifier.fit trains a logistic model (skipped if scikit-learn absent).
  - compute_ece / brier_score sanity.
  - TemperatureScaler reduces ECE on an over-confident synthetic set.
"""

import sys
import os
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "engine"))

from fusion import (  # noqa: E402
    FusionFeatureBuilder, FusionClassifier, FUSION_VECTOR_DIM, FEATURE_NAMES,
)
from calibration import (  # noqa: E402
    compute_ece, brier_score, reliability_bins, TemperatureScaler,
)


def _det(ai_pct, human_pct, stats=None):
    return SimpleNamespace(
        ai_percentage=ai_pct, human_percentage=human_pct,
        statistical_features=stats or {},
    )


# ── Fusion feature builder ──────────────────────────────────────────────────

def test_vector_dim_and_names_match():
    b = FusionFeatureBuilder()
    ff = b.build(_det(90, 10), {})
    assert ff.vector.shape == (FUSION_VECTOR_DIM,)
    assert len(FEATURE_NAMES) == FUSION_VECTOR_DIM
    d = ff.as_dict()
    assert d["neural_ai_prob"] == pytest.approx(0.90)
    assert d["neural_uncertainty"] == pytest.approx(1.0 - 0.80)


def test_extracts_plugin_features():
    additional = {
        "perplexity": {"feature_values": {"proxy_perplexity_mean": 3.2,
                                          "low_perplexity_ratio": 0.8,
                                          "token_entropy_mean": 2.5}},
        "reasoning": {"feature_values": {"backtracking_density": 0.07,
                                         "cot_scaffold_density": 0.1}},
        "hallucination": {"overall_risk": 0.6,
                          "category_scores": {"semantic_incoherence": 0.7,
                                              "vagueness": 0.3, "repetition": 0.2}},
        "hybrid_segment": {"global_ai_score": 80.0,
                           "feature_vector": {"ai_segment_ratio": 0.75,
                                              "breakpoint_count": 2.0,
                                              "longest_ai_run": 3.0}},
        "reference_check": {"feature_values": {"fabricated_ratio": 0.5,
                                               "chimeric_ratio": 0.1,
                                               "verified_ratio": 0.4}},
    }
    det = _det(70, 30, stats={"burstiness": 0.1, "lexical_diversity": 0.5,
                              "avg_sentence_length": 18.0})
    d = FusionFeatureBuilder().build(det, additional).as_dict()
    assert d["ppl_proxy_mean"] == pytest.approx(3.2 / 15.0)  # M-12: [1,15] scale → [0,1]
    assert d["rsn_backtracking"] == pytest.approx(0.07)
    assert d["hal_semantic_incoherence"] == pytest.approx(0.7)
    assert d["hyb_global_ai"] == pytest.approx(0.80)          # 80/100
    assert d["ref_fabricated_ratio"] == pytest.approx(0.5)
    assert d["sty_burstiness"] == pytest.approx(0.1)


def test_reasoning_feature_details_path():
    # ReasoningRiskClassifier full path uses feature_details = {name: {"value": x}}
    additional = {"reasoning": {"feature_details": {
        "backtracking_density": {"value": 0.05}}}}
    d = FusionFeatureBuilder().build(_det(50, 50), additional).as_dict()
    assert d["rsn_backtracking"] == pytest.approx(0.05)


def test_robust_to_missing_and_nonsense():
    d = FusionFeatureBuilder().build(_det(50, 50), {"perplexity": "garbage"}).as_dict()
    assert d["ppl_proxy_mean"] == 0.0
    assert d["neural_ai_prob"] == pytest.approx(0.5)


# ── Fusion classifier (untrained) ───────────────────────────────────────────

def test_untrained_passthrough_mode():
    clf = FusionClassifier(untrained_mode="passthrough")
    res = clf.predict_proba(_det(88, 12), {})
    assert res.source == "neural_passthrough"
    assert res.calibrated is False
    assert res.probability == pytest.approx(0.88)
    assert clf.is_trained is False


def test_untrained_heuristic_is_default_and_bounded():
    clf = FusionClassifier()  # default = heuristic fusion
    res = clf.predict_proba(_det(88, 12), {})
    assert res.source == "heuristic_fusion"
    assert res.calibrated is False
    # de-overconfidence: a raw 0.88 neural prob is softened toward 0.5 with no other signal.
    assert 0.5 < res.probability < 0.88


def test_heuristic_plugins_move_the_score():
    clf = FusionClassifier()
    base = clf.predict_proba(_det(60, 40), {}).probability
    # Strong model-agnostic evidence (fabricated citations) should push toward AI.
    strong = clf.predict_proba(_det(60, 40), {
        "reference_check": {"feature_values": {"fabricated_ratio": 1.0}},
    }).probability
    assert strong > base
    # Reference verification should temper toward human.
    tempered = clf.predict_proba(_det(60, 40), {
        "reference_check": {"feature_values": {"verified_ratio": 1.0}},
    }).probability
    assert tempered < base


def test_fit_trains_logistic():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 400
    X = np.zeros((n, FUSION_VECTOR_DIM))
    y = np.zeros(n, dtype=int)
    neural_idx = FEATURE_NAMES.index("neural_ai_prob")
    for i in range(n):
        ai = i >= n // 2
        y[i] = int(ai)
        X[i, neural_idx] = (0.85 if ai else 0.15) + rng.normal(0, 0.05)
    clf = FusionClassifier().fit(X, y)
    assert clf.is_trained
    pv_ai = clf.predict_proba_vec(np.array([0.9 if k == neural_idx else 0.0
                                            for k in range(FUSION_VECTOR_DIM)]))
    pv_hu = clf.predict_proba_vec(np.array([0.1 if k == neural_idx else 0.0
                                            for k in range(FUSION_VECTOR_DIM)]))
    assert pv_ai.source == "logistic"
    assert pv_ai.probability > pv_hu.probability


# ── Calibration metrics ─────────────────────────────────────────────────────

def test_ece_zero_for_calibrated():
    # Predicted prob equals empirical frequency in each bin → ECE ~ 0.
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 20000)
    labels = (rng.uniform(0, 1, 20000) < probs).astype(int)
    assert compute_ece(probs, labels, n_bins=15) < 0.02


def test_brier_bounds():
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)
    assert brier_score(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(1.0)


def test_reliability_bins_shape():
    bins = reliability_bins(np.array([0.1, 0.4, 0.9]), np.array([0, 0, 1]), n_bins=5)
    assert len(bins) == 5
    assert all(len(t) == 3 for t in bins)


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(2)
    n = 20000
    z = rng.normal(0, 1.5, n)
    true_p = 1.0 / (1.0 + np.exp(-z))
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    # Over-confident reported probs: sharpen logits by 1/0.4 (true temperature 0.4).
    over_p = 1.0 / (1.0 + np.exp(-(z / 0.4)))

    ece_before = compute_ece(over_p, labels)
    ts = TemperatureScaler().fit(over_p, labels)
    ece_after = compute_ece(ts.apply_array(over_p), labels)

    assert ts.temperature > 1.0                  # softening an over-confident model
    assert ece_after < ece_before
    assert ece_after < 0.03


# ── Fase-2 heuristic-fusion safety (N-01/N-02/M-4/M-5/M-21) ─────────────────

def _formal_human_signals():
    """Genre-correlated signals a formal human academic essay plausibly triggers."""
    return {
        "discourse_structure": {"uniformity": 0.6},
        "reasoning": {"feature_values": {"cot_scaffold_density": 0.5,
                                         "backtracking_density": 0.3}},
        "hallucination": {"overall_risk": 0.5},
        "semantic_consistency": {"strong_contradiction_ratio": 0.2},
    }


def test_formal_human_not_flipped_to_ai():
    """N-01: accumulated genre-correlated heuristics must NOT flip an 80%-human
    neural verdict to AI-Generated (the Fase-2 flip pathway)."""
    clf = FusionClassifier()
    res = clf.predict_proba(_det(20, 80), _formal_human_signals())
    assert res.probability < 0.5


def test_corroboration_rule_caps_single_family():
    """M-1: a single active family gets the reduced positive budget (≤ 0.35)."""
    clf = FusionClassifier()
    res = clf.predict_proba(_det(50, 50), {
        "discourse_structure": {"uniformity": 1.0},
        "reasoning": {"feature_values": {"cot_scaffold_density": 1.0,
                                         "backtracking_density": 1.0}},
    })
    adj = res.contributions["_adjustment_clamped"]
    assert adj <= 0.35 + 1e-9
    assert res.contributions["_active_families"] == 1.0


def test_reference_evidence_corroborates_alone():
    """M-1: external ground truth (fabricated citations) unlocks the full budget."""
    clf = FusionClassifier()
    res = clf.predict_proba(_det(50, 50), {
        "reference_check": {"feature_values": {"fabricated_ratio": 1.0}},
    })
    assert res.contributions["_active_families"] >= 2.0
    assert res.contributions["_adjustment_clamped"] == pytest.approx(0.6)


def test_hyb_ai_ratio_removed_from_adjustment():
    """N-02: the hybrid-segment ratio comes from the SAME neural ensemble — it must
    stay in the vector but never move the heuristic score."""
    clf = FusionClassifier()
    base = clf.predict_proba(_det(60, 40), {}).probability
    with_hyb = clf.predict_proba(_det(60, 40), {
        "hybrid_segment": {"global_ai_score": 100.0,
                           "feature_vector": {"ai_segment_ratio": 1.0}},
    }).probability
    assert with_hyb == pytest.approx(base)


def test_language_gate_excludes_english_lexicon_features():
    """M-5/M-18: only the EN-regex reasoning features are gated on non-English text
    (discourse/semantic carry their own es/fr/pt lexicons and stay active)."""
    clf = FusionClassifier()
    signals = {"reasoning": {"feature_values": {"cot_scaffold_density": 0.5,
                                                "backtracking_density": 0.3}}}
    en = clf.predict_proba(_det(50, 50), {**signals,
                                          "language": {"lang": "en"}})
    es = clf.predict_proba(_det(50, 50), {**signals,
                                          "language": {"lang": "es"}})
    assert es.probability < en.probability  # gated features no longer push toward AI
    assert "rsn_cot_scaffold_excluded_lang" in es.contributions
    assert "rsn_cot_scaffold" in en.contributions


def test_pro_human_terms_temper_score():
    """M-4: organic-writing stylometry (burstiness, hapax) pushes toward human."""
    clf = FusionClassifier()
    base = clf.predict_proba(_det(60, 40), {}).probability
    human_style = clf.predict_proba(
        _det(60, 40, stats={"burstiness": 0.9, "hapax_legomena_ratio": 0.8}), {},
    ).probability
    assert human_style < base


def test_contributions_are_surfaced():
    """M-21/N-15: per-term log-odds contributions must reach the output dict."""
    clf = FusionClassifier()
    res = clf.predict_proba(_det(70, 30), {
        "reference_check": {"feature_values": {"fabricated_ratio": 0.5}},
    })
    d = res.to_dict()
    assert "contributions" in d and d["contributions"]
    assert "neural_softened" in d["contributions"]
    assert d["contributions"]["ref_fabricated_ratio"] > 0


def test_weights_roundtrip_and_env_loading(tmp_path, monkeypatch):
    """M-19 wiring: trained weights persist to JSON and get_fusion_classifier
    loads them (calibrated logistic) when FUSION_WEIGHTS_PATH is set."""
    pytest.importorskip("sklearn")
    import json
    import fusion as fusion_mod
    from calibration import TemperatureScaler

    rng = np.random.default_rng(5)
    n = 300
    X = rng.normal(0, 1, (n, FUSION_VECTOR_DIM))
    y = (X[:, FEATURE_NAMES.index("neural_ai_prob")] > 0).astype(int)
    clf = FusionClassifier().fit(X, y)
    clf.attach_calibrator(TemperatureScaler(temperature=1.5, fitted=True))

    path = tmp_path / "fusion_weights.json"
    path.write_text(json.dumps(clf.to_payload()))

    monkeypatch.setenv("FUSION_WEIGHTS_PATH", str(path))
    monkeypatch.setattr(fusion_mod, "_shared_fusion", None)  # reset singleton
    loaded = fusion_mod.get_fusion_classifier()
    assert loaded.is_trained
    res = loaded.predict_proba_vec(X[0])
    assert res.source == "logistic"
    assert res.calibrated is True

    # Schema-drift protection: wrong feature list must be refused.
    bad = clf.to_payload()
    bad["feature_names"] = bad["feature_names"][:-1]
    with pytest.raises(ValueError, match="schema mismatch"):
        FusionClassifier().load_payload(bad)

    monkeypatch.setattr(fusion_mod, "_shared_fusion", None)  # don't leak to other tests


def test_calibrator_attaches_to_fusion():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(0, 1, (n, FUSION_VECTOR_DIM))
    y = (X[:, FEATURE_NAMES.index("neural_ai_prob")] > 0).astype(int)
    clf = FusionClassifier().fit(X, y)
    ts = TemperatureScaler(temperature=2.0, fitted=True)
    clf.attach_calibrator(ts)
    res = clf.predict_proba_vec(X[0])
    assert res.calibrated is True
    assert 0.0 <= res.probability <= 1.0
