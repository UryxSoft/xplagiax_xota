"""
fusion.py — Late-fusion feature assembly + (uncalibrated) meta-classifier scaffold.
=====================================================================================

WHY THIS EXISTS
---------------
The forensic pipeline historically derived its verdict from the neural ensemble ONLY;
every other plugin (perplexity, reasoning, hallucination, hybrid-segment, reference)
was rendered as narrative "evidence" but never entered a decision formula. This module
is the scaffolding for a REAL late-fusion classifier: it assembles the existing per-plugin
feature vectors into one fixed-schema vector that a trained meta-classifier can consume.

⚠ STATUS: FRAMEWORK ONLY — NOT CALIBRATED, NOT WIRED INTO THE PRODUCTION VERDICT.
Until a labelled corpus is available (see docs/EXPERIMENTAL_PROTOCOL.md), `FusionClassifier`
is a transparent NEURAL PASS-THROUGH: predict_proba() returns the neural AI probability and
reports `calibrated=False`. Call `.fit(X, y)` with labelled data to train a logistic model
(requires scikit-learn) and `TemperatureScaler`/`compute_ece` (calibration.py) to calibrate it.

DESIGN
------
The builder consumes the orchestrator's output — `detection_result` (DetectionResult) and
`additional_analyses` (the dict assembled by PluginOrchestrator.run_with_result) — so no model
is re-run. Missing signals default to 0.0, making the vector robust to degraded pipelines and
unit-testable with synthetic dicts.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# FUSION VECTOR SCHEMA — single source of truth (name -> index)
# ============================================================================
# Each feature is sourced from an existing plugin output. Keeping a small, named,
# interpretable schema (rather than concatenating all 70+ raw dims) keeps the
# meta-classifier auditable and the ablation tractable.

_FUSION_SCHEMA: Tuple[str, ...] = (
    # ── Neural ensemble (the only currently trustworthy signal) ──
    "neural_ai_prob",            # ai% / 100  ∈ [0,1]
    "neural_uncertainty",        # 1 - |ai-human|/100  ∈ [0,1]
    # ── Perplexity profiler ──
    "ppl_proxy_mean",
    "ppl_low_ratio",
    "ppl_valley_count",
    "ppl_burstiness",
    "ppl_curvature",
    "ppl_entropy_mean",
    # ── Reasoning profiler ──
    "rsn_backtracking",
    "rsn_cot_scaffold",
    "rsn_entropy_norm",
    "rsn_type_token_ratio",
    # ── Hallucination profiler (category scores) ──
    "hal_overall",
    "hal_semantic_incoherence",
    "hal_vagueness",
    "hal_repetition",
    # ── Hybrid-segment detector ──
    "hyb_global_ai",             # 0..100 -> normalised to 0..1 on build
    "hyb_ai_ratio",
    "hyb_breakpoints",
    "hyb_longest_ai_run",
    # ── Reference validator ──
    "ref_fabricated_ratio",
    "ref_chimeric_ratio",
    "ref_verified_ratio",
    # ── Stylometric profiler ──
    "sty_burstiness",
    "sty_lexical_diversity",
    "sty_avg_sentence_len",
    # ── Tier-1 model-agnostic signals (survive paraphrasing; help vs frontier models) ──
    "author_outlier_ratio",      # fraction of style-divergent chunks (splice/mix signal)
    "dsc_uniformity",            # templated discourse structure ∈ [0,1]
    "sem_contradiction_ratio",   # STRONG contradictions (numeric/NLI) / sentences ∈ [0,1]
    # ── Fase-2 additions (M-4 pro-human term, M-5 language gate) ──
    "sty_hapax_ratio",           # hapax legomena ratio ∈ [0,1] — higher in organic human prose
    "lang_en",                   # 1.0 if document language is English, else 0.0 (gates EN-lexicon features)
)

FUSION_VECTOR_DIM: int = len(_FUSION_SCHEMA)
FEATURE_NAMES: Tuple[str, ...] = _FUSION_SCHEMA


def feature_names() -> Tuple[str, ...]:
    """Ordered fusion feature names matching the assembled vector."""
    return FEATURE_NAMES


# ============================================================================
# Small extraction helpers (navigate the loosely-typed plugin dicts safely)
# ============================================================================

def _num(d: Any, *keys: str, default: float = 0.0) -> float:
    """Return the first numeric value found among keys in dict d, else default."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return default


def _norm01(value: float, scale: float) -> float:
    """[M-12] value/scale clipped to [0,1] — saturating normalisation."""
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _reasoning_feature(reasoning: Dict[str, Any], name: str) -> float:
    """
    Reasoning analysis stores features either as a flat `feature_values` dict
    (orchestrator partial path) or as `feature_details` = {name: {"value": x}}
    (ReasoningRiskClassifier full path). Handle both.
    """
    if not isinstance(reasoning, dict):
        return 0.0
    fv = reasoning.get("feature_values")
    if isinstance(fv, dict) and name in fv:
        return _num(fv, name)
    fd = reasoning.get("feature_details")
    if isinstance(fd, dict) and name in fd and isinstance(fd[name], dict):
        return _num(fd[name], "value")
    return 0.0


# ============================================================================
# Feature builder
# ============================================================================

@dataclass
class FusionFeatures:
    vector: np.ndarray
    names: Tuple[str, ...]

    def as_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}


class FusionFeatureBuilder:
    """Assemble the fixed-schema fusion vector from orchestrator output."""

    def build(self, detection_result: Any,
              additional_analyses: Optional[Dict[str, Any]] = None) -> FusionFeatures:
        aa = additional_analyses or {}

        # ── Neural ──
        ai_pct = float(getattr(detection_result, "ai_percentage", 50) or 50)
        human_pct = float(getattr(detection_result, "human_percentage", 50) or 50)
        neural_ai_prob = ai_pct / 100.0
        neural_uncertainty = 1.0 - abs(ai_pct - human_pct) / 100.0

        # ── Perplexity ──
        ppl = aa.get("perplexity", {})
        ppl_fv = ppl.get("feature_values", ppl) if isinstance(ppl, dict) else {}

        # ── Reasoning ──
        rsn = aa.get("reasoning", {})

        # ── Hallucination ──
        hal = aa.get("hallucination", {})
        hal_cats = hal.get("category_scores", {}) if isinstance(hal, dict) else {}

        # ── Hybrid segment ──
        hyb = aa.get("hybrid_segment", {})
        hyb_fv = hyb.get("feature_vector", {}) if isinstance(hyb, dict) else {}

        # ── Reference ──
        ref = aa.get("reference_check", {})
        ref_fv = ref.get("feature_values", ref) if isinstance(ref, dict) else {}

        # ── Stylometric ──
        sty = getattr(detection_result, "statistical_features", {}) or {}

        # ── Tier-1 model-agnostic signals ──
        aus = aa.get("author_signature", {}) if isinstance(aa.get("author_signature"), dict) else {}
        dsc = aa.get("discourse_structure", {}) if isinstance(aa.get("discourse_structure"), dict) else {}
        sem = aa.get("semantic_consistency", {}) if isinstance(aa.get("semantic_consistency"), dict) else {}
        lang = aa.get("language", {}) if isinstance(aa.get("language"), dict) else {}

        # [Fase-2 M-12] Every feature is normalised to [0,1] with a documented scale so
        # the vector is homogeneous (interpretable trained weights, tractable ablation).
        # Scales: ppl proxy lives on the invented [1,15] scale; curvature is clamped to
        # [-10,10] upstream; entropy ~[0,10] bits; sentence length saturates at 40 words;
        # breakpoint/run counts saturate at 6/10 (documents rarely exceed these).
        values: Dict[str, float] = {
            "neural_ai_prob":          float(np.clip(neural_ai_prob, 0.0, 1.0)),
            "neural_uncertainty":      float(np.clip(neural_uncertainty, 0.0, 1.0)),
            "ppl_proxy_mean":          _norm01(_num(ppl_fv, "proxy_perplexity_mean"), 15.0),
            "ppl_low_ratio":           _num(ppl_fv, "low_perplexity_ratio"),
            "ppl_valley_count":        _norm01(_num(ppl_fv, "perplexity_valley_count"), 10.0),
            "ppl_burstiness":          _norm01(_num(ppl_fv, "burstiness_perplexity"), 2.0),
            "ppl_curvature":           _norm01(_num(ppl_fv, "curvature_score") + 10.0, 20.0),
            "ppl_entropy_mean":        _norm01(_num(ppl_fv, "token_entropy_mean"), 10.0),
            "rsn_backtracking":        _reasoning_feature(rsn, "backtracking_density"),
            "rsn_cot_scaffold":        _reasoning_feature(rsn, "cot_scaffold_density"),
            "rsn_entropy_norm":        _reasoning_feature(rsn, "word_entropy_normalised"),
            "rsn_type_token_ratio":    _reasoning_feature(rsn, "type_token_ratio"),
            "hal_overall":             _num(hal, "overall_risk"),
            "hal_semantic_incoherence": _num(hal_cats, "semantic_incoherence"),
            "hal_vagueness":           _num(hal_cats, "vagueness"),
            "hal_repetition":          _num(hal_cats, "repetition"),
            "hyb_global_ai":           _num(hyb, "global_ai_score") / 100.0,
            "hyb_ai_ratio":            _num(hyb_fv, "ai_segment_ratio"),
            "hyb_breakpoints":         _norm01(_num(hyb_fv, "breakpoint_count"), 6.0),
            "hyb_longest_ai_run":      _norm01(_num(hyb_fv, "longest_ai_run"), 10.0),
            "ref_fabricated_ratio":    _num(ref_fv, "fabricated_ratio"),
            "ref_chimeric_ratio":      _num(ref_fv, "chimeric_ratio"),
            "ref_verified_ratio":      _num(ref_fv, "verified_ratio"),
            "sty_burstiness":          _norm01(_num(sty, "burstiness", "burstiness_score"), 1.0),
            "sty_lexical_diversity":   _num(sty, "lexical_diversity", "vocabulary_richness"),
            "sty_avg_sentence_len":    _norm01(_num(sty, "avg_sentence_length"), 40.0),
            "author_outlier_ratio":    _num(aus, "outlier_ratio"),
            "dsc_uniformity":          _num(dsc, "uniformity"),
            # M-6: only STRONG contradictions (numeric mismatch / NLI) feed the fusion;
            # the noisy negation-flip heuristic stays report-only.
            "sem_contradiction_ratio": _num(sem, "strong_contradiction_ratio"),
            "sty_hapax_ratio":         _num(sty, "hapax_legomena_ratio"),
            "lang_en":                 1.0 if str(lang.get("lang", "en")).lower().startswith("en") else 0.0,
        }

        vec = np.array([values[n] for n in _FUSION_SCHEMA], dtype=np.float64)
        return FusionFeatures(vector=vec, names=FEATURE_NAMES)


# ============================================================================
# Meta-classifier (uncalibrated scaffold)
# ============================================================================

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class FusionResult:
    probability: float                 # P(AI) ∈ [0,1]
    calibrated: bool                   # False until trained + calibrated
    source: str                        # "neural_passthrough" | "heuristic_fusion" | "logistic"
    features: Dict[str, float] = field(default_factory=dict)
    # [Fase-2 M-21/N-15] Per-term log-odds contributions: positive pushed toward AI,
    # negative toward human, zero-suffixed "_excluded_lang" entries were gated out.
    # This is the explainability artifact — surface it, never discard it.
    contributions: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
            "calibrated": self.calibrated,
            "source": self.source,
            "note": self.note,
            "features": {k: round(v, 6) for k, v in self.features.items()},
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
        }


# ============================================================================
# Default (untrained) heuristic fusion — MODEL-AGNOSTIC, BOUNDED, UNCALIBRATED
# ============================================================================
# This is the interim fusion used until a labelled corpus enables fit(). It makes the
# plugins ACTUALLY contribute to the score (the audit's core complaint was that they were
# decorative) while keeping the neural ensemble dominant and bounding every adjustment so
# no single weak signal can flip the verdict. The adjustments use MODEL-AGNOSTIC signals
# (fabricated citations, hallucination, reasoning markers, perplexity, ensemble
# disagreement) — features that do not depend on which LLM produced the text — so they give
# *some* lift against frontier models the 2023 neural net never saw. It is NOT calibrated
# and is NOT a substitute for retraining (see docs/EXPERIMENTAL_PROTOCOL.md §Retraining).

# Anti-overconfidence prior: soften the neural log-odds. Transformer softmax is
# systematically over-confident; T>1 pulls extreme probabilities toward 0.5. This is a
# documented heuristic prior, NOT data-calibrated temperature scaling.
DEFAULT_NEURAL_TEMPERATURE: float = 1.6

# Cap on the neural log-odds magnitude BEFORE temperature. The neural ai% is rounded to
# integers, so confident outputs saturate at exactly 0/100 → infinite log-odds, which no
# temperature can soften (the audit's "falsely crisp 100%"). Capping at ±4.0 (≈ prob
# 0.018..0.982) encodes that an OOD-blind 2023 detector should never claim literal certainty,
# and leaves headroom for plugins + temperature to move the score.
_NEURAL_LOGIT_CAP: float = 4.0

# Bounded log-odds weights for model-agnostic adjustments (each input is in [0,1]).
# [Fase-2 N-01/N-02] Removed from this dict:
#   - hyb_ai_ratio: produced by the SAME 3-model ensemble as neural_ai_prob — counting it
#     here double-counted the neural evidence (it stays in the vector for the TRAINED
#     fusion, where the logistic regression absorbs the collinearity).
#   - ppl_low_ratio: Tier-1 proxy is derived from hapax/TTR (not perplexity) and dies
#     under any edit; too fragile to move a verdict.
_HEURISTIC_WEIGHTS: Dict[str, float] = {
    "ref_fabricated_ratio":   1.6,   # strong: verified-absent citations (model-agnostic)
    "ref_chimeric_ratio":     0.9,
    "hal_overall":            0.6,   # moderate: internal incoherence
    "rsn_cot_scaffold":       0.5,   # reasoning-model scaffolding
    "rsn_backtracking":       0.5,
    "dsc_uniformity":         0.7,   # Tier-1: templated discourse (survives paraphrasing)
    "sem_contradiction_ratio": 0.6,  # Tier-1: STRONG internal contradiction (numeric/NLI)
    "author_outlier_ratio":   0.4,   # Tier-1: style splice (mild mixed-authorship signal)
}

# [Fase-2 M-4] Pro-human terms (negative log-odds). Small, bounded: high burstiness and a
# rich hapax profile are organic-writing signals; verified citations are external ground
# truth. Without these, every active default signal could only PUSH toward AI (N-01).
_HEURISTIC_HUMAN_WEIGHTS: Dict[str, float] = {
    "sty_burstiness":     -0.35,
    "sty_hapax_ratio":    -0.25,
    "ref_verified_ratio": -0.60,
}

# [Fase-2 M-5] Features whose computation depends on ENGLISH lexicons. On non-English
# documents they are structurally unreliable, so the gate zeroes them out of the
# adjustment (reported as excluded). [M-18] discourse_analyzer and semantic_consistency
# now carry es/fr/pt lexicons and select them by detected language, so only the
# reasoning-profiler regexes (English CoT markers) remain gated.
_LEXICON_EN_FEATURES = frozenset({
    "rsn_cot_scaffold", "rsn_backtracking",
})

# [Fase-2 M-1] Corroboration families. Signals inside one family measure the same
# underlying phenomenon (e.g. "formal register"), so they corroborate each other only
# weakly. The positive adjustment is allowed its full budget only when at least TWO
# independent families are active; reference evidence (external ground truth) counts as
# two families on its own.
_FAMILY_MAP: Dict[str, str] = {
    "ref_fabricated_ratio":    "reference",
    "ref_chimeric_ratio":      "reference",
    "rsn_cot_scaffold":        "structure",
    "rsn_backtracking":        "structure",
    "dsc_uniformity":          "structure",
    "hal_overall":             "coherence",
    "sem_contradiction_ratio": "coherence",
    "author_outlier_ratio":    "authorship",
}
_FAMILY_ACTIVE_MIN: float = 0.15    # summed log-odds for a family to count as "active"

# [Fase-2 M-1] ASYMMETRIC adjustment clamps (audit principle #1: cut FP before recall).
# Positive (toward-AI) budget: +0.35 with a single active family, +0.6 with corroboration.
# Negative (toward-human) budget: −1.2 — evidence of humanity may temper more than
# heuristics may accuse.
_HEURISTIC_ADJ_POS_SINGLE: float = 0.35
_HEURISTIC_ADJ_POS_CORROBORATED: float = 0.6
_HEURISTIC_ADJ_NEG: float = -1.2


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def heuristic_fusion(features: Dict[str, float],
                     temperature: float = DEFAULT_NEURAL_TEMPERATURE
                     ) -> Tuple[float, Dict[str, float]]:
    """
    Bounded, model-agnostic, UNCALIBRATED fusion of the assembled features → P(AI).

    Returns (probability, contributions_in_logodds). The neural term (temperature-softened)
    dominates; plugin adjustments are summed in log-odds under an ASYMMETRIC clamp:
    the toward-AI budget is small and only grows when ≥2 independent signal families
    corroborate each other, while toward-human evidence keeps a larger budget
    (Fase-2 N-01 fix — accumulated genre-correlated heuristics must not flip a
    human verdict on their own).
    """
    neural = float(features.get("neural_ai_prob", 0.5))
    capped = float(np.clip(_logit(neural), -_NEURAL_LOGIT_CAP, _NEURAL_LOGIT_CAP))
    base_lo = capped / max(temperature, 1e-6)

    is_english = float(features.get("lang_en", 1.0)) >= 0.5

    contributions: Dict[str, float] = {"neural_softened": base_lo}
    adj = 0.0
    family_lo: Dict[str, float] = {}
    for feat, w in _HEURISTIC_WEIGHTS.items():
        if not is_english and feat in _LEXICON_EN_FEATURES:
            contributions[f"{feat}_excluded_lang"] = 0.0
            continue
        v = float(np.clip(features.get(feat, 0.0), 0.0, 1.0))
        c = w * v
        if c != 0.0:
            contributions[feat] = c
            fam = _FAMILY_MAP.get(feat)
            if fam is not None:
                family_lo[fam] = family_lo.get(fam, 0.0) + c
        adj += c

    # Pro-human terms (negative log-odds), always active.
    for feat, w in _HEURISTIC_HUMAN_WEIGHTS.items():
        v = float(np.clip(features.get(feat, 0.0), 0.0, 1.0))
        c = w * v
        if c != 0.0:
            contributions[feat] = c
        adj += c

    # Corroboration rule: full positive budget only with ≥2 active families;
    # external ground truth (reference) corroborates on its own.
    active = sum(1 for lo in family_lo.values() if lo >= _FAMILY_ACTIVE_MIN)
    if family_lo.get("reference", 0.0) >= _FAMILY_ACTIVE_MIN:
        active += 1
    pos_cap = _HEURISTIC_ADJ_POS_CORROBORATED if active >= 2 else _HEURISTIC_ADJ_POS_SINGLE

    adj = float(np.clip(adj, _HEURISTIC_ADJ_NEG, pos_cap))
    contributions["_active_families"] = float(active)
    contributions["_adjustment_clamped"] = adj
    p = 1.0 / (1.0 + math.exp(-(base_lo + adj)))
    return float(np.clip(p, 0.0, 1.0)), contributions


class FusionClassifier:
    """
    Late-fusion meta-classifier over the assembled fusion vector.

    DEFAULT (untrained): transparent neural pass-through — predict_proba() returns the
    neural AI probability and reports calibrated=False. This deliberately avoids inventing
    a new pseudo-score before training data exists (a core finding of the audit).

    TRAINED: call fit(X, y) with a labelled corpus to learn a logistic model (requires
    scikit-learn). Optionally attach a calibrator (see calibration.TemperatureScaler) so
    predict_proba() returns a calibrated probability and reports calibrated=True.
    """

    _NEURAL_IDX = _FUSION_SCHEMA.index("neural_ai_prob")

    def __init__(self, untrained_mode: str = "heuristic") -> None:
        """
        untrained_mode : behaviour before fit():
            "heuristic"   — bounded model-agnostic fusion (default; plugins contribute).
            "passthrough" — return the neural probability unchanged.
        """
        self._builder = FusionFeatureBuilder()
        self._untrained_mode = untrained_mode
        self._weights: Optional[np.ndarray] = None   # set by fit()
        self._bias: float = 0.0
        self._mean: Optional[np.ndarray] = None      # standardisation
        self._std: Optional[np.ndarray] = None
        self._calibrator: Any = None                 # optional, exposes .apply(p)->p
        self._trained: bool = False

    # ── Inference ──────────────────────────────────────────────────────────
    def predict_proba_vec(self, vec: np.ndarray) -> FusionResult:
        names = FEATURE_NAMES
        feat = {n: float(v) for n, v in zip(names, vec)}

        if not self._trained or self._weights is None:
            if self._untrained_mode == "passthrough":
                p = float(np.clip(vec[self._NEURAL_IDX], 0.0, 1.0))
                return FusionResult(
                    probability=p, calibrated=False, source="neural_passthrough",
                    features=feat,
                    note="Neural pass-through. Train with fit(X, y) to enable fusion.",
                )
            p, contrib = heuristic_fusion(feat)
            return FusionResult(
                probability=p, calibrated=False, source="heuristic_fusion",
                features=feat, contributions=contrib,
                note="Bounded model-agnostic heuristic fusion (UNCALIBRATED, asymmetric "
                     "clamp + corroboration rule). Plugins contribute but the neural "
                     "ensemble dominates. Train with fit(X, y) on a labelled corpus + "
                     "calibrate to replace this.",
            )

        x = vec.astype(np.float64)
        if self._mean is not None and self._std is not None:
            x = (x - self._mean) / self._std
        z = float(np.dot(self._weights, x) + self._bias)
        p = _sigmoid(z)
        calibrated = False
        if self._calibrator is not None:
            p = float(self._calibrator.apply(p))
            calibrated = True
        # Per-term contributions for the trained path: weight × standardized input.
        contrib = {n: float(w * xi) for n, w, xi in zip(names, self._weights, x)}
        contrib["_bias"] = self._bias
        return FusionResult(
            probability=float(np.clip(p, 0.0, 1.0)),
            calibrated=calibrated,
            source="logistic",
            features=feat, contributions=contrib,
            note="Trained logistic fusion." + ("" if calibrated else " NOT calibrated."),
        )

    def predict_proba(self, detection_result: Any,
                      additional_analyses: Optional[Dict[str, Any]] = None) -> FusionResult:
        """Build the fusion vector from orchestrator output and score it."""
        ff = self._builder.build(detection_result, additional_analyses)
        return self.predict_proba_vec(ff.vector)

    # ── Training (requires scikit-learn) ────────────────────────────────────
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int],
            standardize: bool = True) -> "FusionClassifier":
        """
        Train a logistic-regression fusion model.

        X : (n_samples, FUSION_VECTOR_DIM) assembled fusion vectors.
        y : (n_samples,) labels — 1 = AI, 0 = human.

        Raises ImportError if scikit-learn is unavailable (kept optional so the
        scaffold imports with zero extra deps).
        """
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.int64)
        if Xa.ndim != 2 or Xa.shape[1] != FUSION_VECTOR_DIM:
            raise ValueError(f"X must be (n, {FUSION_VECTOR_DIM}), got {Xa.shape}")

        if standardize:
            self._mean = Xa.mean(axis=0)
            self._std = Xa.std(axis=0)
            self._std[self._std < 1e-9] = 1.0
            Xs = (Xa - self._mean) / self._std
        else:
            self._mean = self._std = None
            Xs = Xa

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FusionClassifier.fit requires scikit-learn. "
                "Install it, or supply weights via set_weights()."
            ) from exc

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(Xs, ya)
        self._weights = model.coef_[0].astype(np.float64)
        self._bias = float(model.intercept_[0])
        self._trained = True
        logger.info("FusionClassifier trained on %d samples (%d features).",
                    Xa.shape[0], Xa.shape[1])
        return self

    def set_weights(self, weights: Sequence[float], bias: float = 0.0,
                    mean: Optional[Sequence[float]] = None,
                    std: Optional[Sequence[float]] = None) -> "FusionClassifier":
        """Inject pre-trained weights (e.g. loaded from disk) without sklearn."""
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != FUSION_VECTOR_DIM:
            raise ValueError(f"weights must have {FUSION_VECTOR_DIM} elements")
        self._weights = w
        self._bias = float(bias)
        self._mean = np.asarray(mean, dtype=np.float64) if mean is not None else None
        self._std = np.asarray(std, dtype=np.float64) if std is not None else None
        self._trained = True
        return self

    def attach_calibrator(self, calibrator: Any) -> "FusionClassifier":
        """Attach an object exposing .apply(prob)->prob (e.g. TemperatureScaler)."""
        self._calibrator = calibrator
        return self

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── Persistence [Fase-2 M-19 wiring] ────────────────────────────────────
    def to_payload(self) -> Dict[str, Any]:
        """Serializable snapshot of a trained model (+ calibrator temperature)."""
        if not self._trained or self._weights is None:
            raise ValueError("Cannot serialize an untrained FusionClassifier")
        temp = getattr(self._calibrator, "temperature", None)
        return {
            "schema": "fusion-weights-v1",
            "feature_names": list(FEATURE_NAMES),
            "weights": [float(w) for w in self._weights],
            "bias": self._bias,
            "mean": [float(m) for m in self._mean] if self._mean is not None else None,
            "std": [float(s) for s in self._std] if self._std is not None else None,
            "temperature": float(temp) if temp is not None else None,
        }

    def load_payload(self, payload: Dict[str, Any]) -> "FusionClassifier":
        """
        Load weights produced by scripts/corpus/train_fusion.py.

        Refuses payloads whose feature_names do not EXACTLY match the current
        _FUSION_SCHEMA — a schema drift means the weights are meaningless and
        silently applying them would corrupt every verdict.
        """
        names = payload.get("feature_names")
        if tuple(names or ()) != FEATURE_NAMES:
            raise ValueError(
                "Fusion weights schema mismatch: trained on "
                f"{len(names or [])} features, current schema has "
                f"{len(FEATURE_NAMES)}. Retrain (scripts/corpus/train_fusion.py) "
                "before deploying."
            )
        self.set_weights(payload["weights"], bias=payload.get("bias", 0.0),
                         mean=payload.get("mean"), std=payload.get("std"))
        temp = payload.get("temperature")
        if temp is not None:
            from calibration import TemperatureScaler
            self.attach_calibrator(TemperatureScaler(temperature=float(temp), fitted=True))
        return self


# ============================================================================
# Shared instance [Fase-2 M-13 + M-19 wiring]
# ============================================================================
# One process-wide classifier. If FUSION_WEIGHTS_PATH points at a weights file
# produced by scripts/corpus/train_fusion.py, it is loaded ONCE at first use and
# every request runs the trained+calibrated logistic fusion instead of the
# interim heuristic. Remember to bump MODEL_VERSION when deploying new weights
# (invalidates the namespaced result caches).

_shared_fusion: Optional[FusionClassifier] = None


def get_fusion_classifier() -> FusionClassifier:
    """Return the process-wide FusionClassifier (created lazily; loads trained
    weights from FUSION_WEIGHTS_PATH when set)."""
    global _shared_fusion
    if _shared_fusion is None:
        clf = FusionClassifier()
        weights_path = os.getenv("FUSION_WEIGHTS_PATH", "")
        if weights_path:
            try:
                with open(weights_path, "r", encoding="utf-8") as fh:
                    clf.load_payload(json.load(fh))
                logger.info(
                    "Trained fusion weights loaded from %s (calibrated=%s). "
                    "Ensure MODEL_VERSION was bumped for this deploy.",
                    weights_path, clf._calibrator is not None,
                )
            except Exception as exc:
                logger.error(
                    "Failed to load fusion weights from %s — falling back to "
                    "heuristic fusion: %s", weights_path, exc,
                )
                clf = FusionClassifier()
        _shared_fusion = clf
    return _shared_fusion
