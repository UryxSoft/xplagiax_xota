"""
calibration.py — Confidence calibration utilities (framework, no training data required).
==========================================================================================

The audit found the production "confidence" is `max(softmax)`, presented as a probability
though transformer softmax is systematically over-confident, with NO calibration and NO ECE
ever measured. These utilities provide the standard tools to fix that once labelled data exists:

  - compute_ece(...)       Expected Calibration Error (binned).
  - reliability_bins(...)  Per-bin confidence/accuracy for reliability diagrams.
  - TemperatureScaler      Single-parameter temperature scaling (Guo et al., 2017).
  - brier_score(...)       Mean squared error of probabilistic predictions.

Pure NumPy — zero heavy deps, fully unit-testable on synthetic data. None of this is wired
into the production verdict yet (Fase 2 is framework-only); see docs/EXPERIMENTAL_PROTOCOL.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_EPS = 1e-12


def _as_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _as_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================================
# Metrics
# ============================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error for binary P(AI).

    ECE = Σ_b (|B_b|/N) · |acc(B_b) − conf(B_b)|

    probs  : predicted P(positive) ∈ [0,1].
    labels : 0/1 ground truth.
    Returns ECE ∈ [0,1] (0 = perfectly calibrated).
    """
    probs = _as_prob(probs)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape or probs.size == 0:
        raise ValueError("probs and labels must be same non-empty shape")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = probs.size
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so p==1.0 is included
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 15) -> List[Tuple[float, float, int]]:
    """Return [(mean_confidence, accuracy, count)] per bin for reliability diagrams."""
    probs = _as_prob(probs)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Tuple[float, float, int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            out.append((float((lo + hi) / 2), float("nan"), 0))
        else:
            out.append((float(probs[mask].mean()), float(labels[mask].mean()), cnt))
    return out


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    probs = _as_prob(probs)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = _as_prob(probs)
    labels = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


# ============================================================================
# Temperature scaling (Guo et al., 2017)
# ============================================================================

@dataclass
class TemperatureScaler:
    """
    Single-parameter temperature scaling for binary P(AI).

    Fits T > 0 by minimising NLL of  sigmoid(logit(p) / T)  against labels via a
    coarse-to-fine 1-D search (no autograd dependency). T > 1 softens an over-confident
    model; T < 1 sharpens an under-confident one.

    Usage::
        ts = TemperatureScaler().fit(val_probs, val_labels)
        calibrated_p = ts.apply(p)              # single probability
        calibrated   = ts.apply_array(probs)    # vectorised
    """

    temperature: float = 1.0
    fitted: bool = False

    def fit(self, probs: np.ndarray, labels: np.ndarray,
            t_min: float = 0.05, t_max: float = 10.0) -> "TemperatureScaler":
        logits = _logit(np.asarray(probs, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.float64)

        def nll_at(t: float) -> float:
            return _nll(_sigmoid(logits / t), labels)

        lo, hi = t_min, t_max
        best_t, best_nll = 1.0, nll_at(1.0)
        # coarse-to-fine grid refinement
        for _ in range(6):
            grid = np.linspace(lo, hi, 25)
            nlls = [nll_at(float(t)) for t in grid]
            j = int(np.argmin(nlls))
            if nlls[j] < best_nll:
                best_nll, best_t = nlls[j], float(grid[j])
            span = (hi - lo) / 25
            lo = max(t_min, best_t - span)
            hi = min(t_max, best_t + span)

        self.temperature = best_t
        self.fitted = True
        return self

    def apply_array(self, probs: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(probs, dtype=np.float64))
        return _sigmoid(z / self.temperature)

    def apply(self, prob: float) -> float:
        """Scale a single probability (interface consumed by FusionClassifier)."""
        z = _logit(np.array([prob], dtype=np.float64))[0]
        return float(_sigmoid(np.array([z / self.temperature]))[0])
