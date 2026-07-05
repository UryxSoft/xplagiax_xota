"""
authorship_consistency.py — Intra-document authorship-consistency core.
========================================================================

Shared core used by BOTH the HTTP plugin (app/plugins/author_signature.py) and the
forensic PluginOrchestrator (so the signal can feed the late-fusion vector). It takes an
already-loaded StylometricProfiler and a text, splits the text into chunks, extracts a
descriptive stylometric vector per chunk, and measures the dispersion of those vectors in
standardised (z-score) space.

Interpretation:
  • Tight clustering  ⇒ a single coherent author (human OR AI — formal/academic human
    writing is also very uniform, so HIGH consistency is NOT by itself an AI indicator).
  • An outlier chunk  ⇒ possible multiple authors, quoted material, or an AI-spliced
    section. This is a *localization* signal, not a verdict.

For fusion, the directional signal is `outlier_ratio` (fraction of chunks that diverge):
style splicing is mild evidence of mixed human+AI authorship.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

# Chunking parameters — chunks must be large enough for stable stylometry.
_TARGET_CHUNK_WORDS = 140
_MIN_CHUNK_WORDS = 70
_MIN_CHUNKS = 3

# Comparable, length-normalised stylometric features from compute_stats().
_FEATURES = (
    "burstiness", "lexical_diversity", "avg_sentence_length",
    "sentence_length_variance", "avg_word_length", "vocabulary_richness",
    "hapax_legomena_ratio", "rare_word_ratio", "comma_rate",
)

# A chunk whose RMS z-score across features exceeds this is flagged as a style outlier.
_OUTLIER_RMS_Z = 1.8


def split_chunks(text: str) -> List[str]:
    """Greedily group paragraphs/sentences into ~_TARGET_CHUNK_WORDS chunks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < _MIN_CHUNKS:
        blocks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    chunks: List[str] = []
    buf: List[str] = []
    count = 0
    for b in blocks:
        buf.append(b)
        count += len(b.split())
        if count >= _TARGET_CHUNK_WORDS:
            chunks.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        tail = " ".join(buf)
        if len(tail.split()) < _MIN_CHUNK_WORDS and chunks:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)
    return [c for c in chunks if len(c.split()) >= _MIN_CHUNK_WORDS]


def compute_authorship_consistency(profiler: Any, text: str) -> Dict[str, Any]:
    """
    Measure intra-document stylometric consistency using a loaded StylometricProfiler.

    Returns a JSON-serialisable dict (always includes `outlier_ratio` ∈ [0,1] for fusion).
    """
    if profiler is None:
        return {"status": "error", "error": "StylometricProfiler not loaded.", "outlier_ratio": 0.0}

    chunks = split_chunks(text)
    if len(chunks) < _MIN_CHUNKS:
        return {
            "status": "inconclusive",
            "reason": (
                f"Need at least {_MIN_CHUNKS} chunks of ≥{_MIN_CHUNK_WORDS} words "
                f"for authorship-consistency analysis (got {len(chunks)})."
            ),
            "chunk_count": len(chunks),
            "outlier_ratio": 0.0,
        }

    try:
        matrix = np.array(
            [[float(profiler.compute_stats(c).get(f, 0.0)) for f in _FEATURES]
             for c in chunks],
            dtype=np.float64,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"Analysis failed: {exc}", "outlier_ratio": 0.0}

    # Standardise each feature across chunks; constant features carry no info.
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    active = std > 1e-9
    z = np.zeros_like(matrix)
    z[:, active] = (matrix[:, active] - mean[active]) / std[active]

    # Per-chunk RMS z-score = how far (in std-devs) the chunk sits from the document's
    # average style. Identical chunks → all std 0 → RMS 0 → consistent.
    n_active = max(int(active.sum()), 1)
    rms = np.sqrt((z ** 2).sum(axis=1) / n_active)
    mean_rms = float(rms.mean())

    outliers = [
        {
            "chunk_index": int(i),
            "rms_zscore": round(float(rms[i]), 3),
            "divergent_features": [
                _FEATURES[j] for j in range(len(_FEATURES))
                if active[j] and abs(z[i, j]) >= 2.0
            ][:5],
            "text_preview": chunks[i][:100],
        }
        for i in range(len(chunks))
        if rms[i] >= _OUTLIER_RMS_Z
    ]

    # Map dispersion → consistency ∈ (0,1]; mean_rms 0 → 1.0 (identical style).
    consistency_score = float(1.0 / (1.0 + mean_rms))
    outlier_ratio = float(len(outliers) / len(chunks))

    if consistency_score >= 0.65 and not outliers:
        level = "HIGH CONSISTENCY — single coherent author"
        interpretation = (
            "The writing style is uniform across the document. This is consistent with a "
            "single author (human or AI) and is NOT by itself an AI indicator — formal and "
            "academic human writing is also highly uniform."
        )
    elif outliers:
        level = "MIXED — style shifts detected"
        interpretation = (
            f"{len(outliers)} of {len(chunks)} chunks diverge stylistically from the "
            f"document baseline. This can indicate multiple authors, quoted material, or an "
            f"AI-spliced section. Use the outlier list to locate WHICH sections differ — "
            f"this does not, on its own, prove AI authorship."
        )
    else:
        level = "MODERATE CONSISTENCY"
        interpretation = (
            "Mild stylistic variation across the document — within the range of normal "
            "single-author writing."
        )

    return {
        "status": "ok",
        "consistency_score": round(consistency_score, 4),
        "consistency_level": level,
        "interpretation": interpretation,
        "mean_rms_zscore": round(mean_rms, 4),
        "max_rms_zscore": round(float(rms.max()), 4),
        "active_features": n_active,
        "chunk_count": len(chunks),
        "outlier_count": len(outliers),
        "outlier_ratio": round(outlier_ratio, 4),
        "outliers": outliers,
    }
