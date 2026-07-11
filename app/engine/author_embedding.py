"""
author_embedding.py — Intra-document authorship consistency via authorship
embeddings (LUAR / PAN-style verification), replacing hand-crafted stylometric
dispersion when available (docs/sota/D_AUTHOR_SIGNATURE.md).

Answers: "did the same person write section 1 and section 4?" — the signal most
robust to paraphrasing: a humanizer rewrites n-grams, not the deep author
signature.

Activation (opt-in — run the D.1 GO/NO-GO benchmark BEFORE trusting it in
Spanish; see scripts/benchmark_luar.py):

    ENABLE_AUTHOR_EMBEDDING=1              enable the engine
    AUTHOR_EMBED_MODEL=rrivera1849/LUAR-MUD  (default; needs `pip install einops`)
    AUTHOR_EMBED_DOWNLOAD=1                allow first-time HF download
                                           (default: local cache only)
    AUTHOR_EMBED_OUTLIER_THRESHOLD=0.60    provisional — calibrate per guide D.3
                                           (percentile 5 of mono-author sims)

Output dict is fusion-compatible: always includes `outlier_ratio` ∈ [0,1]
(consumed by fusion.py feature `author_outlier_ratio`) and mirrors the key
names of authorship_consistency.compute_authorship_consistency() so the
forensic report renders either source unchanged.

Falls back gracefully: when disabled/unavailable the orchestrator keeps using
the stylometric implementation (authorship_consistency.py).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # engine modules bare-import each other in prod; app.* path in tests
    from hybrid_segment_detector import TextSegmenter, _strip_references_section
except ImportError:  # pragma: no cover
    from app.engine.hybrid_segment_detector import TextSegmenter, _strip_references_section

# ── Chunking parameters ──────────────────────────────────────────────────────
# LUAR-style encoders need substantial text per unit: 300-500 words. Below
# ~100 words the embedding is noise (guide D, error #2).
TARGET_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 100
MIN_CHUNKS = 3          # fewer chunks → nothing to compare → reliable: False
EMBED_BATCH = 8
MAX_SEQ_TOKENS = 512

_MODEL_NAME = os.getenv("AUTHOR_EMBED_MODEL", "rrivera1849/LUAR-MUD")
_ENABLED = os.getenv("ENABLE_AUTHOR_EMBEDDING", "0") == "1"

_model = None
_tokenizer = None
_available = False

if _ENABLED:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _local_only = os.getenv("AUTHOR_EMBED_DOWNLOAD", "0") != "1"
        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME, trust_remote_code=True, local_files_only=_local_only)
        _model = AutoModel.from_pretrained(
            _MODEL_NAME, trust_remote_code=True, local_files_only=_local_only)
        _model.eval()
        # CoW-friendly: loaded at import (preload_app) so forked workers share pages.
        _available = True
        logger.info("Author-embedding engine loaded (%s)", _MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 — optional engine, degrade silently
        logger.warning("Author-embedding engine unavailable (%s): %s",
                       _MODEL_NAME, exc)


def is_available() -> bool:
    return _available


# ── Result cache (same pattern as detector_final._FAST_CACHE) ────────────────
_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 300.0
_CACHE_MAX = 20


def _build_chunks(text: str) -> List[str]:
    """Section-aware chunks of ~TARGET_CHUNK_WORDS words.

    Bibliography is stripped (reference lists are not authored prose) and
    paragraphs are grouped, never split, so a chunk stays a coherent unit of
    one register. Trailing fragments below MIN_CHUNK_WORDS merge backwards.
    """
    body = _strip_references_section(text)
    paras = [p for p, _s, _e in TextSegmenter.split_paragraphs(body)]

    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0
    for p in paras:
        cur.append(p)
        cur_words += len(p.split())
        if cur_words >= TARGET_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
    if cur:
        tail = " ".join(cur)
        if len(tail.split()) >= MIN_CHUNK_WORDS or not chunks:
            chunks.append(tail)
        else:
            chunks[-1] += " " + tail
    return chunks


def embed_chunks(texts: List[str]) -> np.ndarray:
    """L2-normalised authorship embeddings, batched. Shape (n_chunks, dim).

    LUAR expects (batch, n_utterances, seq_len); each chunk is one utterance.
    """
    import torch

    outs: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            enc = _tokenizer(batch, padding="max_length", truncation=True,
                             max_length=MAX_SEQ_TOKENS, return_tensors="pt")
            enc = {k: v.unsqueeze(1) for k, v in enc.items()}  # utterance dim
            emb = _model(**enc)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            outs.append(emb.cpu().numpy())
    return np.concatenate(outs, axis=0)


def analyze_document(
    text: str,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Authorship-consistency analysis over embedding space.

    embed_fn is injectable for testing; defaults to the module LUAR encoder.
    Always returns `outlier_ratio` (fusion contract), even on error paths.
    """
    if embed_fn is None:
        if not _available:
            return {"status": "error", "method": "embedding",
                    "error": "author-embedding engine not loaded "
                             "(set ENABLE_AUTHOR_EMBEDDING=1)",
                    "outlier_ratio": 0.0}
        embed_fn = embed_chunks

        key = hashlib.sha256(text.encode()).hexdigest()
        now = time.monotonic()
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
            if hit is not None and now - hit[1] < _CACHE_TTL:
                return hit[0]
    else:
        key = None

    chunks = _build_chunks(text)
    if len(chunks) < MIN_CHUNKS:
        return {
            "status": "ok", "method": "embedding", "reliable": False,
            "chunk_count": len(chunks), "outlier_count": 0,
            "outlier_ratio": 0.0, "outliers": [],
            "interpretation": (
                f"Document too short for authorship comparison: {len(chunks)} "
                f"chunk(s) of ≥{MIN_CHUNK_WORDS} words (minimum {MIN_CHUNKS})."
            ),
        }

    E = np.asarray(embed_fn(chunks), dtype=np.float64)
    # Defensive re-normalisation — cosine math below assumes unit vectors.
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    E = E / norms

    centroid = E.mean(axis=0)
    c_norm = np.linalg.norm(centroid)
    centroid = centroid / (c_norm if c_norm > 1e-12 else 1.0)
    sims = E @ centroid                                   # (n,)

    threshold = float(os.getenv("AUTHOR_EMBED_OUTLIER_THRESHOLD", "0.60"))
    outlier_idx = [int(i) for i in np.where(sims < threshold)[0]]

    adjacent = (E[:-1] * E[1:]).sum(axis=1)               # sim(chunk_i, chunk_i+1)
    break_i = int(np.argmin(adjacent))

    mean_sim = float(sims.mean())
    outlier_ratio = len(outlier_idx) / len(chunks)

    if outlier_idx:
        level = "MIXED — authorship signature breaks detected"
        interpretation = (
            f"{len(outlier_idx)} of {len(chunks)} chunks diverge from the "
            f"document's authorship signature (embedding similarity < "
            f"{threshold:.2f}). This can indicate multiple authors, heavily "
            f"quoted material, or an AI-spliced section. Localization aid, "
            f"not a verdict."
        )
    elif mean_sim >= 0.80:
        level = "HIGH CONSISTENCY — single coherent author signature"
        interpretation = (
            "Authorship embeddings are uniform across the document. Consistent "
            "with a single author (human or AI); NOT by itself an AI indicator."
        )
    else:
        level = "MODERATE CONSISTENCY"
        interpretation = (
            "Mild signature variation across sections — within the range of "
            "normal single-author writing."
        )

    result = {
        "status": "ok",
        "method": "embedding",
        "model": _MODEL_NAME,
        "reliable": True,
        "chunk_count": len(chunks),
        "mean_self_similarity": round(mean_sim, 4),
        "consistency_score": round(max(0.0, mean_sim), 4),
        "consistency_level": level,
        "interpretation": interpretation,
        "outlier_threshold": threshold,
        "outlier_count": len(outlier_idx),
        "outlier_ratio": round(outlier_ratio, 4),
        "outliers": [
            {
                "chunk_index": i,
                "similarity": round(float(sims[i]), 4),
                "text_preview": chunks[i][:100],
            }
            for i in outlier_idx
        ],
        "max_break": {
            "after_chunk": break_i,
            "adjacent_similarity": round(float(adjacent[break_i]), 4),
        },
    }

    if key is not None:
        with _CACHE_LOCK:
            if len(_CACHE) >= _CACHE_MAX:
                oldest = min(_CACHE, key=lambda k: _CACHE[k][1])
                del _CACHE[oldest]
            _CACHE[key] = (result, time.monotonic())
    return result
