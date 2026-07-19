"""
Unit tests for app/engine/author_embedding.py — pure math paths via an
injected embed_fn; no model download required.
"""

import numpy as np
import pytest

from app.engine.author_embedding import (
    MIN_CHUNKS,
    _build_chunks,
    analyze_document,
)


def _para(marker: str, words: int = 130) -> str:
    return f"{marker} " + " ".join(
        f"palabra{k} de relleno académico" for k in range(words // 4)
    ) + "."


def _long_text(n_paras: int = 12) -> str:
    return "\n\n".join(_para(f"Sección {i}.") for i in range(n_paras))


class TestBuildChunks:

    def test_groups_to_target_words(self):
        chunks = _build_chunks(_long_text())
        assert len(chunks) >= MIN_CHUNKS
        # No chunk (except possibly a merged tail) is tiny
        assert all(len(c.split()) >= 100 for c in chunks)

    def test_short_text_single_chunk(self):
        chunks = _build_chunks("Un párrafo corto de quince palabras que no da "
                               "para segmentar en absoluto nada más.")
        assert len(chunks) == 1


class TestAnalyzeDocument:

    def test_uniform_document_no_outliers(self):
        def embed_uniform(texts):
            v = np.zeros((len(texts), 8)); v[:, 0] = 1.0
            return v

        res = analyze_document(_long_text(), embed_fn=embed_uniform)
        assert res["status"] == "ok"
        assert res["reliable"] is True
        assert res["outlier_count"] == 0
        assert res["outlier_ratio"] == 0.0
        assert res["mean_self_similarity"] == pytest.approx(1.0)
        assert "HIGH CONSISTENCY" in res["consistency_level"]

    def test_divergent_chunk_detected(self):
        def embed_marked(texts):
            out = np.zeros((len(texts), 8))
            for i, t in enumerate(texts):
                # Orthogonal signature for the injected section
                out[i, 1 if "INJERTO" in t else 0] = 1.0
            return out

        paras = [_para(f"Sección {i}.") for i in range(12)]
        # One full chunk (~3-4 paragraphs) of foreign authorship
        paras[4] = _para("INJERTO uno.")
        paras[5] = _para("INJERTO dos.")
        paras[6] = _para("INJERTO tres.")
        paras[7] = _para("INJERTO cuatro.")
        res = analyze_document("\n\n".join(paras), embed_fn=embed_marked)

        assert res["status"] == "ok"
        assert res["outlier_count"] >= 1
        assert 0.0 < res["outlier_ratio"] < 1.0
        assert "MIXED" in res["consistency_level"]
        assert any("INJERTO" in o["text_preview"] for o in res["outliers"])
        assert "max_break" in res

    def test_too_short_is_unreliable_not_a_verdict(self):
        res = analyze_document("Texto corto.", embed_fn=lambda t: np.ones((len(t), 4)))
        assert res["status"] == "ok"
        assert res["reliable"] is False
        assert res["outlier_ratio"] == 0.0   # neutral for fusion

    def test_engine_unavailable_keeps_fusion_contract(self):
        # No embed_fn injected and model not loaded (ENABLE_AUTHOR_EMBEDDING
        # unset in the test environment) → error dict must still carry the
        # fusion feature key.
        res = analyze_document(_long_text())
        assert "outlier_ratio" in res
        assert res["outlier_ratio"] == 0.0
