"""
Engine unit tests — pure logic, no ML model loading required.
"""

import pytest


# ── StylometricProfiler._split_sentences (EC-01) ──────────────────────────

class TestSplitSentences:

    @pytest.fixture(autouse=True)
    def import_fn(self):
        try:
            from app.engine.stylometric_profiler import _split_sentences
            self.split = _split_sentences
        except ImportError:
            pytest.skip("stylometric_profiler not importable in this environment")

    def test_basic_split(self):
        sents = self.split("Hello world. How are you? Fine.")
        assert len(sents) >= 2

    def test_abbreviation_not_split(self):
        sents = self.split("Dr. Smith works at MIT. He is great.")
        assert len(sents) == 2

    def test_decimal_not_split(self):
        sents = self.split("The value is 3.14. That is pi.")
        assert len(sents) == 2

    def test_all_caps_acronym_not_split(self):
        """EC-01: NASA. CIA. should not be sentence boundaries."""
        sents = self.split("The NASA. and CIA. work together. Great.")
        # NASA. and CIA. should be protected — result should be 2 sentences
        assert len(sents) == 2

    def test_empty_string(self):
        assert self.split("") == []


# ── HybridSegmentAnalyzer minimum word count (EC-02) ──────────────────────

class TestHybridMinWords:

    @pytest.fixture(autouse=True)
    def import_cls(self):
        try:
            from app.engine.hybrid_segment_detector import (
                HybridSegmentAnalyzer, MIN_WINDOW_WORDS,
            )
            self.MIN = MIN_WINDOW_WORDS

            class _FakeClassifier:
                def __call__(self, text):
                    return (50.0, 50.0)

            self.analyzer = HybridSegmentAnalyzer(classify_fn=_FakeClassifier())
        except ImportError:
            pytest.skip("hybrid_segment_detector not importable")

    def test_short_text_returns_inconclusive(self):
        """< MIN_WINDOW_WORDS → INCONCLUSIVE with explicit message."""
        short = "word " * (self.MIN - 10)
        result = self.analyzer.analyze(short)
        assert result.classification == "INCONCLUSIVE"
        assert str(self.MIN) in result.interpretation

    def test_empty_text_returns_inconclusive(self):
        result = self.analyzer.analyze("")
        assert result.classification == "INCONCLUSIVE"


# ── PerplexityProfiler language detection (EC-03) ─────────────────────────

class TestPerplexityLanguageDetection:

    @pytest.fixture(autouse=True)
    def import_profiler(self):
        try:
            from app.engine.perplexity_profiler import PerplexityProfiler
            self.profiler = PerplexityProfiler()
        except ImportError:
            pytest.skip("perplexity_profiler not importable")

    def test_non_english_sets_language_warning(self):
        """Spanish text should trigger language_warning flag."""
        es_text = ("El perro come el hueso. La gata duerme en el sofá. "
                   "Los niños juegan en el parque. " * 10)
        result = self.profiler.compute_stats(es_text)
        # Only check if enough tokens were processed
        if result.get("tokens_analysed", 0) > 0:
            assert result.get("language_warning") == "non_english"

    def test_english_no_language_warning(self):
        """English text should NOT trigger language_warning."""
        en_text = ("The quick brown fox jumps over the lazy dog. "
                   "She is reading a book in the library. " * 10)
        result = self.profiler.compute_stats(en_text)
        assert "language_warning" not in result


# ── CitationDetector Vancouver orphan fix (EC-05) ─────────────────────────

class TestVancouverOrphan:

    @pytest.fixture(autouse=True)
    def import_detector(self):
        try:
            from app.antiplagio.citation.detector import CitationDetector
            self.detector = CitationDetector()
        except ImportError:
            pytest.skip("CitationDetector not importable")

    def test_vancouver_numeric_no_bibliography_not_orphan(self):
        """[1] [2] without bibliography section should NOT be flagged as orphans."""
        text = (
            "The study showed significant results [1]. "
            "Further analysis confirmed this finding [2]. "
            "Additional evidence supports the conclusion [3]."
        )
        result = self.detector.analyze(text)
        assert len(result.orphan_citations) == 0


# ── ReasoningRiskClassifier academic calibration (EC-04) ──────────────────

class TestReasoningClassifierThresholds:

    @pytest.fixture(autouse=True)
    def import_classifier(self):
        try:
            from app.engine.forensic_reports import ReasoningRiskClassifier
            self.clf = ReasoningRiskClassifier()
        except ImportError:
            pytest.skip("forensic_reports not importable")

    def test_academic_causal_density_not_high(self):
        """causal_density=0.08 (typical academic) should not hit HIGH threshold."""
        thr = self.clf._THR["causal_density"]
        # New upper bound is 0.12 — 0.08 should be MEDIUM or LOW
        assert thr[1] >= 0.10, f"causal_density upper threshold too low: {thr[1]}"

    def test_consequence_density_threshold_raised(self):
        thr = self.clf._THR["consequence_density"]
        assert thr[1] >= 0.09, f"consequence_density upper threshold too low: {thr[1]}"
