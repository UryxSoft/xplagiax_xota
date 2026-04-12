"""
hybrid_segment_detector.py  (xota_ensemble_v6 — inference/)
=============================================================
Per-paragraph AI/Human heatmap via sliding-window ModernBERT.

Architecture
------------
  1. Segment text into overlapping windows (~300 words, 50% overlap).
  2. Each window passed through the injected classify_fn (4-model ensemble).
  3. Overlap scores averaged -> one score per paragraph.
  4. Breakpoint detection via gradient thresholding.
  5. 10-dimensional feature vector for Late Fusion.

Integration
-----------
  - Requires classify_fn callable with signature:
        classify_fn(text: str) -> (float, float)  # (human%, ai%)
    In production this is detector_final.classify_segment().
  - Called from plugin_orchestrator.py.
  - Results consumed by forensic_reports.py for heatmap rendering.

Changelog
---------
  v1.0  Initial implementation.

(c) 2025-2026 XplagiaX — Research use only.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_WORDS: int = 300
WINDOW_OVERLAP: float = 0.50
MIN_WINDOW_WORDS: int = 80
MIN_PARAGRAPH_WORDS: int = 15

THRESHOLD_AI: float = 70.0
THRESHOLD_UNCERTAIN: float = 30.0

BREAKPOINT_DELTA: float = 25.0

# Detect references/bibliography section header
_REF_HEADER_RE = re.compile(
    r"^\s*(?:References|Bibliography|Works Cited|Literature Cited"
    r"|Bibliograf[íi]a|Referencias)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_references_section(text: str) -> str:
    """Remove bibliography section so it is not analysed as paragraphs."""
    search_start = len(text) * 3 // 10
    matches = list(_REF_HEADER_RE.finditer(text, search_start))
    if matches:
        cut_pos = matches[-1].start()
        stripped = text[:cut_pos].strip()
        if len(stripped.split()) >= 20:
            return stripped
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WindowResult:
    window_index: int
    start_word: int
    end_word: int
    human_pct: float
    ai_pct: float
    word_count: int


@dataclass
class ParagraphScore:
    paragraph_index: int
    text: str
    start_word: int
    end_word: int
    word_count: int
    ai_score: float
    human_score: float
    zone: str
    contributing_windows: int


@dataclass
class Breakpoint:
    after_paragraph: int
    delta: float
    from_zone: str
    to_zone: str


@dataclass
class HybridAnalysisResult:
    paragraph_scores: List[ParagraphScore]
    window_results: List[WindowResult]
    breakpoints: List[Breakpoint]
    global_ai_score: float
    classification: str
    risk_level: str
    interpretation: str
    feature_vector: Dict[str, float] = field(default_factory=dict)
    total_words: int = 0
    total_paragraphs: int = 0
    total_windows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "risk_level": self.risk_level,
            "global_ai_score": round(self.global_ai_score, 4),
            "interpretation": self.interpretation,
            "total_words": self.total_words,
            "total_paragraphs": self.total_paragraphs,
            "total_windows": self.total_windows,
            "breakpoint_count": len(self.breakpoints),
            "feature_vector": {k: round(v, 6) for k, v in self.feature_vector.items()},
            "paragraph_scores": [
                {
                    "index": p.paragraph_index,
                    "text_preview": p.text[:80],
                    "word_count": p.word_count,
                    "ai_score": round(p.ai_score, 2),
                    "human_score": round(p.human_score, 2),
                    "zone": p.zone,
                    "windows": p.contributing_windows,
                }
                for p in self.paragraph_scores
            ],
            "breakpoints": [
                {
                    "after_paragraph": b.after_paragraph,
                    "delta": round(b.delta, 2),
                    "from_zone": b.from_zone,
                    "to_zone": b.to_zone,
                }
                for b in self.breakpoints
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# TextSegmenter
# ═══════════════════════════════════════════════════════════════════════════

class TextSegmenter:

    @staticmethod
    def split_paragraphs(text: str) -> List[Tuple[str, int, int]]:
        raw_paras = re.split(r"\n\s*\n|\n(?=[A-Z])", text.strip())
        raw_paras = [p.strip() for p in raw_paras if p.strip()]

        merged: List[str] = []
        for p in raw_paras:
            wc = len(p.split())
            if merged and wc < MIN_PARAGRAPH_WORDS:
                merged[-1] = merged[-1] + " " + p
            else:
                merged.append(p)

        result: List[Tuple[str, int, int]] = []
        offset = 0
        for p in merged:
            wc = len(p.split())
            result.append((p, offset, offset + wc))
            offset += wc
        return result

    @staticmethod
    def build_windows(total_words: int) -> List[Tuple[int, int]]:
        step = max(1, int(WINDOW_WORDS * (1 - WINDOW_OVERLAP)))
        windows: List[Tuple[int, int]] = []
        start = 0
        while start < total_words:
            end = min(start + WINDOW_WORDS, total_words)
            if (end - start) >= MIN_WINDOW_WORDS:
                windows.append((start, end))
            if end >= total_words:
                break
            start += step
        return windows


# ═══════════════════════════════════════════════════════════════════════════
# WindowClassifier
# ═══════════════════════════════════════════════════════════════════════════

class WindowClassifier:

    def __init__(self, classify_fn: Callable[[str], Tuple[float, float]]) -> None:
        self._classify_fn = classify_fn

    def classify_windows(
        self, words: List[str], windows: List[Tuple[int, int]]
    ) -> List[WindowResult]:
        results: List[WindowResult] = []
        for idx, (start, end) in enumerate(windows):
            window_text = " ".join(words[start:end])
            try:
                human_pct, ai_pct = self._classify_fn(window_text)
            except Exception as exc:
                logger.warning("Window %d classification failed: %s", idx, exc)
                human_pct, ai_pct = 50.0, 50.0

            results.append(WindowResult(
                window_index=idx, start_word=start, end_word=end,
                human_pct=human_pct, ai_pct=ai_pct, word_count=end - start,
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# ParagraphMapper
# ═══════════════════════════════════════════════════════════════════════════

class ParagraphMapper:

    @staticmethod
    def map_to_paragraphs(
        paragraphs: List[Tuple[str, int, int]],
        window_results: List[WindowResult],
    ) -> List[ParagraphScore]:
        scores: List[ParagraphScore] = []

        for p_idx, (p_text, p_start, p_end) in enumerate(paragraphs):
            overlapping_ai: List[float] = []
            overlapping_human: List[float] = []

            for wr in window_results:
                overlap_start = max(wr.start_word, p_start)
                overlap_end = min(wr.end_word, p_end)
                if overlap_end > overlap_start:
                    overlap_words = overlap_end - overlap_start
                    para_words = max(p_end - p_start, 1)
                    weight = overlap_words / para_words
                    overlapping_ai.append(wr.ai_pct * weight)
                    overlapping_human.append(wr.human_pct * weight)

            if overlapping_ai:
                total_ai = sum(overlapping_ai)
                total_human = sum(overlapping_human)
                denom = total_ai + total_human
                ai_avg = (total_ai / denom * 100) if denom > 0 else 50.0
                human_avg = 100.0 - ai_avg
            else:
                ai_avg = 50.0
                human_avg = 50.0

            zone = (
                "AI" if ai_avg >= THRESHOLD_AI
                else "HUMAN" if ai_avg < THRESHOLD_UNCERTAIN
                else "UNCERTAIN"
            )

            scores.append(ParagraphScore(
                paragraph_index=p_idx, text=p_text,
                start_word=p_start, end_word=p_end,
                word_count=p_end - p_start,
                ai_score=round(ai_avg, 2), human_score=round(human_avg, 2),
                zone=zone, contributing_windows=len(overlapping_ai),
            ))

        return scores


# ═══════════════════════════════════════════════════════════════════════════
# BreakpointDetector
# ═══════════════════════════════════════════════════════════════════════════

class BreakpointDetector:

    @staticmethod
    def detect(paragraph_scores: List[ParagraphScore]) -> List[Breakpoint]:
        breakpoints: List[Breakpoint] = []
        for i in range(len(paragraph_scores) - 1):
            curr = paragraph_scores[i]
            nxt = paragraph_scores[i + 1]
            delta = abs(nxt.ai_score - curr.ai_score)
            if delta >= BREAKPOINT_DELTA:
                breakpoints.append(Breakpoint(
                    after_paragraph=i, delta=delta,
                    from_zone=curr.zone, to_zone=nxt.zone,
                ))
        return breakpoints


# ═══════════════════════════════════════════════════════════════════════════
# HybridRiskClassifier
# ═══════════════════════════════════════════════════════════════════════════

class HybridRiskClassifier:

    @staticmethod
    def classify(
        paragraph_scores: List[ParagraphScore],
        breakpoints: List[Breakpoint],
    ) -> Tuple[str, str, str]:
        if not paragraph_scores:
            return ("INCONCLUSIVE", "LOW", "Insufficient text for segment analysis.")

        total_words = sum(p.word_count for p in paragraph_scores)
        ai_words = sum(p.word_count for p in paragraph_scores if p.zone == "AI")
        human_words = sum(p.word_count for p in paragraph_scores if p.zone == "HUMAN")
        uncertain_words = sum(p.word_count for p in paragraph_scores if p.zone == "UNCERTAIN")

        ai_ratio = ai_words / max(total_words, 1)
        human_ratio = human_words / max(total_words, 1)
        n_bp = len(breakpoints)

        if ai_ratio >= 0.85:
            return (
                "FULLY AI-GENERATED", "HIGH",
                f"Virtually all text ({ai_ratio:.0%} by word count) shows AI-generated "
                f"characteristics across {len(paragraph_scores)} paragraphs. "
                f"No significant human-authored segments detected."
            )

        if human_ratio >= 0.85:
            return (
                "FULLY HUMAN-WRITTEN", "LOW",
                f"The text is predominantly human-authored ({human_ratio:.0%} by word count). "
                f"No significant AI-generated segments detected across "
                f"{len(paragraph_scores)} paragraphs."
            )

        if ai_ratio >= 0.25 and human_ratio >= 0.25 and n_bp >= 1:
            return (
                "HYBRID \u2014 MIXED AUTHORSHIP", "HIGH",
                f"Clear mixed authorship detected: {ai_ratio:.0%} AI-generated, "
                f"{human_ratio:.0%} human-written, "
                f"{uncertain_words / max(total_words, 1):.0%} uncertain. "
                f"{n_bp} authorship transition{'s' if n_bp > 1 else ''} identified. "
                f"The text contains distinct AI and human segments."
            )

        if ai_ratio >= 0.15 and n_bp >= 1:
            return (
                "LIKELY AI-ASSISTED", "MEDIUM",
                f"Partial AI involvement detected: {ai_ratio:.0%} of the text shows "
                f"AI characteristics with {n_bp} transition point{'s' if n_bp > 1 else ''}. "
                f"AI was likely used to draft or expand specific sections."
            )

        if ai_ratio >= 0.05:
            return (
                "LIKELY HUMAN WITH AI ELEMENTS", "LOW",
                f"Predominantly human-authored with minor AI indicators ({ai_ratio:.0%}). "
                f"Could reflect light AI editing or natural overlap with AI-typical patterns."
            )

        return (
            "FULLY HUMAN-WRITTEN", "LOW",
            f"No meaningful AI-generated segments detected. All "
            f"{len(paragraph_scores)} paragraphs classified as human-authored."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature Vector Builder (10-dimensional)
# ═══════════════════════════════════════════════════════════════════════════

def _build_feature_vector(
    paragraph_scores: List[ParagraphScore],
    breakpoints: List[Breakpoint],
) -> Dict[str, float]:
    if not paragraph_scores:
        return {k: 0.0 for k in [
            "segment_count", "global_ai_score", "ai_segment_ratio",
            "human_segment_ratio", "uncertain_segment_ratio",
            "max_ai_score", "min_ai_score", "score_std",
            "breakpoint_count", "longest_ai_run",
        ]}

    total_words = sum(p.word_count for p in paragraph_scores)
    ai_words = sum(p.word_count for p in paragraph_scores if p.zone == "AI")
    human_words = sum(p.word_count for p in paragraph_scores if p.zone == "HUMAN")
    uncertain_words = sum(p.word_count for p in paragraph_scores if p.zone == "UNCERTAIN")

    ai_scores = [p.ai_score for p in paragraph_scores]

    longest_ai = 0
    current_run = 0
    for p in paragraph_scores:
        if p.zone == "AI":
            current_run += 1
            longest_ai = max(longest_ai, current_run)
        else:
            current_run = 0

    return {
        "segment_count": float(len(paragraph_scores)),
        "global_ai_score": sum(p.ai_score * p.word_count for p in paragraph_scores) / max(total_words, 1),
        "ai_segment_ratio": ai_words / max(total_words, 1),
        "human_segment_ratio": human_words / max(total_words, 1),
        "uncertain_segment_ratio": uncertain_words / max(total_words, 1),
        "max_ai_score": max(ai_scores),
        "min_ai_score": min(ai_scores),
        "score_std": statistics.stdev(ai_scores) if len(ai_scores) >= 2 else 0.0,
        "breakpoint_count": float(len(breakpoints)),
        "longest_ai_run": float(longest_ai),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HybridSegmentAnalyzer — Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

class HybridSegmentAnalyzer:
    """
    Main analyser. Requires a classify_fn callable:
        classify_fn(text: str) -> Tuple[float, float]  # (human%, ai%)

    In production, use detector_final.classify_segment as classify_fn.

    Usage
    -----
        from hybrid_segment_detector import HybridSegmentAnalyzer
        from detector_final import classify_segment

        analyzer = HybridSegmentAnalyzer(classify_fn=classify_segment)
        result   = analyzer.analyze("Full text here...")
        heatmap  = result.to_dict()
    """

    def __init__(self, classify_fn: Callable[[str], Tuple[float, float]]) -> None:
        self._segmenter = TextSegmenter()
        self._classifier = WindowClassifier(classify_fn)
        self._mapper = ParagraphMapper()
        self._bp_detector = BreakpointDetector()
        self._risk_classifier = HybridRiskClassifier()

    def analyze(self, text: str) -> HybridAnalysisResult:
        if not text or not text.strip():
            return HybridAnalysisResult(
                paragraph_scores=[], window_results=[], breakpoints=[],
                global_ai_score=0.0, classification="INCONCLUSIVE",
                risk_level="LOW",
                interpretation="No text provided for segment analysis.",
                feature_vector=_build_feature_vector([], []),
            )

        # [FIX v3.9] Strip bibliography before segmenting — avoids counting
        # APA references as AI paragraphs (they look like AI to ModernBERT).
        body_text = _strip_references_section(text)
        paragraphs = self._segmenter.split_paragraphs(body_text)
        words = body_text.split()
        total_words = len(words)

        logger.info("HybridSegment: %d words, %d paragraphs",
                     total_words, len(paragraphs))

        windows = self._segmenter.build_windows(total_words)
        logger.info("HybridSegment: %d windows (size=%d, overlap=%.0f%%)",
                     len(windows), WINDOW_WORDS, WINDOW_OVERLAP * 100)

        window_results = self._classifier.classify_windows(words, windows)
        paragraph_scores = self._mapper.map_to_paragraphs(paragraphs, window_results)
        breakpoints = self._bp_detector.detect(paragraph_scores)

        classification, risk_level, interpretation = (
            self._risk_classifier.classify(paragraph_scores, breakpoints)
        )

        feature_vector = _build_feature_vector(paragraph_scores, breakpoints)

        if total_words > 0:
            global_ai = sum(
                p.ai_score * p.word_count for p in paragraph_scores
            ) / total_words
        else:
            global_ai = 0.0

        result = HybridAnalysisResult(
            paragraph_scores=paragraph_scores,
            window_results=window_results,
            breakpoints=breakpoints,
            global_ai_score=round(global_ai, 4),
            classification=classification,
            risk_level=risk_level,
            interpretation=interpretation,
            feature_vector=feature_vector,
            total_words=total_words,
            total_paragraphs=len(paragraphs),
            total_windows=len(windows),
        )

        logger.info(
            "HybridSegment: classification=%s risk=%s ai=%.1f%% "
            "breakpoints=%d paragraphs=%d windows=%d",
            classification, risk_level, global_ai,
            len(breakpoints), len(paragraphs), len(windows),
        )

        return result
