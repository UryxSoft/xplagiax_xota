"""
discourse_analyzer.py — Argumentative / rhetorical structure uniformity.
========================================================================

WHAT IT MEASURES (and why it is model-agnostic)
-----------------------------------------------
LLM prose tends to be *templated*: evenly-sized paragraphs, heavy use of formal transition
connectives ("however", "moreover", "in conclusion"), explicit enumeration scaffolding
("Firstly… Secondly… Finally…"), repetitive sentence openings, and a tidy summarising
closer. These structural habits are imposed by instruction-tuning / RLHF and are therefore
LARGELY INDEPENDENT of which model produced the text — and, crucially, they SURVIVE
paraphrasing and "humanizer" tools (which rewrite words but preserve structure). That makes
discourse uniformity one of the few signals that gives lift against frontier models we never
trained on.

OUTPUT — a `uniformity` score ∈ [0,1] where HIGHER = more templated/LLM-like. This is a
soft, UNCALIBRATED structural prior, NOT a verdict: careful human technical/academic writing
is also structured, so it must be combined with other signals (it feeds the late-fusion
vector with a bounded weight). Every sub-feature is reported with the matched markers so a
human reviewer can see exactly WHY the score is what it is.

Pure-Python, deterministic, O(n) in tokens — no model load, no network. English-centric
marker lexicons (matches the rest of the English-trained pipeline).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# ── Discourse-marker lexicons ────────────────────────────────────────────────
_TRANSITION_MARKERS = (
    "however", "moreover", "furthermore", "additionally", "consequently",
    "therefore", "thus", "nevertheless", "nonetheless", "in addition",
    "on the other hand", "in contrast", "for instance", "for example",
    "as a result", "in particular", "notably", "importantly", "subsequently",
)
_ENUMERATION_MARKERS = (
    "firstly", "secondly", "thirdly", "fourthly", "finally", "lastly",
    "first of all", "to begin with", "next", "then",
)
_CONCLUSION_MARKERS = (
    "in conclusion", "in summary", "to summarize", "to summarise",
    "overall", "in essence", "ultimately", "to conclude", "all in all",
)

# Heuristic saturation scales — value/scale clipped to 1.0 (declared uncalibrated).
_TRANSITION_SCALE = 0.45     # transitions per sentence at which the feature saturates
_ENUMERATION_SCALE = 3.0     # number of ordinal scaffolding hits to saturate
_NUMBERED_LIST_SCALE = 4.0   # number of "1." / "- " list lead-ins to saturate

# Sub-feature weights for the aggregate uniformity (sum = 1.0).
_WEIGHTS = {
    "connective_density":  0.28,
    "paragraph_uniformity": 0.22,
    "enumeration_scaffold": 0.20,
    "opening_repetition":  0.18,
    "conclusion_marker":   0.12,
}

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _count_markers(low: str, markers) -> List[str]:
    """Return the list of markers (with multiplicity) found in lowercased text."""
    hits: List[str] = []
    for m in markers:
        # word-boundary match for single words; substring for multi-word phrases
        if " " in m:
            hits += [m] * low.count(m)
        else:
            hits += [m] * len(re.findall(rf"\b{re.escape(m)}\b", low))
    return hits


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)


class DiscourseAnalyzer:
    """Stateless analyzer — safe to share across threads/workers."""

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"status": "error", "error": "empty text", "uniformity": 0.0}

        low = text.lower()
        sents = _sentences(text)
        paras = _paragraphs(text)
        n_sent = max(len(sents), 1)

        if len(sents) < 4:
            return {
                "status": "inconclusive",
                "reason": f"Need ≥4 sentences for discourse analysis (got {len(sents)}).",
                "sentence_count": len(sents),
                "uniformity": 0.0,
            }

        # 1) Connective density — formal transitions per sentence.
        trans_hits = _count_markers(low, _TRANSITION_MARKERS)
        connective_density = _clip01((len(trans_hits) / n_sent) / _TRANSITION_SCALE)

        # 2) Paragraph uniformity — low coefficient-of-variation of paragraph word counts.
        if len(paras) >= 3:
            lengths = [len(p.split()) for p in paras]
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                var = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
                cv = (var ** 0.5) / mean_len
                paragraph_uniformity = _clip01(1.0 - cv)   # cv→0 means perfectly even paras
            else:
                paragraph_uniformity = 0.0
        else:
            paragraph_uniformity = 0.0  # too few paragraphs to judge

        # 3) Enumeration scaffolding — ordinal words + numbered/bulleted list lead-ins.
        enum_hits = _count_markers(low, _ENUMERATION_MARKERS)
        numbered = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+", text))
        enumeration_scaffold = _clip01(
            0.6 * (len(enum_hits) / _ENUMERATION_SCALE)
            + 0.4 * (numbered / _NUMBERED_LIST_SCALE)
        )

        # 4) Opening repetition — fraction of repeated sentence-opening bigrams.
        openings: List[str] = []
        for s in sents:
            toks = _WORD_RE.findall(s.lower())
            if toks:
                openings.append(" ".join(toks[:2]) if len(toks) >= 2 else toks[0])
        if openings:
            uniq = len(set(openings))
            opening_repetition = _clip01(1.0 - uniq / len(openings))
        else:
            opening_repetition = 0.0

        # 5) Conclusion marker — explicit summarising closer present.
        concl_hits = _count_markers(low, _CONCLUSION_MARKERS)
        conclusion_marker = 1.0 if concl_hits else 0.0

        features = {
            "connective_density":  round(connective_density, 4),
            "paragraph_uniformity": round(paragraph_uniformity, 4),
            "enumeration_scaffold": round(enumeration_scaffold, 4),
            "opening_repetition":  round(opening_repetition, 4),
            "conclusion_marker":   round(conclusion_marker, 4),
        }
        uniformity = sum(_WEIGHTS[k] * v for k, v in features.items())
        uniformity = round(_clip01(uniformity), 4)

        if uniformity >= 0.55:
            level = "HIGH — strongly templated structure"
            interpretation = (
                "The argumentative structure is highly templated (even paragraphs, heavy "
                "connectives, enumeration/closing scaffolding). This pattern is common in "
                "LLM output, but disciplined human academic/technical writing can also score "
                "here — treat as a structural prior, not proof."
            )
        elif uniformity >= 0.3:
            level = "MODERATE — some structural regularity"
            interpretation = (
                "Moderate structural regularity — within the range of organised human writing."
            )
        else:
            level = "LOW — irregular/organic structure"
            interpretation = (
                "Irregular, organic discourse structure — more typical of spontaneous human "
                "writing than of templated generation."
            )

        return {
            "status": "ok",
            "uniformity": uniformity,
            "level": level,
            "interpretation": interpretation,
            "features": features,
            "evidence": {
                "transition_markers": sorted(set(trans_hits))[:10],
                "enumeration_markers": sorted(set(enum_hits))[:10],
                "conclusion_markers": sorted(set(concl_hits))[:5],
                "numbered_list_items": numbered,
            },
            "sentence_count": len(sents),
            "paragraph_count": len(paras),
        }
