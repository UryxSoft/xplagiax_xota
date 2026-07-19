"""
semantic_consistency.py — Internal contradiction detection.
===========================================================

WHAT IT MEASURES (and why it is model-agnostic)
-----------------------------------------------
A document that contradicts ITSELF (asserts X in one place and ¬X in another, or gives two
different numbers for the same quantity) is exhibiting a coherence failure. LLMs — even
frontier ones — produce these self-contradictions far more often than careful humans,
because they generate locally-fluent text without a global truth model. A contradiction is
a contradiction regardless of which model wrote it, so this signal does NOT depend on the
2023 training distribution and gives lift against unseen models.

It is, however, EVIDENCE OF INCOHERENCE rather than a direct "AI" label (humans contradict
themselves too, e.g. across edits), so it feeds the late-fusion vector with a bounded weight
and is reported with the exact contradicting sentence pairs for human verification.

TWO TIERS
---------
• Heuristic (default, dependency-free): flags sentence pairs that talk about the SAME thing
  (high content-word overlap) but disagree — a negation-polarity flip, or the same subject
  with a different number. Conservative thresholds keep false positives low.
• NLI (optional, set SEMANTIC_NLI=1): lazily loads a cross-encoder NLI model and scores
  pairs for entailment/contradiction. Off by default to avoid surprise model downloads.

Output — `contradiction_ratio` ∈ [0,1] (contradictory pairs / sentences) + the pairs.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Cap sentences compared so the O(n²) pairing stays bounded on long documents.
_MAX_SENTENCES = 120
_MIN_SENTENCES = 4

# Heuristic thresholds (declared uncalibrated).
_OVERLAP_MIN = 0.5          # Jaccard of content words for "same topic"
_MIN_CONTENT_WORDS = 4      # ignore very short sentences (unreliable overlap)

# [Fase-2 M-18] Per-language negation cues (en/es/fr/pt); unsupported → en.
_NEGATION_CUES_BY_LANG = {
    "en": (
        "not", "no", "never", "cannot", "n't", "without", "none", "nobody",
        "nothing", "neither", "nor", "fails", "fail", "lacks", "lack", "absent",
        "unable", "impossible", "false", "incorrect",
    ),
    "es": (
        "no", "nunca", "jamás", "sin", "ningún", "ninguna", "ninguno", "nadie",
        "nada", "tampoco", "incapaz", "imposible", "falso", "falsa", "incorrecto",
        "incorrecta", "carece", "carecen", "ausente",
    ),
    "fr": (
        "ne", "pas", "jamais", "sans", "aucun", "aucune", "personne", "rien",
        "non", "ni", "incapable", "impossible", "faux", "fausse", "incorrect",
        "incorrecte", "absent", "absente",
    ),
    "pt": (
        "não", "nunca", "jamais", "sem", "nenhum", "nenhuma", "ninguém", "nada",
        "tampouco", "incapaz", "impossível", "falso", "falsa", "incorreto",
        "incorreta", "carece", "carecem", "ausente",
    ),
}
_NEGATION_CUES = _NEGATION_CUES_BY_LANG["en"]  # back-compat default

_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "from", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their", "his", "her", "they",
    "we", "you", "he", "she", "i", "which", "who", "whom", "whose", "what", "when",
    "where", "how", "than", "then", "so", "such", "also", "can", "could", "would",
    "should", "may", "might", "will", "shall", "do", "does", "did", "has", "have",
    "had", "there", "here", "more", "most", "some", "any", "all", "into", "about",
}

# [Fase-2 M-18] Accent-aware word regex so es/fr/pt content words are captured.
_WORD_RE = re.compile(r"[a-záéíóúüñàâçèêëîïôùûœãõ']+|\d+(?:\.\d+)?", re.IGNORECASE | re.UNICODE)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _lang_stopwords(lang: str) -> Set[str]:
    """Function-word set for content-word filtering; reuses lang_detect's lists."""
    if lang == "en":
        return _STOPWORDS
    try:
        from lang_detect import _STOPWORDS as _LD
        return set(_LD.get(lang, ())) or _STOPWORDS
    except Exception:
        return _STOPWORDS


def _content_words(sent: str, stopwords: Optional[Set[str]] = None) -> Set[str]:
    sw = stopwords if stopwords is not None else _STOPWORDS
    return {w for w in (t.lower() for t in _WORD_RE.findall(sent))
            if w not in sw and len(w) > 2}


def _has_negation(sent: str, cues=_NEGATION_CUES) -> bool:
    low = " " + sent.lower() + " "
    return any((cue if cue == "n't" else f" {cue} ") in (low if cue != "n't" else sent.lower())
               for cue in cues)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ── Optional NLI backend (lazy) ──────────────────────────────────────────────
_nli_pipe = None
_nli_tried = False


def _get_nli():
    global _nli_pipe, _nli_tried
    if _nli_tried:
        return _nli_pipe
    _nli_tried = True
    try:
        from transformers import pipeline
        model = os.getenv("SEMANTIC_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
        _nli_pipe = pipeline("text-classification", model=model, top_k=None)
        logger.info("Semantic NLI model loaded: %s", model)
    except Exception as exc:  # noqa: BLE001 — degrade to heuristic
        logger.warning("Semantic NLI unavailable, using heuristic: %s", exc)
        _nli_pipe = None
    return _nli_pipe


class SemanticConsistencyAnalyzer:
    """Stateless analyzer — safe to share across threads/workers."""

    def analyze(self, text: str) -> Dict[str, Any]:
        sents = _sentences(text)
        if len(sents) < _MIN_SENTENCES:
            return {
                "status": "inconclusive",
                "reason": f"Need ≥{_MIN_SENTENCES} sentences (got {len(sents)}).",
                "contradiction_ratio": 0.0,
            }
        sents = sents[:_MAX_SENTENCES]
        use_nli = os.getenv("SEMANTIC_NLI", "0") == "1" and _get_nli() is not None

        # [Fase-2 M-18] Language-aware stopwords + negation cues (en/es/fr/pt).
        try:
            from lang_detect import detect_language
            lang = detect_language(text).get("lang", "en")
        except Exception:
            lang = "en"
        cues = _NEGATION_CUES_BY_LANG.get(lang, _NEGATION_CUES_BY_LANG["en"])
        stopwords = _lang_stopwords(lang)

        words = [_content_words(s, stopwords) for s in sents]
        contradictions: List[Dict[str, Any]] = []

        for i in range(len(sents)):
            if len(words[i]) < _MIN_CONTENT_WORDS:
                continue
            for j in range(i + 1, len(sents)):
                if len(words[j]) < _MIN_CONTENT_WORDS:
                    continue
                overlap = _jaccard(words[i], words[j])
                if overlap < _OVERLAP_MIN:
                    continue
                verdict = self._pair_contradicts(sents[i], sents[j], words[i], words[j],
                                                 use_nli, cues)
                if verdict is not None:
                    reason, strong = verdict
                    contradictions.append({
                        "sentence_a": sents[i][:160],
                        "sentence_b": sents[j][:160],
                        "overlap": round(overlap, 3),
                        "reason": reason,
                        # [Fase-2 M-6] strong = numeric mismatch or NLI — specific enough
                        # to feed the fusion. Negation-polarity flips are report-only
                        # evidence (legitimate rhetorical contrast triggers them).
                        "strong": strong,
                    })

        # Deduplicate overlapping reports (keep at most a handful for readability).
        ratio = min(1.0, len(contradictions) / len(sents))
        strong_count = sum(1 for c in contradictions if c.get("strong"))
        strong_ratio = min(1.0, strong_count / len(sents))
        if contradictions:
            level = "CONTRADICTIONS FOUND"
            interpretation = (
                f"{len(contradictions)} internal contradiction(s) detected. Self-contradiction "
                f"is a coherence failure common in LLM output, but can also occur in human "
                f"drafts — review the listed pairs to judge."
            )
        else:
            level = "COHERENT"
            interpretation = "No internal contradictions detected by the current method."

        return {
            "status": "ok",
            "method": "nli" if use_nli else "heuristic",
            "language": lang,
            "contradiction_ratio": round(ratio, 4),
            "strong_contradiction_ratio": round(strong_ratio, 4),
            "strong_contradiction_count": strong_count,
            "contradiction_count": len(contradictions),
            "level": level,
            "interpretation": interpretation,
            "contradictions": contradictions[:10],
            "sentences_analyzed": len(sents),
        }

    # ── pair-level decision ──────────────────────────────────────────────────
    def _pair_contradicts(self, a: str, b: str, wa: Set[str], wb: Set[str],
                          use_nli: bool, cues=_NEGATION_CUES,
                          ) -> Optional[Tuple[str, bool]]:
        """Return (reason, strong) or None. strong=True only for numeric/NLI evidence."""
        if use_nli:
            label = self._nli_contradiction(a, b)
            if label is not None:
                return label, True
            # fall through to heuristic as a cheap second opinion
        # Heuristic 2 first: same subject, different number (specific → strong).
        nums_a = set(_NUM_RE.findall(a))
        nums_b = set(_NUM_RE.findall(b))
        if nums_a and nums_b and nums_a != nums_b and (wa & wb):
            shared = (wa & wb) - {n for n in nums_a | nums_b}
            if len(shared) >= _MIN_CONTENT_WORDS:
                return (f"numeric mismatch ({sorted(nums_a)} vs {sorted(nums_b)}) "
                        f"on shared subject"), True
        # Heuristic 1: negation-polarity flip on shared topic (noisy → weak, report-only).
        neg_a, neg_b = _has_negation(a, cues), _has_negation(b, cues)
        if neg_a != neg_b:
            return "negation polarity flip on shared content", False
        return None

    def _nli_contradiction(self, a: str, b: str) -> Optional[str]:
        pipe = _get_nli()
        if pipe is None:
            return None
        try:
            out = pipe({"text": a, "text_pair": b})
            scores = {d["label"].lower(): d["score"] for d in (out[0] if isinstance(out[0], list) else out)}
            c = scores.get("contradiction", 0.0)
            if c >= 0.6 and c >= max(scores.values()):
                return f"NLI contradiction (p={c:.2f})"
        except Exception as exc:  # noqa: BLE001
            logger.debug("NLI scoring failed: %s", exc)
        return None
