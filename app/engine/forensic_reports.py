"""
Forensic Report Generator v3.9
================================
Changelog (v3.5 -> v3.9):
  [FIX]  Verdict: Now displays both Human Score and AI Score instead of
         a single "Confidence" percentage that only showed one side.
  [FIX]  Key Evidence: Moved _collect_evidence() to AFTER hallucination,
         reasoning, and watermark analyses so all data sources are available.
         Lowered thresholds: sentence AI 0.85→0.60, uniform scores 0.7→0.5,
         stylometric burst 0.10→0.12, lex_div 0.35→0.40, hapax 0.25→0.30,
         stat deviation 0.40→0.30. Added human-supporting evidence (Source 1b)
         so evidence section is never empty. Added hallucination category
         evidence (Source 8) and moderate reasoning/hallucination evidence.
  [FIX]  Evidence cards now include "explanation" fields for all evidence
         types (hallucination, reasoning, watermark) with human-readable text.
  - All v3.5 logic preserved. ReasoningRiskClassifier unchanged.
"""

import base64, hashlib, io, json, logging, re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

_FORENSIC_DISCLAIMER = (
    "RESEARCH OUTPUT ONLY \u2014 Not suitable as primary or sole evidence in "
    "academic integrity proceedings, legal decisions, or employment actions. "
    "Results must be reviewed by a qualified human expert."
)


@dataclass
class WordAttribution:
    word: str; position: int; ai_score: float; confidence: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class SentenceAttribution:
    text: str; position: int; ai_score: float; confidence: float
    word_attributions: List[WordAttribution]; key_indicators: List[str]


@dataclass
class ForensicReport:
    report_id: str; generated_at: str; text_hash: str; word_count: int
    verdict: str; confidence: float; neural_score: float
    statistical_score: float; stylometric_score: float
    reasoning_score: float; watermark_score: float
    sentence_attributions: List[SentenceAttribution]
    human_baseline_comparison: Dict[str, Tuple[float, float]]
    evidence_points: List[Dict[str, Any]]
    hallucination_risk: Optional[Dict[str, Any]] = None
    reasoning_analysis: Optional[Dict[str, Any]] = None
    stylometric_stats: Optional[Dict[str, float]] = None   # [NEW v3.5]
    perplexity_analysis: Optional[Dict[str, Any]] = None   # [NEW v3.7]
    hybrid_analysis: Optional[Dict[str, Any]] = None      # [NEW v3.9]
    reference_analysis: Optional[Dict[str, Any]] = None   # [NEW v3.9]
    executive_summary: Optional[str] = None                 # [NEW v3.5]
    heatmap_b64: Optional[str] = None
    confidence_chart_b64: Optional[str] = None
    comparison_chart_b64: Optional[str] = None


#: [D-3] Display-only disclaimer for the word/sentence heatmaps. These attributions
#: are nudged by the hardcoded buzzword lexicon below, which is biased against formal,
#: academic, legal and non-native (ESL) writing. They are a localization aid ONLY and
#: do not contribute to the verdict (see generate_report). Reports should surface this.
ATTRIBUTION_DISCLAIMER = (
    "Heatmap por palabra/oración: ayuda visual heurística (no calibrada, sesgada "
    "contra escritura formal/académica/legal/ESL). No determina el veredicto."
)


class AttributionCalculator:
    """Per-word/sentence colouring for the heatmap. DISPLAY-ONLY (see ATTRIBUTION_DISCLAIMER)."""

    AI_INDICATOR_WORDS = {
        "furthermore": 0.8, "moreover": 0.8, "additionally": 0.7,
        "consequently": 0.85, "therefore": 0.7, "thus": 0.7,
        "delve": 0.95, "utilize": 0.7, "leverage": 0.75,
        "robust": 0.7, "comprehensive": 0.7, "innovative": 0.65,
        "streamline": 0.75, "optimize": 0.7, "enhance": 0.65,
        "facilitate": 0.75, "implement": 0.65, "integrate": 0.65,
        "whilst": 0.8, "hence": 0.75, "thereby": 0.8,
        "aforementioned": 0.85, "pertaining": 0.8, "regarding": 0.6,
    }
    HUMAN_INDICATOR_WORDS = {
        "kinda": 0.9, "gonna": 0.9, "wanna": 0.9, "like": 0.3,
        "basically": 0.4, "actually": 0.4, "honestly": 0.5,
        "literally": 0.5, "seriously": 0.5, "anyway": 0.6,
        "whatever": 0.7, "stuff": 0.6, "thing": 0.4, "things": 0.4,
        "guy": 0.6, "cool": 0.5, "awesome": 0.5, "crazy": 0.5,
    }

    # [FIX v3.9] Detect and strip references/bibliography section.
    # Searches from 30% through the text for the last occurrence of a
    # reference header. Everything after that header is removed before
    # sentence splitting and word attribution (avoids counting author
    # initials as separate sentences).
    _REF_HEADER_RE = re.compile(
        r"^\s*(?:References|Bibliography|Works Cited|Literature Cited"
        r"|Bibliograf[íi]a|Referencias)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    @classmethod
    def strip_references_section(cls, text: str) -> str:
        """Remove bibliography section from text for attribution analysis."""
        # Only search in the last 70% of the text
        search_start = len(text) * 3 // 10
        matches = list(cls._REF_HEADER_RE.finditer(text, search_start))
        if matches:
            cut_pos = matches[-1].start()
            stripped = text[:cut_pos].strip()
            if len(stripped.split()) >= 20:  # safety: keep at least 20 words
                return stripped
        return text

    def calculate_word_attributions(self, text, overall_ai_score):
        # [FIX v3.8] Strip zero-width Unicode chars that cause matplotlib
        # Glyph warnings (U+200B ZERO WIDTH SPACE, U+FEFF BOM, etc.)
        _ZW = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]')
        words = text.split()
        if not words: return []
        attrs = []
        for i, word in enumerate(words):
            display_word = _ZW.sub('', word)  # clean for rendering
            if not display_word:
                continue  # skip pure zero-width tokens
            cw = re.sub(r"[^\w]", "", word.lower())
            bs = overall_ai_score
            if cw in self.AI_INDICATOR_WORDS:
                ws = min(1.0, bs + self.AI_INDICATOR_WORDS[cw] * 0.3); conf = 0.8
            elif cw in self.HUMAN_INDICATOR_WORDS:
                ws = max(0.0, bs - self.HUMAN_INDICATOR_WORDS[cw] * 0.3); conf = 0.8
            else:
                ws = bs; conf = 0.5
            pf = 1.0 - (i / len(words)) * 0.1
            ws = float(np.clip(ws * pf, 0.0, 1.0))
            attrs.append(WordAttribution(word=display_word, position=i, ai_score=ws,
                confidence=conf, features={"ai_indicator": cw in self.AI_INDICATOR_WORDS,
                "human_indicator": cw in self.HUMAN_INDICATOR_WORDS}))
        return attrs

    def calculate_sentence_attributions(self, text, overall_ai_score):
        # [FIX v3.9] Strip references section before splitting sentences.
        # Prevents author initials ("D. R. E.") from creating 40+ fragments.
        clean_text = self.strip_references_section(text)

        # [FIX v3.9] Smarter sentence split: require period/!/? followed by
        # whitespace + uppercase letter or end of string. Avoids splitting on
        # abbreviations (D. R. E.), "et al.", "Dr.", "e.g.", "i.e." etc.
        # First protect common abbreviations
        protected = clean_text
        protected = re.sub(r'\b([A-Z])\.\s*(?=[A-Z]\.)', r'\1DOTPROTECT', protected)  # initials
        protected = re.sub(r'\bet\s+al\.', 'et alDOTPROTECT', protected)
        protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Jr|Sr|St|Inc|Ltd|Vol|No|pp|ed|eds)\.',
                           r'\1DOTPROTECT', protected, flags=re.IGNORECASE)
        protected = re.sub(r'\b(e\.g|i\.e|cf|vs|etc)\.',
                           lambda m: m.group(0).replace('.', 'DOTPROTECT'), protected,
                           flags=re.IGNORECASE)

        sentences = [s.strip().replace('DOTPROTECT', '.')
                     for s in re.split(r'[.!?]+(?:\s|$)', protected) if s.strip()]

        attrs = []
        for i, sent in enumerate(sentences):
            wa = self.calculate_word_attributions(sent, overall_ai_score)
            ss = float(np.mean([w.ai_score for w in wa])) if wa else overall_ai_score
            conf = float(np.mean([w.confidence for w in wa])) if wa else 0.5
            inds = []
            ls = sent.lower()
            for w, st in self.AI_INDICATOR_WORDS.items():
                if w in ls and st > 0.7: inds.append(f"AI indicator: '{w}'")
            if re.search(r"\b(first|second|third|finally)\b", ls): inds.append("Sequential structure")
            if re.search(r"\b(therefore|thus|hence|consequently)\b", ls): inds.append("Logical connector")
            attrs.append(SentenceAttribution(text=sent, position=i, ai_score=ss,
                confidence=conf, word_attributions=wa, key_indicators=inds))
        return attrs


class HeatmapGenerator:
    def __init__(self):
        self.cmap = LinearSegmentedColormap.from_list(
            "ai_detection", [(0, "#2ecc71"), (0.5, "#f1c40f"), (1, "#e74c3c")])

    def generate_word_heatmap(self, word_attributions, max_words_per_line=15, figsize=(14, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        words = [w.word for w in word_attributions]
        scores = [w.ai_score for w in word_attributions]
        x, y, lh = 0.02, 0.95, 0.06
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        for word, score in zip(words, scores):
            color = self.cmap(score)
            to = ax.text(x, y, word + " ", fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor="none"),
                verticalalignment="top")
            bb = to.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            x += bb.width + 0.01
            if x > 0.95:
                x = 0.02; y -= lh
                if y < 0.1: break
        ax.legend(handles=[
            mpatches.Patch(facecolor="#2ecc71", label="Human-like (0.0-0.3)"),
            mpatches.Patch(facecolor="#f1c40f", label="Uncertain (0.3-0.7)"),
            mpatches.Patch(facecolor="#e74c3c", label="AI-like (0.7-1.0)"),
        ], loc="lower center", ncol=3, frameon=False, fontsize=9)
        ax.set_title("Word-Level AI Attribution Heatmap [HEURISTIC]", fontsize=14, fontweight="bold")
        return fig

    def generate_sentence_chart(self, sentence_attributions, figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        positions = range(len(sentence_attributions))
        scores = [s.ai_score for s in sentence_attributions]
        ax.bar(positions, scores, color=[self.cmap(s) for s in scores], edgecolor="white", linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Decision threshold")
        ax.set_xlabel("Sentence Position", fontsize=11); ax.set_ylabel("AI Probability Score", fontsize=11)
        ax.set_title("Sentence-by-Sentence AI Detection Scores [HEURISTIC]", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1); ax.legend(loc="upper right"); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        return fig

    def generate_comparison_chart(self, metrics, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        labels = list(metrics.keys())
        tv = [metrics[l][0] for l in labels]; bv = [metrics[l][1] for l in labels]
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, tv, w, label="Analysed Text", color="#3498db", edgecolor="white")
        ax.bar(x + w/2, bv, w, label="Human Baseline", color="#2ecc71", edgecolor="white")
        ax.set_xlabel("Metric", fontsize=11); ax.set_ylabel("Value", fontsize=11)
        ax.set_title("Statistical Comparison vs Human Writing", fontsize=14, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.legend()
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True); plt.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════════════════════
# ReasoningRiskClassifier (unchanged from v3.4)
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningRiskClassifier:
    """
    Heuristic classifier for the 15-dim vector from ReasoningProfiler.
    Display/reporting layer — does NOT modify reasoning_profiler.py.
    """

    _WEIGHTS = {
        "backtracking_density":    0.26,
        "cot_scaffold_density":    0.23,
        "consequence_density":     0.09,
        "causal_density":          0.07,
        "sequence_density":        0.07,
        "contrast_density":        0.05,
        "word_entropy_normalised": 0.07,
        "type_token_ratio":        0.05,
        "paragraph_length_cv":     0.03,
    }
    _INVERSE_WEIGHT = 0.08

    # EC-04 / 4.2-Bias-2: causal/consequence/contrast/sequence thresholds raised to
    # reduce false positives on formal academic text, which is inherently dense in
    # logical connectors. backtracking + cot_scaffold thresholds are unchanged —
    # those markers are unique to reasoning models, not academic prose.
    _THR = {
        "type_token_ratio":         (0.35, 0.72),
        "mean_sentence_length":     (12.0, 26.0),
        "std_sentence_length":      (4.0,  14.0),
        "mean_word_length":         (3.8,  5.8),
        "punctuation_ratio":        (0.02, 0.06),
        "stopword_ratio":           (0.30, 0.52),
        "consequence_density":      (0.03, 0.10),  # was (0.02, 0.06)
        "causal_density":           (0.03, 0.12),  # was (0.02, 0.07)
        "contrast_density":         (0.03, 0.09),  # was (0.02, 0.06)
        "sequence_density":         (0.02, 0.09),  # was (0.01, 0.05)
        "backtracking_density":     (0.01, 0.07),  # unchanged — reasoning-model unique
        "cot_scaffold_density":     (0.02, 0.10),  # unchanged — reasoning-model unique
        "intuition_leap_density":   (0.01, 0.04),
        "paragraph_length_cv":      (0.18, 0.55),
        "word_entropy_normalised":  (0.68, 0.90),
    }

    _HIGH   = 0.55
    _MEDIUM = 0.28

    _EXPL = {
        "backtracking_density": {
            "display": "Self-Correction Density", "group": "CoT & Self-Correction",
            "high":   "Very high self-correction density (value: {v:.6f}). Phrases such as 'wait', 'let me reconsider', 'actually that is incorrect', 'I made an error' appear with significant frequency. Strongest single marker of a reasoning-optimised model (o1, o3, DeepSeek-R1, QwQ). Standard models rarely produce this at detectable density.",
            "medium": "Moderate self-correction language (value: {v:.6f}). Could indicate a reasoning model on a moderate task, a standard model with 'think step by step' instructions, or a careful human author revising mid-composition.",
            "low":    "Minimal self-correction language (value: {v:.6f}). Consistent with standard autoregressive models (GPT-4o, Claude 3.x, Gemini 1.5) or typical human prose.",
        },
        "cot_scaffold_density": {
            "display": "Chain-of-Thought Scaffolding Density", "group": "CoT & Self-Correction",
            "high":   "Dense CoT scaffolding (value: {v:.6f}). 'step by step', 'let me think', 'working through this', 'step N:', 'from this we can conclude'. Characteristic of models trained with extended thinking budgets.",
            "medium": "Moderate CoT scaffolding (value: {v:.6f}). May reflect a reasoning model, a prompted standard model, or a methodical human author.",
            "low":    "Negligible CoT scaffolding (value: {v:.6f}). Typical of conversational AI or informal human text.",
        },
        "consequence_density": {
            "display": "Logical Consequence Connector Density", "group": "Logical Connectors",
            "high":   "Dense logical consequence connectors (value: {v:.6f}). 'therefore', 'thus', 'consequently', 'hence', 'accordingly'. Signals deductive reasoning chains.",
            "medium": "Moderate consequence language (value: {v:.6f}).",
            "low":    "Sparse consequence connectors (value: {v:.6f}). Narrative or descriptive prose.",
        },
        "causal_density": {
            "display": "Causal Connector Density", "group": "Logical Connectors",
            "high":   "High causal language density (value: {v:.6f}). 'because', 'due to', 'since', 'owing to', 'given that'. Reasoning models produce dense causal chains when constructing derivations.",
            "medium": "Moderate causal language (value: {v:.6f}).",
            "low":    "Low causal density (value: {v:.6f}). Narrative style predominates.",
        },
        "contrast_density": {
            "display": "Contrast Connector Density", "group": "Logical Connectors",
            "high":   "High contrast language (value: {v:.6f}). 'however', 'nevertheless', 'despite', 'although'. Signals dialectical reasoning.",
            "medium": "Moderate contrastive language (value: {v:.6f}).",
            "low":    "Sparse contrast markers (value: {v:.6f}). Monological style.",
        },
        "sequence_density": {
            "display": "Sequential Structure Density", "group": "Logical Connectors",
            "high":   "Heavy sequential framing (value: {v:.6f}). 'first', 'second', 'third', 'finally', 'subsequently'. Strongly characteristic of step-by-step reasoning model output.",
            "medium": "Moderate sequential structure (value: {v:.6f}).",
            "low":    "Non-sequential prose (value: {v:.6f}). No explicit step enumeration.",
        },
        "intuition_leap_density": {
            "display": "Intuitive Assertion Density [INVERSE SIGNAL]", "group": "Style Markers",
            "high":   "Frequent intuitive assertions (value: {v:.6f}). 'obviously', 'clearly', 'of course', 'naturally'. INVERSE signal: high density here is more consistent with human writing or standard AI — reasoning models prefer explicit derivation.",
            "medium": "Moderate intuitive language (value: {v:.6f}). Does not strongly indicate or rule out a reasoning model.",
            "low":    "Minimal intuitive leaps (value: {v:.6f}). Expected profile for reasoning models (o1, DeepSeek-R1, QwQ) that prefer explicit derivation over bare assertion.",
        },
        "type_token_ratio": {
            "display": "Vocabulary Diversity (TTR = |V|/N)", "group": "Lexical Quality",
            "high": "High lexical diversity TTR={v:.4f}. Rich, varied vocabulary.",
            "medium": "Moderate vocabulary diversity TTR={v:.4f}.",
            "low": "Low lexical diversity TTR={v:.4f}. Vocabulary repetition detected.",
        },
        "word_entropy_normalised": {
            "display": "Normalised Word Entropy H(words)/log\u2082(|V|)", "group": "Lexical Quality",
            "high":   "High normalised word entropy H_norm={v:.4f}. Word distribution spread broadly — rich, varied text.",
            "medium": "Moderate word entropy H_norm={v:.4f}.",
            "low":    "Low normalised entropy H_norm={v:.4f}. Concentrated, repetitive word distribution.",
        },
        "paragraph_length_cv": {
            "display": "Paragraph Length CV (\u03c3/\u03bc)", "group": "Structural Variety",
            "high":   "High paragraph length variability CV={v:.4f}. Reasoning models often produce structurally heterogeneous paragraphs — brief assertions alternating with extended derivations.",
            "medium": "Moderate paragraph length variation CV={v:.4f}.",
            "low":    "Highly uniform paragraph lengths CV={v:.4f}. Common in templated AI output.",
        },
        "mean_sentence_length": {
            "display": "Mean Sentence Length (words/sentence)", "group": "Stylometric",
            "high": "Long mean sentence length \u03bc={v:.1f}. Complex, multi-clause constructions.",
            "medium": "Moderate sentence length \u03bc={v:.1f}.",
            "low": "Short mean sentence length \u03bc={v:.1f}. Terse, direct prose.",
        },
        "std_sentence_length": {
            "display": "Sentence Length Std. Deviation \u03c3", "group": "Stylometric",
            "high": "High sentence length variance \u03c3={v:.1f}.",
            "medium": "Moderate sentence length variation \u03c3={v:.1f}.",
            "low": "Uniform sentence lengths \u03c3={v:.1f}. Highly regular pattern.",
        },
        "mean_word_length": {
            "display": "Mean Word Length (chars/token)", "group": "Stylometric",
            "high": "Long average word length \u03bc={v:.2f}. Dense technical vocabulary.",
            "medium": "Moderate word length \u03bc={v:.2f}.",
            "low": "Short average word length \u03bc={v:.2f}. Informal register.",
        },
        "punctuation_ratio": {
            "display": "Punctuation Density (punct/chars)", "group": "Stylometric",
            "high": "High punctuation density r={v:.4f}. Complex sentence structure.",
            "medium": "Moderate punctuation r={v:.4f}.",
            "low": "Sparse punctuation r={v:.4f}. Linear sentence structure.",
        },
        "stopword_ratio": {
            "display": "Stopword Ratio (stopwords/tokens)", "group": "Stylometric",
            "high": "High stopword density r={v:.4f}. Functional language dominates.",
            "medium": "Moderate stopword ratio r={v:.4f}.",
            "low": "Low stopword density r={v:.4f}. Content-dense, technical writing.",
        },
    }

    def classify(self, vec, feature_names):
        features = dict(zip(feature_names, vec.tolist()))
        score  = self._score(features)
        level  = self._level(score)
        return {
            "ai_score":        score,
            "risk_level":      level,
            "feature_details": self._feature_details(features),
            "group_scores":    self._group_scores(features),
            "top_signals":     self._top_signals(features),
            "interpretation":  self._interpretation(score, features),
        }

    def _norm(self, feat, val):
        thr = self._THR.get(feat, (0.0, 1.0))
        return min(1.0, val / max(thr[1], 1e-9))

    def _score(self, features):
        s = sum(w * self._norm(f, features.get(f, 0.0)) for f, w in self._WEIGHTS.items())
        inv = self._norm("intuition_leap_density", features.get("intuition_leap_density", 0.0))
        s += self._INVERSE_WEIGHT * max(0.0, 1.0 - inv)
        return round(min(1.0, max(0.0, s)), 4)

    def _level(self, score):
        if score >= self._HIGH:   return "HIGH \u2014 Reasoning Model"
        if score >= self._MEDIUM: return "MEDIUM \u2014 Possible Reasoning Model"
        return "LOW \u2014 Standard Model or Human"

    def _feat_level(self, feat, val):
        thr = self._THR.get(feat, (0.0, 1.0))
        if val >= thr[1]: return "high"
        if val >= thr[0]: return "medium"
        return "low"

    def _feature_details(self, features):
        details = {}
        for feat, val in features.items():
            em = self._EXPL.get(feat)
            if em is None: continue
            thr = self._THR.get(feat, (0.0, 1.0))
            lev = self._feat_level(feat, val)
            et = em.get(lev, "")
            details[feat] = {
                "display_name": em["display"], "group": em.get("group", "Other"),
                "value": round(val, 6), "level": lev,
                "explanation": et.format(v=val) if "{v" in et else et,
                "threshold_low": thr[0], "threshold_high": thr[1],
            }
        return details

    def _top_signals(self, features, k=5):
        scored = []
        for feat, w in self._WEIGHTS.items():
            val = features.get(feat, 0.0); norm = self._norm(feat, val)
            lev = self._feat_level(feat, val); em = self._EXPL.get(feat, {})
            et = em.get(lev, "")
            scored.append({"feature": feat, "display_name": em.get("display", feat),
                "group": em.get("group", ""), "raw_value": round(val, 6),
                "normalised": round(norm, 4), "weight": w, "level": lev,
                "explanation": (et.format(v=val) if "{v" in et else et)[:280]})
        iv = features.get("intuition_leap_density", 0.0)
        norm = self._norm("intuition_leap_density", iv); lev = self._feat_level("intuition_leap_density", iv)
        ie = self._EXPL.get("intuition_leap_density", {}); et = ie.get(lev, "")
        scored.append({"feature": "intuition_leap_density",
            "display_name": ie.get("display", "Intuitive Assertion Density"),
            "group": ie.get("group", "Style Markers"), "raw_value": round(iv, 6),
            "normalised": round(norm, 4), "weight": self._INVERSE_WEIGHT, "level": lev,
            "explanation": (et.format(v=iv) if "{v" in et else et)[:280]})
        scored.sort(key=lambda x: x["normalised"], reverse=True)
        return scored[:k]

    def _group_scores(self, features):
        n = lambda f: self._norm(f, features.get(f, 0.0))
        return {
            "CoT & Self-Correction": round(n("backtracking_density")*0.55 + n("cot_scaffold_density")*0.45, 4),
            "Logical Connectors":   round(n("consequence_density")*0.30 + n("causal_density")*0.25 + n("contrast_density")*0.20 + n("sequence_density")*0.25, 4),
            "Lexical Richness":     round(n("type_token_ratio")*0.50 + n("word_entropy_normalised")*0.50, 4),
            "Structural Variety":   round(n("paragraph_length_cv"), 4),
            "Intuitive Assertions (inverse)": round(max(0.0, 1.0 - n("intuition_leap_density")), 4),
        }

    def _interpretation(self, score, features):
        bt  = features.get("backtracking_density", 0.0)
        cot = features.get("cot_scaffold_density", 0.0)
        seq = features.get("sequence_density", 0.0)
        con = features.get("consequence_density", 0.0)
        ent = features.get("word_entropy_normalised", 0.0)
        inv = features.get("intuition_leap_density", 0.0)
        if score >= self._HIGH:
            parts = []
            if bt  >= self._THR["backtracking_density"][1]:  parts.append(f"self-correction (density={bt:.4f})")
            if cot >= self._THR["cot_scaffold_density"][1]:  parts.append(f"CoT scaffolding (density={cot:.4f})")
            if seq >= self._THR["sequence_density"][1]:       parts.append(f"step enumeration (density={seq:.4f})")
            sig = "; ".join(parts) if parts else f"combined score={score:.2f}"
            return (f"Strong reasoning-model signature (overall score={score:.2f}). Dominant signals: {sig}. "
                    f"Characteristic of o1, o3-mini, DeepSeek-R1, QwQ — trained via process reward models or "
                    f"MCTS-style search for explicit multi-step deliberation.")
        if score >= self._MEDIUM:
            return (f"Moderate reasoning-model indicators (overall score={score:.2f}). "
                    f"CoT scaffolding={cot:.4f}, consequence connectors={con:.4f}, word entropy={ent:.4f}. "
                    f"Compatible with a reasoning-capable model, a standard model with step-by-step "
                    f"system-prompt instructions, or a methodical human author.")
        return (f"Low reasoning-model indicators (overall score={score:.2f}). "
                f"Self-correction={bt:.4f}, CoT scaffolding={cot:.4f}, intuitive assertions={inv:.4f}. "
                f"Consistent with a standard autoregressive model without extended chain-of-thought "
                f"inference, or with natural human prose.")


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Hallucination category explanation map
# ═══════════════════════════════════════════════════════════════════════════

_HAL_CATEGORY_EXPLANATIONS = {
    "lexical_risk": {
        "name": "Language Confidence Patterns",
        "desc": "Measures hedging phrases, overconfident assertions, and negation patterns.",
        "high": "The text contains many hedging or overconfident phrases that are typical of AI-generated content trying to appear authoritative while being uncertain.",
        "medium": "Some hedging or overconfidence markers detected. This level is common in both careful human writing and AI output.",
        "low": "Very few problematic lexical patterns. The language confidence level appears natural.",
    },
    "entity_anomaly": {
        "name": "Named References Check",
        "desc": "Checks for unusual patterns in how people, places, and organizations are mentioned.",
        "high": "Unusual patterns in how names and organizations are mentioned — such as names without context or entities that don't typically co-occur. This is a common sign of fabricated details.",
        "medium": "Some mild irregularities in how names are used. May reflect a dense-information writing style.",
        "low": "Entity usage appears natural and consistent throughout the text.",
    },
    "entropy": {
        "name": "Word Pattern Regularity",
        "desc": "Measures the randomness and predictability of word usage patterns.",
        "high": "The word distribution is highly unusual — either too predictable (low entropy) or too chaotic (high entropy), which can indicate machine-generated filler content.",
        "medium": "Word patterns are in a moderate range. Neither strongly indicative of human or AI authorship.",
        "low": "Word distribution patterns are within normal human writing range.",
    },
    "semantic_incoherence": {
        "name": "Logical Flow Between Sentences",
        "desc": "Evaluates whether sentences logically connect to each other.",
        "high": "Several sentences appear disconnected from their neighbors. This is a common sign where AI generates plausible-sounding but logically unrelated statements.",
        "medium": "Some mild coherence gaps between sentences. Could reflect topic transitions or paragraph breaks.",
        "low": "Strong logical flow between sentences. Ideas connect naturally throughout the text.",
    },
    "vagueness": {
        "name": "Specificity of Claims",
        "desc": "Detects vague quantifiers and lack of specific details.",
        "high": "The text relies heavily on vague language ('many', 'some', 'various') instead of specific facts. AI models often use vague language to avoid committing to verifiable claims.",
        "medium": "Moderate use of vague language. This is common in introductory or summary-style writing.",
        "low": "The text is specific and detailed, with concrete facts and precise language.",
    },
    "repetition": {
        "name": "Repetitive Patterns",
        "desc": "Detects repeated phrases, self-referential patterns, and redundant mentions.",
        "high": "Significant repetition detected — phrases or sentence structures are being reused in ways that suggest automated generation.",
        "medium": "Some repetition present. May reflect emphasis in human writing or mild AI generation patterns.",
        "low": "Minimal repetition. The text uses varied phrasing and avoids redundancy.",
    },
    "structural_anomaly": {
        "name": "Sentence Structure Uniformity",
        "desc": "Checks sentence length uniformity, modal verb usage, and superlative frequency.",
        "high": "The text shows unusually uniform sentence structures or excessive use of certain word types — patterns associated with AI that follows a formulaic template.",
        "medium": "Some structural patterns detected but within acceptable ranges.",
        "low": "Natural structural variation in sentence construction and word choice.",
    },
    "imprecision": {
        "name": "Precision of Facts & Dates",
        "desc": "Evaluates the precision of numbers, dates, and temporal references.",
        "high": "The text avoids or misuses specific numbers and dates. AI often substitutes vague references for precise ones to avoid verifiable errors.",
        "medium": "Moderate level of precision. Some specific facts present alongside general statements.",
        "low": "The text includes precise numerical and temporal details, consistent with informed human writing.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Stylometric metric explanation map
# ═══════════════════════════════════════════════════════════════════════════

_STYLO_EXPLANATIONS = {
    "burstiness": {
        "name": "Sentence Length Variation",
        "desc": "How much sentence lengths vary throughout the text. High variation is typical of natural human writing (mixing short and long sentences), while low variation suggests machine-generated uniformity.",
        "thresholds": (0.15, 0.30),
    },
    "lexical_diversity": {
        "name": "Lexical Diversity",
        "desc": "How varied the vocabulary is. Higher values mean the author uses many different words; lower values suggest repetitive word choice, which is more common in AI output.",
        "thresholds": (0.40, 0.70),
    },
    "avg_sentence_length": {
        "name": "Average Sentence Length",
        "desc": "The average number of words per sentence. AI models often produce sentences that cluster around 15-20 words, while human writing shows more variation.",
        "thresholds": (12.0, 22.0),
    },
    "sentence_length_variance": {
        "name": "Sentence Length Variance",
        "desc": "The statistical spread in sentence lengths. Low variance means sentences are all about the same length (common in AI text); high variance means a natural mix of short and long sentences.",
        "thresholds": (20.0, 80.0),
    },
    "avg_word_length": {
        "name": "Average Word Length",
        "desc": "The average number of characters per word. Unusually high values may indicate dense technical jargon or AI tendency to use longer, more formal words.",
        "thresholds": (3.8, 5.5),
    },
    "vocabulary_richness": {
        "name": "Vocabulary Richness",
        "desc": "Ratio of unique words to total words. A richer vocabulary suggests more diverse language use, which is more typical of experienced human writers.",
        "thresholds": (0.40, 0.70),
    },
    "hapax_legomena_ratio": {
        "name": "Unique Word Ratio",
        "desc": "Proportion of words that appear only once. A higher ratio indicates more unique word choices — a strong human writing indicator. AI tends to reuse words more frequently.",
        "thresholds": (0.30, 0.60),
    },
    "rare_word_ratio": {
        "name": "Rare Word Ratio",
        "desc": "Proportion of uncommon words. Higher values suggest specialized vocabulary or creative writing; very low values may indicate AI's tendency toward common, 'safe' word choices.",
        "thresholds": (0.05, 0.20),
    },
    "comma_rate": {
        "name": "Comma Rate",
        "desc": "Frequency of comma usage. AI models sometimes overuse commas for list-like structures, while human writers show more varied punctuation patterns.",
        "thresholds": (0.02, 0.06),
    },
    "complex_sentence_ratio": {
        "name": "Complex Sentence Ratio",
        "desc": "Proportion of sentences with multiple clauses. Higher ratios indicate more complex sentence construction, which can be a sign of experienced human writing.",
        "thresholds": (0.20, 0.50),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Sentence explanation helper
# ═══════════════════════════════════════════════════════════════════════════

def _explain_sentence(sent_attr: SentenceAttribution) -> str:
    """Generate a plain-language explanation for why a sentence is suspicious."""
    score = sent_attr.ai_score
    inds  = sent_attr.key_indicators

    # Build a specific reason based on indicators
    reasons = []
    for ind in inds[:3]:
        if "AI indicator" in ind:
            word = ind.split("'")[1] if "'" in ind else "detected"
            reasons.append(f"uses the word '{word}', which is strongly associated with AI writing")
        elif "Sequential" in ind:
            reasons.append("uses sequential structure markers (first, second, finally) common in AI output")
        elif "Logical connector" in ind:
            reasons.append("uses formal logical connectors (therefore, thus) at AI-typical density")
        else:
            reasons.append(ind.lower())

    if not reasons:
        if score >= 0.85:
            reasons.append("the overall phrasing and word choices closely match AI-generated patterns")
        elif score >= 0.65:
            reasons.append("the sentence structure and vocabulary have some AI-typical characteristics")
        elif score >= 0.45:
            reasons.append("the patterns are ambiguous — could be either human or AI")
        else:
            reasons.append("the writing style appears natural and human-like")

    return "; ".join(reasons).capitalize() + "."


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Executive Summary generator
# ═══════════════════════════════════════════════════════════════════════════

def _generate_executive_summary(report: ForensicReport) -> str:
    """
    [v3.9] Professor-friendly executive summary.

    Produces structured HTML bullet points in plain language.
    Explains contradictions, avoids jargon, and focuses on
    actionable conclusions.
    """
    # ── Confidence values ──
    ai_pct    = report.confidence * 100
    human_pct = (1 - report.confidence) * 100

    # ── 1. VERDICT ──
    if "AI" in report.verdict:
        verdict_line = (
            f"<strong>This text was most likely written by an AI tool</strong> "
            f"like ChatGPT, Claude, or similar (AI confidence: {ai_pct:.1f}%)."
        )
        verdict_icon = "🔴"
    elif "Human" in report.verdict:
        verdict_line = (
            f"<strong>This text appears to be written by a human</strong> "
            f"(human confidence: {human_pct:.1f}%)."
        )
        verdict_icon = "🟢"
    elif "Hybrid" in report.verdict:
        verdict_line = (
            f"<strong>This text appears to be a mix of human and AI writing</strong> — "
            f"some sections look AI-generated, others look human-written."
        )
        verdict_icon = "🟡"
    else:
        verdict_line = (
            f"<strong>The analysis is inconclusive</strong> — the system cannot determine "
            f"with confidence whether this was written by a human or by AI."
        )
        verdict_icon = "⚪"

    # ── 2. KEY INDICATORS (bullet list) ──
    indicators = []

    # Sentence uniformity
    if report.sentence_attributions:
        scores = [s.ai_score for s in report.sentence_attributions]
        total  = len(scores)
        high_ai = sum(1 for s in scores if s > 0.7)
        score_std = float(np.std(scores)) if len(scores) >= 2 else 0.0

        if high_ai == total:
            indicators.append(
                f"Every sentence ({total} total) shows AI-generated characteristics. "
                f"Human writing naturally varies from sentence to sentence — "
                f"this level of uniformity is a strong AI hallmark."
            )
        elif high_ai > total * 0.6:
            indicators.append(
                f"{high_ai} out of {total} sentences ({high_ai*100//total}%) show "
                f"AI-generated patterns, while the rest appear more human-like."
            )
        elif high_ai > 0:
            indicators.append(
                f"Only {high_ai} out of {total} sentences show clear AI patterns. "
                f"This suggests selective AI use for specific parts."
            )

        if score_std < 0.05 and total > 3:
            indicators.append(
                f"All sentence scores are nearly identical (variation: {score_std:.3f}). "
                f"Human writing shows much more variation — this uniformity "
                f"suggests the entire text was produced in a single AI session."
            )

    # Writing style
    stats = report.stylometric_stats or {}
    if stats:
        burst = stats.get("burstiness", -1)
        if burst != -1 and burst < 0.12:
            indicators.append(
                "The writing style is unusually uniform throughout — "
                "sentence lengths barely vary. Human writers naturally mix "
                "short and long sentences."
            )
        lex_div = stats.get("lexical_diversity", 0)
        if lex_div > 0.75:
            indicators.append(
                f"Vocabulary richness is high ({lex_div:.2f}). Note: modern AI "
                f"models also produce varied vocabulary, so this alone does not "
                f"indicate human writing."
            )

    # Segment analysis
    if report.hybrid_analysis:
        ha = report.hybrid_analysis
        ha_class = ha.get("classification", "")
        p_scores = ha.get("paragraph_scores", [])
        ai_paras = sum(1 for p in p_scores if p.get("zone") == "AI")
        human_paras = sum(1 for p in p_scores if p.get("zone") == "HUMAN")
        uncertain_paras = sum(1 for p in p_scores if p.get("zone") == "UNCERTAIN")
        total_paras = len(p_scores)
        n_bp = ha.get("breakpoint_count", 0)
        global_ai = ha.get("global_ai_score", 0)

        if "HYBRID" in ha_class or "AI-ASSISTED" in ha_class:
            indicators.append(
                f"Per-paragraph analysis found <strong>{ai_paras} AI-written</strong> "
                f"and <strong>{total_paras - ai_paras} human-written</strong> paragraphs, "
                f"with {n_bp} clear transition point{'s' if n_bp != 1 else ''} where "
                f"the writing style shifts. This suggests the student used AI for "
                f"specific sections."
            )
        elif "FULLY AI" in ha_class:
            indicators.append(
                f"All {total_paras} paragraphs show AI-generated characteristics — "
                f"no human-authored sections detected."
            )
        elif uncertain_paras > 0 and global_ai > 30:
            # Segment says HUMAN but global AI is significant → explain
            indicators.append(
                f"Per-paragraph analysis found {human_paras} human-like and "
                f"{uncertain_paras} uncertain paragraph{'s' if uncertain_paras != 1 else ''} "
                f"(global AI score: {global_ai:.0f}%). "
                f"No single paragraph crossed the 70% AI threshold individually, "
                f"but the overall AI signal is elevated."
            )

    # ── 3. CONTRADICTIONS (critical for professor understanding) ──
    contradictions = []

    # Perplexity vs Neural contradiction
    ppl_score = 0.0
    if report.perplexity_analysis:
        ppl_score = report.perplexity_analysis.get("ai_score", 0.0)
    neural_score = report.neural_score

    if neural_score > 0.7 and ppl_score < 0.3:
        contradictions.append(
            f"The <em>text predictability analysis</em> ({ppl_score:.0%}) suggests human-like text, "
            f"while the <em>neural classifier</em> ({neural_score:.0%}) says AI. "
            f"<strong>Why?</strong> Modern AI (GPT-4, Claude) produces text that older "
            f"detection methods cannot distinguish from human writing. The neural classifier, "
            f"which was trained specifically on modern AI outputs, is more reliable here."
        )
    elif neural_score < 0.3 and ppl_score > 0.7:
        contradictions.append(
            f"The <em>text predictability analysis</em> ({ppl_score:.0%}) flags AI-like patterns, "
            f"but the <em>neural classifier</em> ({neural_score:.0%}) says human. "
            f"This can happen with highly formal or technical human writing that mimics "
            f"AI's statistical regularity. Trust the neural classifier for the final verdict."
        )

    # Segment vs Neural contradiction
    if report.hybrid_analysis:
        ha = report.hybrid_analysis
        ha_class = ha.get("classification", "")
        global_ai = ha.get("global_ai_score", 0)

        if "HUMAN" in ha_class and neural_score > 0.7:
            contradictions.append(
                f"The <em>per-paragraph segment analysis</em> classifies the text as "
                f"\"human-written\" (global AI: {global_ai:.0f}%), while the "
                f"<em>neural classifier</em> scores it at {neural_score:.0%} AI. "
                f"<strong>Why?</strong> The neural classifier sees the full text at once "
                f"and detects overall AI patterns. The segment analysis evaluates smaller "
                f"windows where no single paragraph crosses the 70% AI threshold — "
                f"the AI signal gets diluted across paragraphs. "
                f"The neural classifier is more reliable for the overall verdict; "
                f"the segment heatmap is more useful for locating <em>which</em> sections "
                f"are suspicious."
            )
        elif "AI" in ha_class and neural_score < 0.3:
            contradictions.append(
                f"The <em>per-paragraph segment analysis</em> classifies segments as "
                f"AI-generated, but the <em>neural classifier</em> scores only {neural_score:.0%}. "
                f"This can occur with highly structured or formal human writing. "
                f"The neural classifier is generally more reliable."
            )

    # Formal/classical text warning
    if report.sentence_attributions:
        avg_word_len = 0
        all_wa = [w for s in report.sentence_attributions for w in s.word_attributions]
        if all_wa:
            avg_word_len = float(np.mean([len(w.word) for w in all_wa]))
        if avg_word_len > 5.5 and "AI" in report.verdict:
            contradictions.append(
                "This text uses unusually long words and formal language. "
                "<strong>Note:</strong> formal academic writing, legal texts, "
                "philosophical works, and translated texts can trigger false positives "
                "because they share statistical patterns with AI output "
                "(uniform sentence structure, formal vocabulary, low colloquial markers). "
                "If the student's source material includes classical or highly formal texts, "
                "this should be taken into account."
            )

    # Hallucination vs overall
    if report.hallucination_risk:
        hal_level = report.hallucination_risk.get("risk_level", "")
        hal_risk = report.hallucination_risk.get("overall_risk", 0)
        if "MEDIUM" in hal_level and "Human" in report.verdict:
            contradictions.append(
                f"Some risk-of-fabrication patterns detected ({hal_risk:.0%}), but the "
                f"overall verdict is human-written. Medium risk can occur in careful "
                f"human academic writing — it is not conclusive on its own."
            )

    # ── 4. ADDITIONAL FINDINGS ──
    findings = []

    # Citations
    if report.reference_analysis:
        ra = report.reference_analysis
        ra_fv = ra.get("feature_values", {})
        total_refs = int(ra_fv.get("total_references", 0))
        fab = int(ra_fv.get("fabricated_count", 0))
        if total_refs > 0 and fab == 0:
            findings.append(
                f"<strong>Citations:</strong> {total_refs} references found, "
                f"all verified as real publications in academic databases."
            )
        elif fab > 0:
            findings.append(
                f"<strong>Citations:</strong> {fab} out of {total_refs} references "
                f"could not be found in academic databases — these may be fabricated. "
                f"AI tools frequently invent plausible-sounding citations."
            )
    elif report.reference_analysis is None:
        pass  # no reference check run, don't mention it

    # Hallucination
    if report.hallucination_risk:
        hal = report.hallucination_risk
        hal_risk = hal.get("overall_risk", 0)
        hal_level = hal.get("risk_level", "")
        if "HIGH" in hal_level:
            findings.append(
                f"<strong>Risk of fabricated content:</strong> HIGH ({hal_risk:.0%}). "
                f"The text shows multiple patterns associated with AI making up facts."
            )
        elif "MEDIUM" in hal_level:
            top_cats = hal.get("top_signals", [])
            cat_names = [s.get("category", "") for s in top_cats[:2]] if isinstance(top_cats, list) else []
            findings.append(
                f"<strong>Risk of fabricated content:</strong> MEDIUM ({hal_risk:.0%}). "
                f"Some anomaly patterns detected but not conclusive."
            )

    # Reasoning model
    if report.reasoning_analysis:
        ra_level = report.reasoning_analysis.get("risk_level", "")
        if "HIGH" in ra_level:
            findings.append(
                "<strong>AI type:</strong> The text shows signs of a <em>reasoning AI model</em> "
                "(like OpenAI o1 or DeepSeek R1) — these models think step-by-step before answering."
            )
        else:
            findings.append(
                "<strong>AI type:</strong> If AI was used, it was likely a standard model "
                "(ChatGPT, Claude, Gemini) — no step-by-step reasoning patterns detected."
            )

    # ── Build HTML ──
    html_parts = []

    # Verdict
    html_parts.append(
        f'<div style="padding:12px 16px;background:#f8f9fa;border-radius:8px;'
        f'margin-bottom:16px;font-size:15px;">'
        f'{verdict_icon} {verdict_line}</div>'
    )

    # Key indicators
    if indicators:
        items = "".join(f"<li>{i}</li>" for i in indicators)
        html_parts.append(
            f'<div style="margin-bottom:14px;">'
            f'<strong style="font-size:14px;">📊 Key Indicators</strong>'
            f'<ul style="margin:8px 0;padding-left:20px;line-height:1.7;">{items}</ul></div>'
        )

    # Contradictions
    if contradictions:
        items = "".join(f"<li>{c}</li>" for c in contradictions)
        html_parts.append(
            f'<div style="margin-bottom:14px;background:#fff8e1;padding:12px 16px;'
            f'border-left:4px solid #f39c12;border-radius:4px;">'
            f'<strong style="font-size:14px;">⚠️ Why Some Indicators Disagree</strong>'
            f'<ul style="margin:8px 0;padding-left:20px;line-height:1.7;">{items}</ul></div>'
        )

    # Additional findings
    if findings:
        items = "".join(f"<li>{f}</li>" for f in findings)
        html_parts.append(
            f'<div style="margin-bottom:14px;">'
            f'<strong style="font-size:14px;">🔍 Additional Findings</strong>'
            f'<ul style="margin:8px 0;padding-left:20px;line-height:1.7;">{items}</ul></div>'
        )

    # Disclaimer
    html_parts.append(
        '<div style="font-size:12px;color:#7f8c8d;margin-top:12px;padding-top:10px;'
        'border-top:1px solid #e0e0e0;">'
        '<em>⚠ This is an automated analysis. It should not be the sole basis for '
        'academic integrity decisions. All findings must be reviewed by a qualified '
        'human who considers the full context of the submission.</em></div>'
    )

    return "\n".join(html_parts)


# ═══════════════════════════════════════════════════════════════════════════
# ForensicReportGenerator
# ═══════════════════════════════════════════════════════════════════════════

class ForensicReportGenerator:
    """
    Generates forensic analysis reports for detected texts.

    Parameters (all optional)
    ─────────────────────────
    detector               : SOTAAIDetector/EnsembleDetector
    profiler               : StylometricProfiler
    hallucination_profiler : HallucinationProfiler
    hallucination_classifier: HallucinationRiskClassifier (auto-created if profiler given)
    reasoning_profiler     : ReasoningProfiler
    reasoning_classifier   : ReasoningRiskClassifier (auto-created if profiler given)
    watermark_decoder      : WatermarkDecoder
    """

    def __init__(self, detector=None, profiler=None,
                 hallucination_profiler=None, hallucination_classifier=None,
                 reasoning_profiler=None, reasoning_classifier=None,
                 watermark_decoder=None):
        self.detector = detector
        self.profiler = profiler
        self.hallucination_profiler = hallucination_profiler
        if hallucination_classifier is not None:
            self.hallucination_classifier = hallucination_classifier
        elif hallucination_profiler is not None:
            try:
                from hallucination_profile import HallucinationRiskClassifier
                self.hallucination_classifier = HallucinationRiskClassifier()
            except ImportError:
                self.hallucination_classifier = None
        else:
            self.hallucination_classifier = None
        self.reasoning_profiler = reasoning_profiler
        if reasoning_classifier is not None:
            self.reasoning_classifier = reasoning_classifier
        elif reasoning_profiler is not None:
            self.reasoning_classifier = ReasoningRiskClassifier()
        else:
            self.reasoning_classifier = None
        self.watermark_decoder = watermark_decoder
        self.attribution_calc = AttributionCalculator()
        self.heatmap_gen = HeatmapGenerator()

    def _bridge_stats_from_result(self, text, detection_result, additional_analyses):
        additional = dict(additional_analyses) if additional_analyses else {}
        merged_stat = {"ppl": 50.0, "burstiness": 0.1, "entropy": 0.0,
            "lexical_diversity": 0.5, "avg_sentence_length": 16.0,
            "sentence_length_variance": 0.0, "flesch_kincaid_grade": 0.0}
        if self.profiler is not None and text:
            try:
                ps = self.profiler.compute_stats(text)
                for k in ["burstiness", "lexical_diversity", "avg_sentence_length", "sentence_length_variance"]:
                    if k in ps: merged_stat[k] = ps[k]
                # [NEW v3.5] Store full stylometric stats for dedicated section
                additional["stylometric_full"] = ps
            except Exception as exc:
                logger.warning("StylometricProfiler.compute_stats() failed: %s", exc)
        if detection_result is not None:
            stats = getattr(detection_result, "statistical_features", None)
            if stats:
                for k in merged_stat:
                    if k in stats and stats[k] != 0.0: merged_stat[k] = stats[k]
                # [NEW v3.5] Use detection_result stats if profiler wasn't run
                if "stylometric_full" not in additional:
                    additional["stylometric_full"] = stats
        merged_stat.update(additional.get("statistical", {}))
        additional["statistical"] = merged_stat
        return additional

    def generate_report(self, text, detection_result=None,
                        additional_analyses=None, generate_visuals=True):
        additional = self._bridge_stats_from_result(text, detection_result, additional_analyses)

        if detection_result is not None:
            raw_ai = (detection_result.raw_scores or {}).get("ai")
            if raw_ai is not None:
                overall_score = float(raw_ai) / 100.0
            elif detection_result.prediction == "AI":
                overall_score = detection_result.confidence / 100.0
            else:
                overall_score = 1.0 - detection_result.confidence / 100.0
            overall_score = float(np.clip(overall_score, 0.0, 1.0))
            verdict = "AI-Generated" if detection_result.prediction == "AI" else "Human-Written"
            neural_score = overall_score
        else:
            overall_score, verdict, neural_score = 0.5, "Inconclusive", 0.5

        # [D-3 FIX] Per-word/sentence attributions are a DISPLAY-ONLY heatmap aid:
        # they start from the neural overall_score and are nudged by a hardcoded
        # buzzword lexicon ("delve"/"furthermore" = AI), which is biased against
        # formal/academic/legal/ESL writing. They must NOT move the verdict.
        # The "Hybrid" decision now comes from the real per-paragraph NEURAL analysis
        # (HybridSegmentAnalyzer), not from lexicon-nudged sentence means.
        sentence_attrs = self.attribution_calc.calculate_sentence_attributions(text, overall_score)
        _hybrid = additional.get("hybrid_segment") or {}
        _hybrid_class = str(_hybrid.get("classification", "")).upper()
        if "HYBRID" in _hybrid_class or "MIXED" in _hybrid_class:
            verdict = "Hybrid"

        stat = additional.get("statistical", {})
        statistical_score = stat.get("score", 0.5)
        stylometric_score = additional.get("stylometric", {}).get("similarity_score", 0.5)
        fk = stat.get("flesch_kincaid_grade", 0.0)
        human_baseline = {
            "Perplexity":           (stat.get("ppl", 50.0), 55.0),
            "Burstiness (CV)":      (stat.get("burstiness", 0.1), 0.25),
            "Lexical Diversity":    (stat.get("lexical_diversity", 0.65), 0.70),
            "Avg Sentence Length":  (stat.get("avg_sentence_length", 18.0), 16.0),
            "Flesch-Kincaid Grade": (fk, 10.0),
        }
        # [FIX v3.7] Moved _collect_evidence AFTER all analyses so that
        # hallucination, reasoning, and watermark data are available for
        # evidence generation. Previously it ran first, missing those signals.

        # Hallucination (unchanged v3.3)
        hallucination_risk = None
        if self.hallucination_profiler and self.hallucination_classifier and text:
            try:
                hs = self.hallucination_profiler.compute_stats(text)
                hallucination_risk = self.hallucination_classifier.classify(hs)
                additional["hallucination"] = hallucination_risk
            except Exception as exc:
                logger.warning("Hallucination analysis failed: %s", exc)

        # Reasoning [v3.4, improved v3.5]
        reasoning_analysis = additional.get("reasoning")
        # [FIX v3.5] If reasoning dict exists but is PARTIAL (missing
        # group_scores/top_signals/feature_details), re-classify with
        # ReasoningRiskClassifier to produce the full structure.
        if reasoning_analysis is not None:
            if "group_scores" not in reasoning_analysis and "feature_values" in reasoning_analysis:
                # Partial dict from old-style orchestrator — re-classify
                try:
                    from reasoning_profiler import FEATURE_NAMES as _RN
                    import numpy as _np
                    fv = reasoning_analysis["feature_values"]
                    vec = _np.array([fv.get(n, 0.0) for n in _RN])
                    clf = self.reasoning_classifier or ReasoningRiskClassifier()
                    reasoning_analysis = clf.classify(vec, _RN)
                    additional["reasoning"] = reasoning_analysis
                except Exception as exc:
                    logger.warning("Reasoning re-classification failed: %s", exc)

        if reasoning_analysis is None and self.reasoning_profiler is not None:
            try:
                from reasoning_profiler import FEATURE_NAMES as _RN
                vec = self.reasoning_profiler.vectorize(text)
                clf = self.reasoning_classifier or ReasoningRiskClassifier()
                reasoning_analysis = clf.classify(vec, _RN)
                additional["reasoning"] = reasoning_analysis
            except Exception as exc:
                logger.warning("Reasoning analysis failed: %s", exc)

        # Watermark [v3.4]
        if additional.get("watermark") is None and self.watermark_decoder is not None:
            try:
                wm = self.watermark_decoder.detect(text)
                additional["watermark"] = wm.to_forensic_dict()
            except Exception as exc:
                logger.warning("Watermark detection failed: %s", exc)

        reasoning_score = additional.get("reasoning", {}).get("ai_score", 0.5)
        watermark_score = additional.get("watermark", {}).get("confidence", 0.0)

        # ── LATE FUSION [Fase 2 activated] ────────────────────────────────────
        # The verdict no longer depends on the neural ensemble ALONE. When FUSION_ACTIVE
        # (default), a bounded, model-agnostic, UNCALIBRATED fusion of all plugin signals
        # produces P(AI); the neural ensemble dominates but plugins now genuinely move the
        # result. neural_score is preserved separately so the report can show both. The
        # confidence is explicitly flagged uncalibrated (no labelled corpus yet).
        # The orchestrator normally computes this already (in additional["fusion"]); we
        # reuse it when present and only compute as a standalone fallback.
        import os as _os
        if _os.getenv("FUSION_ACTIVE", "1") == "1" and detection_result is not None:
            try:
                fused = additional.get("fusion")
                if fused is None:
                    from fusion import FusionClassifier
                    fused = FusionClassifier().predict_proba(detection_result, additional).to_dict()
                    additional["fusion"] = fused
                overall_score = float(np.clip(fused["probability"], 0.0, 1.0))
                # Keep an explicit Hybrid verdict (set from per-paragraph analysis); else
                # let the fused probability decide AI vs Human.
                if verdict != "Hybrid" and verdict != "Inconclusive":
                    verdict = "AI-Generated" if overall_score >= 0.5 else "Human-Written"
            except Exception as exc:
                logger.warning("Fusion scoring failed, falling back to neural-only: %s", exc)

        # [FIX v3.7] Now collect evidence AFTER all analyses are complete
        evidence_points = self._collect_evidence(sentence_attrs, additional)

        # Append high-level evidence from hallucination/reasoning/watermark
        if hallucination_risk and hallucination_risk.get("risk_level") == "HIGH":
            evidence_points.append({"type": "high_hallucination_risk",
                "overall_risk": hallucination_risk["overall_risk"],
                "risk_level": hallucination_risk["risk_level"],
                "top_signals": hallucination_risk["top_signals"],
                "interpretation": "Text exhibits multiple hallucination risk indicators.",
                "explanation": (
                    f"Overall hallucination risk is HIGH ({hallucination_risk['overall_risk']:.0%}). "
                    f"Multiple categories flagged — the text exhibits patterns commonly "
                    f"associated with AI-generated content that may contain fabricated details."
                )})
        # [NEW v3.7] Also collect MEDIUM hallucination risk as moderate evidence
        elif hallucination_risk and hallucination_risk.get("risk_level") == "MEDIUM":
            high_cats = [c for c, v in hallucination_risk.get("category_scores", {}).items() if v >= 0.6]
            if high_cats:
                cat_names = [_HAL_CATEGORY_EXPLANATIONS.get(c, {}).get("name", c) for c in high_cats]
                evidence_points.append({"type": "elevated_hallucination_categories",
                    "categories": cat_names,
                    "risk_level": "MEDIUM",
                    "explanation": (
                        f"While overall hallucination risk is MEDIUM, {len(high_cats)} "
                        f"individual {'category scores' if len(high_cats) > 1 else 'category score'} "
                        f"HIGH: {', '.join(cat_names)}. These specific categories warrant attention."
                    )})

        if reasoning_analysis and reasoning_analysis.get("risk_level", "").startswith("HIGH"):
            evidence_points.append({"type": "high_reasoning_model_signal",
                "ai_score": reasoning_analysis["ai_score"],
                "risk_level": reasoning_analysis["risk_level"],
                "interpretation": reasoning_analysis["interpretation"],
                "explanation": (
                    f"Reasoning model detection scored {reasoning_analysis['ai_score']:.0%} — "
                    f"strong chain-of-thought scaffolding and self-correction markers "
                    f"characteristic of reasoning-optimized models (o1, DeepSeek-R1, QwQ)."
                )})
        # [NEW v3.7] Moderate reasoning evidence
        elif reasoning_analysis and "MEDIUM" in reasoning_analysis.get("risk_level", ""):
            evidence_points.append({"type": "moderate_reasoning_model_signal",
                "ai_score": reasoning_analysis["ai_score"],
                "risk_level": reasoning_analysis["risk_level"],
                "explanation": (
                    f"Reasoning model detection scored {reasoning_analysis['ai_score']:.0%} (moderate). "
                    f"Some chain-of-thought patterns detected but not conclusive — "
                    f"could indicate a reasoning model, prompted standard model, or "
                    f"methodical human author."
                )})

        wm = additional.get("watermark", {})
        if wm.get("detected"):
            evidence_points.append({"type": "candidate_watermark_signal",
                "scheme": wm.get("scheme_type"), "confidence": wm.get("confidence"),
                "note": "EXPERIMENTAL watermark signal.",
                "explanation": (
                    f"A candidate digital watermark was detected (confidence: "
                    f"{wm.get('confidence', 0):.0%}, scheme: {wm.get('scheme_type', 'unknown')}). "
                    f"This is experimental and should be verified independently."
                )})

        # [NEW v3.7] Perplexity-based evidence
        ppl_data = additional.get("perplexity", {})
        ppl_score = ppl_data.get("ai_score", 0.0)
        ppl_level = ppl_data.get("risk_level", "")
        if "HIGH" in ppl_level:
            ppl_fv = ppl_data.get("feature_values", {})
            evidence_points.append({"type": "high_perplexity_ai_signal",
                "ai_score": ppl_score,
                "risk_level": ppl_level,
                "proxy_perplexity_mean": ppl_fv.get("proxy_perplexity_mean", 0.0),
                "explanation": (
                    f"Perplexity analysis scored {ppl_score:.0%} — the text's "
                    f"word patterns are highly predictable to a language model, "
                    f"characteristic of machine-generated content. Mean proxy "
                    f"perplexity: {ppl_fv.get('proxy_perplexity_mean', 0):.1f}."
                )})
        ppl_valleys = int(ppl_data.get("feature_values", {}).get("perplexity_valley_count", 0))
        if ppl_valleys >= 2:
            evidence_points.append({"type": "perplexity_valley_detected",
                "valley_count": ppl_valleys,
                "explanation": (
                    f"{ppl_valleys} distinct low-perplexity regions detected in the "
                    f"perplexity curve. Each valley represents a section where the "
                    f"text becomes highly predictable (AI-like), suggesting hybrid "
                    f"authorship with AI-generated segments interspersed with human writing."
                )})

        # [NEW v3.9] Hybrid segment evidence
        hyb_data = additional.get("hybrid_segment", {})
        hyb_class = hyb_data.get("classification", "")
        hyb_risk = hyb_data.get("risk_level", "")
        if "HYBRID" in hyb_class or "AI-ASSISTED" in hyb_class:
            p_scores = hyb_data.get("paragraph_scores", [])
            ai_paras = sum(1 for p in p_scores if p.get("zone") == "AI")
            n_bp = hyb_data.get("breakpoint_count", 0)
            evidence_points.append({"type": "hybrid_authorship_detected",
                "classification": hyb_class,
                "risk_level": hyb_risk,
                "ai_paragraphs": ai_paras,
                "breakpoints": n_bp,
                "explanation": (
                    f"Per-paragraph sliding-window analysis detected {hyb_class}. "
                    f"{ai_paras} paragraph{'s' if ai_paras != 1 else ''} classified "
                    f"as AI-generated with {n_bp} authorship transition{'s' if n_bp != 1 else ''}. "
                    f"Global AI score: {hyb_data.get('global_ai_score', 0):.1f}%."
                )})
        fv = hyb_data.get("feature_vector", {})
        longest_ai_run = int(fv.get("longest_ai_run", 0))
        if longest_ai_run >= 3:
            evidence_points.append({"type": "ai_cluster_detected",
                "longest_ai_run": longest_ai_run,
                "explanation": (
                    f"A cluster of {longest_ai_run} consecutive AI-classified paragraphs "
                    f"was detected, indicating a sustained block of AI-generated content "
                    f"rather than scattered AI edits."
                )})

        # [NEW v3.9] Reference validation evidence
        ref_data = additional.get("reference_check", {})
        ref_score = ref_data.get("ai_score", 0.0)
        ref_level = ref_data.get("risk_level", "")
        ref_fv = ref_data.get("feature_values", {})
        fab_count = int(ref_fv.get("fabricated_count", 0))
        chim_count = int(ref_fv.get("chimeric_count", 0))
        orn_count = int(ref_fv.get("ornamental_count", 0))
        if fab_count >= 1:
            evidence_points.append({"type": "fabricated_citations_detected",
                "fabricated_count": fab_count,
                "ai_score": ref_score,
                "explanation": (
                    f"{fab_count} citation{'s' if fab_count != 1 else ''} could not be found "
                    f"in CrossRef, Semantic Scholar, or OpenAlex databases. Fabricated "
                    f"references are a strong indicator of AI-generated academic text, "
                    f"as LLMs frequently hallucinate plausible-sounding citations."
                )})
        if chim_count >= 1:
            evidence_points.append({"type": "chimeric_citations_detected",
                "chimeric_count": chim_count,
                "explanation": (
                    f"{chim_count} citation{'s' if chim_count != 1 else ''} appear chimeric — "
                    f"combining real author names with fabricated titles or mixing attributes "
                    f"from different real papers. This is characteristic of LLM hallucination."
                )})
        if orn_count >= 2:
            evidence_points.append({"type": "ornamental_references",
                "ornamental_count": orn_count,
                "explanation": (
                    f"{orn_count} references appear ornamental — cited in the reference list "
                    f"but not meaningfully integrated into the text's arguments. AI-generated "
                    f"papers frequently include padding references to appear well-researched."
                )})

        # [NEW v3.5] Extract stylometric stats
        stylometric_stats = additional.get("stylometric_full")

        # [NEW v3.7] Extract perplexity analysis
        perplexity_analysis = additional.get("perplexity")

        # [NEW v3.9] Extract hybrid segment analysis
        hybrid_analysis = additional.get("hybrid_segment")

        # [NEW v3.9] Extract reference analysis
        reference_analysis = additional.get("reference_check")

        report_id = hashlib.md5(f"{text[:100]}{datetime.now().isoformat()}".encode()).hexdigest()[:12].upper()
        report = ForensicReport(
            report_id=report_id, generated_at=datetime.now().isoformat(),
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            word_count=len(text.split()), verdict=verdict, confidence=overall_score,
            neural_score=neural_score, statistical_score=statistical_score,
            stylometric_score=stylometric_score, reasoning_score=reasoning_score,
            watermark_score=watermark_score, sentence_attributions=sentence_attrs,
            human_baseline_comparison=human_baseline, evidence_points=evidence_points,
            hallucination_risk=hallucination_risk, reasoning_analysis=reasoning_analysis,
            stylometric_stats=stylometric_stats,
            perplexity_analysis=perplexity_analysis,
            hybrid_analysis=hybrid_analysis,
            reference_analysis=reference_analysis,
        )

        # [NEW v3.5] Generate executive summary
        report.executive_summary = _generate_executive_summary(report)

        if generate_visuals:
            awa = [w for s in sentence_attrs for w in s.word_attributions]
            report.heatmap_b64           = self._fig_to_base64(self.heatmap_gen.generate_word_heatmap(awa[:200]))
            report.confidence_chart_b64  = self._fig_to_base64(self.heatmap_gen.generate_sentence_chart(sentence_attrs))
            report.comparison_chart_b64  = self._fig_to_base64(self.heatmap_gen.generate_comparison_chart(human_baseline))
        return report

    def _collect_evidence(self, sentence_attrs, additional):
        """
        [FIX v3.6] Completely rewritten evidence collection.

        v3.4/v3.5 bug: required BOTH ai_score > 0.8 AND key_indicators non-empty.
        key_indicators only fires on 23 specific words (furthermore, delve, etc.),
        so natural-sounding AI text produces ZERO evidence despite 95%+ scores.

        v3.6 fix: evidence is generated from MULTIPLE independent sources with
        OR logic, not AND. Any single qualifying condition produces evidence.
        """
        ev = []

        # ── Source 1: High-confidence AI sentences ────────────────────
        # [FIX v3.7] Lowered threshold 0.85 → 0.60 so moderate-AI sentences
        # generate evidence. Texts where ALL sentences score 0.35 still won't
        # fire, but anything in the uncertain-to-AI zone will.
        for a in sentence_attrs:
            if a.ai_score > 0.60:
                ev.append({
                    "type": "high_confidence_ai_sentence",
                    "sentence_position": a.position,
                    "score": round(a.ai_score, 4),
                    "indicators": a.key_indicators if a.key_indicators else ["Elevated AI probability (score-based)"],
                    "excerpt": a.text[:120] + "..." if len(a.text) > 120 else a.text,
                    "explanation": (
                        f"Sentence {a.position+1} scores {a.ai_score:.0%} AI probability. "
                        + (f"Flagged indicators: {', '.join(a.key_indicators[:3])}."
                           if a.key_indicators else
                           "No specific trigger words detected — the score comes from "
                           "the overall AI classification and the sentence's "
                           "statistical conformity to AI-generated patterns.")
                    ),
                })
                if len(ev) >= 5:  # cap at 5 sentence-level evidence items
                    break

        # ── Source 1b: Human-supporting sentences ─────────────────────
        # [NEW v3.7] When no AI sentences found, collect human-supporting evidence
        # so evidence section is never empty for any classification.
        if not ev:
            human_sents = [a for a in sentence_attrs if a.ai_score < 0.40]
            if human_sents:
                avg_human = float(np.mean([a.ai_score for a in human_sents]))
                ev.append({
                    "type": "human_writing_indicators",
                    "sentence_count": len(human_sents),
                    "total_sentences": len(sentence_attrs),
                    "avg_ai_score": round(avg_human, 4),
                    "explanation": (
                        f"{len(human_sents)} of {len(sentence_attrs)} sentences score below "
                        f"the AI threshold (average AI score: {avg_human:.0%}). These sentences "
                        f"exhibit natural language patterns consistent with human authorship — "
                        f"varied word choice, organic sentence structure, and absence of "
                        f"formulaic AI patterns."
                    ),
                })

        # ── Source 2: Score uniformity (ALL sentences score similarly) ─
        # [FIX v3.7] Lowered mean threshold 0.7→0.5 and std 0.05→0.08
        if len(sentence_attrs) >= 3:
            scores = [s.ai_score for s in sentence_attrs]
            score_std = float(np.std(scores))
            score_mean = float(np.mean(scores))
            if score_std < 0.08 and score_mean > 0.5:
                ev.append({
                    "type": "uniform_high_ai_scores",
                    "mean_score": round(score_mean, 4),
                    "std_dev": round(score_std, 4),
                    "sentence_count": len(sentence_attrs),
                    "explanation": (
                        f"All {len(sentence_attrs)} sentences score within a very narrow "
                        f"band (mean={score_mean:.0%}, std={score_std:.4f}). Human writing "
                        f"typically shows much more variation between sentences. This "
                        f"uniformity is a strong indicator of single-source AI generation."
                    ),
                })

        # ── Source 3: No human indicator words found ──────────────────
        # [FIX v3.7] Skip if Source 1b already produced human-supporting evidence,
        # because "human writing indicators" + "absent human markers" reads as
        # contradictory to non-technical readers. Source 3 measures informal/slang
        # words specifically; Source 1b measures sentence-level AI scores.
        has_human_evidence = any(e.get("type") == "human_writing_indicators" for e in ev)
        has_human_words = any(
            any(w.features.get("human_indicator") for w in s.word_attributions)
            for s in sentence_attrs
        )
        if not has_human_words and len(sentence_attrs) >= 3 and not has_human_evidence:
            ev.append({
                "type": "absent_human_markers",
                "explanation": (
                    "The text contains zero informal or colloquial markers "
                    "(contractions, slang, casual expressions) across all "
                    f"{len(sentence_attrs)} sentences. Human writing almost always "
                    "includes some informal elements. Their complete absence is "
                    "consistent with AI-generated text that maintains a uniform "
                    "formal register throughout."
                ),
            })

        # ── Source 4: Low perplexity (when available) ─────────────────
        st = additional.get("statistical", {})
        ppl = st.get("ppl", 50)
        if ppl < 20:
            ev.append({
                "type": "low_perplexity",
                "value": ppl,
                "human_baseline": 55,
                "explanation": (
                    f"Text perplexity is {ppl:.1f}, far below the human baseline "
                    f"of ~55. Low perplexity means the text is highly predictable "
                    f"— a language model can easily guess what comes next. This "
                    f"is a strong statistical indicator of AI generation."
                ),
            })

        # ── Source 5: Stylometric anomalies ───────────────────────────
        # [FIX v3.7] Broadened thresholds and added high-variance detection
        stylo = additional.get("stylometric_full", {})
        if stylo:
            burst = stylo.get("burstiness", 0.5)
            lex_d = stylo.get("lexical_diversity", 0.5)
            hapax = stylo.get("hapax_legomena_ratio", 0.5)
            slv   = stylo.get("sentence_length_variance", -1)
            anomalies = []
            if burst < 0.12:
                anomalies.append(f"very low burstiness ({burst:.3f})")
            if lex_d < 0.40:
                anomalies.append(f"low lexical diversity ({lex_d:.3f})")
            if hapax < 0.30:
                anomalies.append(f"low hapax ratio ({hapax:.3f})")
            # [FIX v3.8] Only report high variance if burstiness is NOT low.
            # Both firing simultaneously is perceptually contradictory.
            if slv > 0 and slv > 200 and burst >= 0.12:
                anomalies.append(f"very high sentence length variance ({slv:.1f})")
            if anomalies:
                ev.append({
                    "type": "stylometric_anomaly",
                    "anomalies": anomalies,
                    "explanation": (
                        "Writing style analysis detected: " + "; ".join(anomalies) + ". "
                        "These patterns deviate from typical human writing baselines "
                        "and are more commonly observed in AI-generated text."
                    ),
                })

        # ── Source 6: Hallucination risk (individual HIGH categories) ──
        # [FIX v3.7] Now accessible because hallucination analysis runs before
        # _collect_evidence. Collects individual HIGH categories as evidence.
        hal = additional.get("hallucination", {})
        if hal:
            for cname, cscore in hal.get("category_scores", {}).items():
                if cscore >= 0.6:
                    cat_info = _HAL_CATEGORY_EXPLANATIONS.get(cname, {})
                    cat_display = cat_info.get("name", cname.replace("_", " ").title())
                    cat_expl = cat_info.get("high", "")
                    ev.append({
                        "type": "high_hallucination_category",
                        "category": cat_display,
                        "score": round(cscore, 4),
                        "explanation": (
                            f"Hallucination category '{cat_display}' scored {cscore:.0%} (HIGH). "
                            f"{cat_expl}"
                        ),
                    })

        # ── Source 7: Statistical comparison vs human baseline ────────
        if st:
            deviations = []
            baseline_map = {
                "burstiness": (0.25, "burstiness"),
                "lexical_diversity": (0.70, "lexical diversity"),
                "avg_sentence_length": (16.0, "average sentence length"),
            }
            for key, (human_val, label) in baseline_map.items():
                val = st.get(key)
                if val is not None and human_val > 0:
                    ratio = abs(val - human_val) / human_val
                    if ratio > 0.30:
                        deviations.append(f"{label} deviates {ratio:.0%} from human baseline")
            if deviations:
                ev.append({
                    "type": "statistical_deviation_from_baseline",
                    "deviations": deviations,
                    "explanation": (
                        "Compared to typical human writing: " + "; ".join(deviations) + ". "
                        "Significant deviations from human baselines are consistent "
                        "with machine-generated text."
                    ),
                })

        return ev

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _level_colors(level):
        return {"high": ("#c0392b","#fff0ee"), "medium": ("#856404","#fff8e1"),
                "low":  ("#1e8449","#eafaf1")}.get(level, ("#555","#f5f5f5"))

    # ── Build reasoning HTML (v3.4 structure, v3.5 fixes) ────────────

    def _build_reasoning_html(self, reasoning):
        if reasoning is None: return ""
        score = reasoning.get("ai_score", 0.0); risk_level = reasoning.get("risk_level", "N/A")
        interp = reasoning.get("interpretation", ""); gs = reasoning.get("group_scores", {})
        top = reasoning.get("top_signals", []); fd = reasoning.get("feature_details", {})
        if "HIGH" in risk_level:   rf,rb,rb2 = "#c0392b","#fff0ee","#e74c3c"
        elif "MEDIUM" in risk_level: rf,rb,rb2 = "#856404","#fff8e1","#f39c12"
        else:                        rf,rb,rb2 = "#1e8449","#eafaf1","#27ae60"

        grows = ""
        for gn, gv in gs.items():
            p = gv*100; bc = "#e74c3c" if p>=55 else "#f39c12" if p>=28 else "#27ae60"
            grows += (f"<tr><td style='width:40%;'>{gn}</td>"
                      f"<td style='text-align:right;'><strong>{p:.1f}%</strong></td>"
                      f"<td><div style='background:#ecf0f1;border-radius:3px;height:14px;'>"
                      f"<div class='animated-bar' style='background:{bc};border-radius:3px;height:14px;width:{max(p,2):.1f}%;'>"
                      f"</div></div></td></tr>\n")

        trows = ""
        for s in top:
            fg,bg = self._level_colors(s.get("level","low"))
            expl = (s.get("explanation") or "")[:280]
            trows += (f"<tr><td><strong style='font-size:13px;'>{s.get('display_name',s.get('feature'))}</strong>"
                      f"<br><small style='color:#7f8c8d;'>{s.get('group')}</small></td>"
                      f"<td style='text-align:right;font-family:monospace;'>{s.get('raw_value',0):.6f}</td>"
                      f"<td style='background:{bg};color:{fg};text-align:center;font-weight:bold;'>{s.get('level','').upper()}</td>"
                      f"<td style='font-size:12px;'>{expl}</td></tr>\n")

        frows = ""
        for feat, det in fd.items():
            fg,bg = self._level_colors(det.get("level","low"))
            expl = (det.get("explanation") or "")[:240]
            frows += (f"<tr><td><strong style='font-size:12px;'>{det.get('display_name',feat)}</strong>"
                      f"<br><small style='color:#7f8c8d;'>{det.get('group')}</small></td>"
                      f"<td style='text-align:right;font-family:monospace;font-size:12px;'>{det.get('value',0):.6f}</td>"
                      f"<td style='background:{bg};color:{fg};text-align:center;font-weight:bold;'>{det.get('level','').upper()}</td>"
                      f"<td style='font-size:11px;color:#555;white-space:nowrap;'>&ge;{det.get('threshold_low',0):.3f} / &ge;{det.get('threshold_high',0):.3f}</td>"
                      f"<td style='font-size:11px;'>{expl}</td></tr>\n")

        return f"""
<h2 class="section-header" onclick="toggleSection('reasoning-section')">
  Advanced: AI Model Type Detection <small style="color:#7f8c8d;">(Technical)</small> <span class="toggle-icon">&#9660;</span>
</h2>
<div id="reasoning-section" class="collapsible-section collapsed">
<div class="disclaimer" style="background:#eef5ff; border:2px solid #3498db; color:#1a5276;">
  <strong>REASONING ANALYSIS \u2014 Zero-Resource, CPU-only</strong><br>
  Detects chain-of-thought scaffolding, self-correction markers, and logical connector patterns
  characteristic of reasoning-optimised LLMs (o1, o3, DeepSeek-R1, QwQ).
  Based on 15 features from <code>ReasoningProfiler</code>. No GPU or external knowledge base required.
</div>
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:220px;">
    <div class="metric-value" style="color:{rf};">{score:.1%}</div>
    <div class="metric-label">Reasoning Model Probability</div>
  </div>
  <div class="metric" style="width:330px; background:{rb}; border:2px solid {rb2};">
    <div class="metric-value" style="color:{rf}; font-size:16px;">{risk_level}</div>
    <div class="metric-label">Classification</div>
  </div>
</div>
<p><strong>Professional Interpretation:</strong><br>{interp}</p>
<h3>Signal Group Scores</h3>
<table class="interactive-table"><thead><tr><th style="width:40%;">Group</th><th style="width:10%;">Score</th><th>Intensity</th></tr></thead>
<tbody>{grows}</tbody></table>
<h3>Top 5 Diagnostic Signals</h3>
<table class="interactive-table"><thead><tr><th style="width:22%;">Signal</th><th style="width:10%;">Raw Value</th>
<th style="width:8%;">Level</th><th>Professional Interpretation</th></tr></thead>
<tbody>{trows}</tbody></table>
<h3>Complete 15-Dimensional Feature Profile</h3>
<p style="color:#7f8c8d;font-size:12px;">All features from <code>ReasoningProfiler</code> (O(n), CPU-only).
<em>LOW = below low threshold | MEDIUM = between thresholds | HIGH = above high threshold.</em></p>
<table class="interactive-table"><thead><tr><th style="width:22%;">Feature</th><th style="width:10%;">Value</th>
<th style="width:7%;">Level</th><th style="width:14%;">Thresholds</th><th>Explanation</th></tr></thead>
<tbody>{frows}</tbody></table>
</div>
"""

    # ── [NEW v3.5] Build hallucination HTML with explanations ─────────

    def _build_hallucination_html(self, hallucination_risk):
        if hallucination_risk is None: return ""
        hr = hallucination_risk
        rc = {"LOW":"#27ae60","MEDIUM":"#f39c12","HIGH":"#e74c3c"}.get(hr.get("risk_level",""),"#7f8c8d")
        risk_level = hr.get("risk_level", "N/A")
        overall    = hr.get("overall_risk", 0)

        cr = ""
        for c,v in sorted(hr.get("category_scores",{}).items(), key=lambda x:x[1], reverse=True):
            cat_info = _HAL_CATEGORY_EXPLANATIONS.get(c, {})
            cat_name = cat_info.get("name", c.replace("_"," ").title())
            cat_desc = cat_info.get("desc", "")
            # Determine level for explanation
            if v >= 0.6:
                cat_expl = cat_info.get("high", "")
                lvl_badge = '<span style="color:#e74c3c;font-weight:bold;">HIGH</span>'
            elif v >= 0.3:
                cat_expl = cat_info.get("medium", "")
                lvl_badge = '<span style="color:#f39c12;font-weight:bold;">MEDIUM</span>'
            else:
                cat_expl = cat_info.get("low", "")
                lvl_badge = '<span style="color:#27ae60;font-weight:bold;">LOW</span>'

            pct = v * 100
            bar_c = "#e74c3c" if v >= 0.6 else "#f39c12" if v >= 0.3 else "#27ae60"
            cr += (f'<tr class="hoverable-row">'
                   f'<td><strong>{cat_name}</strong><br>'
                   f'<small style="color:#7f8c8d;">{cat_desc}</small></td>'
                   f'<td style="text-align:right;"><strong>{v:.2%}</strong></td>'
                   f'<td>{lvl_badge}</td>'
                   f'<td><div style="background:#ecf0f1;border-radius:3px;height:12px;">'
                   f'<div class="animated-bar" style="background:{bar_c};border-radius:3px;height:12px;width:{max(pct,2):.1f}%;"></div>'
                   f'</div></td>'
                   f'<td style="font-size:12px;">{cat_expl}</td>'
                   f'</tr>\n')

        sh = "".join(f'<li><code>{s["feature"]}</code> = {s["value"]:.4f}</li>' for s in hr.get("top_signals",[])[:5])

        return f"""
<h2 class="section-header" onclick="toggleSection('hallucination-section')">
  Risk of Fabricated Content <span class="toggle-icon">&#9660;</span>
</h2>
<div id="hallucination-section" class="collapsible-section">
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:200px;">
    <div class="metric-value" style="color:{rc};">{overall:.0%}</div>
    <div class="metric-label">Overall Risk ({risk_level})</div>
  </div>
</div>
<table class="interactive-table">
  <thead><tr><th>Category</th><th style="width:8%;">Score</th><th style="width:7%;">Level</th><th style="width:12%;">Intensity</th><th>What This Means</th></tr></thead>
  <tbody>{cr}</tbody>
</table>
<p><strong>Top Statistical Signals:</strong></p><ul>{sh}</ul>
<p style="color:#7f8c8d;font-size:12px;"><em>Zero-resource analysis. Scores indicate statistical anomaly patterns, not confirmed factual errors.</em></p>
</div>
"""

    # ── [NEW v3.5] Build stylometric HTML section ────────────────────

    def _build_stylometric_html(self, stylometric_stats):
        if not stylometric_stats: return ""

        rows = ""
        for key in ["burstiness", "lexical_diversity", "avg_sentence_length",
                     "sentence_length_variance", "avg_word_length", "vocabulary_richness",
                     "hapax_legomena_ratio", "rare_word_ratio", "comma_rate",
                     "complex_sentence_ratio"]:
            val = stylometric_stats.get(key)
            if val is None: continue
            info = _STYLO_EXPLANATIONS.get(key, {})
            name = info.get("name", key.replace("_", " ").title())
            desc = info.get("desc", "")
            thr  = info.get("thresholds", (0.0, 1.0))

            if val >= thr[1]:
                lvl = "HIGH"; fg = "#e74c3c"; bg = "#fff0ee"
            elif val >= thr[0]:
                lvl = "MEDIUM"; fg = "#856404"; bg = "#fff8e1"
            else:
                lvl = "LOW"; fg = "#1e8449"; bg = "#eafaf1"

            rows += (f'<tr class="hoverable-row">'
                     f'<td><strong>{name}</strong></td>'
                     f'<td style="text-align:right;font-family:monospace;">{val:.4f}</td>'
                     f'<td style="background:{bg};color:{fg};text-align:center;font-weight:bold;">{lvl}</td>'
                     f'<td style="font-size:12px;">{desc}</td>'
                     f'</tr>\n')

        return f"""
<h2 class="section-header" onclick="toggleSection('stylometric-section')">
  Writing Style Analysis <span class="toggle-icon">&#9660;</span>
</h2>
<div id="stylometric-section" class="collapsible-section">
<div class="disclaimer" style="background:#f0f8e7; border:2px solid #27ae60; color:#1a5216;">
  <strong>WRITING STYLE ANALYSIS</strong><br>
  How the text's vocabulary, sentence structure, and writing patterns compare to
  typical human and AI baselines.
</div>
<table class="interactive-table">
  <thead><tr><th style="width:22%;">Metric</th><th style="width:10%;">Value</th><th style="width:8%;">Level</th><th>What This Means</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
"""

    # ── [NEW v3.7] Build perplexity analysis HTML section ────────────

    def _build_perplexity_html(self, perplexity_analysis):
        if not perplexity_analysis: return ""
        pa = perplexity_analysis
        score = pa.get("ai_score", 0.0)
        risk_level = pa.get("risk_level", "N/A")
        interp = pa.get("interpretation", "")
        tier = pa.get("tier", "tier1")
        stats = pa.get("stats", pa)
        fd = pa.get("feature_details", {})

        if "HIGH" in risk_level:   rf, rb, rb2 = "#c0392b", "#fff0ee", "#e74c3c"
        elif "MEDIUM" in risk_level: rf, rb, rb2 = "#856404", "#fff8e1", "#f39c12"
        elif "INSUFFICIENT" in risk_level: rf, rb, rb2 = "#7f8c8d", "#f5f5f5", "#95a5a6"
        else:                        rf, rb, rb2 = "#1e8449", "#eafaf1", "#27ae60"

        tier_badge = ("&#x1F4BB; Tier 1 (CPU — N-gram Proxy)" if tier == "tier1"
                      else "&#x1F680; Tier 2 (GPT-2 Token-Level)")

        # Feature detail rows
        frows = ""
        for feat, det in fd.items():
            fg, bg = self._level_colors(det.get("level", "low"))
            expl = (det.get("explanation") or "")[:280]
            frows += (f'<tr class="hoverable-row">'
                      f'<td><strong>{det.get("display_name", feat)}</strong></td>'
                      f'<td style="text-align:right;font-family:monospace;">{det.get("value", 0):.4f}</td>'
                      f'<td style="background:{bg};color:{fg};text-align:center;font-weight:bold;">'
                      f'{det.get("level", "").upper()}</td>'
                      f'<td style="font-size:12px;">{expl}</td></tr>\n')

        # Window perplexity mini-chart (text-based bar chart)
        window_ppls = stats.get("window_ppls", [])
        wrows = ""
        if window_ppls:
            max_ppl = max(window_ppls) if window_ppls else 1.0
            for i, ppl in enumerate(window_ppls):
                pct = (ppl / max_ppl * 100) if max_ppl > 0 else 0
                bc = "#e74c3c" if ppl < 4.0 else "#f39c12" if ppl < 7.0 else "#27ae60"
                wrows += (f'<tr><td style="width:8%;">W{i+1}</td>'
                          f'<td style="text-align:right;font-family:monospace;">{ppl:.2f}</td>'
                          f'<td><div style="background:#ecf0f1;border-radius:3px;height:14px;">'
                          f'<div class="animated-bar" style="background:{bc};border-radius:3px;'
                          f'height:14px;width:{max(pct,2):.1f}%;"></div></div></td></tr>\n')

        window_table = ""
        if wrows:
            window_table = f"""
<h3>Per-Window Perplexity Curve</h3>
<p style="color:#7f8c8d;font-size:12px;">Each window = 5-10 sentences with overlap.
<em>Red = AI-typical low perplexity | Green = human-typical high perplexity.</em></p>
<table class="interactive-table">
<thead><tr><th style="width:8%;">Window</th><th style="width:12%;">PPL Score</th><th>Relative Level</th></tr></thead>
<tbody>{wrows}</tbody></table>"""

        return f"""
<h2 class="section-header" onclick="toggleSection('perplexity-section')">
  Advanced: Text Predictability Analysis <small style="color:#7f8c8d;">(Technical)</small> <span class="toggle-icon">&#9660;</span>
</h2>
<div id="perplexity-section" class="collapsible-section collapsed">
<div class="disclaimer" style="background:#f0e6ff; border:2px solid #8e44ad; color:#4a235a;">
  <strong>PERPLEXITY ANALYSIS — {tier_badge}</strong><br>
  Measures how "predictable" the text is to a language model. AI-generated text tends to be
  highly predictable (low perplexity) because it follows statistical patterns learned during training.
  Based on DetectGPT, Fast-DetectGPT, and LLMDet research methodologies.
</div>
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:220px;">
    <div class="metric-value" style="color:{rf};">{score:.1%}</div>
    <div class="metric-label">AI Perplexity Score</div>
  </div>
  <div class="metric" style="width:330px; background:{rb}; border:2px solid {rb2};">
    <div class="metric-value" style="color:{rf}; font-size:16px;">{risk_level}</div>
    <div class="metric-label">Classification</div>
  </div>
</div>
<p><strong>Professional Interpretation:</strong><br>{interp}</p>
<h3>Feature Analysis</h3>
<table class="interactive-table">
  <thead><tr><th style="width:22%;">Feature</th><th style="width:10%;">Value</th>
  <th style="width:8%;">Level</th><th>What This Means</th></tr></thead>
  <tbody>{frows}</tbody>
</table>
{window_table}
<p style="color:#7f8c8d;font-size:12px;"><em>Window count: {stats.get('window_count', 0)} |
Tokens analysed: {stats.get('tokens_analysed', 0)} |
Engine: {tier}</em></p>
</div>
"""

    # ── [NEW v3.9] Build hybrid segment heatmap HTML ─────────────────

    def _build_hybrid_heatmap_html(self, hybrid_analysis):
        """Render per-paragraph heatmap with colored bars."""
        if not hybrid_analysis:
            return ""
        ha = hybrid_analysis
        classification = ha.get("classification", "N/A")
        risk_level = ha.get("risk_level", "N/A")
        interp = ha.get("interpretation", "")
        global_ai = ha.get("global_ai_score", 0.0)
        n_bp = ha.get("breakpoint_count", 0)
        p_scores = ha.get("paragraph_scores", [])

        if "HIGH" in risk_level:   rf, rb, rb2 = "#c0392b", "#fff0ee", "#e74c3c"
        elif "MEDIUM" in risk_level: rf, rb, rb2 = "#856404", "#fff8e1", "#f39c12"
        else:                        rf, rb, rb2 = "#1e8449", "#eafaf1", "#27ae60"

        # Zone color mapping
        def _zone_colors(zone):
            if zone == "AI":        return "#e74c3c", "#fff0ee", "AI-Generated"
            if zone == "HUMAN":     return "#27ae60", "#eafaf1", "Human-Written"
            return "#f39c12", "#fff8e1", "Uncertain"

        # Build paragraph heatmap bars
        para_bars = ""
        for p in p_scores:
            zc, zbg, zlabel = _zone_colors(p.get("zone", "UNCERTAIN"))
            ai_pct = p.get("ai_score", 50)
            human_pct = p.get("human_score", 50)
            wc = p.get("word_count", 0)
            preview = p.get("text_preview", "")[:70]
            idx = p.get("index", 0) + 1

            para_bars += f"""
<div class="hoverable-row" style="display:flex;align-items:center;gap:10px;padding:8px 12px;
  margin:3px 0;border-radius:6px;border-left:5px solid {zc};background:{zbg};
  transition:transform 0.15s,box-shadow 0.15s;">
  <div style="min-width:28px;font-weight:700;color:{zc};font-size:14px;">P{idx}</div>
  <div style="flex:1;">
    <div style="display:flex;height:22px;border-radius:4px;overflow:hidden;background:#ecf0f1;">
      <div class="animated-bar" style="width:{ai_pct:.1f}%;background:#e74c3c;"></div>
      <div class="animated-bar" style="width:{human_pct:.1f}%;background:#27ae60;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:#555;margin-top:2px;">
      <span title="{preview}...">{preview[:50]}{'...' if len(preview) > 50 else ''}</span>
      <span>{wc} words</span>
    </div>
  </div>
  <div style="min-width:70px;text-align:center;">
    <span style="background:{zc};color:white;padding:2px 8px;border-radius:10px;font-size:11px;
      font-weight:700;">{ai_pct:.0f}% AI</span>
  </div>
  <div style="min-width:80px;text-align:right;">
    <span style="background:{zbg};color:{zc};padding:2px 8px;border-radius:10px;font-size:11px;
      font-weight:600;border:1px solid {zc};">{zlabel}</span>
  </div>
</div>"""

        # Build breakpoint markers
        bp_html = ""
        breakpoints = ha.get("breakpoints", [])
        if breakpoints:
            bp_rows = ""
            for bp in breakpoints:
                bp_rows += (
                    f'<tr class="hoverable-row">'
                    f'<td>P{bp.get("after_paragraph", 0)+1} → P{bp.get("after_paragraph", 0)+2}</td>'
                    f'<td style="font-weight:700;">±{bp.get("delta", 0):.1f}%</td>'
                    f'<td>{bp.get("from_zone", "?")} → {bp.get("to_zone", "?")}</td>'
                    f'</tr>\n'
                )
            bp_html = f"""
<h3>Authorship Transitions (Breakpoints)</h3>
<p style="color:#7f8c8d;font-size:12px;">Points where the AI score changes
significantly between adjacent paragraphs, indicating an authorship switch.</p>
<table class="interactive-table">
<thead><tr><th>Location</th><th>Score Change</th><th>Transition</th></tr></thead>
<tbody>{bp_rows}</tbody></table>"""

        # Feature vector table
        fv = ha.get("feature_vector", {})
        fv_rows = ""
        fv_display_names = {
            "segment_count": "Total Segments",
            "global_ai_score": "Global AI Score",
            "ai_segment_ratio": "AI Segment Ratio",
            "human_segment_ratio": "Human Segment Ratio",
            "uncertain_segment_ratio": "Uncertain Segment Ratio",
            "max_ai_score": "Max AI Score (any paragraph)",
            "min_ai_score": "Min AI Score (any paragraph)",
            "score_std": "Score Std Dev (paragraph variability)",
            "breakpoint_count": "Breakpoint Count",
            "longest_ai_run": "Longest Consecutive AI Paragraphs",
        }
        for feat, val in fv.items():
            display = fv_display_names.get(feat, feat.replace("_", " ").title())
            # Color high-risk values
            if feat in ("ai_segment_ratio", "global_ai_score") and val > 60:
                fg, bg = "#c0392b", "#fff0ee"
            elif feat == "breakpoint_count" and val >= 2:
                fg, bg = "#856404", "#fff8e1"
            elif feat == "longest_ai_run" and val >= 3:
                fg, bg = "#c0392b", "#fff0ee"
            else:
                fg, bg = "#2c3e50", "transparent"
            fv_rows += (
                f'<tr class="hoverable-row">'
                f'<td>{display}</td>'
                f'<td style="text-align:right;font-family:monospace;color:{fg};'
                f'background:{bg};">{val:.4f}</td></tr>\n'
            )

        # Summary metrics
        ai_paras = sum(1 for p in p_scores if p.get("zone") == "AI")
        human_paras = sum(1 for p in p_scores if p.get("zone") == "HUMAN")
        uncertain_paras = sum(1 for p in p_scores if p.get("zone") == "UNCERTAIN")

        return f"""
<h2 class="section-header" onclick="toggleSection('hybrid-section')">
  Segment Heatmap — Per-Paragraph AI Detection <span class="toggle-icon">&#9660;</span>
</h2>
<div id="hybrid-section" class="collapsible-section">
<div class="disclaimer" style="background:#f0e6ff; border:2px solid #8e44ad; color:#4a235a;">
  <strong>SEGMENT ANALYSIS — Sliding Window ModernBERT</strong><br>
  Each paragraph is classified independently using overlapping 300-word windows passed through
  the 4-model ModernBERT ensemble. This reveals <strong>where</strong> in the document AI was
  used, not just whether it was used. Red = AI-generated, Green = Human-written, Yellow = Uncertain.
</div>
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:180px;">
    <div class="metric-value" style="color:{rf};">{global_ai:.1f}%</div>
    <div class="metric-label">Global AI Score</div>
  </div>
  <div class="metric" style="width:280px; background:{rb}; border:2px solid {rb2};">
    <div class="metric-value" style="color:{rf}; font-size:15px;">{classification}</div>
    <div class="metric-label">Classification</div>
  </div>
  <div class="metric" style="width:130px;">
    <div class="metric-value" style="color:#e74c3c;">{ai_paras}</div>
    <div class="metric-label">AI Paragraphs</div>
  </div>
  <div class="metric" style="width:130px;">
    <div class="metric-value" style="color:#27ae60;">{human_paras}</div>
    <div class="metric-label">Human Paragraphs</div>
  </div>
  <div class="metric" style="width:130px;">
    <div class="metric-value" style="color:#f39c12;">{uncertain_paras}</div>
    <div class="metric-label">Uncertain</div>
  </div>
</div>

<p><strong>Professional Interpretation:</strong><br>{interp}</p>

<h3>Paragraph Heatmap</h3>
<p style="color:#7f8c8d;font-size:12px;">
  Each bar shows the AI (red) vs Human (green) score distribution for that paragraph.
  <strong>🟢 HUMAN</strong> (AI &lt; 30%) &nbsp;
  <strong>🟡 UNCERTAIN</strong> (30–70%) &nbsp;
  <strong>🔴 AI</strong> (AI ≥ 70%)
</p>
<div style="margin:15px 0;">
{para_bars}
</div>

{bp_html}

<h3>Feature Vector (10-dimensional)</h3>
<table class="interactive-table">
<thead><tr><th style="width:55%;">Feature</th><th style="width:15%;">Value</th></tr></thead>
<tbody>{fv_rows}</tbody></table>

<p style="color:#7f8c8d;font-size:12px;"><em>
  Paragraphs: {ha.get('total_paragraphs', 0)} |
  Windows: {ha.get('total_windows', 0)} |
  Breakpoints: {n_bp} |
  Window size: 300 words / 50% overlap
</em></p>
</div>
"""

    # ── [NEW v3.5] Build executive summary HTML ──────────────────────

    def _build_reference_html(self, reference_analysis):
        """Render citation validation results."""
        if not reference_analysis:
            return ""
        ra = reference_analysis
        score = ra.get("ai_score", 0.0)
        risk_level = ra.get("risk_level", "N/A")
        interp = ra.get("interpretation", "")
        refs = ra.get("references", [])
        fv = ra.get("feature_values", {})

        if "HIGH" in risk_level:   rf, rb, rb2 = "#c0392b", "#fff0ee", "#e74c3c"
        elif "MEDIUM" in risk_level: rf, rb, rb2 = "#856404", "#fff8e1", "#f39c12"
        else:                        rf, rb, rb2 = "#1e8449", "#eafaf1", "#27ae60"

        total_refs = int(fv.get("total_references", 0))
        fab_count = int(fv.get("fabricated_count", 0))
        chim_count = int(fv.get("chimeric_count", 0))
        verified_count = int(fv.get("verified_count", 0))
        orn_count = int(fv.get("ornamental_count", 0))

        # Per-reference rows
        ref_rows = ""
        for r in refs[:20]:
            status = (r.get("status") or "unknown").upper()
            s_score = r.get("confidence_score", 0)
            if status == "VERIFIED":
                sc, sbg = "#27ae60", "#eafaf1"
            elif status == "NOT_FOUND":
                sc, sbg = "#e74c3c", "#fff0ee"
            elif r.get("is_chimeric"):
                sc, sbg = "#e67e22", "#fef5e7"
                status = "CHIMERIC"
            elif status == "PARTIAL":
                sc, sbg = "#f39c12", "#fff8e1"
            else:
                sc, sbg = "#7f8c8d", "#f5f5f5"

            title_preview = r.get("title", "")[:70]
            authors = r.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(authors[:3])
            authors = str(authors)[:40]
            year = r.get("year", "")

            ref_rows += (
                f'<tr class="hoverable-row">'
                f'<td style="max-width:300px;">{title_preview}{"..." if len(r.get("title","")) > 70 else ""}</td>'
                f'<td>{authors}</td>'
                f'<td style="text-align:center;">{year}</td>'
                f'<td style="text-align:center;font-family:monospace;">{s_score:.0f}</td>'
                f'<td style="background:{sbg};color:{sc};text-align:center;font-weight:700;">'
                f'{status}</td></tr>\n'
            )

        ref_table = ""
        if ref_rows:
            ref_table = f"""
<h3>Per-Reference Validation</h3>
<p style="color:#7f8c8d;font-size:12px;">Each reference checked against CrossRef,
Semantic Scholar, and OpenAlex. Score = best Levenshtein match (100 = exact).</p>
<table class="interactive-table">
<thead><tr><th style="width:40%;">Title</th><th style="width:18%;">Authors</th>
<th style="width:7%;">Year</th><th style="width:8%;">Score</th>
<th style="width:12%;">Status</th></tr></thead>
<tbody>{ref_rows}</tbody></table>"""

        # Feature details
        fd = ra.get("feature_details", {})
        frows = ""
        for feat, det in fd.items():
            fg, bg = self._level_colors(det.get("level", "low"))
            expl = (det.get("explanation") or "")[:280]
            frows += (f'<tr class="hoverable-row">'
                      f'<td><strong>{det.get("display_name", feat)}</strong></td>'
                      f'<td style="text-align:right;font-family:monospace;">{det.get("value", 0):.4f}</td>'
                      f'<td style="background:{bg};color:{fg};text-align:center;font-weight:bold;">'
                      f'{det.get("level", "").upper()}</td>'
                      f'<td style="font-size:12px;">{expl}</td></tr>\n')

        feat_table = ""
        if frows:
            feat_table = f"""
<h3>Feature Analysis</h3>
<table class="interactive-table">
<thead><tr><th style="width:22%;">Feature</th><th style="width:10%;">Value</th>
<th style="width:8%;">Level</th><th>What This Means</th></tr></thead>
<tbody>{frows}</tbody></table>"""

        return f"""
<h2 class="section-header" onclick="toggleSection('reference-section')">
  Citation Validation <span class="toggle-icon">&#9660;</span>
</h2>
<div id="reference-section" class="collapsible-section">
<div class="disclaimer" style="background:#e8f4fd; border:2px solid #2980b9; color:#1a5276;">
  <strong>CITATION VALIDATION — CrossRef / Semantic Scholar / OpenAlex</strong><br>
  Verifies whether references cited in the text actually exist in academic databases.
  AI-generated papers frequently contain <strong>fabricated citations</strong> — plausible-sounding
  but non-existent references — or <strong>chimeric citations</strong> that mix real authors with
  fabricated titles.
</div>
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:180px;">
    <div class="metric-value" style="color:{rf};">{score:.1%}</div>
    <div class="metric-label">Citation Risk Score</div>
  </div>
  <div class="metric" style="width:280px; background:{rb}; border:2px solid {rb2};">
    <div class="metric-value" style="color:{rf}; font-size:16px;">{risk_level}</div>
    <div class="metric-label">Classification</div>
  </div>
  <div class="metric" style="width:100px;">
    <div class="metric-value">{total_refs}</div>
    <div class="metric-label">Total Refs</div>
  </div>
  <div class="metric" style="width:100px;">
    <div class="metric-value" style="color:#27ae60;">{verified_count}</div>
    <div class="metric-label">Verified</div>
  </div>
  <div class="metric" style="width:100px;">
    <div class="metric-value" style="color:#e74c3c;">{fab_count}</div>
    <div class="metric-label">Fabricated</div>
  </div>
  <div class="metric" style="width:100px;">
    <div class="metric-value" style="color:#e67e22;">{chim_count}</div>
    <div class="metric-label">Chimeric</div>
  </div>
</div>
<p><strong>Professional Interpretation:</strong><br>{interp}</p>
{ref_table}
{feat_table}
<p style="color:#7f8c8d;font-size:12px;"><em>
  References analysed: {total_refs} |
  Verified: {verified_count} | Fabricated: {fab_count} |
  Chimeric: {chim_count} | Ornamental: {orn_count}
</em></p>
</div>
"""

    def _build_executive_summary_html(self, summary):
        if not summary: return ""
        return f"""
<h2 class="section-header" onclick="toggleSection('summary-section')">
  Executive Summary &amp; Key Findings <span class="toggle-icon">&#9660;</span>
</h2>
<div id="summary-section" class="collapsible-section">
<div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-left:5px solid #3498db; padding:20px; border-radius:0 8px 8px 0; margin:15px 0; line-height:1.7; font-size:14px;">
  {summary}
</div>
</div>
"""

    # ═══════════════════════════════════════════════════════════════════
    # export_html — MAIN OUTPUT METHOD (v3.5 overhaul)
    # ═══════════════════════════════════════════════════════════════════

    def export_html(self, report, output_path):
        vc = ("ai" if "AI" in report.verdict else "human" if "Human" in report.verdict
              else "hybrid" if "Hybrid" in report.verdict else "inconclusive")

        # [FIX v3.7] Show only the relevant score based on verdict
        if "AI" in report.verdict:
            verdict_score_line = f"AI Score: {report.confidence*100:.1f}%"
        elif "Human" in report.verdict:
            verdict_score_line = f"Human Score: {(1-report.confidence)*100:.1f}%"
        else:
            verdict_score_line = (f"Human Score: {(1-report.confidence)*100:.1f}%"
                                  f" &nbsp;|&nbsp; AI Score: {report.confidence*100:.1f}%")

        # [FIX v3.6] Evidence rendering — human-friendly cards instead of raw JSON
        evhtml = ""
        for e in report.evidence_points[:8]:
            etype = e.get("type", "unknown").replace("_", " ").title()
            expl  = e.get("explanation", "")
            # Build detail line from non-meta fields
            detail_items = []
            for k, v in e.items():
                if k in ("type", "explanation", "indicators"):
                    continue
                if isinstance(v, float):
                    detail_items.append(f"<strong>{k.replace('_',' ').title()}:</strong> {v:.4f}")
                elif isinstance(v, list):
                    detail_items.append(f"<strong>{k.replace('_',' ').title()}:</strong> {', '.join(str(x) for x in v[:5])}")
                elif isinstance(v, str) and len(v) < 200:
                    detail_items.append(f"<strong>{k.replace('_',' ').title()}:</strong> {v}")
            detail_line = " &nbsp;|&nbsp; ".join(detail_items) if detail_items else ""

            evhtml += (
                f'<div class="evidence">'
                f'<span class="evidence-title">{etype}</span><br>'
                f'{f"<p style=&quot;font-size:13px;margin:8px 0 4px;&quot;>{expl}</p>" if expl else ""}'
                f'{f"<p style=&quot;font-size:11px;color:#666;margin:4px 0;&quot;>{detail_line}</p>" if detail_line else ""}'
                f'</div>'
            )

        # [FIX v3.9] Sentence breakdown: top 5 by AI score, deduplicated explanations.
        # Only show sentences with unique explanations — avoids 53 identical rows.
        sorted_sents = sorted(report.sentence_attributions, key=lambda s: s.ai_score, reverse=True)
        seen_explanations = set()
        srows = ""
        shown = 0
        for s in sorted_sents:
            if shown >= 5:
                break
            explanation = _explain_sentence(s)
            # Skip if we already showed an identical explanation
            if explanation in seen_explanations and shown >= 2:
                continue
            seen_explanations.add(explanation)
            shown += 1
            score_pct = s.ai_score * 100
            # Color the score cell
            if score_pct >= 70:
                sc_bg = "rgba(231,76,60,0.15)"
            elif score_pct >= 40:
                sc_bg = "rgba(243,156,18,0.15)"
            else:
                sc_bg = "rgba(39,174,96,0.15)"
            srows += (
                f'<tr class="hoverable-row">'
                f'<td>{s.position+1}</td>'
                f'<td style="background:{sc_bg};text-align:center;font-weight:700;">{score_pct:.0f}%</td>'
                f'<td style="max-width:350px;">{s.text[:120]}{"..." if len(s.text) > 120 else ""}</td>'
                f'<td style="font-size:12px;">{explanation}</td></tr>\n'
            )

        hm_img = (f'<img src="data:image/png;base64,{report.heatmap_b64}" alt="Heatmap">' if report.heatmap_b64 else "<p><em>Not generated</em></p>")
        sc_img = (f'<img src="data:image/png;base64,{report.confidence_chart_b64}" alt="Sentence Chart">' if report.confidence_chart_b64 else "<p><em>Not generated</em></p>")
        cc_img = (f'<img src="data:image/png;base64,{report.comparison_chart_b64}" alt="Comparison">' if report.comparison_chart_b64 else "<p><em>Not generated</em></p>")

        # Build section HTMLs
        hal_sec  = self._build_hallucination_html(report.hallucination_risk)
        rsn_sec  = self._build_reasoning_html(report.reasoning_analysis)
        styl_sec = self._build_stylometric_html(report.stylometric_stats)
        ppl_sec  = self._build_perplexity_html(report.perplexity_analysis)
        hyb_sec  = self._build_hybrid_heatmap_html(report.hybrid_analysis)
        ref_sec  = self._build_reference_html(report.reference_analysis)
        exec_sec = self._build_executive_summary_html(report.executive_summary)

        # ── Score cards with animated gauges ──
        def _gauge(val, label, color=None):
            pct = val * 100
            if color is None:
                color = "#e74c3c" if pct >= 70 else "#f39c12" if pct >= 40 else "#27ae60"
            return f"""
            <div class="score-card">
              <div class="gauge-ring" style="--pct:{pct:.0f}; --clr:{color};">
                <span class="gauge-val">{pct:.0f}%</span>
              </div>
              <div class="metric-label">{label}</div>
            </div>"""

        # [NEW v3.7] Extract perplexity score for gauge
        ppl_gauge_score = 0.0
        if report.perplexity_analysis:
            ppl_gauge_score = report.perplexity_analysis.get("ai_score", 0.0)

        # [NEW v3.9] Extract hybrid segment score for gauge
        hyb_gauge_score = 0.0
        if report.hybrid_analysis:
            hyb_gauge_score = report.hybrid_analysis.get("global_ai_score", 0.0) / 100.0

        # [NEW v3.9] Extract reference score for gauge
        ref_gauge_score = 0.0
        if report.reference_analysis:
            ref_gauge_score = report.reference_analysis.get("ai_score", 0.0)

        score_cards = (
            _gauge(report.neural_score, "Neural Score")
            + _gauge(report.statistical_score, "Statistical")
            + _gauge(report.reasoning_score, "Reasoning")
            + _gauge(ppl_gauge_score, "Perplexity")
            + _gauge(hyb_gauge_score, "Segment AI")
            + _gauge(ref_gauge_score, "Citations")
            + _gauge(report.watermark_score, "Watermark")
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>AI Detection Forensic Report \u2014 {report.report_id}</title>
<style>
/* ── Base ─────────────────────────────────────────── */
*{{box-sizing:border-box;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;padding:40px;background:#f0f2f5;color:#2c3e50;line-height:1.6;}}
.container{{max-width:960px;margin:0 auto;background:white;padding:0;box-shadow:0 4px 24px rgba(0,0,0,.08);border-radius:12px;overflow:hidden;}}

/* ── Header ───────────────────────────────────────── */
.report-header{{background:linear-gradient(135deg,#1a2a3a 0%,#2c3e50 50%,#34495e 100%);color:white;padding:40px;}}
.report-header h1{{margin:0 0 10px;font-size:28px;font-weight:700;letter-spacing:-0.5px;}}
.report-header .meta{{font-size:13px;opacity:0.8;}}
.report-header .meta strong{{color:#3498db;}}

/* ── Content ──────────────────────────────────────── */
.content{{padding:30px 40px 40px;}}
h2{{color:#2c3e50;margin-top:35px;font-size:20px;border-bottom:2px solid #3498db;padding-bottom:8px;}}
h3{{color:#34495e;margin-top:20px;font-size:16px;}}

/* ── Section toggle ───────────────────────────────── */
.section-header{{cursor:pointer;user-select:none;transition:color 0.2s;position:relative;}}
.section-header:hover{{color:#3498db;}}
.toggle-icon{{float:right;font-size:14px;transition:transform 0.3s;}}
.collapsible-section{{max-height:4000px;overflow:hidden;transition:max-height 0.5s ease-in-out, opacity 0.3s;opacity:1;}}
.collapsible-section.collapsed{{max-height:0;opacity:0;padding:0;margin:0;}}

/* ── Verdict ──────────────────────────────────────── */
.verdict{{font-size:22px;padding:20px;border-radius:10px;text-align:center;margin:20px 0;font-weight:700;animation:fadeSlideIn 0.6s ease-out;}}
.verdict.ai{{background:linear-gradient(135deg,#fee 0%,#fdd 100%);border:2px solid #e74c3c;color:#c0392b;}}
.verdict.human{{background:linear-gradient(135deg,#efe 0%,#dfd 100%);border:2px solid #2ecc71;color:#27ae60;}}
.verdict.hybrid{{background:linear-gradient(135deg,#fef 0%,#edf 100%);border:2px solid #9b59b6;color:#8e44ad;}}
.verdict.inconclusive{{background:#eee;border:2px solid #95a5a6;color:#7f8c8d;}}

/* ── Score cards with animated gauge rings ─────────── */
.score-cards{{display:flex;justify-content:center;flex-wrap:wrap;gap:20px;margin:25px 0;}}
.score-card{{text-align:center;padding:15px;animation:fadeSlideIn 0.5s ease-out;}}
.gauge-ring{{
  width:90px;height:90px;border-radius:50%;margin:0 auto 8px;
  background:conic-gradient(var(--clr) calc(var(--pct) * 1%), #ecf0f1 0);
  display:flex;align-items:center;justify-content:center;
  position:relative;
  animation:gaugeIn 1s ease-out;
}}
.gauge-ring::after{{
  content:'';position:absolute;width:70px;height:70px;border-radius:50%;background:white;
}}
.gauge-val{{
  position:relative;z-index:1;font-size:18px;font-weight:700;color:var(--clr);
}}

/* ── Metrics ──────────────────────────────────────── */
.metric{{display:inline-block;width:150px;padding:15px;margin:10px;background:#f8f9fa;border-radius:8px;text-align:center;transition:transform 0.2s,box-shadow 0.2s;}}
.metric:hover{{transform:translateY(-3px);box-shadow:0 4px 12px rgba(0,0,0,0.1);}}
.metric-value{{font-size:28px;font-weight:bold;color:#2c3e50;}}
.metric-label{{font-size:11px;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;}}

/* ── Charts ───────────────────────────────────────── */
.chart{{text-align:center;margin:20px 0;}} .chart img{{max-width:100%;border:1px solid #e0e0e0;border-radius:8px;transition:transform 0.2s;}}
.chart img:hover{{transform:scale(1.02);}}

/* ── Evidence ─────────────────────────────────────── */
.evidence{{background:#fff3cd;border:1px solid #ffc107;padding:15px;margin:10px 0;border-radius:8px;transition:transform 0.2s;}}
.evidence:hover{{transform:translateX(4px);}}
.evidence pre{{white-space:pre-wrap;font-size:12px;margin:6px 0 0;}}
.evidence-title{{font-weight:bold;color:#856404;}}

/* ── Disclaimer ───────────────────────────────────── */
.disclaimer{{background:#fff3cd;border:2px solid #fd7e14;padding:16px;border-radius:8px;margin:20px 0;font-size:13px;}}

/* ── Tables ───────────────────────────────────────── */
table{{width:100%;border-collapse:collapse;margin:15px 0;}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #e0e0e0;font-size:13px;}}
th{{background:linear-gradient(135deg,#2c3e50,#34495e);color:white;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:0.3px;position:sticky;top:0;}}
.interactive-table tbody tr{{transition:background 0.15s,transform 0.15s;}}
.interactive-table tbody tr:hover,.hoverable-row:hover{{background:#f0f7ff;transform:translateX(3px);}}

/* ── Animated progress bars ───────────────────────── */
.animated-bar{{animation:barGrow 0.8s ease-out;}}

/* ── Footer ───────────────────────────────────────── */
.footer{{text-align:center;color:#7f8c8d;font-size:12px;margin-top:40px;padding:25px 40px;border-top:1px solid #e0e0e0;background:#f8f9fa;}}

/* ── Animations ───────────────────────────────────── */
@keyframes fadeSlideIn{{from{{opacity:0;transform:translateY(15px);}}to{{opacity:1;transform:translateY(0);}}}}
@keyframes gaugeIn{{from{{background:conic-gradient(var(--clr) 0%,#ecf0f1 0);}}}}
@keyframes barGrow{{from{{width:0!important;}}}}

/* ── Print ────────────────────────────────────────── */
@media print{{
  body{{padding:0;background:white;}}
  .container{{box-shadow:none;}}
  .section-header{{cursor:default;}}
  .collapsible-section{{max-height:none!important;opacity:1!important;}}
  .toggle-icon{{display:none;}}
}}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="report-header">
  <h1>AI Detection Forensic Report</h1>
  <div class="meta">
    <strong>Report ID:</strong> {report.report_id} &nbsp;|&nbsp;
    <strong>Generated:</strong> {report.generated_at} &nbsp;|&nbsp;
    <strong>Text Hash:</strong> {report.text_hash} &nbsp;|&nbsp;
    <strong>Words:</strong> {report.word_count}
  </div>
</div>

<div class="content">

<!-- Disclaimer -->
<div class="disclaimer"><strong>RESEARCH OUTPUT ONLY</strong><br>{_FORENSIC_DISCLAIMER}</div>

<!-- Verdict -->
<div class="verdict {vc}"><strong>VERDICT: {report.verdict.upper()}</strong><br>{verdict_score_line}</div>

<!-- Executive Summary -->
{exec_sec}

<!-- Score Cards (animated gauge rings) -->
<h2>Detection Scores</h2>
<div class="score-cards">
{score_cards}
</div>

<!-- Charts -->
<h2 class="section-header" onclick="toggleSection('heatmap-section')">
  Word-Level AI Patterns <span class="toggle-icon">&#9660;</span>
</h2>
<div id="heatmap-section" class="collapsible-section"><div class="chart">{hm_img}</div>
  <p style="font-size:0.8em;color:#888;font-style:italic;margin-top:6px">{ATTRIBUTION_DISCLAIMER}</p></div>

<h2 class="section-header" onclick="toggleSection('sentence-chart-section')">
  Sentence-by-Sentence AI Scores <span class="toggle-icon">&#9660;</span>
</h2>
<div id="sentence-chart-section" class="collapsible-section"><div class="chart">{sc_img}</div></div>

<h2 class="section-header" onclick="toggleSection('comparison-section')">
  Comparison vs. Typical Human Writing <span class="toggle-icon">&#9660;</span>
</h2>
<div id="comparison-section" class="collapsible-section"><div class="chart">{cc_img}</div></div>

<!-- Stylometric Analysis [NEW v3.5] -->
{styl_sec}

<!-- Hallucination [IMPROVED v3.5] -->
{hal_sec}

<!-- Reasoning [FIXED v3.5] -->
{rsn_sec}

<!-- Perplexity Analysis [NEW v3.7] -->
{ppl_sec}

<!-- Hybrid Segment Heatmap [NEW v3.9] -->
{hyb_sec}

<!-- Citation Validation [NEW v3.9] -->
{ref_sec}

<!-- Key Evidence -->
<h2 class="section-header" onclick="toggleSection('evidence-section')">
  Key Evidence <span class="toggle-icon">&#9660;</span>
</h2>
<div id="evidence-section" class="collapsible-section">{evhtml if evhtml else '<p style="color:#7f8c8d;"><em>No high-confidence evidence points detected.</em></p>'}</div>

<!-- Sentence Breakdown [v3.9 — top 5, deduplicated, Why Suspicious column] -->
<h2 class="section-header" onclick="toggleSection('breakdown-section')">
  Most Suspicious Sentences (Top 5) <span class="toggle-icon">&#9660;</span>
</h2>
<div id="breakdown-section" class="collapsible-section">
<p style="color:#7f8c8d;font-size:12px;">Showing the sentences with the highest AI probability scores.
Only unique patterns shown — duplicate explanations are merged.</p>
<table class="interactive-table"><thead><tr>
  <th style="width:5%;">#</th><th style="width:8%;">AI Score</th><th style="width:50%;">Sentence</th><th>Why Suspicious?</th>
</tr></thead><tbody>{srows}</tbody></table>
</div>

<!-- What To Do With This Information [NEW v3.9] -->
<h2 class="section-header" onclick="toggleSection('action-section')">
  What To Do With This Information <span class="toggle-icon">&#9660;</span>
</h2>
<div id="action-section" class="collapsible-section">
<div style="background:linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%); border-left:5px solid #2980b9; padding:20px; border-radius:0 8px 8px 0; margin:15px 0; line-height:1.8; font-size:14px;">

<p style="font-weight:700;font-size:15px;margin-bottom:12px;">Recommended Steps for Educators</p>

<p><strong>1. Do not act automatically.</strong> A high AI score does not equal a proven violation.
This tool provides <em>indicators</em>, not proof. AI detection has known limitations —
false positives occur, especially with formal academic writing.</p>

<p><strong>2. Talk to the student.</strong> Ask about their writing process:
Did they take notes or make drafts beforehand? Can they explain their main arguments
in their own words? Do they have sources they can point to? A conversation often reveals
more than any automated tool.</p>

<p><strong>3. Consider the context.</strong> Is this work consistent with the student's
usual performance? Is the level of writing unusually polished or different from their
previous submissions? Does the student typically write in a formal or informal style?</p>

<p><strong>4. Consider assisted use.</strong> The student may have used AI as a starting
point and then edited the result. Depending on your institution's policies, this may
or may not be a violation. Check your academic integrity guidelines.</p>

<p><strong>5. Document.</strong> If you decide to begin a formal review process, keep this
report as <em>supporting evidence</em> — not as the sole basis for any decision.
Combine it with other evidence (drafts, version history, conversation notes).</p>

</div>
</div>

</div><!-- .content -->

<!-- Footer -->
<div class="footer">
  <p>XplagiaX SOTA AI Detector v3.9 | Word Count: {report.word_count} | {report.generated_at}</p>
  <p>{_FORENSIC_DISCLAIMER}</p>
</div>

</div><!-- .container -->

<!-- Section toggle script -->
<script>
function toggleSection(id) {{
  var el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('collapsed');
  // Rotate toggle icon
  var header = el.previousElementSibling;
  if (!header) return;
  var icon = header.querySelector('.toggle-icon');
  if (icon) {{
    icon.style.transform = el.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
  }}
}}
</script>
</body></html>"""

        with open(output_path, "w", encoding="utf-8") as f: f.write(html)
        logger.info("HTML report exported: %s", output_path)
        return output_path

    def export_json(self, report, output_path):
        data = {"report_id": report.report_id, "generated_at": report.generated_at,
            "text_hash": report.text_hash, "word_count": report.word_count,
            "verdict": report.verdict, "confidence": report.confidence,
            "disclaimer": _FORENSIC_DISCLAIMER,
            "scores": {"neural": report.neural_score, "statistical": report.statistical_score,
                "stylometric": report.stylometric_score, "reasoning": report.reasoning_score,
                "watermark": report.watermark_score,
                "perplexity": report.perplexity_analysis.get("ai_score", 0.0) if report.perplexity_analysis else 0.0,
                "segment_ai": report.hybrid_analysis.get("global_ai_score", 0.0) / 100.0 if report.hybrid_analysis else 0.0,
                "citations": report.reference_analysis.get("ai_score", 0.0) if report.reference_analysis else 0.0},
            "evidence_points": report.evidence_points,
            "hallucination_risk": report.hallucination_risk,
            "stylometric_stats": report.stylometric_stats,
            "executive_summary": report.executive_summary,
            "sentence_scores": [{"position": s.position, "score": s.ai_score,
                "indicators": s.key_indicators,
                "explanation": _explain_sentence(s)} for s in report.sentence_attributions]}
        if report.reasoning_analysis is not None:
            ra = report.reasoning_analysis
            data["reasoning_analysis"] = {
                "ai_score": ra.get("ai_score", 0.0), "risk_level": ra.get("risk_level", "N/A"),
                "interpretation": ra.get("interpretation", ""),
                "group_scores": ra.get("group_scores", {}),
                "top_signals": [{"feature": s["feature"], "display_name": s["display_name"],
                    "raw_value": s["raw_value"], "level": s["level"],
                    "explanation": s.get("explanation", "")} for s in ra.get("top_signals", [])],
                "feature_values": {k: v["value"] for k, v in ra.get("feature_details", {}).items()}}
        if report.perplexity_analysis is not None:
            pa = report.perplexity_analysis
            data["perplexity_analysis"] = {
                "ai_score": pa.get("ai_score", 0.0),
                "risk_level": pa.get("risk_level", "N/A"),
                "tier": pa.get("tier", "tier1"),
                "interpretation": pa.get("interpretation", ""),
                "feature_values": pa.get("feature_values", {}),
                "window_count": pa.get("window_count", 0),
            }
        if report.hybrid_analysis is not None:
            data["hybrid_analysis"] = report.hybrid_analysis
        if report.reference_analysis is not None:
            ra = report.reference_analysis
            data["reference_analysis"] = {
                "ai_score": ra.get("ai_score", 0.0),
                "risk_level": ra.get("risk_level", "N/A"),
                "interpretation": ra.get("interpretation", ""),
                "feature_values": ra.get("feature_values", {}),
                "references": ra.get("references", []),
            }
        with open(output_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        logger.info("JSON report exported: %s", output_path)
        return output_path
