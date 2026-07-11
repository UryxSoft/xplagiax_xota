"""
plugin_orchestrator.py  (xota_ensemble_v6 — inference/)
=========================================================
Thin coordination layer. Contains NO plugin business logic.

Responsibility
--------------
  1. Load plugins once at startup (controlled by PluginConfig flags).
  2. Call each plugin's existing public method(s) exactly as documented.
  3. Assemble the additional_analyses dict for ForensicReportGenerator.
  4. Call ForensicReportGenerator.generate_report() and export.

What this file does NOT do
---------------------------
  * Does not re-implement any plugin algorithm.
  * Does not subclass any plugin.
  * Does not contain classification logic from any plugin module.

The only computation here is _compute_reasoning_score() — a minimal
weighted aggregation converting ReasoningProfiler's raw 15-dim vector
into a single float. This is necessary because ReasoningProfiler is a
pure extractor by design (no classifier). The weights and thresholds
are local constants in this file, not borrowed from any plugin.

Plugin call map
---------------
  Plugin                        Method(s) called
  ─────────────────────────────────────────────────────────────────
  StylometricProfiler           .compute_stats(text)
  HallucinationProfiler         passed to ForensicReportGenerator.__init__
  HallucinationRiskClassifier   passed to ForensicReportGenerator.__init__
  ReasoningProfiler             .vectorize(text), .feature_names()
  PerplexityProfiler            .compute_stats(text)  [NEW v3.7]
  PerplexityRiskClassifier      .classify(stats)      [NEW v3.7]
  HybridSegmentAnalyzer         .analyze(text)        [NEW v3.9]
  ReferenceValidator            .compute_stats(text)  [NEW v3.9]
  ReferenceRiskClassifier       .classify(stats)      [NEW v3.9]
  WatermarkDecoder              .detect(text) -> .to_forensic_dict()
  ForensicReportGenerator       .generate_report(...) -> .export_html/json()

Changelog (v3.4 -> v3.5)
-------------------------
  [FIX]  ForensicReportGenerator now receives reasoning_profiler via
         __init__ so _build_reasoning_html() can populate group_scores,
         top_signals, and feature_details tables (previously empty).
  [FIX]  Reasoning dict produced by orchestrator now uses
         ReasoningRiskClassifier.classify() (from forensic_reports.py)
         to generate the FULL reasoning analysis structure instead of
         the partial {ai_score, risk_level, feature_values} dict.

Usage
-----
    from plugin_orchestrator import PluginOrchestrator, PluginConfig
    from detector_final import classify_text

    # Full pipeline from raw text
    orch   = PluginOrchestrator(PluginConfig(enable_watermark=True))
    result = orch.run("Paste text here...")

    # Pre-computed detection (avoids re-running the 4 models)
    msg, fig, det = classify_text("Paste text here...")
    result        = orch.run_with_result("Paste text here...", det)

    print(orch.summary(result))
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _unique_report_path() -> str:
    """DT-12: unique per-instance path prevents concurrent report overwrites."""
    report_dir = os.path.join(tempfile.gettempdir(), "xplagiax_reports")
    return os.path.join(report_dir, f"forensic_{uuid.uuid4().hex[:8]}.html")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PluginConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PluginConfig:
    """
    Activation flags for every plugin in the pipeline.

    Parameters
    ----------
    enable_stylometric      : Run StylometricProfiler.compute_stats().
    enable_hallucination    : Pass HallucinationProfiler + Classifier to
                              ForensicReportGenerator (runs internally).
    enable_reasoning        : Call ReasoningProfiler.vectorize() and include
                              results in additional_analyses["reasoning"].
    enable_watermark        : Call WatermarkDecoder.detect() on every call.
                              Loads GPT-2 — disabled by default.
    enable_forensic_report  : Generate and export the forensic report.
    forensic_output_path    : File path for the exported report.
    forensic_output_format  : "html" (default) or "json".
    watermark_device        : Torch device string for WatermarkDecoder.
                              None = auto-detect.
    """
    enable_stylometric:     bool = True
    enable_hallucination:   bool = True
    enable_reasoning:       bool = True
    enable_perplexity:      bool = True     # [NEW v3.7] Perplexity profiler
    enable_hybrid_segment:  bool = True     # [NEW v3.9] Per-paragraph heatmap
    enable_reference_check: bool = False    # [NEW v3.9] Citation validator (requires network)
    enable_author_signature: bool = True    # [Tier1] Intra-document authorship consistency
    enable_discourse:        bool = True    # [Tier1] Discourse-structure uniformity (model-agnostic)
    enable_semantic_consistency: bool = True  # [Tier1] Internal-contradiction detection
    enable_watermark:       bool = False
    enable_forensic_report: bool = True
    forensic_output_path:   str  = field(default_factory=_unique_report_path)
    forensic_output_format: str  = "html"
    watermark_device:       Optional[str] = None
    perplexity_dict_path:   Optional[str] = None   # [NEW v3.7] Pre-built n-gram dict
    perplexity_tier2:       bool = True             # [NEW v3.7] Auto-enable GPT-2
    reference_network:      bool = True             # [NEW v3.9] Call CrossRef/S2/OpenAlex APIs


# ═══════════════════════════════════════════════════════════════════════════
# PluginOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class PluginOrchestrator:
    """
    Thin pipeline coordinator. Instantiate once; call run() per text.

    Parameters
    ----------
    config : PluginConfig with activation flags.
    """

    # Weights for _compute_reasoning_score().
    # 9 positive features (sum=0.92) + 1 inverse feature (0.08) = 1.00
    _RSN_WEIGHTS: Dict[str, float] = {
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
    _RSN_HIGH_THRESHOLDS: Dict[str, float] = {
        "backtracking_density":    0.07,
        "cot_scaffold_density":    0.10,
        "consequence_density":     0.06,
        "causal_density":          0.07,
        "sequence_density":        0.05,
        "contrast_density":        0.06,
        "word_entropy_normalised": 0.90,
        "type_token_ratio":        0.72,
        "paragraph_length_cv":     0.55,
        "intuition_leap_density":  0.04,
    }

    def __init__(self, config: Optional[PluginConfig] = None) -> None:
        self.config = config or PluginConfig()
        self._stylometric:              Any = None
        self._hallucination_profiler:   Any = None
        self._hallucination_classifier: Any = None
        self._reasoning_profiler:       Any = None
        self._reasoning_classifier:     Any = None      # [NEW v3.5]
        self._perplexity_profiler:      Any = None      # [NEW v3.7]
        self._perplexity_classifier:    Any = None      # [NEW v3.7]
        self._hybrid_analyzer:          Any = None      # [NEW v3.9]
        self._reference_validator:      Any = None      # [NEW v3.9]
        self._reference_classifier:     Any = None      # [NEW v3.9]
        self._discourse_analyzer:       Any = None      # [Tier1]
        self._semantic_analyzer:        Any = None      # [Tier1]
        self._watermark_decoder:        Any = None
        self._forensic_generator:       Any = None
        self._init_plugins()

    def _init_plugins(self) -> None:
        """Load each enabled plugin once. Import failures are logged and skipped."""
        cfg = self.config

        if cfg.enable_stylometric:
            try:
                from stylometric_profiler import StylometricProfiler
                self._stylometric = StylometricProfiler()
                logger.info("StylometricProfiler loaded")
            except ImportError as exc:
                logger.warning("StylometricProfiler unavailable: %s", exc)

        if cfg.enable_hallucination:
            try:
                from hallucination_profile import (
                    HallucinationProfiler,
                    HallucinationRiskClassifier,
                )
                self._hallucination_profiler   = HallucinationProfiler()
                self._hallucination_classifier = HallucinationRiskClassifier()
                logger.info("HallucinationProfiler + Classifier loaded")
            except ImportError as exc:
                logger.warning("HallucinationProfiler unavailable: %s", exc)

        if cfg.enable_reasoning:
            try:
                from reasoning_profiler import ReasoningProfiler
                self._reasoning_profiler = ReasoningProfiler()
                logger.info("ReasoningProfiler loaded")
            except ImportError as exc:
                logger.warning("ReasoningProfiler unavailable: %s", exc)

            # [NEW v3.5] Load ReasoningRiskClassifier for full analysis
            try:
                from forensic_reports import ReasoningRiskClassifier
                self._reasoning_classifier = ReasoningRiskClassifier()
                logger.info("ReasoningRiskClassifier loaded")
            except ImportError as exc:
                logger.warning("ReasoningRiskClassifier unavailable: %s", exc)

        if cfg.enable_watermark:
            try:
                import torch
                from watermark_decoder import WatermarkDecoder
                device = torch.device(cfg.watermark_device) if cfg.watermark_device else None
                self._watermark_decoder = WatermarkDecoder(device=device)
                logger.info("WatermarkDecoder loaded")
            except ImportError as exc:
                logger.warning("WatermarkDecoder unavailable: %s", exc)

        # [NEW v3.7] PerplexityProfiler + PerplexityRiskClassifier
        if cfg.enable_perplexity:
            try:
                from perplexity_profiler import PerplexityProfiler, PerplexityRiskClassifier
                self._perplexity_profiler = PerplexityProfiler(
                    ngram_dict_path=cfg.perplexity_dict_path,
                    enable_tier2=cfg.perplexity_tier2,
                )
                self._perplexity_classifier = PerplexityRiskClassifier()
                tier = getattr(self._perplexity_profiler, "tier", "tier1")
                logger.info("PerplexityProfiler loaded (%s)", tier)
            except ImportError as exc:
                logger.warning("PerplexityProfiler unavailable: %s", exc)

        # [NEW v3.9] HybridSegmentAnalyzer — per-paragraph AI/human heatmap
        if cfg.enable_hybrid_segment:
            try:
                from hybrid_segment_detector import HybridSegmentAnalyzer
                from detector_final import classify_batch, classify_segment
                self._hybrid_analyzer = HybridSegmentAnalyzer(
                    classify_fn=classify_segment,
                    classify_batch_fn=classify_batch,
                )
                logger.info("HybridSegmentAnalyzer loaded")
            except ImportError as exc:
                logger.warning("HybridSegmentAnalyzer unavailable: %s", exc)

        # [NEW v3.9] ReferenceValidator + ReferenceRiskClassifier
        if cfg.enable_reference_check:
            try:
                from reference_validator import ReferenceValidator, ReferenceRiskClassifier
                self._reference_validator = ReferenceValidator(
                    enable_network=cfg.reference_network,
                )
                self._reference_classifier = ReferenceRiskClassifier()
                logger.info("ReferenceValidator loaded (network=%s)", cfg.reference_network)
            except ImportError as exc:
                logger.warning("ReferenceValidator unavailable: %s", exc)

        if cfg.enable_discourse:
            try:
                from discourse_analyzer import DiscourseAnalyzer
                self._discourse_analyzer = DiscourseAnalyzer()
                logger.info("DiscourseAnalyzer loaded")
            except Exception as exc:  # noqa: BLE001 — pure-Python, but stay defensive
                logger.warning("DiscourseAnalyzer unavailable: %s", exc)

        if cfg.enable_semantic_consistency:
            try:
                from semantic_consistency import SemanticConsistencyAnalyzer
                self._semantic_analyzer = SemanticConsistencyAnalyzer()
                logger.info("SemanticConsistencyAnalyzer loaded")
            except Exception as exc:  # noqa: BLE001
                logger.warning("SemanticConsistencyAnalyzer unavailable: %s", exc)

        if cfg.enable_forensic_report:
            try:
                from forensic_reports import ForensicReportGenerator
                # [CHANGED v3.5] Now passes reasoning_profiler and
                # reasoning_classifier so the generator can build full
                # reasoning HTML with group_scores, top_signals, feature_details.
                self._forensic_generator = ForensicReportGenerator(
                    profiler=self._stylometric,
                    hallucination_profiler=self._hallucination_profiler,
                    hallucination_classifier=self._hallucination_classifier,
                    reasoning_profiler=self._reasoning_profiler,
                    reasoning_classifier=self._reasoning_classifier,
                )
                logger.info("ForensicReportGenerator loaded")
            except ImportError as exc:
                logger.warning("ForensicReportGenerator unavailable: %s", exc)

    def run(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline: call classify_text() then all enabled plugins.

        Returns dict with keys:
            "detection_result"    : DetectionResult
            "additional_analyses" : dict of plugin outputs
            "forensic_report"     : ForensicReport | None
        """
        # [C-04/§13 FIX] Use the document-level aggregate (covers the FULL text)
        # instead of classify_text(), which truncates to the model's ~512-token
        # window and would make the forensic verdict ignore most of a long document.
        from detector_final import classify_text_aggregate
        detection_result = classify_text_aggregate(text)
        return self.run_with_result(text, detection_result)

    def run_with_result(self, text: str, detection_result: Any) -> Dict[str, Any]:
        """
        Run all enabled plugins against a pre-computed DetectionResult.

        Use this when classify_text() has already been called (e.g. in Gradio)
        to avoid re-running the 4-model ensemble.

        detection_result.statistical_features is populated in-place by
        StylometricProfiler so callers can access stats without opening the report.
        """
        additional: Dict[str, Any] = {}

        # ── StylometricProfiler ───────────────────────────────────────
        if self._stylometric is not None:
            try:
                stats = self._stylometric.compute_stats(text)
                detection_result.statistical_features = stats
                logger.debug(
                    "Stylometric: burstiness=%.3f vocab=%.3f hapax=%.3f",
                    stats.get("burstiness", 0.0),
                    stats.get("vocabulary_richness", 0.0),
                    stats.get("hapax_legomena_ratio", 0.0),
                )
            except Exception as exc:
                logger.warning("StylometricProfiler.compute_stats() failed: %s", exc)

        # ── ReasoningProfiler ─────────────────────────────────────────
        # [CHANGED v3.5] Now uses ReasoningRiskClassifier.classify() to
        # produce the FULL reasoning analysis dict (with group_scores,
        # top_signals, feature_details, interpretation) instead of the
        # partial dict that was causing empty tables in the HTML report.
        if self._reasoning_profiler is not None:
            try:
                from reasoning_profiler import FEATURE_NAMES as _RN
                vec        = self._reasoning_profiler.vectorize(text)
                feat_names = self._reasoning_profiler.feature_names()

                if self._reasoning_classifier is not None:
                    # Full classification with group_scores, top_signals, etc.
                    reasoning_analysis = self._reasoning_classifier.classify(
                        vec, feat_names,
                    )
                    additional["reasoning"] = reasoning_analysis
                    logger.debug(
                        "Reasoning (full): score=%.4f level=%s",
                        reasoning_analysis.get("ai_score", 0.0),
                        reasoning_analysis.get("risk_level", "N/A"),
                    )
                else:
                    # Fallback: partial dict (legacy behaviour)
                    feat_values: Dict[str, float] = dict(zip(feat_names, vec.tolist()))
                    ai_score    = self._compute_reasoning_score(feat_values)
                    risk_level  = self._classify_reasoning_risk(ai_score)
                    additional["reasoning"] = {
                        "ai_score":       ai_score,
                        "risk_level":     risk_level,
                        "feature_values": feat_values,
                    }
                    logger.debug("Reasoning (partial): score=%.4f level=%s",
                                 ai_score, risk_level)
            except Exception as exc:
                logger.warning("ReasoningProfiler.vectorize() failed: %s", exc)

        # ── PerplexityProfiler [NEW v3.7] ─────────────────────────────
        if self._perplexity_profiler is not None:
            try:
                ppl_stats = self._perplexity_profiler.compute_stats(text)

                if self._perplexity_classifier is not None:
                    ppl_analysis = self._perplexity_classifier.classify(ppl_stats)
                    # Merge raw stats into the analysis dict
                    ppl_analysis["window_ppls"] = ppl_stats.get("window_ppls", [])
                    ppl_analysis["tokens_analysed"] = ppl_stats.get("tokens_analysed", 0)
                    ppl_analysis["feature_values"] = {
                        k: v for k, v in ppl_stats.items()
                        if isinstance(v, (int, float))
                    }
                    additional["perplexity"] = ppl_analysis
                else:
                    additional["perplexity"] = ppl_stats

                logger.debug(
                    "Perplexity (%s): score=%.4f level=%s ppl_mean=%.2f",
                    ppl_stats.get("tier", "tier1"),
                    additional["perplexity"].get("ai_score", 0.0),
                    additional["perplexity"].get("risk_level", "N/A"),
                    ppl_stats.get("proxy_perplexity_mean", 0.0),
                )
            except Exception as exc:
                logger.warning("PerplexityProfiler.compute_stats() failed: %s", exc)

        # ── HybridSegmentAnalyzer [NEW v3.9] ──────────────────────────
        if self._hybrid_analyzer is not None:
            try:
                hybrid_result = self._hybrid_analyzer.analyze(text)
                additional["hybrid_segment"] = hybrid_result.to_dict()
                logger.debug(
                    "HybridSegment: classification=%s risk=%s ai=%.1f%% "
                    "breakpoints=%d paragraphs=%d windows=%d",
                    hybrid_result.classification,
                    hybrid_result.risk_level,
                    hybrid_result.global_ai_score,
                    len(hybrid_result.breakpoints),
                    hybrid_result.total_paragraphs,
                    hybrid_result.total_windows,
                )
            except Exception as exc:
                logger.warning("HybridSegmentAnalyzer.analyze() failed: %s", exc)

        # ── ReferenceValidator [NEW v3.9] ─────────────────────────────
        if self._reference_validator is not None:
            try:
                ref_stats = self._reference_validator.compute_stats(text)

                if self._reference_classifier is not None:
                    ref_analysis = self._reference_classifier.classify(ref_stats)
                    # Map validation_results to 'references' for HTML builder
                    ref_analysis["references"] = ref_analysis.get("validation_results", [])
                    ref_analysis["feature_values"] = {
                        k: ref_stats[k] for k in ref_stats
                        if isinstance(ref_stats[k], (int, float))
                    }
                    additional["reference_check"] = ref_analysis
                else:
                    additional["reference_check"] = ref_stats

                logger.debug(
                    "ReferenceValidator: score=%.4f level=%s refs=%d fabricated=%d",
                    additional["reference_check"].get("ai_score", 0.0),
                    additional["reference_check"].get("risk_level", "N/A"),
                    ref_stats.get("total_references", 0),
                    ref_stats.get("fabricated_count", 0),
                )
            except Exception as exc:
                logger.warning("ReferenceValidator.compute_stats() failed: %s", exc)

        # ── WatermarkDecoder ──────────────────────────────────────────
        if self._watermark_decoder is not None:
            try:
                sig = self._watermark_decoder.detect(text)
                additional["watermark"] = sig.to_forensic_dict()
                logger.debug(
                    "Watermark: detected=%s confidence=%.4f scheme=%s",
                    sig.detected, sig.confidence, sig.scheme_type,
                )
            except Exception as exc:
                logger.warning("WatermarkDecoder.detect() failed: %s", exc)

        # ── Tier-1 model-agnostic signals (feed the fusion + reported standalone) ──
        # Authorship consistency: embedding engine (LUAR, opt-in via
        # ENABLE_AUTHOR_EMBEDDING=1 — see docs/sota/D_AUTHOR_SIGNATURE.md) with
        # fallback to the stylometric implementation. Both emit `outlier_ratio`,
        # so the fusion feature is source-agnostic.
        if self.config.enable_author_signature:
            _authsig_done = False
            try:
                import author_embedding
                if author_embedding.is_available():
                    _authsig = author_embedding.analyze_document(text)
                    if _authsig.get("status") == "ok":
                        additional["author_signature"] = _authsig
                        _authsig_done = True
            except Exception as exc:
                logger.warning("author embedding failed: %s", exc)
            if not _authsig_done and self._stylometric is not None:
                try:
                    from authorship_consistency import compute_authorship_consistency
                    additional["author_signature"] = compute_authorship_consistency(
                        self._stylometric, text)
                except Exception as exc:
                    logger.warning("authorship_consistency failed: %s", exc)

        if self._discourse_analyzer is not None:
            try:
                additional["discourse_structure"] = self._discourse_analyzer.analyze(text)
            except Exception as exc:
                logger.warning("discourse analysis failed: %s", exc)

        if self._semantic_analyzer is not None:
            try:
                additional["semantic_consistency"] = self._semantic_analyzer.analyze(text)
            except Exception as exc:
                logger.warning("semantic consistency failed: %s", exc)

        # ── LATE FUSION (model-agnostic, bounded, UNCALIBRATED) ───────────
        # Compute the fused P(AI) here, in the pipeline coordinator that owns `additional`,
        # so it is BOTH visible in the returned additional_analyses AND consumed by the
        # forensic verdict. The verdict no longer depends on the neural ensemble alone.
        if os.getenv("FUSION_ACTIVE", "1") == "1" and detection_result is not None:
            try:
                from fusion import FusionClassifier
                _fres = FusionClassifier().predict_proba(detection_result, additional)
                additional["fusion"] = _fres.to_dict()
            except Exception as exc:
                logger.warning("Fusion scoring failed: %s", exc)

        # ── ForensicReportGenerator ───────────────────────────────────
        forensic_report = None
        if self._forensic_generator is not None:
            try:
                forensic_report = self._forensic_generator.generate_report(
                    text=text,
                    detection_result=detection_result,
                    additional_analyses=additional,
                    generate_visuals=True,
                )
                path = self.config.forensic_output_path
                if self.config.forensic_output_format == "json":
                    self._forensic_generator.export_json(forensic_report, path)
                else:
                    self._forensic_generator.export_html(forensic_report, path)
                logger.info(
                    "Forensic report -> %s  verdict=%s  confidence=%.1f%%",
                    path, forensic_report.verdict, forensic_report.confidence * 100,
                )
            except Exception as exc:
                logger.warning("ForensicReportGenerator failed: %s", exc)

        return {
            "detection_result":    detection_result,
            "additional_analyses": additional,
            "forensic_report":     forensic_report,
        }

    # ── Reasoning score helpers ────────────────────────────────────────
    # These exist solely because ReasoningProfiler produces a raw 15-dim
    # vector with no aggregated score. The weights are local constants here,
    # not borrowed from any plugin module.

    @classmethod
    def _compute_reasoning_score(cls, features: Dict[str, float]) -> float:
        """Normalised weighted sum of reasoning marker features -> [0, 1]."""
        score = 0.0
        for feat, weight in cls._RSN_WEIGHTS.items():
            val  = features.get(feat, 0.0)
            high = cls._RSN_HIGH_THRESHOLDS.get(feat, 1.0)
            score += weight * min(1.0, val / max(high, 1e-9))
        # Inverse intuition component (weight 0.08)
        inv_val  = features.get("intuition_leap_density", 0.0)
        inv_high = cls._RSN_HIGH_THRESHOLDS["intuition_leap_density"]
        inv_norm = min(1.0, inv_val / max(inv_high, 1e-9))
        score   += 0.08 * max(0.0, 1.0 - inv_norm)
        return round(min(1.0, max(0.0, score)), 4)

    @staticmethod
    def _classify_reasoning_risk(score: float) -> str:
        if score >= 0.55:
            return "HIGH \u2014 Reasoning Model"
        if score >= 0.28:
            return "MEDIUM \u2014 Possible Reasoning Model"
        return "LOW \u2014 Standard Model or Human"

    # ── Utilities ──────────────────────────────────────────────────────

    def active_plugins(self) -> List[str]:
        """Return names of successfully loaded plugins."""
        active: List[str] = []
        if self._stylometric            is not None: active.append("StylometricProfiler")
        if self._hallucination_profiler is not None: active.append("HallucinationProfiler")
        if self._reasoning_profiler     is not None: active.append("ReasoningProfiler")
        if self._reasoning_classifier   is not None: active.append("ReasoningRiskClassifier")
        if self._perplexity_profiler    is not None: active.append("PerplexityProfiler")
        if self._hybrid_analyzer        is not None: active.append("HybridSegmentAnalyzer")
        if self._reference_validator    is not None: active.append("ReferenceValidator")
        if self._watermark_decoder      is not None: active.append("WatermarkDecoder")
        if self._forensic_generator     is not None: active.append("ForensicReportGenerator")
        return active

    def summary(self, result: Dict[str, Any]) -> str:
        """Plain-text summary of a run() result dict."""
        sep   = "\u2550" * 58
        lines = [sep, "  PLUGIN ORCHESTRATOR \u2014 RESULT SUMMARY", sep]

        det = result.get("detection_result")
        if det is not None:
            unc = "\u26a0 UNCERTAINTY" if det.uncertainty_zone else "\u2713 decisive"
            lines.append(
                f"  Detection  : {det.prediction:6s}  confidence={det.confidence:.1f}%  [{unc}]"
            )
            if det.detected_model:
                lines.append(f"  Likely LLM : {det.detected_model}")
            lines.append(
                f"  Scores     : human={det.raw_scores.get('human',0):.1f}%  "
                f"ai={det.raw_scores.get('ai',0):.1f}%"
            )

        aa = result.get("additional_analyses", {})

        sf = getattr(det, "statistical_features", {}) or {}
        if sf:
            lines += ["", "  Stylometric:",
                f"    burstiness={sf.get('burstiness',0):.3f}  "
                f"vocab={sf.get('vocabulary_richness',0):.3f}  "
                f"hapax={sf.get('hapax_legomena_ratio',0):.3f}"]

        rsn = aa.get("reasoning")
        if rsn:
            lines += ["", "  Reasoning:",
                f"    score={rsn['ai_score']:.4f}  level={rsn['risk_level']}"]
            fv = rsn.get("feature_values") or {}
            if not fv:
                # v3.5 full dict: extract from feature_details
                fd = rsn.get("feature_details", {})
                fv = {k: v.get("value", 0.0) for k, v in fd.items()}
            top = sorted(fv.items(), key=lambda x: x[1], reverse=True)[:3]
            for feat, val in top:
                lines.append(f"    {feat:<36s} = {val:.6f}")

        ppl = aa.get("perplexity")
        if ppl:
            lines += ["", "  Perplexity:",
                f"    score={ppl.get('ai_score',0):.4f}  level={ppl.get('risk_level','N/A')}  "
                f"tier={ppl.get('tier','?')}",
                f"    ppl_mean={ppl.get('feature_values',{}).get('proxy_perplexity_mean',0):.2f}  "
                f"entropy={ppl.get('feature_values',{}).get('token_entropy_mean',0):.2f}  "
                f"windows={ppl.get('window_count',0)}"]

        hyb = aa.get("hybrid_segment")
        if hyb:
            lines += ["", "  Hybrid Segment:",
                f"    classification={hyb.get('classification','N/A')}  "
                f"risk={hyb.get('risk_level','N/A')}",
                f"    global_ai={hyb.get('global_ai_score',0):.1f}%  "
                f"paragraphs={hyb.get('total_paragraphs',0)}  "
                f"windows={hyb.get('total_windows',0)}  "
                f"breakpoints={hyb.get('breakpoint_count',0)}"]

        ref = aa.get("reference_check")
        if ref:
            lines += ["", "  Reference Validation:",
                f"    score={ref.get('ai_score',0):.4f}  level={ref.get('risk_level','N/A')}",
                f"    total_refs={ref.get('feature_values',{}).get('total_references',0):.0f}  "
                f"fabricated={ref.get('feature_values',{}).get('fabricated_count',0):.0f}  "
                f"chimeric={ref.get('feature_values',{}).get('chimeric_count',0):.0f}"]

        wm = aa.get("watermark")
        if wm:
            lines += ["", "  Watermark:",
                f"    detected={wm.get('detected')}  "
                f"confidence={wm.get('confidence',0):.4f}  "
                f"scheme={wm.get('scheme_type','none')}"]

        fr = result.get("forensic_report")
        if fr is not None:
            lines += ["", "  Forensic Report:",
                f"    verdict={fr.verdict}  neural={fr.neural_score:.2f}  "
                f"reasoning={fr.reasoning_score:.2f}  watermark={fr.watermark_score:.2f}",
                f"    saved \u2192 {self.config.forensic_output_path}"]

        lines += ["", f"  Active plugins: {', '.join(self.active_plugins())}", sep]
        return "\n".join(lines)

    def export_html(self, forensic_report: Any, output_path: str) -> None:
        """Export a ForensicReport to HTML. Public wrapper — avoids callers touching _forensic_generator."""
        if self._forensic_generator is None:
            raise RuntimeError("ForensicReportGenerator not loaded in this orchestrator")
        self._forensic_generator.export_html(forensic_report, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton factory
# ═══════════════════════════════════════════════════════════════════════════

_shared_orchestrator: Optional[PluginOrchestrator] = None


def initialize_orchestrator(config: PluginConfig) -> PluginOrchestrator:
    """Create and cache the shared orchestrator. Call once at gunicorn preload."""
    global _shared_orchestrator
    _shared_orchestrator = PluginOrchestrator(config)
    return _shared_orchestrator


def get_orchestrator() -> Optional[PluginOrchestrator]:
    """Return the shared orchestrator, or None if not yet initialized."""
    return _shared_orchestrator
