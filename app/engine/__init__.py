"""
app/engine/__init__.py — XplagiaX AI Detection Engine.

This package contains the core detection files UNMODIFIED from
the XplagiaX project.  They import each other directly (e.g.
`from detector_final import classify_text`), so we add this
directory to sys.path to preserve those internal imports.

Files in this directory:
    detector_final.py           4-model ModernBERT ensemble
    forensic_reports.py         HTML/JSON forensic report generator (v3.9)
    plugin_orchestrator.py      Pipeline coordinator
    perplexity_profiler.py      Token-level perplexity analysis
    hybrid_segment_detector.py  Per-paragraph AI/human heatmap
    reference_validator.py      Citation existence verification
    stylometric_profiler.py     Writing style fingerprinting
    hallucination_profile.py    Fabrication risk detection
    reasoning_profiler.py       Reasoning-model detection
    watermark_decoder.py        Digital watermark detection
"""
 
import os
import sys

# ── 1. Add engine dir to sys.path ─────────────────────────────
# DT-08: Engine files use bare imports (e.g. `from stylometric_profiler import ...`)
# because they were originally standalone inference scripts. This sys.path injection
# is the intentional adapter that makes bare imports resolve correctly when the
# engine package is imported as part of the Flask app. Do NOT change engine files
# to `from app.engine.*` imports — that would break standalone inference usage.
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
    if hasattr(sys.modules[__name__], '__path__'):
        pkg_path = sys.modules[__name__].__path__[0]
        if pkg_path not in sys.path:
            sys.path.insert(0, pkg_path)

import logging
_logger = logging.getLogger(__name__)
_logger.info("Engine directory added to sys.path: %s", _ENGINE_DIR)

# ── 2. Verify torch is importable ─────────────────────────────
# With torch>=2.4 and transformers>=4.48 we no longer need to monkey-patch
# is_torch_available(). Log a warning if torch fails to import so the
# engine degrades gracefully rather than crashing the whole app.
try:
    import torch  # noqa: F401
    _logger.info("PyTorch %s loaded on engine init.", torch.__version__)
except ImportError:
    _logger.warning(
        "torch not installed — engine will run in CPU-only / limited mode."
    )
