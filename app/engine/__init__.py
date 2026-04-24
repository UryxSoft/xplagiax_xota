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
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
    # Ensure it's also available via the package path
    if hasattr(sys.modules[__name__], '__path__'):
        pkg_path = sys.modules[__name__].__path__[0]
        if pkg_path not in sys.path:
            sys.path.insert(0, pkg_path)

import logging
_logger = logging.getLogger(__name__)
_logger.info("Engine directory added to sys.path: %s", _ENGINE_DIR)
 
# ── 2. Ensure transformers sees torch as available ────────────
# transformers >=4.48 has a lazy-loader that checks is_torch_available()
# via BACKENDS_MAPPING at attribute-access time.  With torch 2.2.2 the
# stdlib version check is fine (_torch_available is True), but we
# reinforce the guard so reloads under Werkzeug can't hit a stale state.
# NOTE: do NOT fake get_torch_version() — code in utils/generic.py uses
# that function to gate torch 2.2+ features; lying about the version
# would cause AttributeErrors when it tries to use 2.4+ APIs on 2.2.2.
try:
    import torch  # noqa
    import transformers.utils.import_utils as _tiu
    # Force is_torch_available to always return True
    _tiu.is_torch_available = lambda: True
    if hasattr(_tiu, "_torch_available"):
        _tiu._torch_available = True
    # Patch BACKENDS_MAPPING so lazy attribute access never blocks on torch
    if hasattr(_tiu, "BACKENDS_MAPPING") and "torch" in _tiu.BACKENDS_MAPPING:
        original = _tiu.BACKENDS_MAPPING["torch"]
        _tiu.BACKENDS_MAPPING["torch"] = (lambda: True, original[1])
except Exception:
    pass
