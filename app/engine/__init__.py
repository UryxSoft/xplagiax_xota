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

# Add this directory to sys.path so XplagiaX modules can import
# each other with their original `from detector_final import ...` style.
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
