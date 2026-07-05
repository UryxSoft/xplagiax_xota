"""
app/plugins/watermark_detection.py — Digital watermark detection.

Wraps WatermarkDecoder to detect presence of statistical watermarks often
embedded by AI labs in Large Language Model outputs.
"""

import logging
import os
from typing import Any, Dict

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

_decoder = None
_available = False

try:
    import torch
    from app.engine.watermark_decoder import WatermarkDecoder
    _decoder = WatermarkDecoder()
    _available = True
    logger.info("WatermarkDecoder loaded")
except Exception as exc:
    logger.warning("WatermarkDecoder not available: %s", exc)


class WatermarkDetectionPlugin(BasePlugin):

    def name(self) -> str:
        return "watermark_detection"

    def health(self) -> bool:
        return _available

    def description(self) -> str:
        return (
            "Detect statistical watermarks embedded in text by AI models "
            "(requires significant text length for high confidence)."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        if not _available:
            return {"error": "WatermarkDecoder not loaded."}

        sig = _decoder.detect(text)
        return sig.to_forensic_dict()
