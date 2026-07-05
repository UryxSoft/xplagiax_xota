"""
app/plugins/base.py — Abstract base class for all analysis plugins.

Every plugin MUST implement:
    name()        → canonical identifier (lowercase, no spaces)
    description() → human-readable one-liner
    analyze(text) → dict with plugin-specific results

Optional override:
    warmup()      → called once at import to pre-load models/data
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePlugin(ABC):
    """Contract that every analysis plugin must satisfy."""

    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier (e.g. 'sentiment', 'keyphrases')."""
        ...

    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description."""
        ...

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Run the plugin's analysis on the input text.

        Parameters
        ----------
        text : str
            Raw input text (can be very large — handle accordingly).

        Returns
        -------
        dict
            Plugin-specific results.  Must be JSON-serialisable.
        """
        ...

    def warmup(self) -> None:
        """
        Optional — pre-load heavy models/data.

        Called during plugin instantiation (at import time under
        gunicorn --preload) so the data lives in the parent process
        and is shared via CoW across all workers.
        """
        pass

    def health(self) -> bool:
        """
        Optional — report whether the plugin's heavy backend loaded successfully.

        Pure-Python plugins are always healthy (default True). Plugins that wrap a
        model/engine should override this to return their module-level availability
        flag, so /ready can fail honestly instead of reporting a plugin as "ready"
        while its model silently failed to load (audit C-09/C-10/C-11).
        """
        return True

    def is_core(self) -> bool:
        """
        Optional — True if this plugin is required for the service to be considered
        ready (e.g. the primary AI detector). If any core plugin is unhealthy, /ready
        returns 503. Default False.
        """
        return False
