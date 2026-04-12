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
