"""
Tests for the plugin registry's shared executor + global request deadline
([C6]/[C8]). Uses stub plugins — no model weights or Flask app required.
"""

import time

import pytest

from app.plugin_registry import PluginRegistry
from app.plugins.base import BasePlugin


class _FastPlugin(BasePlugin):
    def name(self):
        return "fast"

    def description(self):
        return "returns instantly"

    def analyze(self, text):
        return {"ok": True}


class _SlowPlugin(BasePlugin):
    """Sleeps far past the test budget."""

    def name(self):
        return "slow"

    def description(self):
        return "sleeps 5s"

    def analyze(self, text):
        time.sleep(5)
        return {"ok": True}


class _BrokenPlugin(BasePlugin):
    def name(self):
        return "broken"

    def description(self):
        return "always raises"

    def analyze(self, text):
        raise ValueError("boom")


@pytest.fixture()
def registry():
    r = PluginRegistry()
    r.register(_FastPlugin())
    r.register(_SlowPlugin())
    r.register(_BrokenPlugin())
    return r


def test_fast_plugin_returns_ok(registry):
    results = registry.run(["fast"], "text", timeout=2)
    assert results["fast"]["status"] == "ok"
    assert results["fast"]["data"] == {"ok": True}


def test_broken_plugin_reports_error_not_500(registry):
    results = registry.run(["broken"], "text", timeout=2)
    assert results["broken"]["status"] == "error"
    assert "boom" in results["broken"]["error"]


def test_deadline_returns_before_slow_plugin_finishes(registry):
    """[C8] The response must come back at the budget, not after the slow
    plugin finishes — the old per-request pool blocked on shutdown(wait=True)."""
    t0 = time.perf_counter()
    results = registry.run(["fast", "slow"], "text", timeout=1)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"response blocked past the deadline ({elapsed:.1f}s)"
    assert results["fast"]["status"] == "ok"
    assert results["slow"]["status"] == "error"
    assert "timed out" in results["slow"]["error"]


def test_unknown_plugin_reports_availability(registry):
    results = registry.run(["nope"], "text", timeout=1)
    assert "error" in results["nope"]
    assert "fast" in results["nope"]["available"]


def test_run_stream_yields_fast_result_despite_slow_sibling(registry):
    seen = dict(registry.run_stream(["fast", "slow"], "text", timeout=1))
    assert seen["fast"]["status"] == "ok"
    assert seen["slow"]["status"] == "error"
