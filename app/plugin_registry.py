"""
app/plugin_registry.py — Auto-discovery plugin system.

Plugins are Python modules in app/plugins/ that define a class
inheriting from BasePlugin.  On startup, discover() imports every
module and registers any BasePlugin subclass it finds.

Adding a new plugin:
    1. Create app/plugins/my_plugin.py
    2. Define a class inheriting from BasePlugin
    3. Implement name(), description(), and analyze(text)
    4. Done — it's auto-registered on next startup.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def adaptive_timeout(
    word_count: int,
    base: int = 30,
    per_kwords: float = 15.0,
    cap: int = 3600,
) -> int:
    """Per-plugin timeout scaled to document size.

    A fixed timeout sized for abstracts (30 s) silently breaks thesis-sized
    inputs: the future times out, the result is discarded, but the plugin
    thread keeps running the full CPU-bound inference anyway — worst of both
    worlds. Short texts keep `base`; every 1 000 words adds `per_kwords`
    seconds (CPU ensemble budget); `cap` bounds the total (sync callers pass
    a low cap, Celery passes one under its own soft time limit).
    """
    scaled = base + (max(0, word_count) / 1000.0) * per_kwords
    return int(min(cap, max(base, scaled)))


class PluginRegistry:
    """Thread-safe registry of analysis plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Any] = {}  # name -> instance

    def register(self, plugin_instance: Any) -> None:
        """Register a plugin by its canonical name."""
        name = plugin_instance.name()
        if name in self._plugins:
            logger.warning("Plugin '%s' already registered — skipping duplicate", name)
            return
        self._plugins[name] = plugin_instance
        logger.info("Registered plugin: %s", name)

    def get(self, name: str) -> Optional[Any]:
        """Get a plugin instance by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """Return sorted list of registered plugin names."""
        return sorted(self._plugins.keys())

    def list_plugins_with_info(self) -> List[Dict[str, str]]:
        """Return list of dicts with name + description for /ready."""
        return [
            {"name": p.name(), "description": p.description()}
            for p in sorted(self._plugins.values(), key=lambda x: x.name())
        ]

    def health_report(self) -> Dict[str, bool]:
        """
        Map plugin name -> health() (audit C-09/C-10/C-11).

        A plugin whose heavy backend silently failed to load reports False here, so
        callers (e.g. /ready) can surface degradation instead of pretending the
        plugin is operational just because it registered.
        """
        report: Dict[str, bool] = {}
        for name, p in self._plugins.items():
            try:
                report[name] = bool(p.health())
            except Exception as exc:  # a broken plugin is, by definition, unhealthy
                logger.warning("health() raised for plugin '%s': %s", name, exc)
                report[name] = False
        return report

    def core_unhealthy(self) -> List[str]:
        """Names of core plugins (is_core() True) whose backend failed to load."""
        down: List[str] = []
        for name, p in self._plugins.items():
            try:
                if p.is_core() and not p.health():
                    down.append(name)
            except Exception:
                down.append(name)
        return down

    def run(self, plugin_names: List[str], text: str,
            timeout: int = 30) -> Dict[str, Any]:
        """
        Execute requested plugins in parallel and return aggregated results.

        Returns dict: {plugin_name: {result or error}}
        """
        results: Dict[str, Any] = {}

        valid: List[tuple] = []  # (pname, plugin, t0)
        for pname in plugin_names:
            plugin = self.get(pname)
            if plugin is None:
                results[pname] = {
                    "error": f"Plugin '{pname}' not found",
                    "available": self.list_plugins(),
                }
            else:
                valid.append((pname, plugin))

        if not valid:
            return results

        # P-03: cap raised to 8 — ML plugins release GIL during C-level inference,
        # so additional threads genuinely run in parallel on multi-core machines.
        max_workers = min(len(valid), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all plugins at once so they run in parallel
            future_to_meta: Dict[Any, tuple] = {}
            for pname, plugin in valid:
                t0 = time.perf_counter()
                future_to_meta[executor.submit(plugin.analyze, text)] = (pname, t0)

            # P-07: per-plugin individual timeout — each plugin gets its full budget.
            # A shared deadline would let a slow plugin (e.g. watermark ~15s) starve
            # fast ones that are checked later in iteration order.
            for future, (pname, t0) in future_to_meta.items():
                try:
                    result = future.result(timeout=timeout)
                    elapsed = time.perf_counter() - t0
                    results[pname] = {
                        "status": "ok",
                        "data": result,
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }
                except FuturesTimeoutError:
                    elapsed = time.perf_counter() - t0
                    logger.error("Plugin '%s' timed out after %ds", pname, timeout)
                    results[pname] = {
                        "status": "error",
                        "error": f"Plugin timed out after {timeout}s",
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    logger.error("Plugin '%s' failed: %s", pname, exc, exc_info=True)
                    results[pname] = {
                        "status": "error",
                        "error": str(exc),
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }

        return results

    def run_stream(self, plugin_names: List[str], text: str, timeout: int = 30):
        """
        Yield (plugin_name, result_dict) as each plugin completes.
        Used by /analyze_stream (SSE) to deliver results incrementally.
        """
        # Yield errors for unknown plugins immediately
        valid: List[tuple] = []
        for pname in plugin_names:
            plugin = self.get(pname)
            if plugin is None:
                yield pname, {
                    "status": "error",
                    "error": f"Plugin '{pname}' not found",
                    "available": self.list_plugins(),
                }
            else:
                valid.append((pname, plugin))

        if not valid:
            return

        max_workers = min(len(valid), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta: Dict[Any, tuple] = {}
            for pname, plugin in valid:
                t0 = time.perf_counter()
                future_to_meta[executor.submit(plugin.analyze, text)] = (pname, t0)

            try:
                for future in as_completed(future_to_meta, timeout=timeout):
                    pname, t0 = future_to_meta[future]
                    elapsed = time.perf_counter() - t0
                    try:
                        result = future.result()
                        yield pname, {
                            "status": "ok",
                            "data": result,
                            "elapsed_ms": round(elapsed * 1000, 1),
                        }
                    except Exception as exc:
                        logger.error("Plugin '%s' failed: %s", pname, exc, exc_info=True)
                        yield pname, {
                            "status": "error",
                            "error": str(exc),
                            "elapsed_ms": round(elapsed * 1000, 1),
                        }
            except FuturesTimeoutError:
                for future, (pname, t0) in future_to_meta.items():
                    if not future.done():
                        logger.error("Plugin '%s' timed out after %ds", pname, timeout)
                        yield pname, {
                            "status": "error",
                            "error": f"Plugin timed out after {timeout}s",
                        }

    def discover(self) -> None:
        """
        Auto-import all modules in app.plugins and register any
        BasePlugin subclass found.
        """
        from app.plugins.base import BasePlugin

        package_path = os.path.join(os.path.dirname(__file__), "plugins")
        if not os.path.isdir(package_path):
            logger.warning("Plugin directory not found: %s", package_path)
            return

        for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                mod = importlib.import_module(f"app.plugins.{modname}")
                # Find all BasePlugin subclasses in the module
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type)
                            and issubclass(attr, BasePlugin)
                            and attr is not BasePlugin):
                        instance = attr()
                        self.register(instance)
            except Exception as exc:
                logger.warning("Failed to load plugin '%s': %s", modname, exc)

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins


# ── Singleton ──────────────────────────────────────────────────────
# Created at import time so gunicorn --preload shares it across workers.
registry = PluginRegistry()
