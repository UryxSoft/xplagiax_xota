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

# ── [C6] Shared plugin executor ────────────────────────────────────
# One process-wide pool instead of a fresh ThreadPoolExecutor per request.
# Two wins:
#   1. No thread create/teardown churn per request.
#   2. The old `with ThreadPoolExecutor(...)` pattern called shutdown(wait=True)
#      on exit, so a plugin that had ALREADY been reported as timed out still
#      blocked the HTTP response until it actually finished. With a shared pool
#      the response returns at the deadline; the stray task drains in background.
# Size via PLUGIN_MAX_WORKERS (default 8 — ML plugins release the GIL during
# C-level inference, so these threads genuinely run in parallel).
_MAX_WORKERS = int(os.getenv("PLUGIN_MAX_WORKERS", "8"))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="plugin")

# ── [C8] Global request deadline (seconds) ─────────────────────────
# Upper bound on how long ONE request may wait for its plugin set, regardless
# of how many plugins were requested. The per-plugin `timeout` still applies;
# the effective wait is min(per-plugin timeout, REQUEST_DEADLINE_SECONDS).
_REQUEST_DEADLINE_S = float(os.getenv("REQUEST_DEADLINE_SECONDS", "60"))


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

        [C8] All plugins are submitted at once and share a single wall-clock
        budget of min(timeout, REQUEST_DEADLINE_SECONDS). Because every plugin
        starts at t0, waiting that budget once covers each plugin's individual
        allowance — a slow plugin can no longer stretch the response beyond the
        deadline, and fast plugins are collected as they finish (as_completed)
        instead of in submission order.

        [C6] Futures run on the shared module-level executor: the response
        returns AT the deadline even if a timed-out plugin is still running
        (previously the per-request pool's shutdown(wait=True) blocked until
        the stray plugin finished).
        """
        results: Dict[str, Any] = {}

        valid: List[tuple] = []  # (pname, plugin)
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

        budget = min(float(timeout), _REQUEST_DEADLINE_S)
        t0 = time.perf_counter()
        future_to_name: Dict[Any, str] = {
            _EXECUTOR.submit(plugin.analyze, text): pname
            for pname, plugin in valid
        }

        try:
            for future in as_completed(future_to_name, timeout=budget):
                pname = future_to_name[future]
                elapsed = time.perf_counter() - t0
                try:
                    result = future.result()
                    results[pname] = {
                        "status": "ok",
                        "data": result,
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }
                except Exception as exc:
                    logger.error("Plugin '%s' failed: %s", pname, exc, exc_info=True)
                    results[pname] = {
                        "status": "error",
                        "error": str(exc),
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }
        except FuturesTimeoutError:
            pass  # deadline reached — unfinished plugins handled below

        elapsed = time.perf_counter() - t0
        for future, pname in future_to_name.items():
            if pname in results:
                continue
            if future.done() and not future.cancelled():
                # Finished in the race window between the deadline firing and
                # this mop-up — its result is real, don't report a false timeout.
                try:
                    results[pname] = {
                        "status": "ok",
                        "data": future.result(),
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }
                except Exception as exc:
                    results[pname] = {
                        "status": "error",
                        "error": str(exc),
                        "elapsed_ms": round(elapsed * 1000, 1),
                    }
                continue
            future.cancel()  # frees queue slots for plugins that never started
            logger.error(
                "Plugin '%s' exceeded the request budget (%.1fs)", pname, budget,
            )
            results[pname] = {
                "status": "error",
                "error": f"Plugin timed out after {budget:g}s",
                "elapsed_ms": round(elapsed * 1000, 1),
            }

        return results

    def run_stream(self, plugin_names: List[str], text: str, timeout: int = 30):
        """
        Yield (plugin_name, result_dict) as each plugin completes.
        Used by /analyze_stream (SSE) to deliver results incrementally.

        [C6/C8] Same shared executor + global deadline semantics as run().
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

        budget = min(float(timeout), _REQUEST_DEADLINE_S)
        t0 = time.perf_counter()
        future_to_name: Dict[Any, str] = {
            _EXECUTOR.submit(plugin.analyze, text): pname
            for pname, plugin in valid
        }

        try:
            for future in as_completed(future_to_name, timeout=budget):
                pname = future_to_name[future]
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
            for future, pname in future_to_name.items():
                if not future.done():
                    future.cancel()
                    logger.error("Plugin '%s' exceeded the request budget (%.1fs)",
                                 pname, budget)
                    yield pname, {
                        "status": "error",
                        "error": f"Plugin timed out after {budget:g}s",
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
