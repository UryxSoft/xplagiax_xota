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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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

        max_workers = min(len(valid), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all plugins at once so they run in parallel
            future_to_meta: Dict[Any, tuple] = {}
            for pname, plugin in valid:
                t0 = time.perf_counter()
                future_to_meta[executor.submit(plugin.analyze, text)] = (pname, t0)

            # Collect with a shared deadline so every plugin gets the full budget
            deadline = time.perf_counter() + timeout
            for future, (pname, t0) in future_to_meta.items():
                remaining = max(0.0, deadline - time.perf_counter())
                try:
                    result = future.result(timeout=remaining)
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
