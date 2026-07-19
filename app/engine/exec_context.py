"""
exec_context.py — Per-request execution context (thread-local).
===============================================================

[Fase-2 M-11 + M-7] Two problems share one mechanism:

M-11 (zombie threads): when a plugin future times out, PluginRegistry reports the error
and returns, but the plugin thread keeps burning CPU on the full inference. The registry
now stamps a DEADLINE into this thread-local before calling plugin.analyze(); the
CPU-heavy loops in detector_final check it between batches and abort cooperatively.

M-7 (async-only reference check): the orchestrator is a preload singleton shared by the
sync (gunicorn) and async (Celery, forked post-preload) paths, so an env var cannot
enable the reference validator for async only. The Celery task stamps async_mode=True
here; the orchestrator reads it at run time and lazily activates the validator.

Thread-local by design: PluginRegistry runs each plugin in its own worker thread, so the
stamp set by the wrapper is visible exactly to that plugin's call stack and nothing else.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

_ctx = threading.local()


class PluginDeadlineExceeded(RuntimeError):
    """Raised by cooperative checkpoints once the caller's timeout has passed."""


def set_context(deadline: Optional[float] = None, async_mode: bool = False) -> None:
    """Stamp the current thread. deadline is time.monotonic()-based, None = no limit."""
    _ctx.deadline = deadline
    _ctx.async_mode = async_mode


def clear_context() -> None:
    _ctx.deadline = None
    _ctx.async_mode = False


def is_async() -> bool:
    """True when the current call stack runs inside the async (Celery) path."""
    return bool(getattr(_ctx, "async_mode", False))


def deadline_exceeded() -> bool:
    deadline = getattr(_ctx, "deadline", None)
    return deadline is not None and time.monotonic() > deadline


def check_deadline() -> None:
    """
    Cooperative cancellation checkpoint for CPU-heavy loops.

    Call between batches: cost is one monotonic read. Raises PluginDeadlineExceeded
    once the registry-reported timeout has passed, so the orphaned thread dies at the
    next batch boundary instead of finishing a full inference nobody will read.
    """
    if deadline_exceeded():
        raise PluginDeadlineExceeded(
            "Plugin deadline exceeded — aborting between batches (result already "
            "reported as timeout to the caller)."
        )
