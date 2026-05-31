"""
gunicorn.conf.py — Production Gunicorn configuration.

Key design decisions:
    - preload_app = True → loads the app ONCE in the master process,
      then forks workers. Heavy models (torch, transformers) loaded at
      module level are shared across workers via Linux Copy-on-Write,
      cutting per-worker memory from ~500 MB to ~50 MB overhead.
    - worker_class = "sync" → PyTorch/transformers use C-level threads
      internally; gevent monkey-patching causes deadlocks with ML workloads.
      sync workers are correct and required for CPU-bound inference.
      NOTE: the `threads = 2` setting below has no effect with sync workers
      (it only applies to gthread workers) and is kept for documentation.
    - workers = 2 (default) → ML workloads are memory-heavy; cap at 2 for
      VPS deployments. Override via WEB_CONCURRENCY env var.
"""

import os
import multiprocessing
import sys
# ── Server socket ─────────────────────────────────────────────────
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5006")

# ── Workers ───────────────────────────────────────────────────────
# ML workloads: each worker costs ~200 MB+ overhead (on top of CoW-shared models).
# Cap at 2 for memory-constrained VPS. Override via WEB_CONCURRENCY env var.
workers = int(os.getenv("WEB_CONCURRENCY",
                        min(multiprocessing.cpu_count(), 2)))
worker_class = "sync"
threads = 2

# ── CRITICAL: Enable preload for CoW memory sharing ──────────────
preload_app = True

# ── Timeouts ──────────────────────────────────────────────────────
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))       # hard kill
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))  # soft shutdown

# ── Keep-alive ────────────────────────────────────────────────────
keepalive = int(os.getenv("KEEPALIVE", "5"))

# ── Max requests (prevent memory leaks) ───────────────────────────
max_requests = int(os.getenv("MAX_REQUESTS", "2000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "200"))

# ── Logging ───────────────────────────────────────────────────────
accesslog = "-"   # stdout
errorlog = "-"    # stderr
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" %(D)sμs'

# ── Security ──────────────────────────────────────────────────────
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

_celery_process = None


# ── Celery bootstrap (B-10: shared by when_ready and child_exit) ──
def _run_celery():
    """Celery worker entrypoint — runs inside a forked child process."""
    from celery.__main__ import main as celery_main
    sys.argv = [
        "celery",
        "-A", "app.celery_app.celery",
        "worker",
        "--loglevel=info",
        "--pool=solo",
        "--concurrency=1",
        "--without-heartbeat",
        "--without-gossip",
        "--without-mingle",
        # Recycle the worker process after 50 tasks — equivalent to gunicorn's
        # max_requests. Prevents Python high-water-mark memory from accumulating
        # indefinitely over long uptimes. gunicorn will restart it via child_exit.
        "--max-tasks-per-child=50",
    ]
    celery_main()


def _spawn_celery_worker(server):
    """Fork a Celery worker from the gunicorn master and update _celery_process."""
    global _celery_process
    _celery_process = multiprocessing.Process(
        target=_run_celery,
        name="celery-worker",
        daemon=True,
    )
    _celery_process.start()
    server.log.info(
        "Celery worker forkeado del master (CoW activo): PID %s",
        _celery_process.pid,
    )


# ── Hooks ─────────────────────────────────────────────────────────
def when_ready(server):
    """
    Se ejecuta DESPUÉS de preload_app. Los modelos ya están en memoria.
    Forkeamos el worker de Celery aquí para que herede las páginas via CoW.
    subprocess.Popen NO funciona para esto (hace exec y pierde la memoria).
    multiprocessing.Process usa os.fork() — el hijo hereda las páginas físicas.

    Set GUNICORN_SPAWN_CELERY=1 to enable. Default is 0 — use the
    docker-compose celery_worker service instead to avoid duplicate workers.
    """
    if os.getenv("GUNICORN_SPAWN_CELERY", "0") != "1":
        server.log.info(
            "GUNICORN_SPAWN_CELERY not set — Celery worker managed externally"
        )
        return
    _spawn_celery_worker(server)


def on_exit(server):
    global _celery_process
    if _celery_process and _celery_process.is_alive():
        server.log.info("Terminando celery worker PID %s", _celery_process.pid)
        _celery_process.terminate()
        _celery_process.join(timeout=10)

def child_exit(server, worker):
    """
    Called when a child process exits.
    If the dead child is our Celery worker, restart it automatically
    so async analysis tasks don't pile up in Redis indefinitely.
    """
    global _celery_process
    if _celery_process and not _celery_process.is_alive():
        server.log.warning(
            "Celery worker (PID %s) murió — reiniciando...",
            _celery_process.pid,
        )
        _spawn_celery_worker(server)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    global _celery_process
    # The Celery process belongs to the master, not to this worker.
    # Nulling the reference prevents Python's multiprocessing atexit handler
    # from calling join() on a process the worker doesn't own, which would
    # raise: AssertionError: can only join a child process
    _celery_process = None
    server.log.info("Worker spawned — PID %s", worker.pid)

def worker_exit(server, worker):
    """Called when a worker exits."""

    server.log.info("Worker exiting — PID %s", worker.pid)
