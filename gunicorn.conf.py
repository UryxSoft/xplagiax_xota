"""
gunicorn.conf.py — Production Gunicorn configuration.

Key design decisions:
    - preload_app = True → loads the app ONCE in the master process,
      then forks workers. Heavy models (torch, transformers) loaded at
      module level are shared across workers via Linux Copy-on-Write,
      cutting per-worker memory from ~50 MB overhead per worker.
    - worker_class = "gthread" → workers are threads within a process;
      all share the master's memory. Safe for CPU-only inference (no CUDA,
      no DataLoader multiprocessing). Keeps RAM at ~2.5 GB vs ~5 GB for sync.
    - workers = 2 (default) → with gthread, each worker adds only ~100-150 MB
      (thread overhead, no model duplication).
    - Celery watchdog thread → daemon thread in master checks every 30s if
      the Celery process is alive and restarts it if not. Uses <1 MB RAM.
"""

import os
import multiprocessing
import sys
import threading

# ── [C6] CPU pool caps — MUST be set before the app (and torch) load ──
# Parallelism already comes from gunicorn workers × threads × the plugin
# executor (PLUGIN_MAX_WORKERS) × inference batching. If each BLAS/OpenMP
# runtime also spawns one thread per core, the box runs cores² runnable
# threads and p99 latency blows up on context switches. torch's own intra-op
# cap (TORCH_NUM_THREADS, detector_final.py) covers torch kernels; these
# cover numpy/scipy/spacy BLAS calls. setdefault → any explicit env wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── Server socket ─────────────────────────────────────────────────
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5006")

# ── Workers ───────────────────────────────────────────────────────
# ML workloads: each worker costs ~200 MB+ overhead (on top of CoW-shared models).
# Cap at 2 for memory-constrained VPS. Override via WEB_CONCURRENCY env var.
workers = int(os.getenv("WEB_CONCURRENCY",
                        min(multiprocessing.cpu_count(), 2)))
worker_class = "gthread"
# Fix #6: more threads per worker. The async enqueue (/analyze_document_async)
# and status polling (/analyze_status) are I/O-bound on Redis — extra threads
# keep them responsive even while the heavy *synchronous* endpoints
# (/analyze_document, /analyze_stream) occupy threads. Threads share the CoW
# model memory, so the RAM cost is only ~1 stack each. Override via GUNICORN_THREADS.
threads = int(os.getenv("GUNICORN_THREADS", "4"))

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
        "--pool=threads",
        "--concurrency=2",
        "--without-heartbeat",
        "--without-gossip",
        "--without-mingle",
        # Threads pool does not support max-tasks-per-child (it's a process-level
        # option). Memory is managed via soft_time_limit in the task instead.

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


def _celery_watchdog(server):
    """
    Daemon thread in master — checks every 30s if Celery is alive.
    Restarts it immediately if it died (OOM, segfault, max-tasks-per-child
    recycle). Uses <1 MB RAM. Prevents tasks from piling up in PENDING state.
    """
    while True:
        threading.Event().wait(10)
        global _celery_process
        if _celery_process is not None and not _celery_process.is_alive():
            server.log.warning(
                "Celery watchdog: PID %s muerto — reiniciando...",
                _celery_process.pid,
            )
            _spawn_celery_worker(server)


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
    t = threading.Thread(target=_celery_watchdog, args=(server,), daemon=True)
    t.start()
    server.log.info("Celery watchdog iniciado — revisión cada 30s")


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
