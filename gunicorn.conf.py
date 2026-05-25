"""
gunicorn.conf.py — Production Gunicorn configuration.

Key design decisions:
    - preload_app = True → loads the app ONCE in the master process,
      then forks workers. Heavy models (torch, spacy, etc.) loaded at
      module level are shared across workers via Linux Copy-on-Write,
      cutting per-worker memory from ~500MB to ~50MB overhead.
    - worker_class = "sync" → PyTorch/transformers use C-level threads
      internally; gevent monkey-patching causes deadlocks with ML workloads.
      sync workers are correct for CPU-bound inference.
    - workers = 4 → 2×CPU for CPU-bound NLP tasks.  Override via
      WEB_CONCURRENCY env var for auto-scaling.
"""

import os
import multiprocessing
import sys
# ── Server socket ─────────────────────────────────────────────────
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5006")

# ── Workers ───────────────────────────────────────────────────────
workers = int(os.getenv("WEB_CONCURRENCY",
                        min(multiprocessing.cpu_count() * 2, 8)))
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
# ── Hooks ─────────────────────────────────────────────────────────
def when_ready(server):
    """
    Se ejecuta DESPUÉS de preload_app. Los modelos ya están en memoria.
    Forkeamos el worker de Celery aquí para que herede las páginas via CoW.
    subprocess.Popen NO funciona para esto (hace exec y pierde la memoria).
    multiprocessing.Process usa os.fork() — el hijo hereda las páginas físicas.
    """
    global _celery_process

    def _run_celery():
        # Este código corre en el proceso hijo forkeado.
        # sys.modules ya tiene todos los imports del padre, incluidos los modelos.
        # Celery no los recarga — los encuentra en sys.modules directamente.
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
        ]
        celery_main()

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


def on_exit(server):
    global _celery_process
    if _celery_process and _celery_process.is_alive():
        server.log.info("Terminando celery worker PID %s", _celery_process.pid)
        _celery_process.terminate()
        _celery_process.join(timeout=10)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned — PID %s", worker.pid)

def worker_exit(server, worker):
    """Called when a worker exits."""
    server.log.info("Worker exiting — PID %s", worker.pid)
