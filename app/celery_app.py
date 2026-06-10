"""
app/celery_app.py — Celery worker entry point.
"""

from celery import Celery
from app import create_app

def make_celery():
    """Create Celery app with Flask context."""
    app = create_app()
    
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )

    # ── Cleanup settings ─────────────────────────────────────────────────────
    # Auto-delete task results from Redis after 10 min.
    # The previous 1-hour TTL allowed hundreds of large results (each with
    # base64 charts) to accumulate in Redis simultaneously under load.
    celery.conf.result_expires = 600
    # ACK the task only after it completes — re-queues on worker crash
    celery.conf.task_acks_late = True
    # Prefetch 1 task at a time (important for memory-heavy ML tasks)
    celery.conf.worker_prefetch_multiplier = 1
    # Re-queue unacknowledged tasks after 360s (slightly > time_limit=300).
    # Default is 3600s (1 hour) — tasks from a dead worker would stay
    # PENDING for up to 1 hour before being retried.
    #
    # Fix #3: socket timeouts so a slow/dead Redis broker fails fast instead of
    # hanging the web request that calls .delay() (the "instant" async endpoint).
    celery.conf.broker_transport_options = {
        'visibility_timeout': 360,
        'socket_timeout': 5,
        'socket_connect_timeout': 5,
    }
    # Retry the broker connection on worker startup (Redis may boot after web).
    celery.conf.broker_connection_retry_on_startup = True
    # Producer-side (.delay()) publish: bound the retry so an unreachable broker
    # raises in ~1s instead of blocking the HTTP handler for the default ~20s+.
    celery.conf.broker_connection_timeout = 5
    celery.conf.task_publish_retry_policy = {
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
    }

    # TaskBase override to ensure Flask context is active during task execution
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    celery.flask_app = app
    return celery

celery = make_celery()

# Import tasks AFTER celery is initialized to avoid circular imports
import app.tasks
