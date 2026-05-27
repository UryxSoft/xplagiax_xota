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
    celery.conf.update(app.config)

    # ── Cleanup settings ─────────────────────────────────────────────────────
    # Auto-delete task results from Redis after 1 hour (prevents Redis bloat)
    celery.conf.result_expires = 3600
    # ACK the task only after it completes — re-queues on worker crash
    celery.conf.task_acks_late = True
    # Prefetch 1 task at a time (important for memory-heavy ML tasks)
    celery.conf.worker_prefetch_multiplier = 1

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
