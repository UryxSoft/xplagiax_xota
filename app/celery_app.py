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
    
    # TaskBase override to ensure Flask context is active during task execution
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    celery.flask_app = app
    
    # Import tasks to ensure they are registered with Celery
    import app.tasks
    
    return celery

celery = make_celery()
