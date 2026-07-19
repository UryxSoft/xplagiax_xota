"""
pytest fixtures — shared across all test modules.
"""

import pytest
from app import create_app


@pytest.fixture(scope="session")
def app():
    """
    Flask test application.
    - API_KEY empty → auth disabled
    - PLUGIN_TIMEOUT 5s → fast failure in tests
    - CACHE_TYPE SimpleCache → no Redis needed in CI
    """
    _app = create_app()
    _app.config.update({
        "TESTING": True,
        "API_KEY": "",
        "PLUGIN_TIMEOUT": 5,
        "CACHE_TYPE": "SimpleCache",
        "WTF_CSRF_ENABLED": False,
        # No Redis in CI/local test runs: an empty REDIS_URL makes /health skip
        # the broker ping instead of returning 503 for an infra dependency the
        # test environment deliberately doesn't have.
        "REDIS_URL": "",
    })
    return _app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def authed_client(app):
    """Client that sends the correct API key header."""
    app.config["API_KEY"] = "test-secret"
    c = app.test_client()
    c.environ_base["HTTP_X_API_KEY"] = "test-secret"
    return c
