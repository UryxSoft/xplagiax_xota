"""
Plugin and auth unit tests.
"""

import pytest


# ── Auth ──────────────────────────────────────────────────────────────────

def test_auth_passes_when_api_key_empty(client):
    """Empty API_KEY disables auth — all requests pass through."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_auth_rejects_missing_key(app):
    """Non-empty API_KEY rejects requests without header."""
    app.config["API_KEY"] = "secret"
    c = app.test_client()
    resp = c.post("/analyze", json={"text": "test", "plugins": ["ai_detection"]})
    assert resp.status_code == 401
    app.config["API_KEY"] = ""


def test_auth_accepts_correct_key(app):
    """Correct X-API-Key header passes auth."""
    app.config["API_KEY"] = "secret"
    c = app.test_client()
    resp = c.post(
        "/analyze",
        json={"text": "test", "plugins": ["ai_detection"]},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code != 401
    app.config["API_KEY"] = ""


def test_auth_rejects_wrong_key(app):
    """Wrong API key returns 401."""
    app.config["API_KEY"] = "correct"
    c = app.test_client()
    resp = c.post(
        "/analyze",
        json={"text": "test", "plugins": ["ai_detection"]},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401
    app.config["API_KEY"] = ""


# ── Plugin registry ───────────────────────────────────────────────────────

def test_registry_loaded_in_app(app):
    with app.app_context():
        registry = app.config.get("PLUGIN_REGISTRY")
        assert registry is not None


def test_registry_list_plugins_returns_list(app):
    with app.app_context():
        registry = app.config["PLUGIN_REGISTRY"]
        names = registry.list_plugins()
        assert isinstance(names, list)


def test_registry_unknown_plugin_returns_error(client):
    resp = client.post("/analyze", json={
        "text": "some text",
        "plugins": ["nonexistent_plugin_xyz"],
    })
    assert resp.status_code == 200
    data = resp.json
    assert "nonexistent_plugin_xyz" in data["results"]
    assert "error" in data["results"]["nonexistent_plugin_xyz"]


# ── max_tokens clamping (B-11) ────────────────────────────────────────────

def test_analyze_document_max_tokens_zero_does_not_crash(client, monkeypatch):
    """max_tokens=0 is clamped to 50 — must not crash."""
    monkeypatch.setattr(
        "app.routes.analyze_long_document",
        lambda *a, **kw: {},
        raising=False,
    )
    resp = client.post("/analyze_document", json={
        "text": "word " * 50,
        "plugins": [],
        "max_tokens": 0,
    })
    assert resp.status_code != 500


def test_analyze_document_max_tokens_large_does_not_crash(client, monkeypatch):
    """max_tokens=99999 is clamped to 512 — must not OOM."""
    monkeypatch.setattr(
        "app.routes.analyze_long_document",
        lambda *a, **kw: {},
        raising=False,
    )
    resp = client.post("/analyze_document", json={
        "text": "word " * 50,
        "plugins": [],
        "max_tokens": 99999,
    })
    assert resp.status_code != 500


# ── /analyze response shape ───────────────────────────────────────────────

def test_analyze_response_shape_with_unknown_plugin(client):
    """Response always contains status, word_count, results, total_elapsed_ms."""
    resp = client.post("/analyze", json={
        "text": "The quick brown fox jumps over the lazy dog.",
        "plugins": ["nonexistent"],
    })
    assert resp.status_code == 200
    data = resp.json
    assert data["status"] == "ok"
    assert isinstance(data["word_count"], int)
    assert isinstance(data["results"], dict)
    assert "total_elapsed_ms" in data
