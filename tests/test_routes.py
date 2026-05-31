"""
Route-level smoke tests.
Verify endpoint contracts and input validation without full ML inference.
"""


# ── /health ───────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "healthy"


# ── /ready ────────────────────────────────────────────────────────────────

def test_ready_returns_valid_status(client):
    resp = client.get("/ready")
    assert resp.status_code in (200, 503)
    data = resp.json
    assert "status" in data


# ── /plugins ──────────────────────────────────────────────────────────────

def test_plugins_list_shape(client):
    resp = client.get("/plugins")
    assert resp.status_code == 200
    data = resp.json
    assert "count" in data
    assert "plugins" in data
    assert isinstance(data["plugins"], list)


# ── /analyze input validation ─────────────────────────────────────────────

def test_analyze_wrong_content_type(client):
    resp = client.post("/analyze", data="not json", content_type="text/plain")
    assert resp.status_code == 415


def test_analyze_missing_text(client):
    resp = client.post("/analyze", json={"plugins": ["ai_detection"]})
    assert resp.status_code == 400
    assert "text" in resp.json.get("error", "").lower()


def test_analyze_missing_plugins(client):
    resp = client.post("/analyze", json={"text": "hello world"})
    assert resp.status_code == 400
    assert "plugins" in resp.json.get("error", "").lower()


def test_analyze_empty_text(client):
    resp = client.post("/analyze", json={"text": "", "plugins": ["ai_detection"]})
    assert resp.status_code == 400


def test_analyze_text_too_large(client):
    resp = client.post("/analyze", json={
        "text": "x" * 600_000,
        "plugins": ["ai_detection"],
    })
    assert resp.status_code == 413


def test_analyze_invalid_json(client):
    resp = client.post("/analyze", data="{bad json", content_type="application/json")
    assert resp.status_code == 400


# ── /analyze_document input validation ───────────────────────────────────

def test_analyze_document_missing_text(client):
    resp = client.post("/analyze_document", json={"plugins": ["ai_detection"]})
    assert resp.status_code == 400


def test_analyze_document_text_too_large(client):
    resp = client.post("/analyze_document", json={
        "text": "x" * 600_000,
        "plugins": ["ai_detection"],
    })
    assert resp.status_code == 413


# ── DT-13: X-Request-ID header ────────────────────────────────────────────

def test_request_id_added_to_response(client):
    resp = client.get("/health")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


def test_request_id_echoed_from_caller(client):
    resp = client.get("/health", headers={"X-Request-ID": "caller-abc"})
    assert resp.headers["X-Request-ID"] == "caller-abc"


# ── /report/<filename> path traversal guard ──────────────────────────────

def test_report_endpoint_not_found(client):
    resp = client.get("/report/nonexistent_report.html")
    assert resp.status_code == 404


def test_report_path_traversal_blocked(client):
    resp = client.get("/report/../etc/passwd")
    # basename() in serve_report strips path components; file won't exist
    assert resp.status_code == 404
