"""
Unit tests for the anti-enshittification drift monitor.

Pure Python — no model weights or Flask app needed, safe for any CI runner.
"""

import pytest

from app.engine import drift_monitor as dm


@pytest.fixture()
def monitor(tmp_path, monkeypatch):
    """Fresh monitor with alerts redirected into tmp and no cooldown wait."""
    monkeypatch.setattr(dm, "ALERT_PATH", str(tmp_path / "alerts.jsonl"))
    monkeypatch.setattr(dm, "ALERT_COOLDOWN_S", 0.0)
    return dm.DriftMonitor()


def test_no_data_status(monitor):
    status = monitor.get_status()
    assert status["status"] == "no_data"
    assert status["samples_total"] == 0


def test_healthy_under_stable_confidence(monitor):
    for i in range(dm.MIN_SAMPLES + 20):
        alert = monitor.record_prediction(0.95, "AI" if i % 2 else "Human")
        assert alert is None or alert["kind"] != "confidence_drop"
    status = monitor.get_status()
    assert status["status"] == "healthy"
    assert status["mean_confidence"] == pytest.approx(0.95)


def test_confidence_drop_triggers_alert(monitor):
    # Establish a high-confidence baseline...
    for i in range(dm.WINDOW_SIZE):
        monitor.record_prediction(0.95, "AI" if i % 2 else "Human")
    # ...then simulate degradation well past the drop threshold.
    alerts = []
    for i in range(dm.WINDOW_SIZE):
        alert = monitor.record_prediction(0.70, "AI" if i % 2 else "Human")
        if alert and alert["kind"] == "confidence_drop":
            alerts.append(alert)
    assert alerts, "sustained confidence drop must raise a confidence_drop alert"
    assert monitor.is_degraded()
    assert monitor.get_status()["status"] == "degraded"
    # The alert carries the pre-drop baseline, not the already-degraded mean.
    assert alerts[0]["baseline"] > alerts[0]["current"]


def test_class_imbalance_triggers_info_alert(monitor):
    alerts = []
    for _ in range(dm.WINDOW_SIZE):
        alert = monitor.record_prediction(0.9, "AI")  # 100% one-sided
        if alert:
            alerts.append(alert)
    assert any(a["kind"] == "class_imbalance" for a in alerts)


def test_alerts_written_as_jsonl(monitor, tmp_path):
    for i in range(dm.WINDOW_SIZE):
        monitor.record_prediction(0.95, "AI" if i % 2 else "Human")
    for i in range(dm.WINDOW_SIZE):
        monitor.record_prediction(0.70, "AI" if i % 2 else "Human")
    import json
    path = tmp_path / "alerts.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert lines and all(json.loads(l)["kind"] for l in lines)


def test_clear_degraded_resets_flag(monitor):
    for i in range(dm.WINDOW_SIZE):
        monitor.record_prediction(0.95, "AI" if i % 2 else "Human")
    for i in range(dm.WINDOW_SIZE):
        monitor.record_prediction(0.70, "AI" if i % 2 else "Human")
    assert monitor.is_degraded()
    monitor.clear_degraded()
    assert not monitor.is_degraded()


def test_record_prediction_never_raises(monitor):
    # Garbage input must be swallowed (fail-open monitoring).
    assert monitor.record_prediction(float("nan"), None, -1) is None or True


def test_singleton_identity():
    assert dm.get_drift_monitor() is dm.get_drift_monitor()
