"""
app/engine/drift_monitor.py — Model drift monitor (anti-enshittification safeguard).

Tracks every ensemble prediction and detects quality degradation BEFORE users do:

  * Confidence drift  — rolling-window mean confidence vs a slow EMA baseline.
    A sustained drop (> DRIFT_CONFIDENCE_DROP) usually means the ensemble is
    seeing out-of-distribution text (e.g. a new LLM family it was never trained
    on) and is no longer sure of its verdicts.
  * Class imbalance   — a window that is > IMBALANCE_RATIO one-sided suggests
    either a skewed traffic mix or a model collapsing onto one label.

Alerts are kept in memory (bounded), appended to DRIFT_ALERT_PATH as JSONL for
external monitoring systems, and surfaced via GET /api/drift-status.

Design notes
------------
* Singleton per process (get_drift_monitor()). Counters are per-worker; that is
  fine for alerting purposes — every worker sees a representative traffic slice.
* Zero inference cost: record_prediction() is O(1) dict/deque work under a lock.
* Fail-open: any internal error is swallowed and logged; monitoring must never
  break the detection path it watches.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rolling window of recent predictions used to compute the current mean.
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "100"))
# Minimum samples before drift checks activate (avoid noise at startup).
MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "30"))
# Alert when window mean confidence drops this much below the EMA baseline.
DRIFT_CONFIDENCE_DROP = float(os.getenv("DRIFT_CONFIDENCE_DROP", "0.05"))
# EMA smoothing for the long-term baseline (small = slow-moving baseline).
BASELINE_ALPHA = float(os.getenv("DRIFT_BASELINE_ALPHA", "0.02"))
# Class-imbalance alert when the minority class share falls below this.
IMBALANCE_RATIO = float(os.getenv("DRIFT_IMBALANCE_RATIO", "0.10"))
# Where JSONL alerts are appended for external monitoring pickup.
ALERT_PATH = os.getenv(
    "DRIFT_ALERT_PATH",
    os.path.join(os.getenv("TMPDIR", "/tmp"), "xplagiax_drift_alerts.jsonl"),
)
MAX_ALERTS_KEPT = 200
# Cooldown between alerts of the same kind (seconds) so a degraded window
# doesn't spam one alert per request.
ALERT_COOLDOWN_S = float(os.getenv("DRIFT_ALERT_COOLDOWN_S", "300"))


class DriftMonitor:
    """Rolling drift detector over ensemble predictions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window: deque = deque(maxlen=WINDOW_SIZE)
        self._alerts: deque = deque(maxlen=MAX_ALERTS_KEPT)
        self._baseline_confidence: Optional[float] = None  # slow EMA
        self._samples_total = 0
        self._degraded = False
        self._last_alert_monotonic: Dict[str, float] = {}

    # ── Recording ──────────────────────────────────────────────────

    def record_prediction(
        self,
        confidence: float,
        prediction: str,
        text_len: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Record one ensemble prediction. Returns the alert dict if this sample
        triggered one, else None. Never raises.

        confidence : winning-class probability in [0, 1].
        prediction : "AI" | "Human" | other label.
        """
        try:
            import time as _time
            with self._lock:
                self._window.append((float(confidence), str(prediction), int(text_len)))
                self._samples_total += 1

                if len(self._window) < MIN_SAMPLES:
                    # Warm-up: just seed the baseline.
                    self._update_baseline()
                    return None

                alert = self._check_drift(_time.monotonic())
                self._update_baseline()
                if alert is not None:
                    self._alerts.append(alert)
                    self._degraded = True
                    self._write_alert(alert)
                    logger.warning("Model drift alert: %s", alert["reason"])
                return alert
        except Exception as exc:  # noqa: BLE001 — monitoring must fail open
            logger.debug("DriftMonitor.record_prediction failed: %s", exc)
            return None

    # ── Internals (call with lock held) ────────────────────────────

    def _window_mean_confidence(self) -> float:
        return sum(c for c, _, _ in self._window) / len(self._window)

    def _update_baseline(self) -> None:
        mean = self._window_mean_confidence()
        if self._baseline_confidence is None:
            self._baseline_confidence = mean
        else:
            # Slow EMA — the baseline follows genuine regime changes over
            # hundreds of samples but is not dragged down by a short bad patch,
            # which is exactly what the drop check needs to detect.
            self._baseline_confidence += BASELINE_ALPHA * (mean - self._baseline_confidence)

    def _cooldown_ok(self, kind: str, now_mono: float) -> bool:
        last = self._last_alert_monotonic.get(kind)
        if last is not None and (now_mono - last) < ALERT_COOLDOWN_S:
            return False
        self._last_alert_monotonic[kind] = now_mono
        return True

    def _check_drift(self, now_mono: float) -> Optional[Dict[str, Any]]:
        baseline = self._baseline_confidence
        current = self._window_mean_confidence()

        # 1. Confidence drop vs slow baseline
        if baseline is not None:
            drop = baseline - current
            if drop > DRIFT_CONFIDENCE_DROP and self._cooldown_ok("confidence", now_mono):
                return {
                    "kind": "confidence_drop",
                    "severity": "warning",
                    "reason": (
                        f"Mean confidence dropped {drop:.1%} below baseline "
                        f"({current:.1%} vs {baseline:.1%} over last {len(self._window)} samples)"
                    ),
                    "baseline": round(baseline, 4),
                    "current": round(current, 4),
                    "samples": len(self._window),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        # 2. Class imbalance in the window
        labels = [p for _, p, _ in self._window]
        ai_share = labels.count("AI") / len(labels)
        minority = min(ai_share, 1.0 - ai_share)
        if minority < IMBALANCE_RATIO and self._cooldown_ok("imbalance", now_mono):
            return {
                "kind": "class_imbalance",
                "severity": "info",
                "reason": (
                    f"Prediction mix is {ai_share:.0%} AI / {1 - ai_share:.0%} Human "
                    f"over last {len(labels)} samples (minority < {IMBALANCE_RATIO:.0%})"
                ),
                "ai_share": round(ai_share, 4),
                "samples": len(labels),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return None

    def _write_alert(self, alert: Dict[str, Any]) -> None:
        try:
            with open(ALERT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert) + "\n")
        except OSError as exc:
            logger.debug("Could not append drift alert to %s: %s", ALERT_PATH, exc)

    # ── Status ─────────────────────────────────────────────────────

    def is_degraded(self) -> bool:
        with self._lock:
            return self._degraded

    def clear_degraded(self) -> None:
        """Operator acknowledgement — resets the degraded flag (alerts are kept)."""
        with self._lock:
            self._degraded = False

    def get_status(self) -> Dict[str, Any]:
        """Snapshot for /api/drift-status. Never raises."""
        with self._lock:
            if not self._window:
                return {
                    "status": "no_data",
                    "samples_total": self._samples_total,
                }
            confidences = [c for c, _, _ in self._window]
            labels = [p for _, p, _ in self._window]
            return {
                "status": "degraded" if self._degraded else "healthy",
                "samples_total": self._samples_total,
                "window_samples": len(self._window),
                "mean_confidence": round(sum(confidences) / len(confidences), 4),
                "baseline_confidence": (
                    round(self._baseline_confidence, 4)
                    if self._baseline_confidence is not None else None
                ),
                "confidence_min": round(min(confidences), 4),
                "confidence_max": round(max(confidences), 4),
                "ai_share": round(labels.count("AI") / len(labels), 4),
                "recent_alerts": list(self._alerts)[-5:],
                "alerts_total": len(self._alerts),
            }


# ── Module-level singleton ─────────────────────────────────────────

_monitor: Optional[DriftMonitor] = None
_monitor_lock = threading.Lock()


def get_drift_monitor() -> DriftMonitor:
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = DriftMonitor()
    return _monitor
