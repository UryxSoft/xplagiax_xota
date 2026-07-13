# Implementation: Anti-Enshittification in xplagiax_xota

> **STATUS: IMPLEMENTED** — the changes below shipped with these deltas from the original plan:
>
> | Plan | Shipped as |
> |---|---|
> | `app/plugins/model_drift_detector.py` (plugin) | `app/engine/drift_monitor.py` (engine singleton — a BasePlugin's `analyze(text)` contract never fit a monitor). Wired into `ai_detection`; exposed at `GET /api/drift-status`. |
> | Remove `citation_check` + `reference_validator` | Done — plugin and engine deleted; orchestrator flags removed. |
> | Uncertainty in API responses | Done — `uncertainty {margin_pct, ensemble_std_pct, in_uncertain_zone}`, `model_version`, `warning` in `ai_detection` data and top-level `/analyze` + `/analyze_document`. |
> | Archive dead code | Done — `app/engine/legacy.py` ([C9]). |
> | Versioning + fallback | Done — `MODEL_FALLBACK_DIR` fallback in `_load_model()`, `get_model_info()`, `scripts/retrain_pipeline.py` (collect/evaluate/train/promote with accuracy gate). |
> | Extras from the audit | [C1] single engine instances (`app/engine/engines.py` + import alias finder), [C2] segment-level inference LRU, [C6] shared plugin executor + BLAS caps, [C8] global request deadline. |
>
> Tests: `tests/test_drift_monitor.py`, `tests/test_registry_deadline.py`, `tests/test_precision_corpus.py` (rolling-corpus gate via `ROLLING_CORPUS_DIR`).

This document shows concrete code changes to implement safeguards against service degradation.

## Change 1: Hook ModelDriftDetector into plugin_orchestrator.py

**File**: `app/engine/plugin_orchestrator.py`

**Change**: After ensemble classification, call `drift_detector.record_prediction()` to track model behavior.

```python
# In the response building section (around line 120-150):

# After getting ensemble_result from detector_final
ensemble_result = detector_final.classify_text(text)
ensemble_confidence = ensemble_result["confidence"]
ensemble_prediction = ensemble_result["prediction"]

# Record for drift detection
from app.plugins.model_drift_detector import ModelDriftDetector
drift_detector = ModelDriftDetector()  # or use singleton pattern
drift_alert = drift_detector.record_prediction(
    confidence=ensemble_confidence,
    prediction=ensemble_prediction,
    text_len=len(text),
)

# Include drift alert in response if triggered
if drift_alert:
    additional["model_drift_alert"] = drift_alert
```

**Better pattern** (singleton, to avoid re-instantiating):

Create `app/engine/_drift_detector.py`:
```python
"""Singleton for ModelDriftDetector across requests."""

_detector = None

def get_drift_detector():
    global _detector
    if _detector is None:
        from app.plugins.model_drift_detector import ModelDriftDetector
        _detector = ModelDriftDetector()
    return _detector
```

Then in `plugin_orchestrator.py`:
```python
from app.engine._drift_detector import get_drift_detector

# ... after ensemble classification ...
drift_detector = get_drift_detector()
drift_alert = drift_detector.record_prediction(
    confidence=ensemble_confidence,
    prediction=ensemble_prediction,
    text_len=len(text),
)
```

---

## Change 2: Add `/api/drift-status` Endpoint

**File**: `app/main.py` (add new route)

```python
from app.engine._drift_detector import get_drift_detector

@app.get("/api/drift-status", tags=["monitoring"])
async def get_drift_status():
    """
    Return current model drift detection status.
    
    Used by monitoring systems to detect model degradation.
    
    Returns:
        {
            "status": "healthy" | "degraded",
            "samples_tracked": int,
            "mean_confidence": float,
            "confidence_range": [float, float],
            "recent_alerts": list,
            "last_alert": ISO timestamp or None
        }
    """
    detector = get_drift_detector()
    return detector.get_status()
```

---

## Change 3: Update API Response to Include Uncertainty

**File**: `app/main.py` (modify `/analyze` and `/analyze_document`)

**Before:**
```json
{
    "prediction": "AI",
    "confidence": 0.95,
    "plugins": {...}
}
```

**After:**
```json
{
    "prediction": "AI",
    "confidence": 0.95,
    "uncertainty": {
        "margin": 0.03,
        "ensemble_std": 0.08,
        "in_uncertain_zone": false
    },
    "model_version": "modernbert-2024-12",
    "warning": null,
    "plugins": {...}
}
```

**Implementation in `detector_final.py`:**

```python
def classify_text(text: str) -> Dict[str, Any]:
    """
    Classify text as AI or Human.
    
    Returns dict with:
    - prediction: "AI" or "Human"
    - confidence: 0.0 to 1.0
    - uncertainty: margin of error + ensemble disagreement
    - warning: drift alert or other warning
    """
    # existing ensemble logic...
    predictions = [...]  # from 3 seeds
    mean_confidence = ...
    
    # Calculate ensemble std (measure of disagreement)
    ensemble_std = np.std([p["confidence"] for p in predictions])
    
    # Uncertainty margin: ±3% for confident, ±5% for uncertain
    if ensemble_std > 0.1:
        margin = 0.05
    else:
        margin = 0.03
    
    # Flag if in uncertain zone (confidence 0.4-0.6)
    in_uncertain_zone = 0.4 <= mean_confidence <= 0.6
    
    return {
        "prediction": ...,
        "confidence": mean_confidence,
        "uncertainty": {
            "margin": margin,
            "ensemble_std": float(ensemble_std),
            "in_uncertain_zone": in_uncertain_zone,
        },
        "model_version": "modernbert-2024-12",
        "warning": None,
    }
```

---

## Change 4: Remove Low-Value Plugins

**Decision**: Remove `citation_check` and `reference_validator` entirely.

**Action**: 
1. Delete or comment out from `app/plugins/__init__.py`
2. Remove from `PLUGIN_TIMEOUTS` in `plugin_orchestrator.py`
3. Update tests to not expect these keys

**Why**: They add 200-800ms latency with < 2% precision improvement. Not worth it.

---

## Change 5: Disable Watermark Detection by Default

**File**: `app/plugins/watermark_detection.py`

Add config check:
```python
import os

WATERMARK_ENABLED = os.getenv("ENABLE_WATERMARK", "false").lower() == "true"

class WatermarkDetectionPlugin(BasePlugin):
    def analyze(self, text: str) -> Dict[str, Any]:
        if not WATERMARK_ENABLED:
            return {"status": "disabled"}
        # ... existing logic ...
```

**In production**: Don't set `ENABLE_WATERMARK` (defaults to false).

**Rationale**: Watermarking is speculative. GPT-4 doesn't embed watermarks. Keeping a disabled plugin costs memory. If watermark becomes real, re-enable. Otherwise, after 6 months, delete.

---

## Change 6: Archive Unused Code (C9)

Create `app/engine/legacy.py`:
```python
"""
Deprecated and unused functions.

Kept for historical reference; should not be used in new code.
"""

# Moved from detector_final.py:
# - analyze_long_document_v1(text)
# - analyze_long_document_v2(text)
# - ensemble_results_v2()
# etc.
```

Remove these functions from `detector_final.py`.

**Why**: God objects hide complexity. If code isn't used, delete it. If it might be needed later, version control has history.

---

## Change 7: Add Config for Plugin Pruning

**File**: `config.py` or `.env`

```python
# Plugins to always enable (core detection)
CORE_PLUGINS = [
    "ai_detection",
    "stylometric_analysis",
    "perplexity_check",
]

# Plugins to enable if available (optional enhancements)
OPTIONAL_PLUGINS = [
    "hallucination_check",  # set ENABLE_HALLUCINATION_CHECK=false to skip
    "reasoning_check",
    "segment_analysis",
]

# Plugins to keep disabled (experimental/removed)
DISABLED_PLUGINS = [
    "watermark_detection",
    "citation_check",
    "reference_validator",
]

# Thresholds for drift alerting
DRIFT_CONFIG = {
    "confidence_drop_threshold": 0.05,  # 5%
    "samples_for_baseline": 100,
    "alert_path": "/tmp/xplagiax_drift_alerts.jsonl",
}
```

---

## Change 8: Monitoring & Alerting Integration

**Example**: Send drift alerts to Datadog/Prometheus.

```python
# In app/engine/_drift_detector.py
from app.monitoring import send_metric

def record_prediction(...):
    # ... existing logic ...
    
    alert = self._check_drift()
    if alert:
        # Log to monitoring system
        send_metric(
            "xplagiax.model_drift.confidence_drop",
            value=alert["baseline"] - alert["current"],
            tags={"severity": alert["severity"]},
        )
        logger.warning("Model drift: %s", alert["reason"])
    
    return alert
```

**In `/api/health` or `/ready`**:
```python
@app.get("/ready", tags=["health"])
async def readiness():
    detector = get_drift_detector()
    status = detector.get_status()
    
    if status["status"] == "degraded":
        # Still return 200 (service is functional)
        # But include warning so monitoring sees it
        return JSONResponse(
            status_code=200,
            content={"ready": True, "warning": "model_degraded"},
        )
    
    return {"ready": True}
```

---

## Rollout Plan

### Week 1
- Add `ModelDriftDetector` plugin ✅ (done)
- Implement singleton in `_drift_detector.py`
- Hook into `plugin_orchestrator.py`
- Add `/api/drift-status` endpoint
- Test locally

### Week 2
- Add uncertainty to API responses
- Remove `citation_check` and `reference_validator`
- Archive unused code to `legacy.py`
- Update docs/README

### Week 3
- Deploy to staging
- Monitor drift metrics for 1 week
- Validate no regressions
- Deploy to production

### Week 4+
- Monitor continuously
- Set up alerting (Slack/PagerDuty)
- Monthly accuracy reports to stakeholders
- Plan model retraining on drift alert

---

## Testing Drift Detection

```python
# Manual test in pytest:

def test_drift_detector_detects_confidence_drop():
    detector = ModelDriftDetector()
    
    # Simulate 100 high-confidence predictions
    for i in range(100):
        detector.record_prediction(0.95, "AI", 1000)
    
    # Simulate drop to 0.85
    for i in range(100):
        alert = detector.record_prediction(0.85, "AI", 1000)
    
    # Should trigger alert on the drop
    assert alert is not None
    assert "Confidence dropped" in alert["reason"]
```

---

## Success Metrics

After implementation, you should see:

1. **Zero surprises**: `/api/drift-status` should show "healthy" unless something is actually wrong
2. **Transparent pricing**: Users can see uncertainty bands, know when the model is unsure
3. **Faster API**: Removed slow plugins → p95 latency drops 20-30%
4. **Confidence calibration**: Model says 95% → turns out right 95% of the time (not 87%)
5. **Proactive alerts**: Drift detected before users complain

---

## Rollback Plan

If drift detection causes issues:

```bash
# Quickly disable it:
export ENABLE_DRIFT_DETECTION=false

# The record_prediction() calls become no-ops
# /api/drift-status returns "monitoring_disabled"
```

No code changes needed — just env var.
