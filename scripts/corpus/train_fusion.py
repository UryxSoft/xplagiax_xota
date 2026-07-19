"""
train_fusion.py — [doc A, pasos A.3-A.5] Train + calibrate the fusion, report REAL metrics.

Pipeline:
  1. Group-aware split by author_id (70/15/15) — no author leaks across splits, so the
     model can't memorize authors (doc A paso A.3.2). Each LLM counts as one "author".
  2. FusionClassifier.fit on train.
  3. TemperatureScaler fitted on validation (Guo 2017).
  4. Test metrics: ROC-AUC, Brier, ECE (pre/post calibration), TPR@FPR=1% (the metric
     that matters — docs/sota rule of gold), reliability table.
  5. Saves weights JSON consumable by FUSION_WEIGHTS_PATH (production wiring in
     app/engine/fusion.py::get_fusion_classifier).

    .venv/bin/python scripts/corpus/train_fusion.py --vectors dataset/vectors \
        --out models/fusion_weights.json

Deploy: set FUSION_WEIGHTS_PATH=/path/fusion_weights.json AND bump MODEL_VERSION.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app", "engine"))

import numpy as np  # noqa: E402


def tpr_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float = 0.01) -> tuple[float, float]:
    """(TPR, threshold) at the largest threshold whose FPR ≤ target."""
    human_scores = np.sort(p[y == 0])
    if human_scores.size == 0:
        return float("nan"), 0.5
    thr = float(np.quantile(human_scores, 1.0 - target_fpr))
    tpr = float((p[y == 1] >= thr).mean()) if (y == 1).any() else float("nan")
    return tpr, thr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors", default="dataset/vectors")
    ap.add_argument("--out", default="models/fusion_weights.json")
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    from fusion import FusionClassifier, FEATURE_NAMES
    from calibration import TemperatureScaler, compute_ece, brier_score, reliability_bins

    X = np.load(os.path.join(args.vectors, "X.npy"))
    meta = [json.loads(l) for l in open(os.path.join(args.vectors, "meta.jsonl"), encoding="utf-8")]
    assert len(meta) == X.shape[0], f"meta ({len(meta)}) != X rows ({X.shape[0]})"
    y = np.array([m["label"] for m in meta], dtype=np.int64)
    groups = np.array([m.get("author_id") or m.get("doc_id") or str(i)
                       for i, m in enumerate(meta)])
    print(f"{X.shape[0]} samples · {X.shape[1]} features · "
          f"{int((y == 1).sum())} AI / {int((y == 0).sum())} human · "
          f"{len(set(groups))} author groups")

    # ── 1. Group-aware 70/15/15 split ─────────────────────────────────────────
    from sklearn.model_selection import GroupShuffleSplit
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_idx, rest_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed)
    val_rel, test_rel = next(gss2.split(X[rest_idx], y[rest_idx], groups[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ── 2. Fit ────────────────────────────────────────────────────────────────
    clf = FusionClassifier().fit(X[train_idx], y[train_idx])

    def probs(idx: np.ndarray) -> np.ndarray:
        return np.array([clf.predict_proba_vec(X[i]).probability for i in idx])

    # ── 3. Calibrate on validation ────────────────────────────────────────────
    p_val = probs(val_idx)
    ts = TemperatureScaler().fit(p_val, y[val_idx])
    clf.attach_calibrator(ts)
    print(f"TemperatureScaler: T={ts.temperature:.3f}")

    # ── 4. Test metrics (raw = calibrator detached; calibrated = attached) ───
    clf._calibrator = None
    p_raw = probs(test_idx)
    clf.attach_calibrator(ts)
    p_cal = probs(test_idx)
    y_test = y[test_idx]

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, p_cal)
    tpr1, thr1 = tpr_at_fpr(y_test, p_cal, 0.01)
    print("\n══ TEST METRICS ══")
    print(f"ROC-AUC          : {auc:.4f}")
    print(f"Brier            : {brier_score(p_cal, y_test):.4f}")
    print(f"ECE  (raw)       : {compute_ece(p_raw, y_test):.4f}")
    print(f"ECE  (calibrated): {compute_ece(p_cal, y_test):.4f}")
    print(f"TPR @ FPR=1%     : {tpr1:.4f}  (threshold={thr1:.3f})")
    print("Reliability (calibrated): conf → acc (n)")
    for conf, acc, n in reliability_bins(p_cal, y_test, n_bins=10):
        if n:
            print(f"  {conf:.2f} → {acc:.2f}  ({n})")

    # Per-language slice (bias check, previous audit §11)
    langs = np.array([m.get("lang", "en") for m in meta])[test_idx]
    for lang in sorted(set(langs)):
        mask = langs == lang
        if mask.sum() >= 30 and len(set(y_test[mask])) == 2:
            print(f"  [{lang}] AUC={roc_auc_score(y_test[mask], p_cal[mask]):.4f} (n={int(mask.sum())})")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    payload = clf.to_payload()
    payload["trained_at"] = date.today().isoformat()
    payload["n_train"] = int(len(train_idx))
    payload["metrics"] = {
        "roc_auc": round(float(auc), 4),
        "ece_calibrated": round(float(compute_ece(p_cal, y_test)), 4),
        "tpr_at_fpr_1pct": round(float(tpr1), 4),
        "threshold_fpr_1pct": round(float(thr1), 4),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWeights -> {args.out}")
    print("Deploy: export FUSION_WEIGHTS_PATH plus a MODEL_VERSION bump.")
    assert list(payload["feature_names"]) == list(FEATURE_NAMES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
