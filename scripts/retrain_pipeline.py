#!/usr/bin/env python3
"""
scripts/retrain_pipeline.py — Model retraining pipeline (anti-drift).

AI-detection models age: every new LLM family (o1, Gemini 3, ...) is
out-of-distribution for the current ensemble, and precision decays silently.
This pipeline gives that decay an operational answer. Run it when
/api/drift-status reports "degraded", or on a monthly schedule.

Stages (run individually or chained with `all`):

  collect    Validate the labeled corpus layout:
                 <corpus>/human/*.txt   texts verified as human-written
                 <corpus>/ai/*.txt      texts verified as AI-generated
             Prints per-class counts and refuses to continue when a class is
             below --min-per-class (a lopsided corpus produces a lopsided model).

  evaluate   Score a weights file (or the CURRENT production ensemble when
             --weights is omitted) against the corpus with the same
             analyze_fast() aggregation the API serves. Writes a metrics JSON
             (accuracy, per-class recall, mean confidence).

  train      Fine-tuning scaffold. Prints the exact recipe (base model,
             hyperparameters, seeds) and exits non-zero — fine-tuning needs a
             GPU box and a transformers Trainer setup that does not belong in
             the serving container. Keeping the stage explicit means the
             pipeline is honest about what is and is not automated here.

  promote    Atomically roll new weights into production:
                 1. evaluate NEW weights on the corpus
                 2. compare against CURRENT production metrics
                 3. refuse when the new accuracy is not >= current + --min-gain
                 4. copy current weights into MODEL_FALLBACK_DIR (rollback set)
                 5. install new weights + write metadata.json (version, date,
                    metrics, corpus fingerprint)
             detector_final._load_model() automatically falls back to
             MODEL_FALLBACK_DIR if the new weights are corrupt, and
             /api/drift-status shows which file each worker actually loaded.

Usage examples:
    python scripts/retrain_pipeline.py collect  --corpus data/gold
    python scripts/retrain_pipeline.py evaluate --corpus data/gold
    python scripts/retrain_pipeline.py promote  --corpus data/gold \\
        --weights /path/new/modernbert.bin --version 2026.08
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "app" / "engine"
DEFAULT_FALLBACK_DIR = os.getenv("MODEL_FALLBACK_DIR", str(ENGINE_DIR / "fallback"))


# ── Corpus helpers ─────────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> List[Tuple[str, str]]:
    """Return [(text, label)] with label in {"Human", "AI"}."""
    samples: List[Tuple[str, str]] = []
    for label, sub in (("Human", "human"), ("AI", "ai")):
        for path in sorted((corpus_dir / sub).glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                samples.append((text, label))
    return samples


def corpus_fingerprint(samples: List[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for text, label in samples:
        h.update(label.encode())
        h.update(hashlib.sha256(text.encode()).digest())
    return h.hexdigest()[:16]


# ── Stages ─────────────────────────────────────────────────────────

def stage_collect(args) -> int:
    corpus = Path(args.corpus)
    counts: Dict[str, int] = {}
    for sub in ("human", "ai"):
        d = corpus / sub
        counts[sub] = len(list(d.glob("*.txt"))) if d.is_dir() else 0
    print(json.dumps({"corpus": str(corpus), "counts": counts}, indent=2))

    problems = [s for s, n in counts.items() if n < args.min_per_class]
    if problems:
        print(
            f"ERROR: classes below --min-per-class={args.min_per_class}: "
            f"{', '.join(problems)}. Label more data before training — a "
            "lopsided corpus trains a lopsided detector.",
            file=sys.stderr,
        )
        return 1
    return 0


def stage_evaluate(args) -> int:
    corpus = Path(args.corpus)
    samples = load_corpus(corpus)
    if not samples:
        print(f"ERROR: no samples under {corpus}/human and {corpus}/ai", file=sys.stderr)
        return 1

    if args.weights:
        # Point the loader at the candidate weights BEFORE the engine imports.
        os.environ["XPLAGIAX_EVAL_WEIGHTS"] = args.weights
        print(
            "NOTE: candidate-weights evaluation loads the file named by "
            "--weights in place of modernbert.bin. The seed-12/seed-22 "
            "companions are kept from production.",
        )

    sys.path.insert(0, str(REPO_ROOT))
    from app.engine.detector_final import analyze_fast  # heavy import — deliberate

    correct = 0
    per_class = {"Human": {"n": 0, "ok": 0}, "AI": {"n": 0, "ok": 0}}
    confidences: List[float] = []
    for text, label in samples:
        result = analyze_fast(text)
        summary = result.get("overall_summary", {})
        pred = summary.get("overall_prediction", "Unknown")
        conf = max(
            summary.get("total_human_percentage", 50),
            summary.get("total_ai_percentage", 50),
        ) / 100.0
        confidences.append(conf)
        per_class[label]["n"] += 1
        if pred == label:
            per_class[label]["ok"] += 1
            correct += 1

    metrics = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "corpus": str(corpus),
        "corpus_fingerprint": corpus_fingerprint(samples),
        "samples": len(samples),
        "accuracy": round(correct / len(samples), 4),
        "recall_human": round(
            per_class["Human"]["ok"] / per_class["Human"]["n"], 4
        ) if per_class["Human"]["n"] else None,
        "recall_ai": round(
            per_class["AI"]["ok"] / per_class["AI"]["n"], 4
        ) if per_class["AI"]["n"] else None,
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "weights": args.weights or "production",
    }
    out = Path(args.metrics_out)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


def stage_train(args) -> int:
    print(
        "Fine-tuning is NOT automated inside the serving repo (needs a GPU box).\n"
        "Recipe used for the production ensemble:\n"
        "  base model     : answerdotai/ModernBERT-base, num_labels=41\n"
        "  objective      : cross-entropy over the 41-class label_mapping\n"
        "                   (index 24 = human) — keep the mapping IDENTICAL\n"
        "  seeds          : train 3 runs (e.g. seed 2 / 12 / 22) for the ensemble\n"
        "  data           : the collect-stage corpus + previous training data;\n"
        "                   ALWAYS include fresh samples of the LLM families that\n"
        "                   triggered the drift alert\n"
        "  export         : torch.save(model.state_dict(), <weights>) per seed\n"
        "Then run:  retrain_pipeline.py promote --weights <new.bin> --version <v>",
        file=sys.stderr,
    )
    return 2


def stage_promote(args) -> int:
    if not args.weights or not args.version:
        print("ERROR: promote requires --weights and --version", file=sys.stderr)
        return 1
    new_weights = Path(args.weights)
    if not new_weights.is_file():
        print(f"ERROR: weights file not found: {new_weights}", file=sys.stderr)
        return 1

    target = ENGINE_DIR / args.target_name
    fallback_dir = Path(args.fallback_dir)
    corpus = Path(args.corpus)
    samples = load_corpus(corpus)
    if not samples:
        print(f"ERROR: promote needs the gold corpus for the accuracy gate", file=sys.stderr)
        return 1

    # 1-2. Evaluate current production, then the candidate.
    import subprocess
    me = Path(__file__)
    cur_metrics_path = "/tmp/xplagiax_metrics_current.json"
    new_metrics_path = "/tmp/xplagiax_metrics_candidate.json"
    for weights, metrics_out in ((None, cur_metrics_path), (str(new_weights), new_metrics_path)):
        cmd = [sys.executable, str(me), "evaluate", "--corpus", str(corpus),
               "--metrics-out", metrics_out]
        if weights:
            cmd += ["--weights", weights]
        # Separate process per evaluation: the engine loads weights at import
        # time, so current and candidate cannot coexist in one interpreter.
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc

    current = json.loads(Path(cur_metrics_path).read_text())
    candidate = json.loads(Path(new_metrics_path).read_text())
    gain = candidate["accuracy"] - current["accuracy"]
    print(f"accuracy: current={current['accuracy']} candidate={candidate['accuracy']} gain={gain:+.4f}")

    # 3. Accuracy gate.
    if gain < args.min_gain:
        print(
            f"REFUSED: candidate gain {gain:+.4f} < --min-gain {args.min_gain}. "
            "Shipping a model that is not measurably better than the current one "
            "is how detection services quietly rot.",
            file=sys.stderr,
        )
        return 3

    # 4. Preserve the rollback set.
    fallback_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copy2(target, fallback_dir / target.name)
        print(f"rollback copy: {target} -> {fallback_dir / target.name}")

    # 5. Install + metadata.
    shutil.copy2(new_weights, target)
    metadata = {
        "version": args.version,
        "promoted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "weights_file": target.name,
        "metrics_current": current,
        "metrics_candidate": candidate,
        "corpus_fingerprint": candidate["corpus_fingerprint"],
    }
    (ENGINE_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"PROMOTED {target.name} -> version {args.version}.\n"
        f"Set MODEL_VERSION={args.version} and MODEL_FALLBACK_DIR={fallback_dir} "
        "in the deployment env, then restart workers."
    )
    return 0


# ── CLI ────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["collect", "evaluate", "train", "promote", "all"])
    p.add_argument("--corpus", default="data/gold", help="labeled corpus dir (human/ + ai/)")
    p.add_argument("--weights", default=None, help="candidate weights file")
    p.add_argument("--version", default=None, help="new MODEL_VERSION on promote")
    p.add_argument("--target-name", default="modernbert.bin",
                   help="production weights filename inside app/engine/")
    p.add_argument("--fallback-dir", default=DEFAULT_FALLBACK_DIR)
    p.add_argument("--metrics-out", default="/tmp/xplagiax_metrics.json")
    p.add_argument("--min-per-class", type=int, default=50)
    p.add_argument("--min-gain", type=float, default=0.0,
                   help="required accuracy improvement to promote (default: no regression)")
    args = p.parse_args()

    stages = {
        "collect": stage_collect,
        "evaluate": stage_evaluate,
        "train": stage_train,
        "promote": stage_promote,
    }
    if args.stage == "all":
        for name in ("collect", "evaluate", "train", "promote"):
            rc = stages[name](args)
            if rc != 0:
                return rc
        return 0
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
