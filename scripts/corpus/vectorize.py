"""
vectorize.py — [doc A, paso A.3.1] Corpus JSONL → fusion feature vectors.

Runs the FULL production pipeline (PluginOrchestrator + FusionFeatureBuilder) over
every corpus sample so training sees exactly the vectors production will produce.
CPU-heavy (neural ensemble per sample) — run overnight / niced:

    nice -n 15 .venv/bin/python scripts/corpus/vectorize.py \
        dataset/human.jsonl dataset/ai_anthropic.jsonl --out dataset/vectors

Resumable: already-vectorized sample keys are skipped on rerun. Output:
    dataset/vectors/X.npy   (n, FUSION_VECTOR_DIM) float64
    dataset/vectors/meta.jsonl  one line per row: label, lang, domain, author_id, doc_id, key
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app", "engine"))

import numpy as np  # noqa: E402


def sample_key(rec: dict) -> str:
    return hashlib.sha256(rec["text"].encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="corpus JSONL files (human + ai)")
    ap.add_argument("--out", default="dataset/vectors")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new samples (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.jsonl")
    x_path = os.path.join(args.out, "X.npy")

    done: set[str] = set()
    rows: list[list[float]] = []
    if os.path.exists(meta_path) and os.path.exists(x_path):
        with open(meta_path, encoding="utf-8") as fh:
            for line in fh:
                done.add(json.loads(line)["key"])
        rows = [list(r) for r in np.load(x_path)]
        print(f"Resuming: {len(done)} samples already vectorized.")

    from plugin_orchestrator import PluginOrchestrator, PluginConfig
    from fusion import FusionFeatureBuilder, FUSION_VECTOR_DIM

    orch = PluginOrchestrator(PluginConfig(enable_forensic_report=False))
    builder = FusionFeatureBuilder()

    new = 0
    t0 = time.perf_counter()
    with open(meta_path, "a", encoding="utf-8") as meta_out:
        for path in args.inputs:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    key = sample_key(rec)
                    if key in done:
                        continue
                    try:
                        result = orch.run(rec["text"])
                        ff = builder.build(result["detection_result"],
                                           result["additional_analyses"])
                    except Exception as exc:
                        print(f"  skip {key}: {exc}", file=sys.stderr)
                        continue
                    rows.append([float(v) for v in ff.vector])
                    meta_out.write(json.dumps({
                        "key": key, "label": int(rec["label"]),
                        "lang": rec.get("lang", "en"), "domain": rec.get("domain", ""),
                        "author_id": rec.get("author_id", ""), "doc_id": rec.get("doc_id", ""),
                    }, ensure_ascii=False) + "\n")
                    done.add(key)
                    new += 1
                    if new % 25 == 0:
                        np.save(x_path, np.asarray(rows, dtype=np.float64))
                        meta_out.flush()
                        rate = new / max(time.perf_counter() - t0, 1e-9)
                        print(f"  {new} new ({rate*3600:.0f}/h) — total {len(rows)}")
                    if args.limit and new >= args.limit:
                        break

    np.save(x_path, np.asarray(rows, dtype=np.float64).reshape(-1, FUSION_VECTOR_DIM))
    print(f"{len(rows)} vectors ({new} new) -> {x_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
