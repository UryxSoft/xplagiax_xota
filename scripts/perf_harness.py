"""
perf_harness.py — [Fase-2 C.4] Baseline CPU/latency measurement for the pipeline.

Measures wall time, CPU time and peak RSS of the full forensic pipeline
(PluginOrchestrator.run) over synthetic documents of increasing size, so that
optimizations (e.g. M-22 hybrid paragraph mode) can be demonstrated with numbers
instead of claims.

Usage (models must be present; runs in-process, no server needed):

    .venv/bin/python scripts/perf_harness.py                 # default sizes
    .venv/bin/python scripts/perf_harness.py 500 2000 8000   # word counts
    HYBRID_WINDOWS=1 .venv/bin/python scripts/perf_harness.py   # legacy windows mode

For flame graphs, wrap with py-spy:

    py-spy record -o profile.svg -- .venv/bin/python scripts/perf_harness.py 2000

Compare runs by exporting the JSON lines this prints (one per size) before and
after a change. KPIs: wall_s, cpu_s, rss_peak_mb.
"""

import json
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "engine"))

_WORDS = (
    "analysis system model result method data study research process value "
    "approach development information structure evidence observation theory "
    "experiment measurement conclusion hypothesis framework parameter variable"
).split()


def synthetic_doc(n_words: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    paras, count = [], 0
    while count < n_words:
        plen = rng.randint(40, 120)
        sent, para = [], []
        for i in range(plen):
            sent.append(rng.choice(_WORDS))
            if len(sent) >= rng.randint(8, 22):
                para.append(" ".join(sent).capitalize() + ".")
                sent = []
        if sent:
            para.append(" ".join(sent).capitalize() + ".")
        paras.append(" ".join(para))
        count += plen
    return "\n\n".join(paras)


def measure(text: str) -> dict:
    from plugin_orchestrator import PluginOrchestrator, PluginConfig
    orch = PluginOrchestrator(PluginConfig(enable_forensic_report=True))
    t0_wall, t0_cpu = time.perf_counter(), time.process_time()
    result = orch.run(text)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 * 1024 if sys.platform == "darwin" else 1024)
    det = result.get("detection_result")
    return {
        "words": len(text.split()),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "rss_peak_mb": round(rss_mb, 1),
        "hybrid_windows_mode": os.getenv("HYBRID_WINDOWS", "0"),
        "verdict_pred": getattr(det, "prediction", None),
    }


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [500, 2000, 8000]
    for n in sizes:
        print(json.dumps(measure(synthetic_doc(n))))
