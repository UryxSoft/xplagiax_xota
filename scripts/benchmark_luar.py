"""
benchmark_luar.py — GO/NO-GO del engine de embeddings de autoría (guía D.1/D.3).

Modos
-----
1. `benchmark`: ¿separa mismo-autor de distinto-autor en TU corpus (español)?
   Entrada: JSONL con {"author_id": ..., "chunk": ...} — 50 autores × ≥4 chunks
   (2 documentos × 2 chunks) recomendado.

       ENABLE_AUTHOR_EMBEDDING=1 AUTHOR_EMBED_DOWNLOAD=1 \
       .venv/bin/python scripts/benchmark_luar.py benchmark bench_autores.jsonl

   Decisión (guía D.1): AUROC ≥ 0.75 GO · 0.65-0.75 señal débil · < 0.65 NO-GO.

2. `threshold`: calibra AUTHOR_EMBED_OUTLIER_THRESHOLD (guía D.3).
   Entrada: JSONL con {"text": ...} — documentos MONO-AUTOR (tesis pre-2022).
   Calcula la distribución de similaridad chunk→centroide y reporta el
   percentil 5 (solo 5% de chunks legítimos caerían como outlier).

       ENABLE_AUTHOR_EMBEDDING=1 \
       .venv/bin/python scripts/benchmark_luar.py threshold tesis_monoautor.jsonl
"""

import itertools
import json
import sys

import numpy as np


def _load_engine():
    sys.path.insert(0, ".")
    sys.path.insert(0, "app/engine")
    from app.engine import author_embedding
    if not author_embedding.is_available():
        sys.exit(
            "Engine no disponible. Ejecuta con ENABLE_AUTHOR_EMBEDDING=1 "
            "(y AUTHOR_EMBED_DOWNLOAD=1 la primera vez para descargar el modelo; "
            "requiere `pip install einops`)."
        )
    return author_embedding


def cmd_benchmark(path: str) -> None:
    eng = _load_engine()
    rows = [json.loads(l) for l in open(path)]
    if len(rows) < 40:
        print(f"AVISO: solo {len(rows)} chunks — resultado poco fiable (usa 200+).")

    E = eng.embed_chunks([r["chunk"] for r in rows])
    authors = [r["author_id"] for r in rows]

    same, diff = [], []
    for i, j in itertools.combinations(range(len(rows)), 2):
        sim = float(E[i] @ E[j])
        (same if authors[i] == authors[j] else diff).append(sim)

    from sklearn.metrics import roc_auc_score
    y = [1] * len(same) + [0] * len(diff)
    auroc = roc_auc_score(y, same + diff)

    print(f"pares mismo-autor : {len(same):6d}  sim media = {np.mean(same):.3f}")
    print(f"pares cross-autor : {len(diff):6d}  sim media = {np.mean(diff):.3f}")
    print(f"AUROC verificación de autoría: {auroc:.3f}")
    if auroc >= 0.75:
        print("→ GO: el modelo separa autores en tu corpus. Sigue a D.2/D.3.")
    elif auroc >= 0.65:
        print("→ SEÑAL DÉBIL: usable con peso bajo; planifica D.4 (encoder propio).")
    else:
        print("→ NO-GO: no integres este modelo para este idioma/dominio. "
              "Prueba otra alternativa o salta a D.4.")


def cmd_threshold(path: str) -> None:
    eng = _load_engine()
    sims_all = []
    n_docs = 0
    for line in open(path):
        text = json.loads(line)["text"]
        chunks = eng._build_chunks(text)
        if len(chunks) < eng.MIN_CHUNKS:
            continue
        E = np.asarray(eng.embed_chunks(chunks), dtype=np.float64)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        c = E.mean(axis=0)
        c /= np.linalg.norm(c)
        sims_all.extend((E @ c).tolist())
        n_docs += 1

    sims = np.array(sims_all)
    p5 = float(np.quantile(sims, 0.05))
    print(f"documentos usados : {n_docs}")
    print(f"chunks totales    : {len(sims)}")
    print(f"similaridad media : {sims.mean():.3f}   p5 = {p5:.3f}   min = {sims.min():.3f}")
    print(f"\n→ exporta: AUTHOR_EMBED_OUTLIER_THRESHOLD={p5:.2f}")
    print("Valida después el recall con injertos sintéticos (guía D.3 paso 2).")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("benchmark", "threshold"):
        sys.exit(__doc__)
    {"benchmark": cmd_benchmark, "threshold": cmd_threshold}[sys.argv[1]](sys.argv[2])
