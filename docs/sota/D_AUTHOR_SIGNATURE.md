# Guía D — Author signature con embeddings de autoría (LUAR)

> **Estado: D.2 IMPLEMENTADO** (2026-07). Código en
> `app/engine/author_embedding.py` (engine), wiring en
> `plugin_orchestrator.py` y `app/plugins/author_signature.py` (con fallback
> al estilométrico), benchmark D.1/D.3 en `scripts/benchmark_luar.py`,
> tests en `tests/test_author_embedding.py`. Apagado por defecto — pendiente
> de TU parte: correr D.1 (GO/NO-GO) con datos reales, calibrar D.3, y
> activar con `ENABLE_AUTHOR_EMBEDDING=1`. Requiere `pip install einops`
> y primera descarga con `AUTHOR_EMBED_DOWNLOAD=1`.

**Objetivo:** sustituir las features estilométricas a mano del plugin
`author_signature` por embeddings de verificación de autoría (estilo
LUAR / PAN). Responde: *"¿la sección 1 y la sección 4 las escribió la misma
persona?"* — la señal más robusta a paráfrasis: el humanizador cambia
n-gramas, no la firma profunda de autor.

La feature `author_outlier_ratio` **ya existe** en el schema de fusion
([fusion.py](../../app/engine/fusion.py), `_FUSION_SCHEMA`) — se sustituye la
implementación por dentro; la fusion no cambia de forma.

**Dónde se ejecuta:**

| Fase | Dónde |
|------|-------|
| D.1 Validar LUAR en español | **Colab** (rápido con GPU, factible en CPU) |
| D.2 Implementar el engine | Local |
| D.3 Calibrar umbral de outlier | Local |
| D.4 (Plan B) Entrenar encoder propio | **Colab/GPU alquilada** + tu BD |

---

## D.1 Validar que LUAR sirve en español (ANTES de integrar nada)

LUAR (`rrivera1849/LUAR-MUD` en HuggingFace) está entrenado con Reddit en
inglés. Puede transferir razonablemente o puede no hacerlo — **mídelo antes
de escribir una línea de integración**.

### Paso 1 — Prepara el mini-benchmark

De tu BD: 50 autores con ≥ 2 documentos cada uno (pre-2022, español).
De cada documento, 2 chunks de ~400 palabras. Total ≈ 200 chunks.

```json
{"author_id": "a1", "doc_id": "d1", "chunk": "..."}
```

### Paso 2 — Notebook Colab

```python
# Celda 1
!pip install -q transformers sentence-transformers

# Celda 2 — cargar LUAR
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "rrivera1849/LUAR-MUD"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()

@torch.inference_mode()
def embed(texts: list[str]) -> torch.Tensor:
    # LUAR espera (batch, n_utterances, seq_len) — usamos 1 utterance por texto
    enc = tok(texts, padding="max_length", truncation=True,
              max_length=512, return_tensors="pt")
    enc = {k: v.unsqueeze(1) for k, v in enc.items()}   # añade dim utterance
    out = model(**enc)
    return torch.nn.functional.normalize(out, dim=-1)

# Celda 3 — métrica: ¿mismo autor queda más cerca que distinto autor?
import itertools, numpy as np, json

chunks = [json.loads(l) for l in open("bench_autores.jsonl")]
E = embed([c["chunk"] for c in chunks]).numpy()

same, diff = [], []
for i, j in itertools.combinations(range(len(chunks)), 2):
    sim = float(E[i] @ E[j])
    if chunks[i]["author_id"] == chunks[j]["author_id"]:
        same.append(sim)
    else:
        diff.append(sim)

print(f"same-author  media={np.mean(same):.3f}")
print(f"cross-author media={np.mean(diff):.3f}")

from sklearn.metrics import roc_auc_score
y = [1]*len(same) + [0]*len(diff)
print("AUROC verificación:", roc_auc_score(y, same + diff))
```

**✅ Checkpoint — decisión GO/NO-GO:**

- AUROC ≥ 0.75 → LUAR transfiere: sigue a D.2.
- AUROC 0.65-0.75 → usable como señal débil; considera D.4 a medio plazo.
- AUROC < 0.65 → NO integres LUAR para español; salta a D.4 (plan B) o usa
  alternativa multilingüe (`sentence-transformers` style-embedding
  `AnnaWegmann/Style-Embedding` — repite este mismo benchmark con ella).

---

## D.2 Implementar el engine

### Paso 1 — `app/engine/author_embedding.py`

Patrones del repo (carga módulo-nivel, CoW, `local_files_only`, cache):

```python
"""Embeddings de autoría para detección de ruptura de firma intra-documento."""

# 1. Carga módulo-nivel con try/except → _available
# 2. embed_chunks(texts: List[str]) -> np.ndarray   (batch de 8, normalizado)
# 3. analyze_document(text: str) -> dict:
#    a. Trocear por secciones/párrafos en chunks de 300-500 palabras
#       (reusa TextSegmenter.split_paragraphs de hybrid_segment_detector
#       y agrupa párrafos hasta ~400 palabras)
#    b. Si < 3 chunks → {"reliable": False} (no hay nada que comparar)
#    c. E = embed_chunks(chunks)
#    d. centroide = E.mean(axis=0, keepdims=True) normalizado
#    e. sims = E @ centroide.T
#    f. outliers = sims < umbral   (D.3)
```

Salida del análisis:

```python
{
    "reliable": True,
    "n_chunks": 14,
    "mean_self_similarity": 0.81,
    "outlier_ratio": 0.21,            # → feature author_outlier_ratio
    "outlier_chunks": [3, 7, 8],      # índices → mapear a offsets de texto
    "max_break": {"after_chunk": 6, "similarity_drop": 0.34},
}
```

### Paso 2 — Conectar al orquestador

En `plugin_orchestrator.py`, el bloque `enable_author_signature` hoy llama a
`compute_authorship_consistency(...)` sobre features estilométricas. Añade
ruta nueva: si `author_embedding._available`, usa embeddings; si no, cae a
la implementación actual (mismo patrón de fallback que en todo el repo).
El dict resultante alimenta `additional["author_signature"]` — la fusion ya
lo consume.

### Paso 3 — RAM/CPU

LUAR-MUD ≈ 330 MB (base RoBERTa-like) — asumible junto al resto. En CPU,
14 chunks × 1 forward ≈ segundos. Cache por sha256 del texto (patrón
`_FAST_CACHE`). Gate por env: `ENABLE_AUTHOR_EMBEDDING=1`.

---

## D.3 Calibrar el umbral de outlier

No inventes el umbral: mídelo.

1. **Documentos mono-autor** (100 tesis pre-2022): calcula la distribución
   de `sims` (similaridad de cada chunk al centroide). El percentil 5 de esa
   distribución = tu umbral base (solo 5% de chunks legítimos caerían).
2. **Documentos sintéticos mezclados**: coge 100 tesis humanas e injértales
   1-3 secciones generadas por IA (de tu corpus A.2). Mide cuántos injertos
   caen bajo el umbral (recall del outlier).

```python
umbral = np.quantile(sims_monoautor, 0.05)
recall = (sims_injertos < umbral).mean()
print(f"umbral={umbral:.3f}  recall de injertos={recall:.2%}")
```

**✅ Checkpoint:** recall de injertos ≥ 0.5 con FPR de chunks ≤ 5%. Si el
recall es bajo, prueba chunks más grandes (500-800 palabras — LUAR mejora
con más texto por unidad).

---

## D.4 Plan B (y el verdadero SOTA): entrenar TU encoder con TU BD

Con 300M de documentos tienes pares mismo-autor a escala que nadie tiene
(tesis + papers del mismo autor, capítulos de la misma tesis). Un encoder
contrastivo entrenado ahí supera a LUAR en tu dominio con seguridad.

Esquema (esto sí es proyecto de GPU seria — A100, días):

1. **Dataset**: pares positivos = 2 fragmentos del mismo autor (distinto
   documento mejor que mismo documento — evita aprender el tema); negativos
   = fragmentos de autores distintos de la MISMA disciplina y época (para
   que no aprenda a distinguir temas ni décadas).
2. **Modelo base**: XLM-RoBERTa-base o mDeBERTa (multilingüe nativo).
3. **Pérdida**: InfoNCE / SupCon con batch grande (256+) y negativos duros
   (misma disciplina).
4. **Entrenamiento**: HuggingFace `sentence-transformers` con
   `MultipleNegativesRankingLoss` es el camino corto y probado.
5. **Evaluación**: el benchmark de D.1 con autores held-out. Meta: AUROC ≥ 0.85.

Colab Pro (A100) sirve para un piloto de 1-5M pares; para el run completo
alquila GPU por horas (Lambda/RunPod). Piloto primero, siempre.

---

## Errores dummy típicos

1. **Integrar LUAR sin el benchmark D.1** → señal muerta en español y no lo
   sabes. GO/NO-GO primero.
2. **Chunks < 100 palabras** → embeddings ruidosos; 300-500 mínimo.
3. **Comparar secciones de naturaleza distinta** — agradecimientos vs
   metodología difieren aunque sea el mismo humano. Excluye del análisis:
   agradecimientos, índices, bibliografía (reusa `_strip_references_section`),
   apéndices de tablas.
4. **Umbral inventado** en vez del percentil medido (D.3).
5. **< 3 chunks y aun así emitir veredicto** → `reliable: False` y fuera.
6. **En D.4, negativos de disciplinas distintas** → el modelo aprende a
   distinguir temas, no autores. Negativos duros: misma disciplina.
7. **Positivos solo intra-documento** → aprende el tema del documento, no
   la firma. Positivos inter-documento del mismo autor.
