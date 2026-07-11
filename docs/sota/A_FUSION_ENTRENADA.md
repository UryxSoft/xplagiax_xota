# Guía A — Fusion entrenada + calibración medible + predicción conformal

**Objetivo:** sustituir el heurístico interino de `FusionClassifier`
(`app/engine/fusion.py`) por una regresión logística entrenada con corpus real,
calibrada (isotónica), con métricas honestas (ECE, Brier, TPR@FPR=1%) y
veredictos con garantía conformal ("IA con cobertura 90%").

**Dónde se ejecuta cada fase:**

| Fase | Dónde |
|------|-------|
| A.1 Construir corpus humano | Local / servidor con acceso a tu BD |
| A.2 Generar clase IA | **Colab o GPU** (modelos abiertos) + APIs (frontera) |
| A.3 Extraer vectores de fusion | Local (usa tu pipeline XplagiaX, CPU, lento) |
| A.4 Entrenar + calibrar | Local (sklearn, CPU, segundos) |
| A.5 Conformal | Local (numpy, trivial) |
| A.6 Persistir + cargar en producción | Local (código en el repo) |

---

## A.0 Prerrequisitos

```bash
# En el venv del proyecto
.venv/bin/pip install scikit-learn numpy pandas
```

Estructura de trabajo (fuera del repo para no ensuciar git):

```
~/xplagiax_corpus/
├── raw/            # documentos fuente
├── dataset/        # JSONL final etiquetado
├── vectors/        # X.npy, y.npy por split
└── models/         # fusion_weights.json, calibrador
```

---

## A.1 Corpus humano (clase 0)

### Paso 1 — Extraer de tu BD solo documentos pre-ChatGPT

**Regla inviolable: solo documentos con fecha de publicación < 2022-11-30.**
Todo lo posterior puede contener texto IA y envenena la clase humana.

```sql
-- Ajusta a tu esquema real
SELECT id, texto, idioma, disciplina, anio, autor_id
FROM documentos
WHERE fecha_publicacion < '2022-11-01'
  AND longitud_palabras BETWEEN 300 AND 200000;
```

### Paso 2 — Muestreo estratificado

No cojas 10.000 tesis de derecho en español y nada más. Estratifica:

- **Idioma**: mínimo español + inglés (tu mercado real primero).
- **Disciplina**: 6-10 áreas (salud, ingeniería, derecho, humanidades…).
- **Longitud**: cortos (300-1K palabras), medios (1K-10K), largos (10K+).
- **Tipo**: tesis, paper, patente.

Objetivo mínimo viable: **10.000 documentos humanos**. Ideal: 50.000+.

### Paso 3 — Trocear documentos largos

El detector opera a nivel párrafo/sección. Trocea tesis en unidades de
500-3.000 palabras (por capítulo/sección). Cada unidad es una muestra.
**Guarda `autor_id` y `documento_id` en cada muestra** — lo necesitas para
el split (Paso A.3.2).

Formato JSONL (`dataset/human.jsonl`):

```json
{"text": "...", "label": 0, "lang": "es", "domain": "derecho", "words": 1240, "author_id": "a-8812", "doc_id": "d-4471", "source": "tesis-pre2022"}
```

**✅ Checkpoint:** `wc -l dataset/human.jsonl` ≥ 10000. Distribución por
idioma/disciplina razonable (`pandas.value_counts()`). Ninguna fecha ≥ 2022-11.

---

## A.2 Clase IA (label 1) — aquí usas Colab

### Paso 1 — Diseña las tareas de generación

Por cada texto humano muestreado, genera una contraparte IA con una de
estas tareas (rota entre ellas):

| Tarea | Prompt base |
|-------|-------------|
| Continuar | "Continúa este texto académico manteniendo el estilo: {primeras 200 palabras}" |
| Reescribir | "Reescribe este texto con tus propias palabras, registro académico: {texto}" |
| Expandir | "Desarrolla este resumen en una sección completa de tesis: {abstract}" |
| Generar desde título | "Escribe la introducción de una tesis titulada: {título}" |
| Parafrasear (ataque) | Ver Guía E — DIPPER y humanizadores |

Así obtienes **pares paralelos** (mismo tema, mismo registro) — el modelo
aprende la diferencia humano/IA real, no diferencias de tema.

### Paso 2 — Canasta de modelos generadores

Reparte la generación (nunca un solo modelo, o el detector solo detecta ESE
modelo):

- **Abiertos (70% del volumen, baratos)**: Llama-3.1-8B-Instruct,
  Qwen2.5-7B-Instruct, DeepSeek-V3 (API barata), Mistral-7B-Instruct.
- **Frontera (30%, API de pago)**: GPT-4o/5, Claude Sonnet/Opus, Gemini.

### Paso 3 — Colab para los modelos abiertos

Notebook Colab (GPU T4 gratis o A100 de pago):

```python
# Celda 1 — instalar
!pip install -q vllm  # o transformers + accelerate si vllm falla en T4

# Celda 2 — generar con vLLM (rápido, batching automático)
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=4096,
          gpu_memory_utilization=0.9)
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1200)

import json
prompts, metas = [], []
with open("human_sample.jsonl") as f:   # súbelo a Colab o monta Drive
    for line in f:
        d = json.loads(line)
        prompts.append(f"Reescribe este texto académico con tus propias palabras:\n\n{d['text'][:3000]}")
        metas.append(d)

outputs = llm.generate(prompts, params)
with open("ai_generated.jsonl", "w") as f:
    for meta, out in zip(metas, outputs):
        f.write(json.dumps({
            "text": out.outputs[0].text.strip(),
            "label": 1, "lang": meta["lang"], "domain": meta["domain"],
            "generator": "qwen2.5-7b", "task": "rewrite",
            "author_id": f"gen-{meta['author_id']}", "doc_id": f"gen-{meta['doc_id']}",
        }, ensure_ascii=False) + "\n")
```

Repite cambiando `model=` por cada generador. Con T4 gratis: ~1-2K
generaciones/hora con 7B. Con A100: 5-10×.

**Trampas de Colab:** la sesión gratis muere a las ~4h y pierde disco —
**guarda a Google Drive cada N muestras** (`from google.colab import drive;
drive.mount('/content/drive')`), no al disco local de la VM.

### Paso 4 — Limpieza de la clase IA

- Elimina generaciones < 150 palabras (rechazos, respuestas vacías).
- Elimina las que empiezan por "Claro, aquí tienes…" / "Sure, here…"
  (o recórtalas) — son artefacto de chat, no texto académico, y el modelo
  aprendería ese atajo trivial.
- Deduplica (hash exacto + near-dup con MinHash si tienes tiempo).

**✅ Checkpoint:** nº muestras IA ≈ nº muestras humanas (balance). Mezcla de
generadores sin que ninguno pase de 30%. Leer 20 muestras al azar a mano:
deben parecer texto académico, no chat.

---

## A.3 Extraer vectores de fusion

### Paso 1 — Une y baraja

```bash
cat dataset/human.jsonl dataset/ai_generated.jsonl | shuf > dataset/full.jsonl
```

### Paso 2 — Split 70/15/15 POR GRUPO, nunca aleatorio puro

**El error que invalida todo el trabajo:** si el mismo autor (o el mismo
documento troceado) cae en train y test, tus métricas mienten.

```python
from sklearn.model_selection import GroupShuffleSplit
import json, numpy as np

rows = [json.loads(l) for l in open("dataset/full.jsonl")]
groups = [r["author_id"] for r in rows]   # agrupa por autor

gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, rest_idx = next(gss.split(rows, groups=groups))
# repite sobre rest_idx para separar cal (15%) y test (15%)
```

Guarda tres ficheros: `train.jsonl`, `cal.jsonl`, `test.jsonl`.

**✅ Checkpoint:** `set(autores_train) & set(autores_test) == set()` (vacío).

### Paso 3 — Corre el pipeline XplagiaX sobre cada documento

Script `scripts/extract_fusion_vectors.py` (créalo en el repo):

```python
"""Extrae vectores de fusion de un JSONL etiquetado. CPU, lento — paciencia."""
import json, sys, numpy as np, os
os.environ["FUSION_ACTIVE"] = "0"          # no contaminar con el heurístico
sys.path.insert(0, ".")
sys.path.insert(0, "app/engine")

from app.engine.plugin_orchestrator import PluginConfig, initialize_orchestrator, get_orchestrator
from app.engine.fusion import FusionClassifier

initialize_orchestrator(PluginConfig(
    enable_reference_check=False,   # sin red durante extracción masiva;
))                                  # actívalo solo si cacheaste las APIs (Guía C)
orch = get_orchestrator()
builder = FusionClassifier()._builder

X, y = [], []
for i, line in enumerate(open(sys.argv[1])):
    d = json.loads(line)
    try:
        result = orch.run(d["text"])
        ff = builder.build(result["detection_result"],
                           result.get("additional_analyses", {}))
        X.append(ff.vector); y.append(d["label"])
    except Exception as exc:
        print(f"[{i}] ERROR: {exc}", file=sys.stderr)
    if i % 100 == 0:
        print(f"{i} procesados"); np.save("X_partial.npy", np.array(X))

np.save(sys.argv[2], np.array(X)); np.save(sys.argv[3], np.array(y))
```

```bash
.venv/bin/python scripts/extract_fusion_vectors.py train.jsonl vectors/X_train.npy vectors/y_train.npy
# idem cal.jsonl y test.jsonl
```

Esto es CPU-bound y tarda (pipeline completo por documento). 10K docs ≈
horas/días. Opciones: correr por lotes en paralelo (varias instancias sobre
shards del JSONL) o alquilar una máquina multi-core temporal. **No es tarea
de Colab** (necesita tu app entera + modelos locales).

**✅ Checkpoint — sanidad de features (crítico):**

```python
import numpy as np
from app.engine.fusion import feature_names
X = np.load("vectors/X_train.npy")
for i, name in enumerate(feature_names()):
    col = X[:, i]
    print(f"{name:28s} min={col.min():.3f} max={col.max():.3f} std={col.std():.4f}")
```

Cualquier feature con `std=0` → ese plugin no corrió o devuelve constante.
**Arréglalo antes de entrenar** (el modelo le pondrá peso 0 y perderás la señal).

---

## A.4 Entrenar + calibrar

### Paso 1 — Entrenar (segundos, local)

```python
import numpy as np
from app.engine.fusion import FusionClassifier
from sklearn.metrics import roc_auc_score

X_tr, y_tr = np.load("vectors/X_train.npy"), np.load("vectors/y_train.npy")
X_te, y_te = np.load("vectors/X_test.npy"),  np.load("vectors/y_test.npy")

fc = FusionClassifier().fit(X_tr, y_tr)

probs_te = np.array([fc.predict_proba_vec(v).probability for v in X_te])
print("AUROC fusion:", roc_auc_score(y_te, probs_te))
```

**✅ Checkpoint:** AUROC fusion > AUROC de la feature neural sola
(`X_te[:, feature_names().index('neural_ai_prob')]` — ajusta el nombre al
schema real). Si no la supera: leakage, feature rota, o corpus demasiado fácil.

### Paso 2 — Calibración isotónica (con el split CAL, jamás train ni test)

```python
from sklearn.isotonic import IsotonicRegression
import pickle

X_cal, y_cal = np.load("vectors/X_cal.npy"), np.load("vectors/y_cal.npy")
probs_cal = np.array([fc.predict_proba_vec(v).probability for v in X_cal])

iso = IsotonicRegression(out_of_bounds="clip").fit(probs_cal, y_cal)
pickle.dump(iso, open("models/isotonic.pkl", "wb"))

class IsotonicCalibrator:                      # interfaz que espera attach_calibrator
    def __init__(self, iso): self.iso = iso
    def apply(self, p: float) -> float: return float(self.iso.predict([p])[0])

fc.attach_calibrator(IsotonicCalibrator(iso))
```

### Paso 3 — Métricas honestas (en TEST)

```python
from sklearn.metrics import brier_score_loss, roc_curve
import numpy as np

probs_cal_te = np.array([fc.predict_proba_vec(v).probability for v in X_te])

# Brier (menor = mejor; <0.10 es bueno)
print("Brier:", brier_score_loss(y_te, probs_cal_te))

# ECE con 10 bins (menor = mejor; <0.05 es bueno)
bins = np.linspace(0, 1, 11); ece = 0.0
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (probs_cal_te >= lo) & (probs_cal_te < hi)
    if m.sum(): ece += m.mean() * abs(probs_cal_te[m].mean() - y_te[m].mean())
print("ECE:", ece)

# LA métrica: TPR@FPR=1%
fpr, tpr, thr = roc_curve(y_te, probs_cal_te)
i = np.searchsorted(fpr, 0.01)
print(f"TPR@FPR=1%: {tpr[i]:.3f}  (umbral={thr[i]:.3f})")
```

Guarda ese umbral: es tu punto de operación de producción.

---

## A.5 Predicción conformal (split conformal, 20 líneas)

```python
# Sobre el split CAL, con probabilidades YA calibradas
probs_cal = np.array([fc.predict_proba_vec(v).probability for v in X_cal])

# No-conformidad = 1 - prob de la clase VERDADERA
s = np.where(y_cal == 1, 1 - probs_cal, probs_cal)

alpha = 0.10                                   # cobertura 90%
n = len(s)
q = np.quantile(s, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
print("q90:", q)                               # guárdalo en el JSON del modelo
```

En producción, para un texto con probabilidad calibrada `p`:

```python
pred_set = []
if 1 - p <= q: pred_set.append("AI")
if p <= q:     pred_set.append("Human")
# {"AI"}          → "IA, cobertura garantizada 90%"
# {"Human"}       → "Humano, cobertura garantizada 90%"
# {"AI","Human"}  → "NO CONCLUYENTE" (honesto — esto también es un resultado)
```

**✅ Checkpoint:** en TEST, la clase verdadera debe estar en el pred_set
~90% de las veces (±2%). Si no, hay bug en el cálculo de `s` o de `q`.

---

## A.6 Persistir y cargar en producción

### Paso 1 — Guarda todo en un JSON + pickle

```python
import json
json.dump({
    "weights": fc._weights.tolist(), "bias": fc._bias,
    "mean": fc._mean.tolist(), "std": fc._std.tolist(),
    "conformal_q90": float(q),
    "threshold_fpr1": float(thr[i]),
    "trained_on": "corpus-v1 2026-07", "n_train": len(y_tr),
}, open("models/fusion_weights.json", "w"))
# isotonic.pkl ya guardado en A.4
```

### Paso 2 — Carga al arranque del orquestador

En `plugin_orchestrator.py`, donde hoy se hace `FusionClassifier().predict_proba(...)`,
carga una vez a nivel módulo si `FUSION_WEIGHTS_PATH` está definido:

```python
_fusion = FusionClassifier()
_wpath = os.getenv("FUSION_WEIGHTS_PATH")
if _wpath and os.path.exists(_wpath):
    w = json.load(open(_wpath))
    _fusion.set_weights(w["weights"], w["bias"], w["mean"], w["std"])
    iso = pickle.load(open(os.path.join(os.path.dirname(_wpath), "isotonic.pkl"), "rb"))
    _fusion.attach_calibrator(IsotonicCalibrator(iso))
```

### Paso 3 — Invalida caches

Sube `MODEL_VERSION` (env) — el cache de resultados está namespaced por
versión y dejará de servir veredictos del heurístico viejo.

**✅ Checkpoint final:** `curl /analyze` con un texto conocido → el campo
`fusion.source` ya no dice heurístico/untrained y `fusion.calibrated: true`.

---

## Errores dummy que invalidan TODO (repaso final)

1. **Split aleatorio en vez de por autor/documento** → métricas infladas, mentira.
2. **Calibrar con train** → ECE de fantasía.
3. **Clase humana con documentos post-2022** → clase contaminada.
4. **Un solo modelo generador** → detector de UN modelo, no de IA.
5. **Features muertas (std=0) sin arreglar** → tiras señal a la basura.
6. **Entrenar con textos de 500 palabras y servir tesis de 80K** → estratifica
   longitudes en el corpus y evalúa por tramo de longitud.
7. **No guardar el umbral de FPR=1%** y usar 0.5 por defecto en producción.
