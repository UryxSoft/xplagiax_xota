# 🧠 Manual de Reentrenamiento — XplagiaX XOTA / Retraining Manual

> **ES** — Parte I (completa, abajo) · **EN** — Part II ([saltar / jump](#part-ii--english))
>
> Última actualización: 2026-07 · Aplica a: rama `auditoria-cientifica-fase1`, v2026.07+

---

# PARTE I — ESPAÑOL

## 0. Por qué existe este manual

Un detector de IA **envejece**. Cada familia nueva de LLMs (GPT-5, Gemini 3, DeepSeek-R2…)
queda *fuera de distribución* para un modelo entrenado en 2023: el detector no "falla con
ruido", falla **en silencio** — sigue emitiendo veredictos confiados mientras su precisión
real se degrada. Por eso el reentrenamiento no es un evento único sino un **ciclo operativo**,
y este sistema tiene dos capas entrenables con costos radicalmente distintos:

| Capa | Qué es | Costo de reentrenar | Frecuencia recomendada |
|---|---|---|---|
| **A. Fusión** (`FusionClassifier`) | Regresión logística de 31 features sobre las salidas de todos los plugins | **Segundos de CPU** (lo caro es el corpus, no el entrenamiento) | Cada vez que cambie el corpus o un plugin |
| **B. Ensemble neural** (3×ModernBERT) | Los 3 transformers que producen la señal dominante | Horas de **GPU** (Colab) + evaluación + promoción | Trimestral, o cuando `/api/drift-status` reporte `degraded` |

**Regla de oro (docs/sota/00_INDICE.md):** nunca reportes "accuracy". La métrica de este
dominio es **TPR@FPR=1%** — cuántos textos IA detectas cuando solo aceptas equivocarte con
1 de cada 100 humanos. Un falso positivo = acusar falsamente a un estudiante.

### ¿Cuándo reentrenar? Las tres señales

1. **`GET /api/drift-status`** devuelve `"status": "degraded"` — la confianza media del
   ensemble cayó respecto a su línea base (síntoma clásico de textos OOD de un LLM nuevo).
2. **`% de veredictos "Inconclusive"`** en el log de métricas (`METRICS_LOG_PATH`) supera
   ~30% sostenido — las señales se están degradando.
3. **Calendario**: revisión trimestral aunque nada haya saltado. El drift es silencioso.

---

## 1. El corpus — el activo que lo decide todo

Ninguna de las dos capas puede entrenarse ni evaluarse sin un **corpus etiquetado**. El 90%
del esfuerzo del reentrenamiento es esto; el entrenamiento en sí es trivial.

### 1.1 Clase HUMANA (label 0) — gratis, ya la tienes

Tu base documental pre-ChatGPT es oro: **todo documento publicado antes de 2022-11-01 es
humano garantizado**. Regla inviolable — cualquier cosa posterior puede contener texto IA y
envenena la clase.

```bash
export CORPUS_DB_DSN="mysql://user:pass@host:3306/dbname"

# Dry-run: imprime la query y sale (ajústala a tu esquema con --query-file)
.venv/bin/python scripts/corpus/extract_human.py --dry-run

# Extracción real, estratificada, troceada en unidades de 500-3000 palabras
.venv/bin/python scripts/corpus/extract_human.py \
    --query-file mi_query.sql \
    --out dataset/human.jsonl \
    --max-per-stratum 800
```

**Qué hace y por qué:**
- **Estratifica** por idioma × disciplina × longitud (`--max-per-stratum` limita cada celda).
  Sin esto, el modelo aprende "derecho en español = humano" en vez de "humano = humano".
- **Trocea** documentos largos en unidades de 500-3000 palabras conservando `doc_id` y
  `author_id` — imprescindible para el split por autor (§3.1).
- **Objetivo mínimo viable: 10.000 muestras humanas.** Ideal: 50.000+.

Formato de salida (JSONL, una muestra por línea):
```json
{"text": "...", "label": 0, "lang": "es", "domain": "derecho", "words": 1240,
 "author_id": "a-8812", "doc_id": "d-4471", "source": "tesis-pre2022"}
```

### 1.2 Clase IA (label 1) — pares paralelos, canasta de modelos

Principio: por cada muestra humana, genera una **contraparte IA del mismo tema y registro**
(pares paralelos). Así la fusión aprende la diferencia humano/IA real, no diferencias de
tema. Y **nunca un solo modelo generador** — o el detector solo detectará ESE modelo.

**Cuota Anthropic (este script, ~30% del volumen)** — usa el Batches API (50% de descuento,
ideal para generación masiva sin requisito de latencia):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python scripts/corpus/generate_ai.py \
    --human dataset/human.jsonl \
    --out dataset/ai_anthropic.jsonl \
    --limit 2000
```

Rota 4 tareas por muestra: *continuar* / *reescribir* / *expandir* / *desde título*.
Es reanudable: las muestras ya generadas se saltan al relanzar.

**Cuota de modelos abiertos (~70% del volumen)** — Llama-3.1-8B, Qwen2.5-7B, Mistral-7B en
**Colab GPU gratuita** (T4). Notebook paso a paso: `docs/sota/A_FUSION_ENTRENADA.md` §A.2
paso 3. Trampa conocida: la sesión gratis de Colab muere a las ~4h — **guarda a Google
Drive cada N muestras**.

**Cuota adversarial (para la suite E)** — parafrasea un subconjunto del corpus IA con
DIPPER/humanizadores (`docs/sota/E_SUITE_ADVERSARIAL.md`). No entra al train; es el examen.

### 1.3 Higiene del corpus (los errores que invalidan todo)

| Error | Consecuencia | Prevención |
|---|---|---|
| Documentos post-2022 en clase humana | Etiquetas envenenadas → techo de precisión falso | Filtro de fecha en la query; verificar `MAX(fecha)` |
| Un solo modelo generador | Detector de UN modelo, no de IA | Canasta ≥4 modelos, ninguno >40% del volumen |
| Mismo autor en train y test | Métricas infladas (memoriza autores) | Split por `author_id` (train_fusion lo hace solo) |
| Duplicados/casi-duplicados | Fuga train→test | Dedupe por hash + shingling antes de vectorizar |
| Clases desbalanceadas por dominio | El modelo aprende el dominio, no la autoría | Pares paralelos + estratificación |

---

## 2. Capa A — Reentrenar la FUSIÓN (CPU, horas de máquina, cero GPU)

### 2.1 Vectorizar el corpus

Cada muestra pasa por el **pipeline de producción completo** (PluginOrchestrator +
FusionFeatureBuilder), así el entrenamiento ve exactamente los vectores que producción
producirá. Es la etapa cara en CPU (ensemble neural por muestra) — nocturna y reanudable:

```bash
nice -n 15 .venv/bin/python scripts/corpus/vectorize.py \
    dataset/human.jsonl dataset/ai_anthropic.jsonl dataset/ai_colab.jsonl \
    --out dataset/vectors
# Salida: dataset/vectors/X.npy (n, 31) + y.npy + meta.jsonl
# Reanudable: al relanzar salta las muestras ya vectorizadas.
```

Estimación: ~1-3 s/muestra en CPU → 20k muestras ≈ 8-16 h de máquina (una noche larga o
dos). `--limit N` permite trocear en tandas.

### 2.2 Entrenar + calibrar + medir

```bash
.venv/bin/python scripts/corpus/train_fusion.py \
    --vectors dataset/vectors \
    --out models/fusion_weights.json
```

Qué hace, en orden, y por qué cada paso importa:

1. **Split por grupos de `author_id`** (70/15/15). Ningún autor cruza splits — sin esto las
   métricas mienten. Cada LLM generador cuenta como un "autor".
2. **`FusionClassifier.fit`** sobre train — regresión logística estandarizada, con
   `class_weight="balanced"`.
3. **`TemperatureScaler.fit`** sobre validación (Guo et al. 2017) — convierte los scores en
   probabilidades honestas. Es lo que hace que "80%" signifique de verdad ~80%.
4. **Métricas sobre test** (nunca tocado hasta aquí): ROC-AUC, Brier, **ECE pre/post
   calibración**, **TPR@FPR=1%**, tabla de fiabilidad por bins.
5. **Guarda `models/fusion_weights.json`** — pesos + media/std + temperatura + hash del
   esquema de features (rechaza cargar si el esquema cambió).

### 2.3 Criterios de aceptación (no despliegues sin esto)

- **ECE post-calibración < 0.05** — si no, la temperatura no basta; revisa el corpus.
- **TPR@FPR=1% mejora** (o al menos iguala) el valor del run anterior. Regístralo siempre.
- **La fusión entrenada supera a la heurística** en el MISMO test set (corre
  `train_fusion.py` reporta ambas). Si no la supera, el corpus es insuficiente — NO
  despliegues, amplía el corpus.

### 2.4 Desplegar

```bash
# 1. Copia el archivo de pesos al servidor
# 2. Apunta la variable y sube la versión (invalida cachés viejas):
export FUSION_WEIGHTS_PATH=/opt/xplagiax/models/fusion_weights.json
export MODEL_VERSION=2026.08
# 3. Reinicia gunicorn. Verifica en el primer análisis:
#    additional.fusion.source == "logistic"  y  calibrated == true
```

**Rollback:** borra/renombra `FUSION_WEIGHTS_PATH` y reinicia — vuelve a la fusión
heurística acotada automáticamente. Sube `MODEL_VERSION` de nuevo.

**Regla del índice SOTA:** cuando integres una señal nueva (Binoculars, etc.), la fusión
**se re-entrena al final** con la señal ya emitiendo — nunca añadas la feature con peso
inventado a mano.

---

## 3. Capa B — Reentrenar el ENSEMBLE NEURAL (GPU/Colab + pipeline de promoción)

Esto es lo único que arregla los **falsos negativos frontier** (GPT-5/Claude/Gemini
actuales): el modelo 2023 es ciego a esas distribuciones y ninguna fusión lo compensa del
todo.

### 3.1 Preparar el corpus "gold" en el layout del pipeline

`scripts/retrain_pipeline.py` espera archivos de texto planos por clase:

```
data/gold/
├── human/  *.txt    ← textos verificados humanos (muestrea de dataset/human.jsonl)
└── ai/     *.txt    ← textos verificados IA (muestrea de los ai_*.jsonl)
```

```bash
# Valida el layout y el balance (rechaza clases con < --min-per-class):
.venv/bin/python scripts/retrain_pipeline.py collect --corpus data/gold
```

### 3.2 Línea base: evalúa los pesos ACTUALES antes de tocar nada

```bash
.venv/bin/python scripts/retrain_pipeline.py evaluate --corpus data/gold \
    --metrics-out /tmp/metrics_current.json
```

Usa la MISMA agregación `analyze_fast()` que sirve el API — mides lo que producción hace,
no un proxy. Guarda ese JSON: es el listón que los pesos nuevos deben superar.

### 3.3 Fine-tune en Colab (GPU)

```bash
# Imprime la receta exacta (modelo base, hiperparámetros, seeds) y sale:
.venv/bin/python scripts/retrain_pipeline.py train --corpus data/gold
```

El stage `train` es **deliberadamente manual**: el fine-tuning necesita GPU y un Trainer de
`transformers` que no pertenece al contenedor de serving. La receta impresa + el notebook de
`docs/sota/A_FUSION_ENTRENADA.md` cubren el Colab. Puntos no negociables:

- **Base**: `answerdotai/ModernBERT-base`, head de 41 clases (mantén `label_mapping` — el
  índice 24 = human es contrato del código).
- **Datos**: corpus gold + el corpus 2024-2026 generado en §1.2. Held-out **temporal**
  (entrena ≤2024, testea 2025-2026) para medir generalización real.
- **Seeds**: entrena los 3 checkpoints con seeds distintas (es lo que da la señal de
  `ensemble_disagreement`).
- Exporta `state_dict` como `modernbert.bin` (+ los dos seeds), a Drive.

### 3.4 Evaluar los pesos candidatos SIN tocar producción

```bash
XPLAGIAX_EVAL_WEIGHTS=/ruta/nuevo/modernbert.bin \
.venv/bin/python scripts/retrain_pipeline.py evaluate --corpus data/gold \
    --metrics-out /tmp/metrics_candidate.json
```

`XPLAGIAX_EVAL_WEIGHTS` sustituye los pesos solo en ese proceso — **nunca la pongas en el
serving**.

### 3.5 Promoción atómica con red de seguridad

```bash
.venv/bin/python scripts/retrain_pipeline.py promote --corpus data/gold \
    --weights /ruta/nuevo/modernbert.bin \
    --version 2026.08 \
    --min-gain 0.01
```

La promoción se **rehúsa** si el candidato no supera a producción por `--min-gain`. Si pasa:

1. Copia los pesos actuales a `MODEL_FALLBACK_DIR` (set de rollback).
2. Instala los nuevos + escribe `metadata.json` (versión, fecha, métricas, fingerprint del
   corpus).
3. `detector_final._load_model()` cae automáticamente al fallback si los pesos nuevos
   están corruptos, y `/api/drift-status` muestra **qué archivo cargó realmente cada
   worker** (`model.weights[].loaded_from`, `fallbacks_used`).

```bash
# Post-deploy, verifica:
curl -s http://localhost:5006/api/drift-status | python3 -m json.tool
# → model.version == "2026.08", fallbacks_used == []
```

**Rollback manual:** copia el archivo de `MODEL_FALLBACK_DIR` de vuelta, baja
`MODEL_VERSION`… o simplemente borra los pesos nuevos: el fallback automático hace el resto.

### 3.6 Después del neural: re-entrena la fusión

Los pesos neurales nuevos cambian la distribución de `neural_ai_prob` → los pesos de fusión
viejos quedan descalibrados. **Siempre**: promote neural → re-vectorizar (§2.1, borra
`dataset/vectors` para forzar) → re-entrenar fusión (§2.2) → desplegar ambos con el mismo
`MODEL_VERSION`.

---

## 4. Validación adversarial (el examen final)

Antes de dar por bueno cualquier reentrenamiento, córrelo contra el set adversarial
(`docs/sota/E_SUITE_ADVERSARIAL.md`): IA parafraseada (DIPPER), humanizada, traducida,
híbridos H+IA. Reporta TPR@FPR=1% **por condición**. Criterio: la degradación bajo
parafraseo no debe superar la del run anterior.

## 5. Monitoreo continuo (cierra el ciclo)

| Herramienta | Qué vigila | Acción si dispara |
|---|---|---|
| `GET /api/drift-status` | Confianza media vs. baseline, share de IA, pesos cargados | `degraded` → §3 completo |
| `METRICS_LOG_PATH` (JSONL) | % Inconclusive, disagreement, distribución de veredictos | >30% Inconclusive sostenido → revisar señales |
| `metadata.json` en engine/ | Qué versión sirve cada despliegue | auditoría |

## 6. Variables de entorno del ciclo (referencia rápida)

| Variable | Capa | Efecto |
|---|---|---|
| `FUSION_WEIGHTS_PATH` | A | Carga pesos entrenados de fusión al arrancar (ausente = heurística) |
| `MODEL_VERSION` | A+B | Versiona cachés Redis + in-process; **súbela en cada despliegue de pesos** |
| `XPLAGIAX_EVAL_WEIGHTS` | B | Solo evaluación offline de candidatos — jamás en serving |
| `MODEL_FALLBACK_DIR` | B | Directorio de pesos "last known good" para fallback automático |
| `METRICS_LOG_PATH` | ops | JSONL de veredictos para monitoreo |
| `CORPUS_DB_DSN` | corpus | DSN MySQL de la BD documental (solo extracción) |
| `ANTHROPIC_API_KEY` | corpus | Generación de clase IA vía Batches API |

---
---

# PART II — ENGLISH

## 0. Why this manual exists

AI detectors **age**. Every new LLM family (GPT-5, Gemini 3, DeepSeek-R2…) is
*out-of-distribution* for a 2023-trained model: the detector doesn't fail loudly, it fails
**silently** — still emitting confident verdicts while real precision decays. Retraining is
therefore an **operational cycle**, not a one-off event, and this system has two trainable
layers with radically different costs:

| Layer | What it is | Retraining cost | Recommended cadence |
|---|---|---|---|
| **A. Fusion** (`FusionClassifier`) | 31-feature logistic regression over all plugin outputs | **Seconds of CPU** (the corpus is the expensive part, not the training) | Whenever the corpus or a plugin changes |
| **B. Neural ensemble** (3×ModernBERT) | The 3 transformers producing the dominant signal | Hours of **GPU** (Colab) + evaluation + promotion | Quarterly, or when `/api/drift-status` reports `degraded` |

**Golden rule (docs/sota/00_INDICE.md):** never report "accuracy". The metric of this
domain is **TPR@FPR=1%** — how many AI texts you catch while accepting being wrong about
only 1 in 100 humans. One false positive = falsely accusing a student.

### When to retrain — the three signals

1. **`GET /api/drift-status`** returns `"status": "degraded"` — mean ensemble confidence
   dropped vs. its baseline (the classic symptom of OOD text from a new LLM).
2. **"Inconclusive" verdict share** in the metrics log (`METRICS_LOG_PATH`) sustained above
   ~30% — signals are degrading.
3. **Calendar**: quarterly review even if nothing fired. Drift is silent.

## 1. The corpus — the asset that decides everything

Neither layer can be trained or evaluated without a **labeled corpus**. 90% of the effort
is here; training itself is trivial.

### 1.1 HUMAN class (label 0) — free, you already own it

Your pre-ChatGPT document DB is gold: **anything published before 2022-11-01 is guaranteed
human**. Inviolable rule — anything later may contain AI text and poisons the class.

```bash
export CORPUS_DB_DSN="mysql://user:pass@host:3306/dbname"
.venv/bin/python scripts/corpus/extract_human.py --dry-run          # inspect query first
.venv/bin/python scripts/corpus/extract_human.py \
    --query-file my_query.sql --out dataset/human.jsonl --max-per-stratum 800
```

Stratifies by language × discipline × length, chunks long docs into 500-3000-word units
preserving `author_id`/`doc_id` (needed for the group split). **Minimum viable: 10,000
human samples. Ideal: 50,000+.**

### 1.2 AI class (label 1) — parallel pairs, model basket

For each sampled human text, generate an AI counterpart **on the same topic and register**
(parallel pairs) so the fusion learns human-vs-AI, not topic-vs-topic. And **never a single
generator model** — or you build a detector of THAT model only.

- **Anthropic share (~30%)** — `scripts/corpus/generate_ai.py` (Message Batches API, 50%
  price, resumable; rotates continue/rewrite/expand/from-title tasks).
- **Open-model share (~70%)** — Llama-3.1-8B / Qwen2.5-7B / Mistral-7B on free Colab T4;
  notebook in `docs/sota/A_FUSION_ENTRENADA.md` §A.2. Save to Drive every N samples
  (free sessions die at ~4h).
- **Adversarial share** — DIPPER-paraphrased/humanized subset for the E suite (exam only,
  never in train).

### 1.3 Corpus hygiene (the mistakes that invalidate everything)

Post-2022 docs in the human class · a single generator model · same author in train and
test · duplicates across splits · domain-imbalanced classes. The table in Part I §1.3
gives cause → consequence → prevention for each.

## 2. Layer A — retraining the FUSION (CPU only)

```bash
# 1. Vectorize (production pipeline per sample; overnight; resumable)
nice -n 15 .venv/bin/python scripts/corpus/vectorize.py \
    dataset/human.jsonl dataset/ai_anthropic.jsonl dataset/ai_colab.jsonl \
    --out dataset/vectors

# 2. Train + calibrate + measure
.venv/bin/python scripts/corpus/train_fusion.py \
    --vectors dataset/vectors --out models/fusion_weights.json
```

`train_fusion.py` performs: group-aware 70/15/15 split by `author_id` (no author leakage),
`FusionClassifier.fit` on train, `TemperatureScaler.fit` on validation (Guo 2017), then
test-set metrics: ROC-AUC, Brier, **ECE pre/post calibration**, **TPR@FPR=1%**, reliability
table. Output JSON embeds weights + mean/std + temperature + a feature-schema hash
(refuses to load if the schema changed).

**Acceptance criteria:** post-calibration ECE < 0.05 · TPR@FPR=1% ≥ previous run · trained
fusion beats the heuristic on the same test set (the script reports both). If it doesn't
beat the heuristic, the corpus is insufficient — grow it, don't deploy.

**Deploy:**
```bash
export FUSION_WEIGHTS_PATH=/opt/xplagiax/models/fusion_weights.json
export MODEL_VERSION=2026.08   # invalidates stale caches
# restart gunicorn; verify additional.fusion.source == "logistic" && calibrated == true
```
**Rollback:** unset `FUSION_WEIGHTS_PATH`, restart → bounded heuristic fusion returns
automatically. Bump `MODEL_VERSION` again.

## 3. Layer B — retraining the NEURAL ENSEMBLE (GPU/Colab + promotion pipeline)

This is the only fix for **frontier false negatives** — the 2023 model is blind to current
LLM distributions and no fusion fully compensates.

```bash
# Layout:  data/gold/human/*.txt  +  data/gold/ai/*.txt
python scripts/retrain_pipeline.py collect  --corpus data/gold          # validate corpus
python scripts/retrain_pipeline.py evaluate --corpus data/gold \
    --metrics-out /tmp/metrics_current.json                             # baseline (prod weights)
python scripts/retrain_pipeline.py train    --corpus data/gold          # prints the exact recipe

# Fine-tune on Colab GPU (recipe + notebook in docs/sota/A_FUSION_ENTRENADA.md):
#   base answerdotai/ModernBERT-base, 41-class head (index 24 = human is a code contract),
#   3 checkpoints with distinct seeds (feeds ensemble_disagreement),
#   temporal held-out (train ≤2024, test 2025-2026).

# Evaluate the candidate WITHOUT touching production:
XPLAGIAX_EVAL_WEIGHTS=/path/new/modernbert.bin \
python scripts/retrain_pipeline.py evaluate --corpus data/gold \
    --metrics-out /tmp/metrics_candidate.json

# Atomic promotion with safety net (refuses unless candidate ≥ current + --min-gain;
# archives current weights to MODEL_FALLBACK_DIR; writes metadata.json):
python scripts/retrain_pipeline.py promote --corpus data/gold \
    --weights /path/new/modernbert.bin --version 2026.08 --min-gain 0.01

# Post-deploy verification:
curl -s http://localhost:5006/api/drift-status | python3 -m json.tool
#   → model.version == "2026.08", fallbacks_used == []
```

`detector_final._load_model()` falls back to `MODEL_FALLBACK_DIR` automatically if the new
weights are corrupt, and `/api/drift-status` shows exactly which file each worker loaded.

**After every neural promotion, retrain the fusion** (new weights shift the
`neural_ai_prob` distribution): delete `dataset/vectors`, re-run §2 steps 1-2, deploy both
under the same `MODEL_VERSION`.

## 4. Adversarial validation (the final exam)

Run every retrained stack against the adversarial suite (`docs/sota/E_SUITE_ADVERSARIAL.md`):
DIPPER-paraphrased, humanized, translated, and hybrid H+AI texts. Report TPR@FPR=1% **per
condition**; degradation under paraphrase must not exceed the previous run's.

## 5. Continuous monitoring · 6. Environment variables

See the tables in Part I §5-§6 — variable names and semantics are identical
(`FUSION_WEIGHTS_PATH`, `MODEL_VERSION`, `XPLAGIAX_EVAL_WEIGHTS`, `MODEL_FALLBACK_DIR`,
`METRICS_LOG_PATH`, `CORPUS_DB_DSN`, `ANTHROPIC_API_KEY`).
