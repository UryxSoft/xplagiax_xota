# Guía B — Binoculars: jubilar GPT-2, perplexity SOTA zero-shot

**Objetivo:** nuevo plugin `binoculars_check` que sustituye la señal tier-2
(GPT-2, 2019) del perplexity profiler por Binoculars
([Hans et al. 2024](https://arxiv.org/abs/2401.12070)): ratio de
cross-perplexity entre dos LLMs pequeños emparentados. SOTA zero-shot,
sin entrenamiento, funciona contra modelos frontera que tu ensamble 2023
nunca vio.

**La idea en una frase:** el texto IA es *predecible para otro LLM*;
el humano sorprende. Score bajo = IA.

**Dónde se ejecuta:**

| Fase | Dónde |
|------|-------|
| B.1 Elegir y descargar el par de modelos | Local (descarga ~6 GB) |
| B.2 Prototipo del score | **Colab** (validar rápido con GPU) |
| B.3 Plugin en el repo | Local |
| B.4 Calibrar umbral | Local (o Colab si el corpus es grande) |
| B.5 Integrar en fusion | Local |

---

## B.1 Elegir el par de modelos

Requisitos INNEGOCIABLES del par:

1. **Mismo tokenizer** (misma familia, mismo vocabulario). Si difieren, la
   cross-perplexity compara distribuciones sobre vocabularios distintos = basura.
2. Un modelo **base** (observador) y su **instruct** (ejecutor), o dos
   variantes cercanas de la misma familia.

Opciones (elige según tu RAM/GPU):

| Par | Tamaño | Nota |
|-----|--------|------|
| `Qwen/Qwen2.5-1.5B` + `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB c/u fp16 | **Recomendado**: multilingüe decente en español |
| `Qwen/Qwen2.5-0.5B` + `-Instruct` | ~1 GB c/u | Si vas justo de RAM; algo menos de señal |
| `tiiuae/falcon-7b` + `falcon-7b-instruct` | ~14 GB c/u | El par del paper original; caro en RAM, flojo en español |

```bash
.venv/bin/pip install -U "transformers>=4.45" accelerate
# Descarga anticipada (los pesos quedan en ~/.cache/huggingface):
.venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in ['Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct']:
    AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)
"
```

**✅ Checkpoint:** ambos tokenizers idénticos:

```python
from transformers import AutoTokenizer
t1 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
t2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
assert t1.get_vocab() == t2.get_vocab(), "PAR INVÁLIDO"
print("Par válido")
```

---

## B.2 Prototipo en Colab (validar antes de integrar)

Notebook Colab con GPU (T4 gratis basta para 1.5B):

```python
# Celda 1
!pip install -q transformers accelerate torch

# Celda 2 — el score de Binoculars
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
OBSERVER = "Qwen/Qwen2.5-1.5B"            # base
PERFORMER = "Qwen/Qwen2.5-1.5B-Instruct"  # instruct

tok = AutoTokenizer.from_pretrained(OBSERVER)
obs = AutoModelForCausalLM.from_pretrained(OBSERVER, torch_dtype=torch.float16).to(DEV).eval()
perf = AutoModelForCausalLM.from_pretrained(PERFORMER, torch_dtype=torch.float16).to(DEV).eval()

@torch.inference_mode()
def binoculars_score(text: str, max_tokens: int = 512) -> float:
    """
    score = PPL_performer(texto) / X-PPL(observer, performer)
    Bajo (~<0.9) = IA probable. Alto (~>1.0) = humano probable.
    """
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=max_tokens).input_ids.to(DEV)
    if ids.shape[1] < 32:
        return float("nan")            # texto demasiado corto: no fiable

    logits_obs = obs(ids).logits[0, :-1]       # (L-1, V)
    logits_perf = perf(ids).logits[0, :-1]
    targets = ids[0, 1:]                       # (L-1,)

    # log-PPL del performer contra los tokens reales
    ce_perf = F.cross_entropy(logits_perf.float(), targets)

    # cross-entropy del performer contra la DISTRIBUCIÓN del observer
    p_obs = F.softmax(logits_obs.float(), dim=-1)          # (L-1, V)
    logq_perf = F.log_softmax(logits_perf.float(), dim=-1)
    x_ce = -(p_obs * logq_perf).sum(dim=-1).mean()

    return float(ce_perf / x_ce)

# Celda 3 — sanity check con 4 textos que conozcas
human_txt = "..."   # pega un párrafo de una tesis pre-2022
ai_txt = "..."      # pega un párrafo generado por ChatGPT
print("humano:", binoculars_score(human_txt))
print("ia:    ", binoculars_score(ai_txt))
```

**✅ Checkpoint:** el texto IA da score CLARAMENTE menor que el humano
(típico: IA 0.6-0.85, humano 0.95-1.15). Si salen iguales: par de modelos mal
elegido o bug en la cross-entropy. **No sigas hasta que esto separe.**

Prueba con 10-20 textos de cada clase, en español e inglés. Apunta los rangos.

---

## B.3 Plugin en el repo

### Paso 1 — Crea `app/engine/binoculars_engine.py`

Misma lógica del prototipo, con los patrones del repo:

- Carga a nivel módulo (CoW-compartido como en `detector_final.py`), con
  `local_files_only=True` y try/except → `_available = False` si faltan pesos.
- En CPU usa `torch_dtype=torch.float32` (fp16 en CPU es lento/impreciso) y
  respeta `TORCH_NUM_THREADS`.
- **Chunking**: textos largos → trocea a 512 tokens por chunk (reusa la
  segmentación por párrafos de `analyze_fast` como referencia), score por
  chunk, promedio ponderado por nº tokens. Devuelve también la lista de
  scores por chunk (para el heatmap).
- **Cache**: copia el patrón `_FAST_CACHE` de `detector_final.py`
  (sha256 del texto, TTL 5 min, LRU 20).

### Paso 2 — Crea `app/plugins/binoculars_check.py`

Calca la estructura de `app/plugins/perplexity_check.py` (BasePlugin,
`name() = "binoculars_check"`, `health()`, `analyze()`). Salida:

```python
{
    "binoculars_score": 0.82,          # promedio ponderado
    "risk_level": "HIGH",              # según umbral calibrado (B.4)
    "chunk_scores": [0.79, 0.85, ...],
    "chunks_below_threshold": 3,
    "n_chunks": 7,
    "reliable": True,                  # False si texto < 100 tokens
}
```

El plugin se auto-registra por el `discover()` del registry — no hay que
tocar nada más para que `"plugins": ["binoculars_check"]` funcione.

### Paso 3 — RAM y despliegue

Dos modelos de 1.5B en fp32 ≈ 12 GB — demasiado junto a tus 1.7 GB de
ModernBERT en un VPS de 4 GB. Opciones en orden de preferencia:

1. **Cuantización int8 con bitsandbytes** (GPU) o **GGUF/llama.cpp** (CPU):
   1.5B int8 ≈ 1.6 GB por modelo. Total ~3.2 GB extra.
2. Usar el par 0.5B: fp32 ≈ 2 GB por modelo, int8 ≈ 0.6 GB.
3. Servicio separado (otro contenedor con más RAM) al que el plugin llama
   por HTTP.

Empieza con el par 0.5B int8 y mide señal; sube a 1.5B si el AUROC lo pide.
Gate por env: `ENABLE_BINOCULARS=1` + `BINOCULARS_MODEL_PAIR=qwen2.5-0.5b`.

---

## B.4 Calibrar el umbral (NO uses el del paper a ciegas)

El paper reporta ~0.90 como corte, pero **eso es con Falcon-7B en inglés
genérico**. Tu par y tu dominio (académico, español) tienen otro rango.

1. Corre `binoculars_score` sobre 500+ textos humanos académicos
   (pre-2022, de tu BD) y 500+ IA (de la Guía A.2). Si aún no tienes el
   corpus A, genera 200 muestras rápidas con ChatGPT/Claude a mano —
   suficiente para un umbral provisional.
2. Elige el umbral en **FPR=1% sobre los humanos**:

```python
import numpy as np
scores_human = np.array([...])   # scores de textos humanos
scores_ai = np.array([...])
thr = np.quantile(scores_human, 0.01)   # solo 1% de humanos por debajo
tpr = (scores_ai < thr).mean()
print(f"umbral={thr:.3f}  TPR@FPR1%={tpr:.2%}")
```

3. Guarda umbral por idioma si difieren (suele pasar): `{"es": 0.87, "en": 0.91}`.

**✅ Checkpoint:** TPR@FPR=1% > 0.6 en texto frontera (GPT-4o/Claude). Si da
menos, prueba el par 1.5B o revisa el chunking.

---

## B.5 Integrar en fusion

1. Añade la feature al schema en `app/engine/fusion.py` (`_FUSION_SCHEMA`):
   `"bin_score"` (normalizada: `min(score, 2.0) / 2.0`) y
   `"bin_chunks_below_ratio"`.
2. En el `FeatureBuilder.build()`, extrae del dict de `additional_analyses`
   (el orquestador debe invocar el engine igual que hace con perplexity —
   añade bloque en `plugin_orchestrator.run()` con flag
   `enable_binoculars`).
3. **Re-entrena la fusion** (Guía A.4) — el vector cambió de dimensión, los
   pesos viejos ya no valen. Sube `MODEL_VERSION`.

---

## Errores dummy típicos

1. **Tokenizers distintos en el par** → score sin sentido. Verifica el assert de B.1.
2. **Texto < 100 tokens** → devuelve `reliable: False` y score neutro; nunca
   un veredicto.
3. **fp16 en CPU** → 10× más lento que fp32 en muchos CPUs. fp16 solo en GPU.
4. **Olvidar el softmax float()** → overflow en fp16, NaNs silenciosos.
5. **Umbral del paper sin calibrar** → FPR desconocido en tu dominio.
6. **No ponderar chunks por tokens** → un chunk de 50 tokens pesa igual que
   uno de 512 y mete ruido.
7. **Cargar los modelos por request** → 30 s de carga cada vez. Nivel módulo,
   una vez.
