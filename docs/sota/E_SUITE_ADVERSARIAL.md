# Guía E — Suite adversarial + la métrica correcta (TPR@FPR=1%)

**Objetivo:** (1) sanitizador anti-evasión antes de TODO análisis;
(2) test set adversarial (paráfrasis, humanizadores, homoglifos, mezclas);
(3) un único script de evaluación que reporta TPR@FPR=1% por idioma,
longitud, disciplina, en limpio Y adversarial; (4) tabla comparativa contra
baselines (Binoculars puro, GPTZero).

Esta guía es transversal: se ejecuta desde que exista el corpus de la
Guía A y se re-corre tras CADA cambio de modelo/fusion. Es tu examen final
permanente.

**Dónde se ejecuta:**

| Fase | Dónde |
|------|-------|
| E.1 Sanitizador | Local (código en el repo) |
| E.2 Set adversarial — DIPPER / back-translation | **Colab** (GPU necesaria) |
| E.3 Script de evaluación | Local |
| E.4 Baselines | Local + API GPTZero |

---

## E.1 Sanitizador pre-análisis (medio día de trabajo)

Los evasores triviales que hoy te vencerían sin tocar el texto "visible":
caracteres de ancho cero entre letras, homoglifos cirílicos (а е о с р
en vez de a e o c p), espacios Unicode raros.

### Paso 1 — `app/engine/text_sanitizer.py`

```python
"""Normalización anti-evasión. SIEMPRE antes de cualquier análisis."""
import unicodedata
import re
from dataclasses import dataclass

# Zero-width y controles invisibles
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿­]")

# Homoglifos frecuentes cirílico/griego → latino (amplía según encuentres)
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "В": "B",
    "Н": "H", "К": "K", "М": "M", "Т": "T", "ο": "o", "ν": "v", "α": "a",
})

@dataclass
class SanitizedText:
    text: str
    zero_width_removed: int
    homoglyphs_replaced: int
    was_suspicious: bool          # evidencia de evasión POR SÍ MISMA

def sanitize(text: str) -> SanitizedText:
    zw = len(_ZERO_WIDTH.findall(text))
    step1 = _ZERO_WIDTH.sub("", text)
    step2 = unicodedata.normalize("NFKC", step1)
    step3 = step2.translate(_HOMOGLYPHS)
    hg = sum(1 for a, b in zip(step2, step3) if a != b)
    return SanitizedText(
        text=step3,
        zero_width_removed=zw,
        homoglyphs_replaced=hg,
        # umbral: unos pocos pueden ser copy-paste legítimo de PDF;
        # decenas es manipulación deliberada
        was_suspicious=(zw > 5 or hg > 10),
    )
```

### Paso 2 — Conectar en la entrada

En `routes.py`, tras validar `text` y ANTES del cache key y de
`registry.run(...)` (los 4 endpoints: analyze, analyze_document,
analyze_document_async, analyze_stream):

```python
from app.engine.text_sanitizer import sanitize
san = sanitize(text)
text = san.text
# incluir en la respuesta:
# "sanitization": {"zero_width_removed": .., "homoglyphs_replaced": ..,
#                  "was_suspicious": true}
```

`was_suspicious=true` debe aparecer en el reporte forense como evidencia:
*"el documento contenía N caracteres invisibles — patrón de evasión de
detectores"*. Eso convence a un comité más que cualquier score.

**✅ Checkpoint:** test unitario — texto con `"h​ola"` y `"сasa"`
(c cirílica) sale limpio y marcado sospechoso con los contadores correctos.

---

## E.2 Construir el test set adversarial

Base: el split **TEST de la Guía A** (nunca train — el adversarial hereda el
mismo aislamiento). De cada texto IA del test, genera variantes:

### Variante 1 — Paráfrasis DIPPER (Colab, GPU obligatoria)

DIPPER (`kalpeshk2011/dipper-paraphraser-xxl`, 11B) es el paraphraser
estándar de los papers de evasión. En Colab necesitas A100 (Colab Pro) o
usa el T5-large alternativo si solo tienes T4.

```python
# Colab — Celda 1
!pip install -q transformers accelerate sentencepiece

# Celda 2
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
model = T5ForConditionalGeneration.from_pretrained(
    "kalpeshk2011/dipper-paraphraser-xxl",
    torch_dtype=torch.bfloat16, device_map="auto")

def dipper(text: str, lex=60, order=60) -> str:
    """lex/order 0-100: agresividad léxica y de reordenación."""
    prompt = f"lexical = {lex}, order = {order} {text}"
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=1024).input_ids.to(model.device)
    out = model.generate(ids, do_sample=True, top_p=0.75, max_length=1024)
    return tok.decode(out[0], skip_special_tokens=True)
```

Genera con dos agresividades: (60,60) moderada y (80,100) fuerte.
**Nota:** DIPPER es inglés. Para español usa la Variante 2 más un LLM
parafraseador (Qwen2.5-7B con prompt "parafrasea manteniendo el
significado") — documenta cuál usaste en cada muestra.

### Variante 2 — Back-translation (Colab, T4 basta)

```python
# es → en → es con NLLB
from transformers import pipeline
en = pipeline("translation", model="facebook/nllb-200-distilled-600M",
              src_lang="spa_Latn", tgt_lang="eng_Latn", device=0)
es = pipeline("translation", model="facebook/nllb-200-distilled-600M",
              src_lang="eng_Latn", tgt_lang="spa_Latn", device=0)
def backtranslate(text): return es(en(text)[0]["translation_text"])[0]["translation_text"]
```

### Variante 3 — Ataques de caracteres (local, trivial)

Inversos del sanitizador E.1: inyecta zero-width cada N caracteres,
sustituye 5% de vocales por homoglifos. **Esta variante debe dar TPR ≈ el
del set limpio** — si baja, tu sanitizador tiene un hueco.

### Variante 4 — Mezclas humano/IA (local)

El caso real más frecuente: estudiante escribe 70%, IA escribe 30%.

```python
# Toma tesis humana del test + secciones IA del test, mezcla por párrafos
# en proporciones 10/30/50%. Etiqueta a nivel párrafo (para evaluar la
# segmentación) y a nivel documento (label = 1 si > 15% IA).
```

### Variante 5 — Humanizadores comerciales (manual, muestra pequeña)

Pasa 50-100 textos IA por 2-3 humanizadores (Undetectable.ai, etc. — cuenta
de pago barata). Manual y tedioso pero es lo que usan los estudiantes reales.

**✅ Checkpoint:** `test_adversarial.jsonl` con campo `attack` en cada
muestra: `dipper60|dipper80|backtrans|chars|mix30|humanizer|clean`.

---

## E.3 El script de evaluación único

`scripts/evaluate.py` — LA fuente de verdad del proyecto. Entrada: un JSONL
etiquetado. Salida: tabla de métricas.

```python
"""
Uso: python scripts/evaluate.py test_adversarial.jsonl resultados.json
Corre el pipeline completo (mismo endpoint que producción) sobre cada texto
y calcula métricas por corte.
"""
import json, sys, numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def tpr_at_fpr(y, scores, target=0.01):
    fpr, tpr, thr = roc_curve(y, scores)
    i = np.searchsorted(fpr, target, side="right") - 1
    return tpr[max(i, 0)], thr[max(i, 0)]

rows = [json.loads(l) for l in open(sys.argv[1])]
# scores = probabilidad calibrada de fusion por documento
# (llama a orch.run() + fusion, o al endpoint HTTP local — mismo código
#  que producción, NUNCA un camino "especial de evaluación")

resultados = {}
for corte in ["all", "lang:es", "lang:en", "attack:clean", "attack:dipper60",
              "attack:dipper80", "attack:backtrans", "attack:chars",
              "attack:mix30", "attack:humanizer",
              "words:<1000", "words:1000-10000", "words:>10000"]:
    sub = filtrar(rows, corte)                    # implementa el filtro
    y = [r["label"] for r in sub]
    s = [r["score"] for r in sub]
    t1, _ = tpr_at_fpr(y, s, 0.01)
    t05, _ = tpr_at_fpr(y, s, 0.005)
    resultados[corte] = {
        "n": len(sub), "auroc": roc_auc_score(y, s),
        "tpr@fpr1%": t1, "tpr@fpr0.5%": t05,
    }
json.dump(resultados, open(sys.argv[2], "w"), indent=2)
```

Reglas:

1. **El score evaluado es el de producción** (fusion calibrada), no un
   atajo. Si evalúas otro camino, mides otra cosa.
2. Guarda `resultados.json` versionado (`eval/2026-07-fusion-v1.json`) —
   el histórico es tu evidencia de progreso y tu detector de regresiones.
3. **El número que importa:** la CAÍDA de `attack:clean` →
   `attack:dipper80`/`humanizer`. Un detector que da 0.95 limpio y 0.30
   con paráfrasis no está listo; publicar solo el 0.95 es autoengaño.

**✅ Checkpoint de honestidad:** `attack:chars` ≈ `attack:clean` (sanitizador
funciona). `attack:clean` en humanos españoles: FPR real ≤ 1% con el umbral
elegido — recalcula sobre humanos españoles solos, no sobre el pool.

---

## E.4 Baselines en la misma tabla

Sin comparación, "superior" no significa nada. Corre sobre EL MISMO
`test_adversarial.jsonl`:

1. **Binoculars puro** (Guía B) — baseline académico fuerte y gratis.
2. **GPTZero** — API pública de pago; muestrea 500-1000 docs si el
   presupuesto no da para todo el set.
3. **Tu ensamble neural solo** (sin fusion) — para demostrar cuánto aporta
   la fusion.

Tabla final (la que enseñas a universidades):

| Sistema | AUROC | TPR@FPR1% (limpio) | TPR@FPR1% (dipper80) | TPR@FPR1% (es) |
|---------|-------|--------------------|-----------------------|----------------|
| XplagiaX fusion | | | | |
| XplagiaX neural solo | | | | |
| Binoculars | | | | |
| GPTZero | | | | |

---

## E.5 Cadencia

- Re-corre `evaluate.py` tras: cada re-fit de fusion, cada modelo nuevo en
  la canasta de generación, cada plugin nuevo integrado.
- Cada 3-6 meses: añade al set adversarial textos del último modelo
  frontera publicado y muestras nuevas de humanizadores (evolucionan).
- Automatiza en CI si puedes: subset de 500 docs, falla el build si
  TPR@FPR1% cae > 3 puntos vs el último resultado versionado.

---

## Errores dummy típicos

1. **Evaluar con accuracy/F1 al 50% de umbral** → métrica de juguete.
   Siempre TPR@FPR fijo.
2. **Adversarial derivado del TRAIN** → contaminación; siempre del test.
3. **Publicar solo el resultado limpio** → te lo tumbarán con un
   parafraseador gratuito en la primera demo hostil.
4. **FPR calculado sobre el pool mezclado** → tu riesgo legal es el FPR en
   humanos reales de tu mercado: mídelo sobre españoles/latinoamericanos,
   académicos, no nativos escribiendo en inglés.
5. **Camino de evaluación distinto al de producción** → mides un sistema
   que no existe.
6. **Comparar con GPTZero una vez y no versionar** → ellos actualizan;
   tu tabla caduca. Fecha en cada tabla.
