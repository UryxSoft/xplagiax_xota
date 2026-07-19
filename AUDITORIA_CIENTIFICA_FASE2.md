# AUDITORÍA CIENTÍFICA — FASE 2 (estado post-fusión)
## XPLAGIAX XOTA · Comité multidisciplinario simulado

> **Alcance:** re-auditoría del sistema COMPLETO tras la implementación del motor de
> late-fusion (`21aef24`), señales Tier-1 (`b7dead8`), fixes de veredicto (`aa924b5`,
> `d34541b`) y timeouts dinámicos (`4531103`). Fundamentada en lectura directa del código
> actual en la rama `auditoria-cientifica-fase1`. NO repite el detalle de
> `AUDITORIA_CIENTIFICA.md` (2026-06-10); lo toma como línea base y audita el **delta**.
>
> Fecha: 2026-07-18 · Comité simulado: Arquitecto IA, NLP Sr., Lingüística Computacional,
> Estilometría, ML Eng., DL Researcher, Ciencia de Datos, Estadístico, Matemático Aplicado,
> IR, Plagio, XAI, IA Responsable, Adversarial, Forense, Ing. Software, Rendimiento,
> Optimización Python, UX, QA.
>
> **Regla aplicada:** ninguna afirmación sin evidencia en código (archivo:línea). Donde no
> hay datos se declara "Información insuficiente".

---

## 0. DICTAMEN EJECUTIVO

**Progreso real y verificado.** De los 17 hallazgos C-01…C-17 de la auditoría anterior, 13
están corregidos en el código actual (ver §2). La fusión tardía **existe y está activa por
defecto** (`plugin_orchestrator.py:511`, `forensic_reports.py:1143-1154`): el veredicto ya
NO depende solo del ensemble neural. El scaffold de calibración (`calibration.py`) es
correcto (ECE, reliability bins, Brier, temperature scaling Guo 2017) y está testeado
(`tests/test_fusion_calibration.py`).

**Nuevo hallazgo central de esta fase:** la fusión heurística interina tiene un **sesgo
direccional estructural hacia "IA"** (N-01) y **cuenta dos veces la evidencia neural**
(N-02). En configuración por defecto, 10 de sus 11 términos solo pueden SUBIR P(IA); el
único término pro-humano pertenece a un plugin desactivado por defecto. Un texto humano
formal con confianza neural ≤ ~85% humano puede ser volteado a "AI-Generated" por la suma
de heurísticos que disparan justamente en escritura académica/formal — la población de
falsos positivos que la auditoría anterior identificó como riesgo crítico. La fusión
resolvió el problema "plugins decorativos" pero reintrodujo el riesgo de FP por otra vía.

**El bloqueante irreducible sigue siendo el mismo:** no existe corpus etiquetado ⇒ no
existe NINGUNA métrica medida (Precision, Recall, F1, ECE, TPR@FPR). Todo score sigue
siendo ordinal, no probabilístico. La ruta ya está documentada (`docs/sota/A_FUSION_ENTRENADA.md`);
esta auditoría no la repite, la referencia.

**Calificación global actualizada** (base anterior entre paréntesis):
arquitectura Flask **7.5** (7.0) · calidad de código **7.0** (6.5) · precisión científica
**4.5** (3.0) · precisión del detector **4.5** (4.0) · explicabilidad **5.5** (3.5) ·
confiabilidad **4.5** (3.0) · escalabilidad **6.5** (5.5).
Justificación: la fusión, las señales Tier-1 con interpretación honesta, la incertidumbre
de ensemble y los fixes de concurrencia son mejoras reales; los scores no suben más porque
nada está calibrado ni medido, y por N-01/N-02.

---

## 1. RECONSTRUCCIÓN DEL PIPELINE REAL (estado actual)

```
Cliente
  │  POST /analyze | /analyze_document | /analyze_document_async | /analyze_stream
  ▼
routes.py ── validación (JSON, 500K chars, API key HMAC, rate-limit)
  │          caché Redis 1h con clave sha256(MODEL_VERSION+text+plugins)  [routes.py:55-61]
  ▼
PluginRegistry.run() ── ThreadPoolExecutor(≤8), timeout adaptativo por tamaño
  │                     [plugin_registry.py:28-44,105-177]
  ▼
full_analysis → PluginOrchestrator.run(text)
  │
  ├─ classify_text_aggregate(text)  [detector_final.py:793]
  │    └─ analyze_fast: split por párrafo → tokenización única → batches ordenados
  │       por longitud → 3×ModernBERT (41 clases) → agregado token-weighted
  │       + ensemble_disagreement (std entre seeds)  [detector_final.py:679-790]
  │
  ├─ StylometricProfiler.compute_stats            (descriptores)
  ├─ ReasoningProfiler + ReasoningRiskClassifier  (regex CoT, pesos a mano)
  ├─ PerplexityProfiler Tier1(proxy)/Tier2(GPT-2) + clasificador
  ├─ HybridSegmentAnalyzer                        (⚠ reutiliza los MISMOS 3 ModernBERT)
  ├─ ReferenceValidator                           (OFF por defecto — ver N-01b)
  ├─ author_signature: LUAR (opt-in) o authorship_consistency estilométrico
  ├─ DiscourseAnalyzer                            (Tier-1, léxico inglés)
  ├─ SemanticConsistencyAnalyzer                  (Tier-1, negación/números, inglés)
  ├─ WatermarkDecoder                             (OFF por defecto)
  │
  ├─ FUSIÓN [FUSION_ACTIVE=1 por defecto]  [plugin_orchestrator.py:511-517]
  │    FusionFeatureBuilder → vector 29-dim (_FUSION_SCHEMA)  [fusion.py:46-84]
  │    FusionClassifier sin entrenar → heuristic_fusion():
  │       logit neural capped ±4 → /T=1.6 → + Σ(w·feature) clamp ±1.2 → sigmoid
  │       [fusion.py:260-325]
  │
  └─ ForensicReportGenerator.generate_report
       verdict = "Hybrid" si HybridSegment lo dice; si no, AI/Human según P_fusión ≥ 0.5
       [forensic_reports.py:1143-1154] → HTML/JSON con path único por request
```

Celery (`tasks.py`): soft/hard limits escalados por palabras al encolar
(`routes.py:335-342`), STARTED state, gc + limpieza de base64. Correcto.

---

## 2. DELTA vs AUDITORÍA ANTERIOR — verificado en código

| ID previo | Estado | Evidencia |
|----|--------|-----------|
| C-01 matplotlib en hilos | ✅ Corregido | `generate_plot=False` por defecto (`detector_final.py:127-137`) |
| C-02 raw_scores redondeados a 0/1 | ✅ Corregido | `round(x*100,2)` con clave "ai" (`detector_final.py:191-194`) |
| C-04 override binario 100% | ✅ Corregido | `validar_veredicto_segmento` ahora anota flags no destructivos (`detector_final.py:390-435`) |
| C-05 redondeo por segmento | ⚠ **Parcial** | ver N-04 |
| C-07 heads 3class vs 41 labels | ⚠ Sin verificar | nombres de dirs siguen `_3class_`; cargan con `num_labels=41` sin fallo aparente. **Información insuficiente** — falta smoke test de shapes |
| C-08 contrato /analyze_status | ❌ Persiste | `response.update(task.info)` mezcla estados (`routes.py:363-371`) |
| C-09/C-10/C-11 fallos silenciosos | ✅ Mayormente | `/ready` honesto con `health_report` + `core_unhealthy` (`routes.py:466-505`); plugins aún degradan con warning (aceptado como diseño, pero ver N-13) |
| C-12 sobresuscripción CPU | ✅ Corregido | `torch.set_num_threads(1)` (`detector_final.py:21-25`) |
| C-14 watermark sin cap | ✅ Mitigado | OFF por defecto |
| C-16/C-17 caché sin versión | ✅ Corregido | `MODEL_VERSION` en ambas claves (`routes.py:52-61`, `detector_final.py:665-676`) — pero ver N-10 |
| D-3 léxico "delve" | ✅ Degradado | display-only, no mueve veredicto (`forensic_reports.py:1057-1062`) |
| §13.1 autoría latente | ✅ Activado | `authorship_consistency.py` + LUAR opcional alimentan fusión |
| §13.4 truncación 512 del veredicto global | ✅ Corregido | `classify_text_aggregate` (`plugin_orchestrator.py:319-321`) |
| Fusión inexistente | ✅ Existe (heurística) | pero ver N-01/N-02 |
| Calibración inexistente | ⚠ Framework listo, **no aplicado** | `calibration.py` no cableado (requiere corpus) |
| Reentrenamiento / corpus | ❌ Pendiente | bloqueante irreducible (docs/sota/A) |
| `not_found→fabricated` en referencias | ❌ Persiste | mitigado indirectamente: plugin OFF por defecto |

---

## 3. HALLAZGOS NUEVOS (N-01 … N-14)

### N-01 🔴 CRÍTICO — Asimetría direccional de la fusión heurística
**Ubicación:** `fusion.py:270-285` (`_HEURISTIC_WEIGHTS`, `_HEURISTIC_VERIFIED_WEIGHT`,
`_HEURISTIC_ADJ_CLAMP`).

Los 10 pesos positivos (todos → IA) suman 6.8 en log-odds; el único término negativo
(→ humano) es `ref_verified_ratio` (−0.6), que proviene de `ReferenceValidator` —
**desactivado por defecto** (`PluginConfig.enable_reference_check=False`,
`plugin_orchestrator.py:114`; `ENABLE_REFERENCE_CHECK=0` en `full_analysis.py:62`).
Consecuencia: en producción por defecto el ajuste solo puede empujar hacia IA.

**Demostración numérica del volteo de veredicto** (matemático aplicado):
- Neural P(IA)=0.20 (80% humano) → logit −1.386 → /T=1.6 → base −0.866.
- Ajuste máximo +1.2 (alcanzable: un ensayo académico humano con
  `dsc_uniformity≈0.6` (+0.42), `hal_overall≈0.5` (+0.30), `rsn_cot≈0.5` (+0.25),
  `rsn_backtracking≈0.3` (+0.15), `ppl_low_ratio≈0.4` (+0.16), `sem_contradiction≈0.2`
  (+0.12) = +1.40 → clamp 1.2).
- Total +0.334 → **P(IA)=0.58 → verdict "AI-Generated"** (`forensic_reports.py:1154`).

Todas esas señales disparan precisamente en las poblaciones FP identificadas en §3 de la
auditoría previa (académico formal, legal, técnico, ESL): conectores formales
(`discourse_analyzer.py:32-45`), scaffolding CoT ("therefore", "step by step"), terminología
repetida (proxy-perplexity). El comentario del código ("no single weak signal can flip the
verdict", `fusion.py:251-252`) es cierto por señal individual y **falso para la suma**.

**Mitigación (compatible, O(1) de costo):**
1. Clamp asimétrico: ajuste positivo ≤ +0.6, negativo hasta −1.2 (principio obligatorio
   n.º 1: reducir FP antes que subir sensibilidad).
2. Regla de corroboración: ajuste > +0.6 solo si ≥2 familias de señal independientes
   superan su umbral (referencia fabricada cuenta doble; discourse+reasoning cuentan como
   UNA familia porque ambas miden "formalidad estructural").
3. Añadir términos pro-humano de costo cero ya disponibles: `sty_burstiness` alto,
   `hapax_legomena_ratio` alto, `author_signature.consistency` con outliers=0
   (hoy solo `outlier_ratio` empuja a IA; su complemento nunca empuja a humano).
4. Banda de incertidumbre en el veredicto (ver N-03).
- **CPU/RAM:** nulo. **Riesgo:** bajo. **Prioridad:** CRÍTICA. **Compatibilidad:** cambia
  solo constantes + una condición en `heuristic_fusion()`.

### N-02 🔴 CRÍTICO — Doble conteo de la evidencia neural en la fusión
**Ubicación:** `fusion.py:277` (`hyb_ai_ratio: 0.6`) + `plugin_orchestrator.py:252-257`.

`HybridSegmentAnalyzer` clasifica párrafos con `classify_batch`/`classify_segment` — los
**mismos 3 ModernBERT** que generan `neural_ai_prob` (vía `classify_text_aggregate` →
`analyze_fast`, también por párrafo). `hyb_ai_ratio` no es señal independiente: es la
fracción de párrafos donde el mismo modelo dijo IA. El comentario "MODEL-AGNOSTIC signals"
(`fusion.py:253`) es incorrecto para este término. Efectos:
- Sobreconfianza: texto que el neural marca IA al 90% recibe además +0.6 por el mismo juicio.
- Amplificación de error: si el neural se equivoca por párrafos en texto humano formal
  (su modo de fallo conocido), el error se cuenta dos veces.

**Mitigación:** sacar `hyb_ai_ratio` del ajuste heurístico (dejarlo en el vector para la
fusión ENTRENADA, donde la regresión logística absorbe la colinealidad). Si se quiere señal
de mezcla, usar `hyb_breakpoints` normalizado (estructura de alternancia — eso sí es
información no contenida en el agregado global). **Costo:** nulo. **Prioridad:** CRÍTICA.

### N-03 🟠 ALTO — Incertidumbre calculada pero ignorada por el veredicto
`ensemble_disagreement` (std entre seeds, `detector_final.py:341-346`) y
`neural_uncertainty` (`fusion.py:151`) se calculan y transportan, pero:
- `heuristic_fusion()` no los usa (`fusion.py:293-325`);
- el veredicto es binario a 0.5 exacto (`forensic_reports.py:1154`);
- `uncertainty_zone` (`detector_final.py:820`) no aparece en el veredicto forense.

Señal de incertidumbre **gratis** (ya computada) desperdiciada. Es exactamente lo que un
detector probabilístico (filosofía del sistema) debe reportar.
**Mitigación:** verdict "Inconclusive" cuando `|P−0.5| < 0.10` **o**
`ensemble_disagreement ≥ 12` (umbral ya usado en `classify_text_aggregate:820`), con texto
explicativo. Reduce FP y FN simultáneamente convirtiendo errores confiados en abstenciones
honestas — el comportamiento que Turnitin/GPTZero adoptaron públicamente para la franja
gris. **Costo:** nulo. **Prioridad:** ALTA. **Complejidad:** ~10 líneas.

### N-04 🟠 MEDIO — El redondeo entero por segmento persiste (C-05 parcial)
`classify_batch` devuelve `round(human_prob*100)` (`detector_final.py:281`) y
`_classify_batch_from_ids` hace `human_pct = round(...)`, `ai_pct = 100 − human_pct`
(`detector_final.py:351-352`). Un párrafo con P=0.996 se muestra "100%" — falsamente
nítido (el mismo defecto que motivó el cap de logits en fusión, que lo re-mitiga después).
El agregado token-weighted promedia enteros: sesgo ≤0.5pp — tolerable, pero gratuito de
eliminar. **Mitigación:** redondear solo en la capa de presentación. **Prioridad:** MEDIA.

### N-05 🟠 ALTO — Señales Tier-1 monolingües sin gate de idioma
`discourse_analyzer.py:32-45` (marcadores), `semantic_consistency.py:47-60` (negaciones,
stopwords) son SOLO inglés. En español/francés/portugués:
- `connective_density≈0`, `_NEGATION_CUES` nunca matchea → señales estructuralmente
  distintas por idioma, con los mismos pesos de fusión;
- `opening_repetition`/`paragraph_uniformity` sí operan → la mezcla de sub-features cambia
  de significado según el idioma sin que la fusión lo sepa.
El pipeline es English-trained (declarado en `discourse_analyzer.py:22-23`) pero el
producto acepta cualquier idioma. **Mitigación mínima:** detección de idioma O(n) barata
(stopword-ratio, sin dependencia nueva); si idioma ≠ en, poner a 0 los features Tier-1
léxico-dependientes y **renormalizar** el ajuste (no dejar que el resto herede su peso), y
reportarlo en la explicación ("señal X no aplicable: idioma"). **Mitigación real:**
léxicos es/fr/pt (esfuerzo bajo: son listas de conectores). **Prioridad:** ALTA para
mercado hispanohablante. **Costo:** despreciable.

### N-06 🟠 MEDIO — Contradicción por flip de negación: heurística débil con peso 0.6
`semantic_consistency.py:177-180`: cualquier par de oraciones con Jaccard≥0.5 donde solo
una contiene negación cuenta como contradicción. La prosa argumentativa humana usa
exactamente ese patrón para el contraste legítimo ("X is not merely A; X is B" vs oración
vecina). Sin validación empírica, alimenta la fusión con peso 0.6 (`fusion.py:279`).
**Mitigación:** contar para la fusión solo (a) mismatch numérico sobre sujeto compartido
(heurística 2, mucho más específica) o (b) contradicción NLI cuando `SEMANTIC_NLI=1`;
el flip de negación queda como evidencia mostrada, no como score. Alternativa mínima: peso
0.6→0.25 hasta validar con corpus. **Prioridad:** MEDIA-ALTA (es un término del volteo N-01).

### N-07 🟡 BAJO — `FusionClassifier()` instanciado por request
`plugin_orchestrator.py:514` y `forensic_reports.py:1148` crean instancia nueva por
análisis. Hoy es gratis (sin estado). Cuando existan pesos entrenados en disco, este patrón
releería el modelo por request. **Mitigación:** singleton a nivel de módulo con
`load_weights()` en preload, ahora que el contrato es estable. **Prioridad:** BAJA (prepara Fase A).

### N-08 🟡 BAJO — Escalas mezcladas en `_FUSION_SCHEMA`
`ppl_proxy_mean` (escala inventada [1,15]), `sty_avg_sentence_len` (~5-40),
`hyb_breakpoints`/`hyb_longest_ai_run` (conteos) conviven con features [0,1]
(`fusion.py:46-84`). Inofensivo hoy (heurística solo consume features [0,1] con clip;
logistic estandariza), pero `hyb_global_ai` sí se normaliza (`fusion.py:197`) y el resto
no — inconsistencia que complicará la ablation y la interpretación de pesos entrenados.
**Mitigación:** normalizar TODO a [0,1] en el builder con rangos documentados. **Prioridad:** BAJA.

### N-09 🟡 BAJO — UX del veredicto "Hybrid"
Con verdict Hybrid, `confidence` sigue siendo la P(IA) global fusionada; el HTML muestra
"Human Score" si P<0.5 (`forensic_reports.py:2179-2181`) bajo un banner "HYBRID". Confuso
para el usuario final. **Mitigación:** para Hybrid mostrar `ai_segment_ratio` y nº de
breakpoints en lugar de un porcentaje global único. **Prioridad:** BAJA (UX).

### N-10 🟡 BAJO — `MODEL_VERSION` desfasada
Default `"2026.06"` (`routes.py:55`) mientras el README declara v2026.07 y la fusión cambió
el resultado de TODOS los análisis (`aa924b5`, `21aef24`). Si el deploy no exporta
`MODEL_VERSION`, la caché Redis puede servir hasta 1h de resultados pre-fusión tras un
deploy. **Mitigación:** bump del default en cada cambio de pesos/fusión/umbral (regla ya
documentada en el propio comentario `routes.py:52-54`; solo falta cumplirla). **Prioridad:** BAJA.

### N-11 🟠 MEDIO — Acumulación de hilos zombis bajo timeouts repetidos
`plugin_registry.py:172-175`: `shutdown(wait=False)` es correcto para no bloquear el
request, pero el hilo del plugin agotado sigue ejecutando inferencia CPU-bound completa.
Bajo carga adversarial (documentos grandes repetidos al endpoint síncrono), los hilos
huérfanos se acumulan y compiten por CPU con requests vivos. Rate-limit 60/min mitiga
parcialmente. **Mitigación:** flag cooperativo de cancelación consultado entre batches en
`analyze_fast`/plugins largos (`threading.Event` en un contexto por request). **Prioridad:**
MEDIA. **Complejidad:** moderada.

### N-12 🟠 MEDIO — El único plugin "científicamente fuerte" está apagado por defecto
La auditoría previa clasificó `reference_validator` como la señal más defendible (ground
truth externo). Sigue OFF por defecto (`ENABLE_REFERENCE_CHECK=0`) por su dependencia de
red — decisión razonable para latencia, pero deja la fusión sin su mejor feature y sin su
único término pro-humano (N-01). **Mitigación:** modo degradado sin red ya soportado
(`enable_network`); activarlo por defecto en el path **async** (Celery, donde la latencia
de red es tolerable) y mantenerlo off en sync. Corregir antes `not_found→fabricated`
(§13.3 previa: exigir miss en ≥2 bases + separar `not_indexed_count`). **Prioridad:** ALTA.

### N-13 🟡 BAJO — Degradación de plugins invisible en el resultado final
`run_with_result` captura excepciones por plugin con warning (`plugin_orchestrator.py:346-505`).
El informe no distingue "señal ausente por fallo" de "señal neutra" (el builder de fusión
rellena 0.0, `fusion.py:23-24` — al menos 0.0 no empuja a IA, mejora sobre el 0.5 previo).
**Mitigación:** lista `degraded_signals` en el dict de fusión y en el reporte HTML — es
además un requisito de defendibilidad forense. **Prioridad:** MEDIA. **Costo:** trivial.

### N-14 ✅ POSITIVOS verificados (para preservar)
- `analyze_fast`: 1 sola tokenización, batching por longitud (30-50% menos FLOPs de
  padding), caché namespaced, disagreement gratis (`detector_final.py:679-790`). Estado del
  arte para CPU en este tamaño de modelo.
- Inyección manual CLS/SEP (`detector_final.py:312-320`) — fix real de un bug que invertía
  veredictos.
- Cap de logits ±4 + T=1.6: elimina el "100% IA" imposible de un detector OOD-blind.
- Señales Tier-1 con interpretación honesta y evidencia inspeccionable (markers, pares de
  oraciones, chunks outliers) — buena práctica XAI.
- `adaptive_timeout` coherente entre sync/async/Celery.
- Suite de tests de fusión/calibración existente.

---

## 4. AUDITORÍA MATEMÁTICA DEL MOTOR DE SCORING ACTUAL

Fórmula real (`fusion.py:293-325`):

```
base = clip(logit(p_neural), ±4.0) / 1.6
adj  = clip( Σᵢ wᵢ·clip(xᵢ,0,1)  +  (−0.6)·ref_verified , ±1.2 )
P(IA) = σ(base + adj)
```

| Propiedad | Evaluación |
|---|---|
| Monotonicidad | ✔ correcta por término |
| Acotación | ✔ P ∈ [σ(−3.7), σ(+3.7)] ≈ [0.024, 0.976] |
| Anti-sobreconfianza | ✔ cap+T bien fundamentados (softmax overconfidence documentado) |
| Independencia de evidencias | ✘ N-02 (hyb=neural), rsn↔dsc correlacionados (ambos = formalidad) |
| Simetría de evidencia | ✘ N-01 (10 términos → IA, 0 activos → humano por defecto) |
| Uso de incertidumbre | ✘ N-03 (disagreement/uncertainty no entran) |
| Calibración | ✘ declarada honestamente como ausente (`calibrated=False`) — correcto |
| Interpretación probabilística del % | ✘ sigue siendo ordinal; el HTML lo muestra como % |

**Veredicto del estadístico:** la estructura (log-odds aditivo acotado) es la correcta —
es exactamente la forma de una regresión logística con pesos fijados a mano, lo que hace
trivial la transición a la versión entrenada. Los defectos son de **parametrización**
(dirección y colinealidad), no de forma. Corregir N-01/N-02/N-03 no requiere corpus.

---

## 5. MATRIZ DE PLUGINS (estado actual, cambios desde la línea base)

| Plugin/Señal | Coste CPU | Independencia real | En fusión | Estado | Acción |
|---|---|---|---|---|---|
| 3×ModernBERT (neural) | Alto (dominante) | — (base) | base_lo | OOD 2026 | Mantener; reentrenar (doc A) |
| hybrid_segment | Alto (2ª pasada neural) | ✘ = neural | 0.6 ⚠ | N-02 | Quitar del ajuste heurístico; conservar breakpoints |
| perplexity T1 proxy | Bajo | media | 0.4 | pseudo-señal (previa §2) | Sustituir por Binoculars (doc B); mientras, peso ≤0.2 |
| perplexity T2 GPT-2 | Medio | media | (vía T1 features) | referencia 2019 obsoleta | idem |
| reasoning (regex CoT) | Bajo | ✘ correl. con discourse | 0.5+0.5 | umbral a mano | Fusionar familia con discourse (una sola contribución) |
| hallucination | Bajo | media | 0.6 | heurístico honesto | Mantener con peso; validar con corpus |
| reference_check | Red | ✔ ALTA (ground truth) | 1.6/0.9/−0.6 | OFF por defecto | N-12: ON en async + fix not_found |
| stylometric | Bajo | media | solo vector | descriptores | Añadir término pro-humano (N-01.3) |
| author_signature (LUAR/stylo) | Medio | ✔ alta | 0.4 | bien diseñado | Mantener |
| discourse_structure | Ínfimo | ✘ = formalidad | 0.7 | inglés-only (N-05) | Gate idioma; agrupar familia |
| semantic_consistency | Bajo (O(n²) cap 120) | ✔ alta si específica | 0.6 | flip-negación débil (N-06) | Restringir a numeric/NLI |
| watermark | Alto | especulativa | no | OFF | Mantener OFF |
| zone_classifier / citation | Bajo | ✔ (dominio plagio) | no | válido | Puente plagio↔IA (mejora M-9) |

---

## 6. FALSOS POSITIVOS / FALSOS NEGATIVOS — estado actual

**FP (humano→IA).** El vector de ataque de la línea base (léxico "delve") está neutralizado,
pero N-01+N-06+N-05 crean un camino nuevo: académico formal + conectores + negaciones
retóricas ⇒ ajuste +≈1.2 ⇒ volteo si neural < ~85% humano. Población de mayor riesgo: la
misma de siempre (académico, legal, ESL formal). **Las mitigaciones N-01.1-4 cierran este
camino sin corpus.**

**FN (IA→humano).** Sin cambios de fondo: modelos 2024-2026 siguen OOD para el neural; la
fusión heurística da algo de lift model-agnóstico (referencias fabricadas, contradicciones,
uniformidad) — dirección correcta, magnitud **no medida** (Información insuficiente hasta
doc E). Parafraseo/humanizadores: `dsc_uniformity` y `sem_contradiction` sobreviven al
parafraseo (bien elegidas); `ppl_*` no.

**Cuantificación:** imposible sin corpus etiquetado. Toda estimación numérica de
Precision/Recall/F1 en este informe sería inventada — prohibido por las reglas de la
auditoría. El protocolo para obtenerlas ya existe (`docs/EXPERIMENTAL_PROTOCOL.md`,
`docs/sota/A`, `docs/sota/E` con TPR@FPR=1% como métrica primaria).

---

## 7. BENCHMARK (solo información pública, sin especular con propietario)

| Práctica observable en competidores | XPLAGIAX hoy | Brecha accionable |
|---|---|---|
| GPTZero: franja explícita "mixed/unclear", abstención | Verdicto binario a 0.5 | N-03 (Inconclusive) — costo cero |
| GPTZero/Copyleaks: resaltado por oración/párrafo | ✔ hybrid_segment + heatmap | Paridad |
| Copyleaks: multilingüe declarado | Inglés-céntrico no declarado al usuario | N-05 + declarar cobertura de idioma en el reporte |
| Originality: score como "probabilidad, no prueba" en UI | Disclaimers en HTML, % prominente | N-09; % con banda de incertidumbre |
| Turnitin: no puntúa documentos <300 palabras; comunica % como "cantidad de texto señalado", no certeza | Analiza cualquier longitud | Umbral mínimo de palabras con abstención explícita |
| Winston: evidencia por factor visible | ✔ evidence_points + markers Tier-1 | Paridad razonable |
| Académico (Binoculars, Fast-DetectGPT): perplejidad par-modelo, zero-shot | proxy T1 + GPT-2 T2 | doc B (pendiente, CPU-viable con modelos ~0.5-1B cuantizados) |
| Académico (calibración): ECE + temperature en validación | framework listo, sin datos | doc A |
| Filosofía llama.cpp aplicable: 1 sola capa de paralelismo, buffers reutilizados | ✔ ya aplicada (torch threads=1, batching, CoW) | Paridad |

---

## 8. PLAN PRIORIZADO (todas las mejoras, una tabla; ordenar por la columna que se necesite)

Leyenda impacto: FP↓ = reduce falsos positivos; FN↓ = reduce falsos negativos.

| # | Mejora | FP↓ | FN↓ | CPU | RAM | Riesgo | Complejidad | Prioridad | Archivos |
|---|--------|-----|-----|-----|-----|--------|-------------|-----------|----------|
| M-1 | Clamp asimétrico + regla de corroboración (N-01) | ●●● | ○ | 0 | 0 | Bajo | Trivial | CRÍTICA | fusion.py |
| M-2 | Quitar `hyb_ai_ratio` del ajuste; usar breakpoints (N-02) | ●●● | ○ | 0 | 0 | Bajo | Trivial | CRÍTICA | fusion.py |
| M-3 | Veredicto "Inconclusive" por banda + disagreement (N-03) | ●●● | ●● | 0 | 0 | Bajo | Baja | CRÍTICA | forensic_reports.py |
| M-4 | Términos pro-humano (burstiness, hapax, consistencia sin outliers) | ●●● | ○ | 0 | 0 | Medio* | Baja | ALTA | fusion.py |
| M-5 | Gate de idioma + renormalización Tier-1 (N-05) | ●● | ● | ~0 | 0 | Bajo | Baja | ALTA | fusion.py, orchestrator |
| M-6 | Restringir sem_contradiction a numeric/NLI (N-06) | ●● | ○ | 0 | 0 | Bajo | Trivial | ALTA | fusion.py o semantic_consistency.py |
| M-7 | reference_check ON en async + fix not_found→not_indexed (N-12) | ●● | ●●● | red | 0 | Medio | Media | ALTA | reference_validator.py, tasks.py |
| M-8 | degraded_signals visibles (N-13) | ● | ● | 0 | 0 | Bajo | Trivial | MEDIA | fusion.py, forensic_reports.py |
| M-9 | Puente plagio↔IA: zonas citadas excluidas del scoring IA | ●● | ○ | 0 | 0 | Medio | Media | MEDIA | forensic_reports.py, citation |
| M-10 | Floats por segmento, redondeo solo en display (N-04) | ● | ● | 0 | 0 | Bajo | Trivial | MEDIA | detector_final.py |
| M-11 | Cancelación cooperativa de hilos huérfanos (N-11) | — | — | ↓ | ↓ | Medio | Media | MEDIA | plugin_registry.py, detector_final.py |
| M-12 | Normalizar _FUSION_SCHEMA a [0,1] (N-08) | — | — | 0 | 0 | Bajo | Baja | BAJA | fusion.py |
| M-13 | Singleton FusionClassifier + load_weights (N-07) | — | — | 0 | 0 | Bajo | Baja | BAJA | fusion.py, orchestrator |
| M-14 | Bump MODEL_VERSION="2026.07" (N-10) | ● | ● | 0 | 0 | Nulo | Trivial | BAJA | routes.py |
| M-15 | Contrato /analyze_status normalizado (C-08) | — | — | 0 | 0 | Bajo | Trivial | BAJA | routes.py |
| M-16 | Smoke test shapes de los 3 checkpoints (C-07) | — | — | 0 | 0 | Bajo | Trivial | MEDIA | tests/ |
| M-17 | UX Hybrid: ratio+breakpoints en vez de % único (N-09) | ● | — | 0 | 0 | Bajo | Baja | BAJA | forensic_reports.py |
| M-18 | Léxicos es/fr/pt para discourse/negación | ●● | ● | 0 | ~0 | Medio | Media | MEDIA | discourse_analyzer.py, semantic_consistency.py |
| M-19 | Corpus etiquetado + fusión entrenada + temperature scaling + ECE | ●●● | ●●● | offline | 0 | Alto | Alta | CRÍTICA (mediano plazo) | docs/sota/A |
| M-20 | Binoculars CPU (sustituye perplexity T1/T2) | ● | ●●● | +Medio | +1-2GB | Alto | Alta | ALTA (mediano plazo) | docs/sota/B |

\* M-4 con riesgo "Medio" solo porque señales pro-humano mal calibradas podrían subir FN;
por eso deben entrar con pesos pequeños (≤0.4) y bajo la misma regla de corroboración.

**Los "Top 20" solicitados** (por impacto, por costo, por facilidad, por FP, por FN, por
reutilización, por no-tocar-modelo) se obtienen ordenando esta tabla por la columna
correspondiente; M-1…M-18 cumplen TODAS: reutilizan código existente, no tocan el modelo
neural, CPU/RAM ≈ 0.

---

## 9. ROADMAP POR SPRINTS

**Sprint 1 (1 semana) — Cerrar el camino de FP de la fusión.** M-1, M-2, M-3, M-6, M-14.
Archivos: `fusion.py`, `forensic_reports.py`, `routes.py`. Pruebas: extender
`test_fusion_calibration.py` con casos de volteo (texto humano formal sintético debe
permanecer Human/Inconclusive). Riesgo: bajo. Beneficio: elimina el hallazgo crítico N-01/N-02.

**Sprint 2 (1 semana) — Simetría y honestidad de señal.** M-4, M-5, M-8, M-10, M-16.
Pruebas: fusión con señales degradadas; textos es/en. Beneficio: FP↓ en ESL/es, evidencia
forense defendible.

**Sprint 3 (2 semanas) — Referencias como pilar.** M-7 (+M-18 si hay tiempo).
Requiere red en workers Celery; circuit breaker ya existe. Beneficio: mejor señal
model-agnóstica del sistema activa en producción; único término pro-humano real.

**Sprint 4 (2-4 semanas) — Corpus y fusión entrenada.** M-19 según `docs/sota/A`:
recolectar/etiquetar, `FusionClassifier.fit()`, `TemperatureScaler`, ECE + reliability,
TPR@FPR=1% como métrica de aceptación. Es el único camino a números reales de
Precision/Recall/F1. M-13 como prerrequisito de despliegue.

**Sprint 5 (2 semanas) — Perplejidad real + suite adversarial.** M-20 (doc B) + doc E
(DIPPER/parafraseo/humanizadores). Re-entrenar la fusión con el feature nuevo (regla de
oro del índice SOTA: A se re-entrena al final).

---

## 10. CONCLUSIÓN DEL COMITÉ

El sistema pasó de "un clasificador 2023 con narrativa forense" a "un motor de fusión
log-odds acotado con señales parcialmente independientes y calibración pendiente". Es la
arquitectura correcta. Los tres defectos que impiden subir de nota — dirección asimétrica
del ajuste (N-01), colinealidad neural (N-02) e incertidumbre ignorada (N-03) — se
corrigen con cambios de constantes y ~30 líneas, sin corpus, sin CPU extra y sin tocar
ninguna API. Después de eso, el único techo real es la ausencia de datos etiquetados: sin
corpus no hay calibración, ni métricas, ni defensa científica del porcentaje mostrado.
El plan A→E de `docs/sota/` sigue siendo la ruta correcta y este comité no lo enmienda —
solo antepone el Sprint 1 como condición para que la fusión activa sea segura en producción.

---
---

# ANEXOS — Partes 7 a 12 y 14 del mandato

---

## ANEXO A — Motor de Scoring: reconstrucción completa (Parte 7)

### A.1 Fórmula conceptual del score (estado actual, exacta)

```
p_neural = agregado token-weighted por párrafo de 3×ModernBERT     [analyze_fast]
base     = clip(logit(p_neural), −4, +4) / T,  T = 1.6             [fusion.py:260-267]
adj      = clip( Σᵢ wᵢ·clip(xᵢ,0,1) − 0.6·ref_verified, −1.2, +1.2 )
P(IA)    = σ(base + adj)
verdict  = "Hybrid"  si HybridSegment lo declara
           "AI-Generated"  si P ≥ 0.5, si no "Human-Written"       [forensic_reports.py:1150-1154]
```
Sin votación, sin cascada, sin pesos dinámicos, sin ajuste por longitud/idioma/tipo
documental (Fase 3: **todas las respuestas son "no"** — es la brecha de la Fase 19, ver A.7).

### A.2 Inventario de evidencias (Fase 2)

| Evidencia | Origen | Costo | Frecuencia | Confiabilidad (juicio del comité) |
|---|---|---|---|---|
| p_neural (3×ModernBERT) | detector_final | ALTO (dominante) | siempre | Media dentro de distribución 2023; baja OOD |
| ensemble_disagreement | gratis con la anterior | 0 | siempre | Alta como señal de OOD — **no consumida** (N-03) |
| hyb_ai_ratio / breakpoints | hybrid_segment | ALTO (2ª pasada neural) | siempre | = neural (N-02) |
| ppl proxy T1 / GPT-2 T2 | perplexity_profiler | Bajo / Medio | siempre | Baja (proxy no es perplejidad; GPT-2 2019) |
| rsn_* (regex CoT) | reasoning_profiler | Bajo | siempre | Media-baja (correlada con registro formal) |
| hal_overall + categorías | hallucination_profile | Bajo | siempre | Media (heurístico honesto) |
| ref_fabricated/chimeric/verified | reference_validator | Red | **OFF por defecto** | ALTA (único ground truth externo) |
| sty_* descriptores | stylometric_profiler | Bajo | siempre | Media (descriptores, no clasificadores) |
| author outlier_ratio | LUAR o stylo-consistency | Medio/Bajo | siempre | Media-alta (localización, no veredicto) |
| dsc_uniformity | discourse_analyzer | Ínfimo | siempre | Media (inglés-only, N-05) |
| sem_contradiction_ratio | semantic_consistency | Bajo (O(n²) cap 120) | siempre | Media si numérica; baja si flip-negación (N-06) |
| watermark | watermark_decoder | Alto | OFF | Especulativa |

### A.3 Matriz de correlación entre evidencias (Fases 4-5; cualitativa — sin corpus no hay r de Pearson)

| Clúster | Miembros | Independencia | Valor único marginal |
|---|---|---|---|
| **Neural** | p_neural, hyb_* | nula entre sí | hyb aporta SOLO localización (breakpoints); su ratio global es redundante |
| **Registro formal** | rsn_cot, rsn_backtracking, dsc_uniformity, hal_vagueness | baja entre sí | contar como UNA familia en el ajuste (mitigación N-01.2) |
| **Diversidad léxica** | sty_lexical_diversity, sty_hapax, rsn_type_token_ratio, ppl_proxy_mean (construida sobre hapax+TTR), hal_unigram_entropy | baja | el proxy T1 es matemáticamente derivado del clúster estilométrico — doble conteo latente si ambos entran a la fusión entrenada sin regularización |
| **Ritmo oracional** | sty_avg_sentence_length, rsn_mean_sentence_length, sty_sentence_length_variance, rsn_std_sentence_length, hal_sentence_length_uniformity, ppl_burstiness | baja | 4 módulos calculan la MISMA estadística por separado (ver Anexo C, P-03) |
| **Independientes** | ref_*, sem (numérica), author_outlier, watermark | ALTA | los únicos con valor marginal irremplazable |

Eliminación simulada (Fase 5/18, razonada): quitar `ref_*` pierde la única evidencia
fuerte y el único término pro-humano; quitar `hyb` del ajuste no pierde nada (N-02);
quitar `ppl_low_ratio` pierde ~0 (señal frágil, peso 0.4); quitar `dsc`/`rsn` a la vez
pierde el único lift model-agnóstico de estructura. Verificación empírica: ablation del
protocolo doc A cuando exista corpus.

### A.4 Matriz de calibración (Fase 7-8, 14) — diseño recomendado

| P(IA) fusionada | Etiqueta | Riesgo FP | Riesgo FN | Acción recomendada |
|---|---|---|---|---|
| ≥ 0.90 | Evidencia fuerte de IA | Bajo* | — | Mostrar evidencias; nunca "prueba" |
| 0.60-0.90 | Evidencia moderada | Medio | Bajo | Mostrar señales a favor/en contra |
| **0.40-0.60** | **Incierto** | Alto | Alto | **Abstención + revisión humana (N-03)** |
| 0.10-0.40 | Evidencia débil de IA | — | Medio | Reportar como "sin evidencia suficiente" |
| < 0.10 | Sin evidencia de IA | — | Alto si editado | Ídem |

\* "Bajo" SOLO tras corregir N-01/N-02; hoy la franja alta es alcanzable por acumulación
heurística. El % mostrado debe etiquetarse "nivel de evidencia (no calibrado)" hasta doc A.

### A.5 Explainable scoring (Fase 15) — **hallazgo nuevo N-15 🟠**
`heuristic_fusion()` YA calcula la contribución en log-odds de cada término
(`contributions`, `fusion.py:307-323`) — exactamente el artefacto que exige la Fase 15
(qué subió, qué bajó, qué fue neutro) — pero `predict_proba_vec` la **descarta**
(`fusion.py:371`: `p, contrib = heuristic_fusion(feat)`; `FusionResult` guarda solo los
features de entrada). Fix trivial: incluir `contributions` en `FusionResult.to_dict()` y
renderizarlas en el HTML como tabla "evidencia → dirección → magnitud". Costo 0. Es la
mejora de explicabilidad de mayor ROI de todo el sistema. → **M-21, prioridad ALTA.**

### A.6 Sensibilidad (Fase 13)
Dominante: p_neural (por diseño). Sobre-influyentes respecto a su confiabilidad:
`ref_fabricated` (1.6 — justificado SI el plugin está bien; hoy `not_found→fabricated`
lo contamina), `dsc_uniformity` (0.7, inglés-only), `sem_contradiction` (0.6, flip débil).
Nunca influyen: watermark (OFF), `ref_*` (OFF por defecto), `neural_uncertainty` (en el
vector, ignorada por la heurística).

### A.7 Sistema de reglas inteligentes (Fase 19) — reglas recomendadas, todas O(n)

| Regla | Condición | Acción | Justificación |
|---|---|---|---|
| R-1 texto corto | < 200 palabras | verdict "Insufficient text", sin % | varianza de todas las señales explota; Turnitin usa umbral público similar |
| R-2 idioma ≠ en | stopword-ratio es/fr/pt | anular features Tier-1 léxicas + renormalizar + declarar | N-05 |
| R-3 código fuente | ratio símbolos/keywords | excluir bloques de código del scoring | código dispara uniformidad/repetición |
| R-4 citas densas | zone_classifier coverage > umbral | excluir zonas citadas del scoring IA | M-9; texto citado no es del autor |
| R-5 evidencia contradictoria | señales fuertes en ambas direcciones | forzar franja Incierto | Parte 10 Fase 3 |
| R-6 OCR degradado | (cuando llegue del servicio OCR) | subir incertidumbre | ruido OCR imita señales IA |
| R-7 seeds en desacuerdo | disagreement ≥ 12 | franja Incierto | N-03, ya calculado |

---

## ANEXO B — Auditoría de Features (Parte 8)

### B.1 Inventario completo (~75 features crudas → 29 en `_FUSION_SCHEMA`)

| Módulo | Features | Clase |
|---|---|---|
| stylometric_profiler (11) | avg_dep_distance, avg_sentence_length, avg_word_length, burstiness, comma_rate, complex_sentence_ratio, hapax_legomena_ratio, lexical_diversity, rare_word_ratio, sentence_length_variance, vocabulary_richness | estilométricas/lingüísticas |
| reasoning_profiler (15) | type_token_ratio, mean/std_sentence_length, mean_word_length, punctuation_ratio, stopword_ratio, {consequence, causal, contrast, sequence, backtracking, cot_scaffold, intuition_leap}_density, paragraph_length_cv, word_entropy_normalised | sintácticas/heurísticas |
| perplexity_profiler (6+) | proxy_perplexity_mean, low_perplexity_ratio, perplexity_valley_count, burstiness_perplexity, curvature_score, token_entropy_mean | estadísticas |
| hallucination_profile (18→6 categorías) | entity(4): entity_density, unique_entity_ratio, person_org_ratio, date_num_ratio · entropy(2): uni/bigram_entropy · cohesion(4): avg/min_jaccard, max_semantic_drop, disconnected_ratio · vagueness(3) · repetition(3) · structural(2) | semánticas/heurísticas |
| hybrid_segment (4+por-párrafo) | global_ai_score, ai_segment_ratio, breakpoint_count, longest_ai_run | neurales |
| reference_validator (6) | total/fabricated/chimeric/verified counts + ratios | metadatos/ground-truth |
| discourse_analyzer (5) | connective_density, paragraph_uniformity, enumeration_scaffold, opening_repetition, conclusion_marker | discursivas |
| semantic_consistency (2) | contradiction_ratio, contradiction_count | semánticas |
| authorship (4) | consistency_score, outlier_ratio, mean/max_rms_zscore | estilométricas |
| neural (4) | ai%, human%, ensemble_disagreement, detected_model | modelo |

### B.2 Veredictos por feature (Fases 4-13, consolidado)

- **Críticas:** p_neural, ensemble_disagreement, ref_fabricated_ratio, author_outlier_ratio.
- **Redundantes (fusionar cálculo, conservar una):** mean/std_sentence_length (reasoning)
  ≡ avg_sentence_length/variance (stylometric); hal_sentence_length_uniformity ≡ inverso de
  lo anterior; rsn_type_token_ratio ≡ lexical_diversity; ppl proxy T1 ≡ combinación de
  hapax+TTR ya existente. **Ninguna debe eliminarse del output** (compatibilidad), pero el
  CÁLCULO debe unificarse (Anexo C P-03) y solo un representante por clúster debe entrar
  al ajuste heurístico.
- **Eliminables del ajuste (no del reporte):** hyb_ai_ratio (N-02), ppl_low_ratio (frágil,
  pseudo-señal), sem flip-negación (N-06).
- **Faltantes de bajo costo (Fase 12), calculables con lo ya existente:** varianza de
  readability por párrafo (FK ya se calcula), entropía por párrafo (unigram_entropy ya
  existe — solo re-scope), estabilidad de function-words entre chunks (authorship ya
  chunkéa), densidad de citas por sección (zone_classifier ya las extrae), ratio
  certeza/hedging (léxico hedge de hallucination ya existe). Cada una: O(n), RAM ~0,
  reutiliza parsing existente. Candidatas al vector de fusión ENTRENADA, no a la heurística.

---

## ANEXO C — Rendimiento (Parte 9)

**Declaración de método:** esta sesión no ejecutó profiling (no se corrió el servicio);
lo que sigue es análisis estático del código + modelo de costos. Los números marcados ≈
deben validarse con la harness de medición C.4. Sin esa harness, cualquier "CPU −X%" sería
inventado. Información insuficiente para: RAM pico real, hit-ratio de Redis, p99.

### C.1 Modelo de costos por etapa (documento de W palabras)

| Etapa | Complejidad | Peso CPU estimado en full_analysis |
|---|---|---|
| 3×ModernBERT vía analyze_fast (pasada 1) | O(W·d) transformer | ~30-35% |
| 3×ModernBERT vía hybrid windows 300w/50% overlap (pasada 2, ≈2×W tokens) | O(2W·d) | **~55-60%** |
| Perplexity T2 (GPT-2) | O(W) | ~5% |
| Todos los profilers regex/estadísticos juntos | O(W) + O(min(n,120)²) semantic | ~3-5% |
| Fusión + reporte HTML | O(W) | <1% |

### C.2 🔴 P-01 (hallazgo nuevo) — Triple pasada neural redundante
`full_analysis` paga ≈3× los tokens del documento por el ensemble:
1. `classify_text_aggregate` → `analyze_fast`: clasifica CADA párrafo (pasada completa).
2. `HybridSegmentAnalyzer.analyze`: construye SUS PROPIAS ventanas de ~300 palabras con
   50% de solape (`hybrid_segment_detector.py:8-9,186-197`) y las re-clasifica con los
   mismos modelos — ≈2× tokens adicionales. Sin caché compartida (el `_FAST_CACHE` opera a
   nivel de documento en analyze_fast, no de segmento).

**Mitigación (M-22):** alimentar el heatmap del hybrid con los scores por párrafo que
`analyze_fast` YA produjo (misma granularidad que `paragraph_scores`), dejando las
ventanas solapadas como refinamiento opcional (`HYBRID_WINDOWS=1`) para documentos cortos.
Reducción esperada del costo dominante: **≈50-60% del CPU total de full_analysis**, cero
pérdida de funcionalidad (los breakpoints se calculan igual sobre scores por párrafo).
Riesgo: medio (suavizado por solape se pierde en modo rápido — mantener flag). Es la
optimización #1 del sistema con diferencia.

### C.3 P-02/P-03 — Menores
- **P-02:** parsing duplicado — stylometric, reasoning, hallucination, discourse, semantic
  y authorship re-tokenizan y re-segmentan oraciones cada uno con su propio regex. Un
  `ParsedDocument` compartido (sentences, words, paragraphs una sola vez) ahorra ~50-70%
  del costo de la etapa de profilers (que es ~3-5% del total — ROI real modesto; hacerlo
  al tocar esos módulos, no como sprint dedicado).
- **P-03:** el mismo refactor elimina las features duplicadas del Anexo B.2.
- Ya óptimos (no tocar): batching por longitud, torch threads=1, CoW preload, caché
  namespaced, gc condicional en Celery, strip de base64.

### C.4 Harness de medición (prerrequisito de cualquier claim)
`py-spy record` sobre un worker + corpus sintético de 10/100/1000 usuarios simulados
(locust contra /analyze_document_async), midiendo: p50/p99, CPU·s/doc, RSS pico,
hit-ratio Redis (`INFO stats`), throughput. KPIs base a registrar ANTES del Sprint de
M-22 para poder demostrar la mejora. Escalabilidad: primer cuello esperado = CPU de
inferencia (P-01); segundo = hilos zombis bajo timeouts (N-11).

---

## ANEXO D — Falsos Positivos y Falsos Negativos (Partes 10-11)

### D.1 Matriz de causas FP → mitigación

| Causa (población humana) | Señales que disparan | Mitigación | Estado |
|---|---|---|---|
| Académico/paper formal | dsc_uniformity, rsn_cot, hal_vagueness | M-1/M-2 clamp asimétrico + familias | Sprint 1 |
| Legal/normativo | dsc + repetición terminológica (ppl proxy) | M-1 + quitar ppl del ajuste | Sprint 1 |
| ESL / traducción humana | sintaxis regular → neural + Tier-1 | R-2 gate idioma (M-5) + banda Incierto (M-3) | Sprint 2 |
| Técnico con terminología repetida | ppl proxy "predecible" | quitar ppl_low_ratio del ajuste | Sprint 1 |
| Prosa editada/pulida | burstiness baja | M-4 términos pro-humano con corroboración | Sprint 2 |
| Texto corto | varianza alta en todo | R-1 abstención < 200 palabras | Sprint 2 |
| Citas densas | uniformidad + refs no indexadas | R-4 + fix not_found→not_indexed (M-7) | Sprint 3 |
| Papers nuevos/preprints citados | `not_found→fabricated` | M-7 (≥2 bases + not_indexed separado) | Sprint 3 |

Principio rector implementado por M-3: ante evidencia débil/contradictoria → "Incierto",
nunca "probablemente IA". Ranking de señales por riesgo FP: 1º dsc_uniformity,
2º sem flip-negación, 3º ppl proxy, 4º rsn_cot, 5º hal_vagueness.

### D.2 Matriz de causas FN → evidencias persistentes

| Causa | Qué señal sobrevive | Acción |
|---|---|---|
| LLM 2024-2026 (OOD) | ref fabricadas, contradicciones internas, uniformidad discursiva, disagreement alto | M-7 ON async; M-3 convierte "Human confiado" en "Incierto" vía disagreement; fondo: doc A reentrenamiento |
| Parafraseo/humanizador | estructura discursiva (sobrevive reescritura), contradicciones, refs | pesos de familia estructural intactos; doc E valida |
| Traducción de IA | refs fabricadas, uniformidad estructural | ídem + R-2 no debe ANULAR señales no-léxicas |
| Híbrido humano+IA | breakpoints, author outlier_ratio, longest_ai_run | ya cubierto (hybrid + authorship); reportar distribución, no % único (M-17) |
| Edición gramatical posterior | refs, contradicciones numéricas | M-7 |
| Multi-modelo en un doc | disagreement por segmento | exponer disagreement por párrafo (ya se calcula en `_classify_batch_from_ids`) — gratis |

Ranking de evidencias más resistentes a evasión (Fase 4 Parte 11): 1º referencias
fabricadas/quiméricas (invariante a reescritura), 2º contradicción numérica interna,
3º uniformidad discursiva, 4º outliers de autoría, 5º neural (solo in-distribution),
último: perplexity proxy (se destruye con cualquier edición).

---

## ANEXO E — Robustez Adversarial (Parte 12)

Nota de alcance: análisis defensivo; no se documentan recetas de evasión. Suite de
validación controlada: `docs/sota/E_SUITE_ADVERSARIAL.md` (DIPPER/parafraseo, métrica
TPR@FPR=1%) — este anexo no la duplica.

### E.1 Matriz de persistencia de señales

| Señal | Parafraseo | Traducción | Humanización | Mezcla H+IA | Estabilidad global |
|---|---|---|---|---|---|
| ref_fabricated/chimeric | ✔ persiste | ✔ | ✔ | ✔ (por sección) | **ALTA** |
| sem contradicción numérica | ✔ | ✔ | ✔ mayormente | ✔ | ALTA |
| dsc_uniformity | ✔ (diseño: sobrevive reescritura léxica) | ~ | ~ | ✔ localiza | MEDIA-ALTA |
| author outlier_ratio | ~ | ✘ | ~ | ✔ (su propósito) | MEDIA |
| neural p | ✘ degrada | ✘ | ✘ | ~ por segmento | BAJA (OOD) |
| ppl_* | ✘ | ✘ | ✘ | ✘ | MUY BAJA |
| rsn regex | ~ | ✘ (léxico en) | ✘ | ~ | BAJA-MEDIA |
| watermark | ✘ | ✘ | ✘ | ✘ | NULA (OFF, correcto) |

### E.2 Conclusión adversarial
La arquitectura de fusión es la defensa correcta: el atacante debe derrotar
simultáneamente familias independientes (refs + estructura + coherencia + autoría), no un
solo clasificador. Requisitos para que eso sea verdad en la práctica: (1) M-7 — la señal
más persistente debe estar ACTIVA; (2) M-1/M-2 — sin ellos la fusión es manipulable en la
otra dirección (FP); (3) degradación honesta — cuando las señales frágiles mueren tras
edición, el sistema debe bajar a "Incierto" (M-3), no fingir confianza. Nivel de
degradación esperado por reescritura progresiva (Fase 6): medible solo con doc E;
Información insuficiente para cifras hoy.

---

## ANEXO F — Plan Maestro Consolidado (Parte 14)

### F.1 Backlog único priorizado (consolida §8 + anexos; sin duplicados)

| ID | Mejora | P | Depende de | Sprint | Esfuerzo | Validación |
|---|---|---|---|---|---|---|
| M-1 | Clamp asimétrico + regla corroboración | **P0** | — | 1 | horas | test volteo humano-formal |
| M-2 | hyb_ai_ratio fuera del ajuste | **P0** | — | 1 | horas | test contribuciones |
| M-3 | Banda Incierto (P±0.10, disagreement≥12) | **P0** | — | 1 | horas | test franja gris |
| M-21 | Exponer `contributions` en FusionResult + HTML (N-15) | **P1** | — | 1 | horas | snapshot HTML |
| M-6 | sem: solo numérica/NLI al ajuste | P1 | — | 1 | horas | unit pares |
| M-14 | MODEL_VERSION=2026.07 | P1 | M-1..M-3 | 1 | minutos | clave caché |
| M-4 | Términos pro-humano acotados | P1 | M-1 | 2 | 1 día | corpus sintético humano |
| M-5 | Gate de idioma + renormalización | P1 | — | 2 | 1-2 días | textos es/en |
| M-8 | degraded_signals visibles | P2 | — | 2 | horas | fallo simulado |
| M-10 | Floats por segmento | P2 | — | 2 | horas | comparación agregados |
| M-16 | Smoke test shapes checkpoints (C-07) | P2 | — | 2 | horas | CI |
| R-1..R-7 | Sistema de reglas (corto/idioma/código/citas/contradicción/OCR/seeds) | P1-P2 | M-3 | 2-3 | 2-3 días | QA por tipo doc |
| M-7 | reference_check ON async + not_indexed | **P1** | fix validator | 3 | 3-5 días | refs reales vs preprints |
| M-9 | Puente plagio↔IA (zonas citadas fuera del scoring) | P2 | R-4 | 3 | 2 días | docs con citas |
| M-22 | **Dedup pasada neural del hybrid (P-01)** | **P1** | harness C.4 | 3 | 2-3 días | equivalencia heatmap + CPU medido |
| C.4 | Harness profiling + KPIs base | P1 | — | 3 | 1 día | py-spy/locust |
| M-11 | Cancelación cooperativa hilos | P2 | — | 4 | 2-3 días | timeout bajo carga |
| M-18 | Léxicos es/fr/pt Tier-1 | P2 | M-5 | 4 | 3-5 días | corpus es |
| M-12/M-13/M-15/M-17 | Normalizar schema, singleton fusión, contrato status, UX Hybrid | P3 | — | 4 | 1-2 días | tests existentes |
| M-19 | **Corpus + FusionClassifier.fit + TemperatureScaler + ECE** | **P0 (mediano plazo)** | doc A; M-22 antes de medir | 5-6 | 2-4 sem | ECE, TPR@FPR=1%, IC95% bootstrap |
| M-20 | Binoculars CPU (doc B) | P1 (mediano) | M-19 corpus para umbral | 6 | 1-2 sem | TPR@FPR vs proxy |
| P-02/P-03 | ParsedDocument compartido | P4 | oportunista | — | al tocar módulos | tests unitarios |

**Quick wins** (≤1 día, alto impacto, cero riesgo arquitectónico): M-1, M-2, M-3, M-21,
M-6, M-14, M-8, M-10, M-16 — todos en `fusion.py`/`forensic_reports.py`/`routes.py`,
ninguno toca el modelo ni las APIs.

### F.2 Grafo de dependencias (crítico)

```
M-1/M-2/M-3 (seguridad de la fusión)
   └─→ M-4 (pro-humano; sin M-1 subiría FN)
   └─→ M-14 (bump versión al desplegar)
C.4 (harness) ─→ M-22 (dedup neural: exige medir antes/después)
fix not_found ─→ M-7 ─→ (mejor feature disponible) ─→ M-19 re-entrena con todo
M-19 (corpus) ─→ calibración real ─→ M-20 umbral Binoculars ─→ doc E validación final
```

### F.3 KPIs y criterios de éxito (Fases 14-15)

| KPI | Base | Objetivo | Cómo |
|---|---|---|---|
| TPR@FPR=1% | sin medir | medir en Sprint 5; toda mejora posterior no puede bajarlo | doc A/E |
| ECE | sin medir | < 0.05 post-calibración | calibration.py |
| % veredictos "Incierto" | 0% (no existe) | 5-15% esperado tras M-3 (monitorear: >30% = señal degradada) | logs |
| CPU·s/doc full_analysis | sin medir | −50% tras M-22 | harness C.4 |
| p99 latencia sync | sin medir | sin regresión en cada sprint | harness C.4 |
| RSS pico worker | sin medir | sin aumento | harness C.4 |
| Volteos por heurística | posible hoy | 0 textos humano-formal del set de control volteados | test Sprint 1 |

Regla de reversión: cada M-x se despliega tras `MODEL_VERSION` bump; rollback = revertir
commit + bump de nuevo (la caché namespaced garantiza consistencia).

### F.4 Cierre
Con las Partes 7-12 y 14 incorporadas, el diagnóstico no cambia — se refuerza: el sistema
tiene la ARQUITECTURA correcta (fusión log-odds acotada, evidencias parcialmente
independientes, calibración lista para datos) y tres defectos de PARAMETRIZACIÓN
corregibles en horas (N-01, N-02, N-03), un desperdicio de explicabilidad ya calculada
(N-15), y un desperdicio de CPU dominante (P-01). El orden óptimo es: asegurar la fusión
(Sprint 1) → reglas e idioma (Sprint 2) → referencias + dedup neural (Sprint 3) →
robustez operativa (Sprint 4) → corpus, calibración y validación adversarial (Sprints 5-6),
que es donde por fin aparecerán números reales de Precision/Recall/F1/ECE — antes de eso,
cualquier cifra sería ficción, y este comité no emite ficción.
