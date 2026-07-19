# Auditoría técnica — Microservicio de detección y sus plugins

**Alcance:** capa de plugins (`app/plugins/*`), infraestructura (`plugin_registry`,
`plugin_orchestrator`, `base`, `routes`, `tasks`) y motores (`app/engine/*`).
**Objetivo:** cuellos de botella, bugs, deuda arquitectónica y optimizaciones, sin
romper contratos públicos.
**Baseline de corrección:** ya aplicado el fix de `[CLS]`/`[SEP]` en `analyze_fast`
(commit previo) — sin él las percentages salían invertidas en `/analyze`.

---

## 0. Resumen ejecutivo (priorizado por impacto)

| # | Severidad | Área | Hallazgo | Impacto |
|---|-----------|------|----------|---------|
| C1 | **Alta** | Arquitectura/Memoria | Doble carga de motores: cada plugin instancia su engine Y el orchestrator instancia otra copia | RSS y tiempo de arranque ~2× en los motores no-singleton |
| C2 | **Alta** | Rendimiento | Inferencia del ensemble ModernBERT duplicada por request entre `ai_detection`, `segment_analysis` y `full_analysis`; el caché solo cubre `analyze_fast` | Latencia multiplicada cuando se piden varios plugins |
| C3 | **Alta** | Rendimiento/Algoritmo | `HybridSegmentAnalyzer` clasifica **ventana por ventana secuencial**; `classify_batch` ya soporta lotes | Latencia O(N) evitable en documentos largos |
| C4 | **Media** | Bug funcional | En `fusion.py` las 4 features de *hallucination* son siempre 0: el orchestrator nunca las mete en `additional` | La fusión ignora una señal que dice usar |
| C5 | **Media** | Rendimiento/Red | `reference_validator` hace red **secuencial** con throttle global de 800 ms y cascada de 3 APIs por cita | Con pocas citas ya supera el timeout de 30 s |
| C6 | **Media** | Concurrencia | `ThreadPoolExecutor` (8 workers) corre plugins ML pesados en paralelo sobre CPU con `torch_num_threads=1` | Oversuscripción y thrashing en CPU |
| C7 | **Media** | Memoria | Dos cargas de `spaCy en_core_web_sm` (stylometric + hallucination) | ~2× memoria del pipeline spaCy |
| C8 | **Baja** | SLA | `registry.run` aplica timeout **por plugin**, no un deadline global | Peor caso ~N×timeout por request |
| C9 | **Baja** | Calidad | Código muerto/duplicado (`analyze_long_document*` ×2, God object `forensic_reports.py` 2590 líneas) | Mantenibilidad |

---

## 1. Infraestructura

### 1.1 `plugin_registry.py`
**Función:** descubrimiento automático + ejecución concurrente de plugins.
**Fortalezas:** auto-discovery limpio, `health()`/`is_core()` para readiness honesto,
manejo de errores por plugin, timeouts.

**Debilidades / hallazgos:**
- **C6 — Oversuscripción CPU.** `max_workers=min(len(valid),8)`. Los plugins ML
  (ModernBERT, GPT-2, spaCy) son CPU-bound; corriendo 8 en paralelo con
  `TORCH_NUM_THREADS=1` cada uno, compiten por los mismos cores y por los mismos
  objetos-modelo. El comentario "liberan GIL" es cierto para el forward de torch,
  pero no elimina la contención de CPU/caché. **Propuesta:** dos pools — uno pequeño
  (`min(cores, 2-4)`) para plugins ML, otro mayor para plugins puros-Python; o
  serializar los que comparten los mismos modelos.
- **C8 — Timeout por plugin, no global.** En `run`, `future.result(timeout=timeout)`
  se aplica a cada future por separado; un request con N plugins lentos puede tardar
  hasta ~N×timeout. **Propuesta:** calcular un `deadline = t0 + timeout` compartido y
  pasar `timeout=deadline-now()` a cada `result()`.
- **C9 — DRY.** `run` y `run_stream` duplican el bloque submit+timing+empaquetado.
  Extraer un helper `_submit(valid, executor)`.

### 1.2 `plugin_orchestrator.py`
**Función:** coordina el pipeline forense completo y ensambla `additional_analyses`.
**Fortalezas:** capa fina sin lógica de negocio, flags de activación, degradación por
plugin, fusión tardía integrada.

**Hallazgos:**
- **C1 — Doble carga (raíz).** `_init_plugins()` instancia su **propio** juego de
  motores (`StylometricProfiler`, `ReasoningProfiler`, `PerplexityProfiler`,
  `HybridSegmentAnalyzer`, `ReferenceValidator`, `HallucinationProfiler`,
  `DiscourseAnalyzer`, `SemanticConsistencyAnalyzer`, `WatermarkDecoder`). Los mismos
  motores ya fueron instanciados por sus plugins adapter al importarse. Los pesos torch
  grandes se comparten (ModernBERT vía módulo, GPT-2 vía singleton `_GPT2Engine`), pero
  los *wrappers*, buffers y estados sí se duplican. **Propuesta:** un contenedor de
  dependencias único (o reusar las instancias del registry) inyectado tanto a los
  plugins como al orchestrator — Single Source of Truth de motores.
- **C4 — Señal hallucination muerta en fusion.** El `HallucinationProfiler` solo se
  pasa al `ForensicReportGenerator.__init__`; **nunca** se hace
  `additional["hallucination"] = ...`. Sin embargo `fusion.FusionFeatureBuilder.build`
  lee `aa.get("hallucination")` para 4 de sus 29 features (`hal_overall`,
  `hal_semantic_incoherence`, `hal_vagueness`, `hal_repetition`, peso heurístico 0.6).
  Resultado: siempre 0. **Propuesta:** en `run_with_result`, ejecutar
  `HallucinationProfiler.compute_stats()` + `Classifier.classify()` y volcar el dict en
  `additional["hallucination"]` (con `category_scores`), como hacen los demás.
- **Acoplamiento de importación.** Usa imports "desnudos" (`from stylometric_profiler
  import ...`). Funciona por la inyección de `sys.path` en `app/engine/__init__.py`,
  pero es frágil: cualquier módulo top-level homónimo colisiona. Documentado como
  intencional (DT-08); mantener, pero aislar en un subpaquete evitaría el riesgo.

### 1.3 `routes.py` / `tasks.py`
**Fortalezas:** validación de entrada, límite de tamaño, caché namespaced por
`MODEL_VERSION`, reuso de segmentos en el task para evitar doble inferencia (bien),
strip de base64 antes de Redis, `gc.collect()` condicionado a texto grande.
**Hallazgos menores:**
- `analyze_document` recalcula `analyze_fast(text)` si `ai_detection` no trae segments;
  correcto, pero conviene compartir el mismo caché que el task.
- `_merge_segment_results` es el punto DRY compartido entre sync y task — bien.

---

## 2. Plugins (capa adapter)

Patrón común: carga perezosa del motor a nivel módulo, `health()` = flag de
disponibilidad, `analyze()` delega. Contratos de E/S estables. Observaciones por plugin:

### 2.1 `ai_detection` (core)
- Delega en `analyze_fast` (parágrafo-aware, caché 5 min). Correcto tras el fix de
  tokens especiales. `confidence = max(human,ai)` sobre enteros redondeados; consistente.
- **C2:** su resultado ya trae `segments`; el task lo reusa, pero `full_analysis` y
  `segment_analysis` recalculan por su cuenta. Unificar.

### 2.2 `segment_analysis`
- **C3 (crítico de rendimiento).** `HybridSegmentAnalyzer` → `WindowClassifier.
  classify_windows` itera y llama `classify_fn` (→ `classify_segment` → `classify_batch
  ([text])`) **una ventana a la vez**. Con overlap 50% y ventanas de 300 palabras, un
  documento de 3 000 palabras genera ~19 ventanas = 19 forward passes de 3 modelos.
  `classify_batch` acepta listas: **una sola llamada por lote** daría el mismo resultado.
  **Propuesta:** construir todos los `window_text`, llamar `classify_batch(all_windows)`
  una vez (con sub-batching interno como en `_classify_batch_from_ids`).

### 2.3 `perplexity_check`
- `PerplexityProfiler` (Tier1 n-gram CPU; Tier2 GPT-2 vía singleton — OK).
- **C1:** instancia duplicada con la del orchestrator; GPT-2 se comparte por singleton,
  pero los diccionarios n-gram y estado sí se duplican.

### 2.4 `reasoning_check`, `stylometric_analysis`, `hallucination_check`,
`author_signature`, `discourse_structure`, `semantic_consistency`
- Puros-Python o spaCy. Correctos y baratos individualmente.
- **C7:** `stylometric_profiler` y `hallucination_profile` cargan cada uno su
  `spacy.load("en_core_web_sm")` a nivel módulo → 2 pipelines spaCy en memoria.
  **Propuesta:** `app/engine/_nlp.py` con un `get_nlp()` singleton importado por ambos.
- `StylometricProfiler` se instancia 3× (stylometric_analysis, author_signature,
  orchestrator) pero comparte `_NLP` de módulo y **no** carga SentenceTransformer por
  defecto (solo aparece en docstring del Protocol) — coste real bajo. No sobredimensionar.

### 2.5 `citation_check`
- **C5.** `reference_validator._APIClient._throttle` usa un `time.sleep` **global** de
  800 ms y una cascada CrossRef→S2→OpenAlex **por cada** referencia, todo secuencial y
  bloqueante. Para K citas, hasta ~3K peticiones seriales; con timeout de plugin de 30 s
  se agota con muy pocas citas. Tiene circuit breaker y guard SSRF (bien), pero:
  **Propuesta:** (a) throttle **por host** en vez de global (las 3 APIs son hosts
  distintos → se pueden solapar); (b) paralelizar referencias con un pool acotado
  respetando el rate-limit por host; (c) caché LRU/persistente de DOIs/títulos ya
  resueltos entre requests.

### 2.6 `watermark_detection`
- Carga GPT-2 (AutoModelForCausalLM). Desactivado por defecto. Dos rutas de carga
  perezosa (modelo de entropía y tokenizer) — **verificar** que no cargue dos veces el
  mismo checkpoint cuando ambas se activan.

### 2.7 `full_analysis`
- Orquesta TODO el pipeline forense por cada llamada, sin granularidad: aunque el caller
  solo quiera la detección, se ejecutan stylometric+reasoning+perplexity+hybrid+
  (reference)+fusión+report HTML. **Propuesta:** parámetro para seleccionar secciones, o
  endpoints diferenciados. Ya limpia reports viejos (`_cleanup_old_reports`) y usa
  `NamedTemporaryFile` único (bien vs. sobrescritura concurrente).

### 2.8 `forensic_report`, `zone_classifier`
- No auditados en profundidad; `forensic_reports.py` (2590 líneas) es un **God object**
  (genera HTML, JSON, gráficos, clasificadores embebidos). **Propuesta:** separar
  render (HTML/JSON) de la lógica de clasificación (`ReasoningRiskClassifier` vive aquí y
  lo importan otros módulos → dependencia invertida rara).

---

## 3. Cuellos de botella (con causa raíz, impacto y prioridad)

| ID | Descripción | Causa raíz | Impacto | Prioridad |
|----|-------------|-----------|---------|-----------|
| C2 | Ensemble ModernBERT corre varias veces por request | Caché solo en `analyze_fast`; `classify_segment`/híbrido sin caché compartido | Latencia ×2–×4 con multi-plugin | Alta |
| C3 | Híbrido clasifica ventana a ventana | Bucle secuencial en vez de `classify_batch(lote)` | +O(N) forward passes | Alta |
| C5 | Validación de citas lenta | `sleep` global 800 ms + cascada 3 APIs secuencial | Timeout con pocas citas | Media |
| C6 | Contención CPU con 8 workers ML | Pool grande + BLAS 1-hilo + modelos compartidos | p99 latencia, thrashing | Media |
| C1 | Doble instanciación de motores | Plugins y orchestrator crean copias separadas | RSS + arranque | Media |

---

## 4. Recomendaciones priorizadas (impacto/esfuerzo)

**Impacto alto / esfuerzo bajo (hacer primero):**
1. **C3** — Batchear las ventanas del híbrido en una sola llamada a `classify_batch`.
   Cambio local, resultado idéntico, gran ganancia de latencia.
2. **C4** — Poblar `additional["hallucination"]` en el orchestrator para que la fusión
   use la señal que ya calcula.
3. **C7** — Singleton único de spaCy compartido por stylometric y hallucination.

**Impacto alto / esfuerzo medio:**
4. **C2** — Caché de inferencia unificado (keyed por texto normalizado + versión de
   modelo) reutilizado por `analyze_fast`, `classify_segment` y el híbrido; que
   `segment_analysis` y `full_analysis` compartan un único resultado híbrido por request.
5. **C1** — Contenedor de dependencias: instanciar cada motor UNA vez e inyectarlo a
   plugins y orchestrator (elimina la doble carga).

**Impacto medio / esfuerzo medio:**
6. **C5** — Throttle por host + paralelismo acotado + caché de citas en
   `reference_validator`.
7. **C6** — Separar pools ML vs. puro-Python en el registry, dimensionados por cores.
8. **C8** — Deadline global además del timeout por plugin.

**Mantenibilidad:**
9. **C9** — Borrar `analyze_long_document` y `analyze_long_documentsd_` (deprecated,
   ~200 líneas muertas); descomponer `forensic_reports.py`.

---

## 5. Compatibilidad

Todas las propuestas preservan las interfaces públicas (`name/description/analyze`,
esquemas de respuesta de `/analyze` y `/analyze_document`) y el comportamiento
funcional. C3, C4 y C7 son las de menor riesgo de regresión y se recomiendan como
primer lote, cada una con verificación end-to-end contra el resultado del código
original (`app2.py`) sobre textos de control (uno humano, uno AI) antes de mergear.

> Nota: este entorno no tiene los pesos de los modelos ni `transformers` con red a
> HuggingFace, por lo que las mejoras se entregan como recomendaciones verificadas por
> lectura de código. La validación numérica debe hacerse en el servidor con los modelos
> presentes.
