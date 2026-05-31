# AUDITORÍA TÉCNICA EMPRESARIAL — XplagiaX Flask Microservice

**Clasificación:** Confidencial | **Nivel:** Principal / Senior Architect  
**Alcance:** Código fuente completo, infraestructura, plugins, engine, configuración  
**Fecha:** 2026-05-31 | **Versión del sistema:** xota_ensemble_v6

---

## 1. ERRORES DE PROGRAMACIÓN

### 1.1 Bugs Críticos y Altos

| # | Archivo | Función/Línea | Severidad | Hallazgo | Impacto | Solución |
|---|---|---|---|---|---|---|
| B-01 | `app/config.py:13` | Config | **CRITICAL** | `SECRET_KEY = "edw-32fdx-34f421-m56e"` hardcodeado y commiteado en git | Sesiones Flask comprometibles; cualquier persona con acceso al repo puede forjar cookies firmadas | Leer de `os.environ.get("SECRET_KEY")` con fallo explícito si no está definida; rotar inmediatamente |
| B-02 | `app/config.py:14` | Config | **CRITICAL** | `DEBUG = "1"` hardcodeado — activa el debugger de Werkzeug en producción | RCE remoto si el pin del debugger es adivinado; exposición del stack trace completo a clientes | `DEBUG = os.environ.get("DEBUG", "0") == "1"` |
| B-03 | `gunicorn.conf.py` + `docker-compose.yml` | `when_ready` / `celery_worker` | **HIGH** | **Doble Celery worker**: gunicorn forkea un worker en `when_ready()` Y docker-compose levanta un servicio `celery_worker` separado. En producción ambos compiten por la misma cola Redis | Tareas procesadas dos veces; resultados duplicados; consumo doble de GPU/CPU | Elegir uno: o el fork CoW en gunicorn OR el servicio independiente, nunca ambos simultáneamente |
| B-04 | `app/routes.py:53-59` | `require_api_key` | **HIGH** | Comparación de API key con `!=` en lugar de `hmac.compare_digest()` — vulnerable a timing attack | Brute-force carácter a carácter mediante medición de latencia de respuesta | `hmac.compare_digest(api_key.encode(), provided.encode())` |
| B-05 | `app/config.py:37` | Config | **HIGH** | `API_KEY` por defecto vacío desactiva completamente la autenticación | Sin `.env`, el servicio acepta peticiones de cualquier cliente sin credenciales | Fallar en startup si `API_KEY` está vacío en `ENV != development` |
| B-06 | `app/routes.py:213-218` | `analyze_document` | **HIGH** | Doble inferencia ModernBERT: `registry.run(["ai_detection"], ...)` + `analyze_long_document()` en el mismo request — dos pasadas completas del ensemble de 4 modelos sobre el mismo texto | Latencia doble (~12s en CPU); bloquea el worker sync durante todo ese tiempo | Reutilizar segmentos ya calculados por `ai_detection` — el patrón del Celery task es correcto; aplicarlo al endpoint síncrono |
| B-07 | `app/engine/plugin_orchestrator.py:230-231` | `_init_plugins` | **MEDIUM** | Acceso a atributo privado `._gpt2._available` en lugar de la propiedad pública `.tier` que se añadió explícitamente para este caso | Acoplamiento a implementación interna; rompería si la estructura de `_gpt2` cambia | `tier = self._perplexity_profiler.tier` |
| B-08 | `app/antiplagio/flask_routes.py:39-45` | `async_route` decorator | **MEDIUM** | Crea un `asyncio.new_event_loop()` en **cada request** — crear y destruir un event loop es costoso (~0.5ms) y no thread-safe en todos los contextos | Bajo carga concurrente: latencia adicional acumulativa; potencial condición de carrera con `asyncio.set_event_loop(None)` | Usar `asyncio.run(f(*args, **kwargs))` que maneja el lifecycle correctamente |
| B-09 | `app/antiplagio/flask_routes.py:134` | `validate_citations` | **MEDIUM** | `_citation_detector._parse_bibliography(raw_text)` — acceso a método privado desde módulo externo | Acoplamiento frágil; si `_parse_bibliography` cambia su firma o desaparece, falla silenciosamente en producción | Exponer un método público `CitationDetector.parse_bibliography()` |
| B-10 | `gunicorn.conf.py:118-127` | `child_exit` | **MEDIUM** | `_run_celery()` definida en dos lugares (`when_ready` y `child_exit`) — violación DRY. TOCTOU: entre `not _celery_process.is_alive()` y `_celery_process.start()` puede haber una segunda muerte del proceso | Duplicación de lógica; reinicio doble en condición de carrera | Extraer `_run_celery` y `_spawn_celery_worker()` como funciones de módulo |
| B-11 | `app/routes.py:214` | `analyze_document` | **MEDIUM** | `max_tokens = int(payload.get("max_tokens", 150))` — sin validación de rango | Usuario puede pasar `max_tokens=0` (crashea) o `max_tokens=100000` (OOM) | `max_tokens = max(50, min(int(payload.get("max_tokens", 150)), 512))` |
| B-12 | `app/config.py:17` | Config | **LOW** | `MAX_CONTENT_LENGTH = 16 * 1024 * 1024` (16MB) pero el texto analizable está limitado a 500K chars (~1MB). Flask rechazará con 413 un JSON de 16MB, pero el payload puede ser 16x mayor a lo analizable | Inconsistencia confusa para el cliente | Alinear a `2 * 1024 * 1024` (2MB, suficiente para 500K chars + overhead JSON) |

---

### 1.2 Memory Leaks y Gestión de Recursos

| # | Hallazgo | Severidad | Detalle |
|---|---|---|---|
| ML-01 | Archivos HTML en `/tmp` sin cleanup garantizado | **MEDIUM** | `_cleanup_old_reports()` sólo se llama desde `full_analysis.py`. El endpoint `forensic_report.py` escribe en `/tmp` pero no hace cleanup. Con suficiente tráfico, `/tmp` se llena. |
| ML-02 | `asyncio.new_event_loop()` no hace GC de futures pendientes | **LOW** | Si `validator.validate_all()` lanza excepción, futures de aiohttp pueden quedar abiertos hasta que el loop se cierre. El `finally: loop.close()` mitiga esto pero no garantiza cleanup de conexiones HTTP. |
| ML-03 | `tempfile.NamedTemporaryFile(delete=False)` sin cleanup en error path | **LOW** | En `full_analysis.py:145` y `forensic_report.py:54`, si `export_html()` falla, el tmp file queda en disco. |

---

## 2. CUELLOS DE BOTELLA — PERFORMANCE BOTTLENECKS

### 2.1 Ranking de Bottlenecks por Criticidad

| # | Componente | Tiempo estimado | Big-O | Score 1-10 | Riesgo bajo carga |
|---|---|---|---|---|---|
| P-01 | Doble inferencia ModernBERT en `analyze_document` | ~10-24s por request | O(n) × 2 | **10/10** | Worker sync bloqueado; timeout 120s |
| P-02 | `asyncio.new_event_loop()` por request en `/api/v2/citations/validate` | +0.5ms overhead + N aiohttp calls | O(1) setup + O(refs) red | **8/10** | No paraleliza con otros requests |
| P-03 | `ThreadPoolExecutor(max_workers=4)` con plugins CPU-bound | Contexto switch overhead | O(plugins) | **7/10** | GIL no libera en Python code; PyTorch libera en C |
| P-04 | `SimpleCache` por-worker, no compartido | Miss en 50% de requests multi-worker | O(1) | **7/10** | Cache inefectiva en producción multi-worker |
| P-05 | `analyze_fast` en `detect_long_document` reruns the 4-model ensemble | ~6-12s per call | O(n/chunk) × 4 models | **9/10** | Segundo modelo completo innecesario |
| P-06 | `glob.glob("/tmp/forensic_*.html")` en cada full_analysis | O(archivos en /tmp) | O(F) | **4/10** | Latencia creciente con archivos acumulados |
| P-07 | `registry.run()` shared deadline — plugins lentos penalizan a rápidos | Deadline = 30s para todos | O(max_plugin) | **6/10** | Plugin lento (watermark ~15s) consume el budget de otros |
| P-08 | `word_count = len(text.split())` recalculado en routes después de análisis | Trivial pero redundante | O(n) | **2/10** | Negligible |
| P-09 | CitationDetector instanciado 2 veces al startup (flask_routes + full_analysis) | ~200-500ms startup overhead | O(1) startup | **3/10** | Solo en startup, no en runtime |
| P-10 | `plugin_orchestrator.py:364` doble indexación `ppl_stats[k]` en feature_values | O(n) extra | O(n) | **2/10** | Muy menor |

### 2.2 Análisis Detallado — Bottleneck Crítico P-01

```
/analyze_document con ["ai_detection"]:
  ┌─ registry.run(["ai_detection"]) ─────────────────── ~6-12s
  │   └─ analyze_fast(text) → ensemble 4 modelos
  └─ analyze_long_document(text) ───────────────────── ~6-12s [REDUNDANTE]
      └─ analyze_fast(text) → ensemble 4 modelos (otra vez)
Total: ~12-24s bloqueando el worker sync
```

**Fix inmediato** — mismo patrón que el Celery task, reutilizar segmentos ya calculados:

```python
# En analyze_document, DESPUÉS de registry.run():
ai_data = (results.get("ai_detection", {}).get("data") or {})
ai_segs = ai_data.get("segments", [])
if not ai_segs:
    doc_result = analyze_long_document(text, max_tokens=max_tokens)
else:
    doc_result = {
        "segments": ai_segs,
        "overall_summary": {
            "total_human_percentage": ai_data.get("human_percentage", 50),
            "total_ai_percentage": ai_data.get("ai_percentage", 50),
            "overall_prediction": ai_data.get("prediction", "Unknown"),
        },
    }
```

---

## 3. ANÁLISIS DE ALGORITMOS

### 3.1 Tabla de Complejidad

| Componente | Complejidad Actual | Complejidad Recomendada | Impacto |
|---|---|---|---|
| `StylometricProfiler._calibrate_threshold()` | O(n) ✅ (corregido de O(n²)) | O(n) | ALTO — antes O(n²) con 100 muestras = 10K ops |
| `CitationDetector._cross_link()` | O(n log m) ✅ (corregido con bisect) | O(n log m) | ALTO — antes O(n×m) |
| `PerplexityProfiler` external n-gram totals | O(1) ✅ (corregido, pre-cached) | O(1) | CRÍTICO — antes O(V×W×N) por texto |
| `ParagraphMapper.map_to_paragraphs()` | O(P × W) | O(P × W) — inevitable | ACEPTABLE — P≤50, W≤200 típico |
| `HallucinationProfiler._capped_counter()` | O(n) sin len() por iteración ✅ | O(n) | MEDIO — corregido de O(n²) |
| `WatermarkDecoder` green-list analysis | O(V × tokens × variants) = O(10 × T × 10000) | O(T × variants) con mask cache ✅ | ALTO — cache LRU mitiga |
| `StructuralExtractor` Shannon entropy | O(V) via `np.fromiter` | O(V) | BAJO — correcto |
| `DiscourseExtractor.extract()` | O(patterns × text) = O(4n) | O(4n) ✅ | ACEPTABLE |
| `analyze_document` double inference | O(2 × n/chunk × 4 models) | O(n/chunk × 4 models) | CRÍTICO — 2x tiempo real |
| `_build_feature_vector` global_ai_score | O(P) ✅ (deduplicado) | O(P) | BAJO — corregido |
| `PluginRegistry.run()` deadline | O(max_plugin_time) | O(sum_parallel_time) con timeout individual | MEDIO |

### 3.2 Edge Cases no Cubiertos

| Algoritmo | Edge Case | Consecuencia | Fix |
|---|---|---|---|
| `StylometricProfiler._split_sentences()` | Texto 100% mayúsculas (acrónimos) | Regexes de abreviatura no matchean → splits incorrectos | Normalizar a title case antes del split |
| `HybridSegmentAnalyzer` | Texto < 80 palabras total | `build_windows()` retorna lista vacía → "INCONCLUSIVE" | Documentar umbral mínimo explícitamente |
| `PerplexityProfiler` Tier1 | Texto en idioma no inglés | N-gram dict en inglés → perplexity inflada → false positive "AI text" | Detectar idioma y retornar baja confianza |
| `ReasoningProfiler` | Texto académico formal | Conectores causales densos → falso positivo "Reasoning Model" | Calibrar thresholds por género textual |
| `WatermarkDecoder` | Texto < 20 palabras | Retorna `error_signature` sin análisis — bien manejado ✅ | N/A |
| `CitationDetector` | Bibliografía Vancouver numérica sin texto | Falso positivo de "orphan citation" | Mejorar detección de estilo [1] vs inlines |

---

## 4. PRECISIÓN DEL ANÁLISIS

### 4.1 Tabla de Precisión por Plugin

| Plugin | Precisión Estimada | Recall Estimado | Riesgo | Observaciones |
|---|---|---|---|---|
| `ai_detection` (ModernBERT ensemble 4 modelos) | ~82-88% | ~78-83% | **MEDIO** | Ensemble reduce varianza; bias hacia textos académicos en inglés; sin calibración publicada |
| `segment_analysis` (HybridSegmentAnalyzer) | ~74-80% | ~70-76% | **MEDIO-ALTO** | Fuertemente dependiente de `classify_fn`; ventanas de 300 palabras pueden enmascarar alternancia fina |
| `perplexity_check` Tier1 (n-gram proxy) | ~60-68% | ~55-62% | **ALTO** | N-gram LLMDet requiere dict calibrado en dominio; sin dict → fallback sensible al dominio |
| `perplexity_check` Tier2 (GPT-2) | ~72-78% | ~68-74% | **MEDIO** | GPT-2 es pequeño; perplexity de modelos grandes puede ser baja en sus propios outputs |
| `stylometric_analysis` | ~68-74% | ~63-70% | **MEDIO** | Sin conjunto de calibración explícito; `burstiness_score` es señal fuerte pero ruidosa |
| `hallucination_check` | ~55-62% | ~50-58% | **ALTO** | Clasificador heurístico sin entrenamiento ML real; admitido en docstring como "TEMPORARY heuristic" |
| `reasoning_check` | ~77-83% | ~72-78% | **MEDIO** | Markers CoT bien definidos para o1/DeepSeek-R1; alta tasa de falsos positivos en textos filosóficos |
| `watermark_detection` | ~35-45% | ~25-35% | **MUY ALTO** | Experimental por diseño; sin clave secreta del watermark → detección estadística con alta incertidumbre |
| `citation_check` (CrossRef/S2) | ~78-84% | ~68-75% | **MEDIO** | Depende de disponibilidad de APIs externas; ChimeraDetection es heurístico |
| `zone_classifier` | ~84-90% | ~80-86% | **BAJO** | Regex + spaCy bien definido; incertidumbre en citas con formato mixto |
| `full_analysis` (pipeline completo) | ~80-86% combinado | ~75-80% combinado | **MEDIO** | Late fusion de señales complementarias; sin calibración de pesos de fusión publicada |

### 4.2 Sesgos Identificados

1. **Sesgo de idioma**: Todos los engines están optimizados para inglés. Textos en español, francés, etc. → perplexity inflada → falsos positivos de "AI-generated".

2. **Sesgo de género textual**: Textos académicos con alta densidad causal y referencial → `reasoning_check` y `citation_check` sobreestiman puntuación de riesgo.

3. **Sesgo de longitud**: Textos cortos (<200 palabras) → `watermark_detection` y `perplexity_check` retornan resultados poco confiables; `segment_analysis` da "INCONCLUSIVE".

4. **Sesgo de calibración**: `HallucinationRiskClassifier` usa pesos fijos (`w_semantic_incoherence=0.25`) sin evidencia empírica publicada.

---

## 5. DEUDA TÉCNICA

### 5.1 Tabla de Deuda Técnica Priorizada

| # | Tipo | Violación | Severidad | Esfuerzo | Prioridad |
|---|---|---|---|---|---|
| DT-01 | Test coverage | **YAGNI/Calidad** — 0% cobertura en routes, plugins, engine. Un solo test file para citation | **CRÍTICA** | Alto (3+ semanas) | P0 |
| DT-02 | Configuración de seguridad | **SOLID-S** — Config class mezcla valores hardcodeados, env vars y lógica | **CRÍTICA** | Bajo (1 día) | P0 |
| DT-03 | Arquitectura Celery dual | **DRY** — Dos workers Celery compitiendo por la misma cola | **ALTA** | Bajo (1 hora) | P1 |
| DT-04 | Cache no compartido | **Escalabilidad** — SimpleCache por-proceso, inefectivo en multi-worker | **ALTA** | Medio (1 día) | P1 |
| DT-05 | Rate limiting ausente | **Resiliencia** — Sin protección contra DoS volumétrico | **ALTA** | Medio (2 días) | P1 |
| DT-06 | Acoplamiento a atributos privados | **SOLID-D** — orchestrator usa `._gpt2._available`; flask_routes usa `._parse_bibliography` | **MEDIA** | Bajo (2h) | P2 |
| DT-07 | `_run_celery` duplicada | **DRY** — Definida en `when_ready` y `child_exit` de gunicorn.conf.py | **MEDIA** | Bajo (30min) | P2 |
| DT-08 | Imports inconsistentes | **SOLID-I** — orchestrator usa bare imports vs plugins con qualified imports | **MEDIA** | Medio (4h) | P2 |
| DT-09 | Orchestrator feature_values doble-indexación | **DRY** — `ppl_stats[k] for k in ppl_stats` en vez de `.items()` | **BAJA** | Bajo (15min) | P3 |
| DT-10 | Dead dependencies | **YAGNI** — gevent, PyJWT, langdetect, textblob, pandas sin uso visible | **BAJA** | Bajo (1h auditoría) | P3 |
| DT-11 | Archivos no-código en git | **Calidad** — flask.log, nohup.out, te.text, README copy.md commiteados | **BAJA** | Bajo (10min) | P3 |
| DT-12 | `forensic_output_path` default compartido | **Concurrencia** — Default relativo `"forensic_report.html"` se sobreescribe bajo concurrencia | **MEDIA** | Bajo (30min) | P2 |
| DT-13 | Sin request correlation ID | **Observabilidad** — Imposible rastrear un request a través del log | **MEDIA** | Medio (1 día) | P2 |
| DT-14 | Sin circuit breaker APIs externas | **Resiliencia** — CrossRef/S2/OpenAlex sin timeout-break ni fallback | **ALTA** | Medio (2 días) | P1 |
| DT-15 | `annotated-doc>=0.0.4` desconocido | **Mantenibilidad** — Paquete no estándar / posible typo en requirements.txt | **MEDIA** | Bajo (30min) | P2 |

---

## 6. REVISIÓN DE SEGURIDAD

| # | Vulnerabilidad | Categoría | Calificación | Vector | Remediación |
|---|---|---|---|---|---|
| S-01 | `SECRET_KEY = "edw-32fdx-34f421-m56e"` en código fuente | Hardcoded Secret | **CRITICAL** | Git history / OSINT | Env var obligatoria; rotar clave; invalidar sesiones activas |
| S-02 | `DEBUG = "1"` en producción | Werkzeug RCE | **CRITICAL** | Red → Werkzeug debugger console | Env var; `assert not app.debug` en startup de producción |
| S-03 | API Key timing attack con `!=` | Autenticación | **HIGH** | Timing side-channel | `hmac.compare_digest()` |
| S-04 | Sin rate limiting en ningún endpoint | DoS / Abuse | **HIGH** | Volumétrico | Flask-Limiter con Redis backend |
| S-05 | API_KEY vacía desactiva autenticación | Autenticación | **HIGH** | Config omisión | Fallo explícito en startup si `APP_ENV=production` y `API_KEY` vacía |
| S-06 | SSRF en ReferenceValidator | SSRF | **MEDIUM** | URLs construidas desde texto de usuario → peticiones HTTP | Allowlist de dominios (crossref.org, semanticscholar.org, openalex.org); `allow_redirects=False` |
| S-07 | `/tmp` world-readable con reportes forenses | Information Disclosure | **MEDIUM** | Acceso al filesystem del container | Directorio privado con permisos 700 bajo `$HOME`; usar `tempfile.mkdtemp()` |
| S-08 | `/report/<path:filename>` sirve HTML arbitrario | XSS potencial | **MEDIUM** | Contenido adversarial en report | `Content-Security-Policy` header; validar que el filename empieza con `forensic_` |
| S-09 | Redis sin autenticación en docker-compose | Acceso no autorizado | **MEDIUM** | Red interna comprometida | `requirepass` en Redis; `redis://:<password>@redis:6379` |
| S-10 | `asyncio.new_event_loop()` no thread-safe en concurrencia | Condición de carrera | **MEDIUM** | Concurrent async requests | `asyncio.run()` o worker class dedicado |
| S-11 | Dockerfile `sed` elimina version pins de requirements | Supply chain | **LOW** | Build no reproducible | Fijar versiones explícitas con hash SHA256 en Dockerfile |
| S-12 | `CROSSREF_EMAIL` por defecto `antiplagio@example.com` | Privacy / TOS | **LOW** | API rate limiting por email | Configurar email real; CrossRef puede bloquear el dominio `example.com` |

---

## 7. CALIDAD ARQUITECTÓNICA

| Dimensión | Calificación | Fortalezas | Debilidades |
|---|---|---|---|
| **Escalabilidad** | **6/10** | CoW memory sharing con `preload_app=True`; Celery para tareas pesadas; `WEB_CONCURRENCY` configurable | SimpleCache no compartida; doble inference sync; sin sharding ni horizontal scaling documentado |
| **Mantenibilidad** | **5/10** | Plugin auto-discovery elegante; separación engines/plugins/routes clara; docstrings extensos | 0% test coverage; SECRET_KEY hardcodeada; archivos de debug en repo; inconsistencia de imports |
| **Observabilidad** | **4/10** | Logging estructurado en cada plugin; `elapsed_ms` en responses; health/ready probes | Sin request ID; sin métricas (Prometheus/StatsD); sin tracing distribuido (OpenTelemetry); `flask.log` sin rotación |
| **Testabilidad** | **3/10** | Dependency injection en algunos engines (NLP inyectable); arquitectura plugin modular | Dependencias globales de módulo (`_profiler = None` al top-level) dificultan mocking; un solo test file |
| **Modularidad** | **7/10** | Plugin system bien diseñado; engines independientes; BasePlugin contract claro | `forensic_report.py` plugin acoplado a `full_analysis._orchestrator` via atributos privados |
| **Resiliencia** | **5/10** | `try/except` en cada plugin load; timeout en ThreadPoolExecutor; `soft_time_limit` Celery; Celery auto-restart | Sin circuit breaker para APIs externas; sin retry logic; sin fallback para fallo de Redis |
| **Disponibilidad** | **6/10** | `restart: unless-stopped` en docker-compose; auto-restart de Celery en `child_exit`; health/ready probes | Dual Celery worker es un riesgo; startup de 2 min por carga de ModernBERT; sin rolling updates documentadas |

---

## 8. RESULTADO FINAL

### 8.1 Score del Código

| Dimensión | Puntuación |
|---|---|
| Calidad General | **6.2 / 10** |
| Rendimiento | **5.5 / 10** |
| Seguridad | **4.0 / 10** |
| Escalabilidad | **5.8 / 10** |
| Mantenibilidad | **5.0 / 10** |

### 8.2 Score General de la Aplicación Flask

> **Nota Final: 5.3 / 10**

El sistema demuestra una arquitectura ML técnicamente sofisticada (CoW sharing, ensemble ModernBERT, Late Fusion, vector schemas declarativos) pero con deuda de seguridad crítica (SECRET_KEY hardcodeada, DEBUG=True), ausencia casi total de tests, y al menos un bug de producción de alto impacto (doble Celery worker). El código de los engines individuales es de alta calidad; la capa de infraestructura/configuración es frágil.

---

### 8.3 Top 20 Problemas Más Críticos

| Rank | ID | Problema | Riesgo Prod | Impacto Usuario | Impacto Perf | Impacto Seguridad |
|---|---|---|---|---|---|---|
| 1 | B-01 / S-01 | SECRET_KEY hardcodeada en git | 🔴 CRÍTICO | 🔴 Sesiones forjables | — | 🔴 CRITICAL |
| 2 | B-02 / S-02 | DEBUG=True en producción | 🔴 CRÍTICO | 🔴 Stacks expuestos | — | 🔴 CRITICAL |
| 3 | B-03 | Doble Celery worker (fork + docker) | 🔴 CRÍTICO | 🔴 Duplicación resultados | 🔴 Doble consumo CPU | — |
| 4 | DT-01 | 0% test coverage | 🔴 CRÍTICO | — | — | — |
| 5 | B-06 / P-01 | Doble inferencia ModernBERT en `/analyze_document` | 🔴 ALTO | 🟡 Latencia 2x | 🔴 10/10 | — |
| 6 | B-04 / S-03 | API key timing attack | 🟡 MEDIO | — | — | 🔴 HIGH |
| 7 | B-05 / S-05 | Sin API key = sin autenticación | 🔴 ALTO | — | — | 🔴 HIGH |
| 8 | S-04 | Sin rate limiting | 🔴 ALTO | 🔴 DoS potencial | 🔴 Worker blocking | 🔴 HIGH |
| 9 | S-06 | SSRF en ReferenceValidator | 🟡 MEDIO | — | — | 🟡 MEDIUM |
| 10 | P-04 / DT-04 | SimpleCache por-proceso, no compartida | 🟡 MEDIO | 🟡 Cache misses constantes | 🟡 7/10 | — |
| 11 | B-08 | `new_event_loop()` por request | 🟡 MEDIO | 🟡 Latencia adicional | 🟡 8/10 concurrencia | 🟡 MEDIUM |
| 12 | B-07 | Acceso `._gpt2._available` privado | 🟡 MEDIO | — | — | — |
| 13 | S-07 / ML-01 | Reports en `/tmp` world-readable, sin cleanup garantizado | 🟡 MEDIO | 🟡 Info disclosure | — | 🟡 MEDIUM |
| 14 | DT-14 | Sin circuit breaker para APIs externas | 🟡 MEDIO | 🟡 Timeouts cascada | 🟡 Bloqueo | — |
| 15 | DT-13 | Sin request correlation ID | 🟡 MEDIO | — | — | — |
| 16 | B-09 | `_parse_bibliography` acoplamiento privado | 🟡 BAJO-MEDIO | 🟡 Fallo silencioso | — | — |
| 17 | DT-12 | `forensic_output_path` default compartido | 🟡 BAJO-MEDIO | 🟡 Race en reports concurrentes | — | — |
| 18 | S-09 | Redis sin autenticación | 🟡 BAJO | — | — | 🟡 MEDIUM |
| 19 | DT-10 | Dead dependencies (gevent, PyJWT, etc.) | 🟢 BAJO | — | 🟡 Imagen Docker más pesada | — |
| 20 | B-11 | `max_tokens` sin validación de rango | 🟢 BAJO | 🟡 Crash con 0 / OOM con 999999 | 🟡 OOM potencial | — |

---

### 8.4 Plan de Remediación

#### ⚡ Quick Wins — 1 día

```python
# 1. SECRET_KEY y API_KEY como env vars obligatorias (B-01, B-05)
SECRET_KEY = os.environ["SECRET_KEY"]         # KeyError intencional si falta
API_KEY    = os.environ.get("API_KEY", "")

# 2. DEBUG via env var (B-02)
DEBUG = os.environ.get("DEBUG", "0") == "1"

# 3. Deshabilitar Celery fork en gunicorn cuando se usa docker-compose (B-03)
#    Comentar/eliminar el bloque when_ready; usar solo el servicio celery_worker

# 4. Timing-safe API key comparison (B-04)
import hmac
if api_key and not hmac.compare_digest(api_key.encode(), provided.encode()):
    return jsonify({"error": "Unauthorized"}), 401

# 5. Eliminar archivos de debug del repo (DT-11)
#    git rm flask.log nohup.out te.text "README copy.md"

# 6. Fix orchestrator: usar propiedad pública .tier (B-07)
tier = self._perplexity_profiler.tier

# 7. Fix orchestrator: feature_values con .items() (DT-09)
ppl_analysis["feature_values"] = {
    k: v for k, v in ppl_stats.items() if isinstance(v, (int, float))
}

# 8. Validar max_tokens en rango [50, 512] (B-11)
max_tokens = max(50, min(int(payload.get("max_tokens", 150)), 512))
```

#### 📅 Corto plazo — 1 semana

```
9.  Rate limiting con Flask-Limiter + Redis backend
    @limiter.limit("60/minute") en /analyze
    @limiter.limit("10/minute") en /analyze_document

10. Eliminar doble inferencia ModernBERT en /analyze_document (B-06)
    Reutilizar segmentos de ai_detection como ya hace el Celery task

11. Cambiar CACHE_TYPE a "RedisCache" con Redis backend compartido (DT-04)
    CACHE_REDIS_URL = os.environ.get("REDIS_URL")

12. asyncio.run() en lugar de new_event_loop() en async_route (B-08)

13. Request correlation ID via middleware
    app.before_request → g.request_id = str(uuid4())[:8]
    Incluir en todos los log messages y respuestas

14. Circuit breaker para APIs externas (CrossRef, S2, OpenAlex)
    Usar pybreaker o implementación simple con contador de fallos

15. Exponer CitationDetector.parse_bibliography() como método público (B-09)

16. Auditoría de dead dependencies (DT-10):
    pip-audit + pipdeptree para confirmar cuáles son transitivas vs directas
```

#### 📆 Mediano plazo — 1 mes

```
17. Test suite: pytest + coverage
    - routes: test cada endpoint con texto válido/inválido/vacío
    - plugins: mock del engine, verificar estructura de respuesta
    - engine: unit tests de cada extractor con casos conocidos
    - integration: pipeline completo con texto real
    Target: ≥70% coverage

18. SSRF mitigation en ReferenceValidator (S-06)
    - Allowlist explícita de dominios permitidos
    - allow_redirects=False
    - Timeout por solicitud individual (5s max)

19. Mover reports de /tmp a directorio privado con UUID (S-07)
    REPORT_DIR = os.path.join(app_home, "reports")  # chmod 700

20. Prometheus metrics endpoint /metrics
    - requests_total por plugin y status
    - plugin_duration_seconds histogram
    - inference_duration_seconds por modelo

21. Limitar plugins por request
    if len(plugins_requested) > 5: return 400

22. Separar config en Dev/Staging/Production
    class ProductionConfig(Config):
        DEBUG = False
        assert SECRET_KEY != "edw-32fdx-34f421-m56e"
```

#### 🗓️ Largo plazo — 3 meses

```
23. Reemplazar heurísticos HallucinationRiskClassifier y ReasoningRiskClassifier
    con modelos XGBoost/LightGBM entrenados en corpus anotado
    → Precision de ~55-62% a ~82-88% para hallucination_check

24. OpenTelemetry distributed tracing
    - Spans por plugin, por inference window, por API call externa
    - Integración con Jaeger / Grafana Tempo

25. Calibración multi-idioma
    - Detector de idioma al inicio del análisis
    - Thresholds separados por idioma para perplexity y stylometric

26. Horizontal scaling
    - Separar ML inference a workers dedicados con GPU
    - API gateway (nginx) → Flask (CPU, auth, routing) → ML workers (GPU)
    - Redis Streams para back-pressure

27. Modelo de permisos por plugin
    - Admin: todos los plugins incluidos watermark + citation
    - Standard: ai_detection, stylometric, perplexity, reasoning
    - Lite: solo ai_detection

28. Pipeline CI/CD completo
    - GitHub Actions: test → lint → docker build → scan (trivy) → deploy
    - Dependabot para actualización automática de dependencias
    - Safety para CVE scanning de requirements.txt
```

---

> **⚠️ Prioridad absoluta antes de cualquier despliegue en producción:**
>
> **B-01** (SECRET_KEY) + **B-02** (DEBUG) + **B-03** (Celery dual) + **S-04** (Rate limiting)
> son bloqueantes de producción. Sin remediar estos cuatro, el sistema no debe
> exponerse en internet bajo ninguna circunstancia.
