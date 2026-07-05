# AUDITORÍA CIENTÍFICA Y ALGORÍTMICA — XplagiaX (detector de texto IA)

> **Alcance:** validez científica, integridad algorítmica, calibración estadística y
> defendibilidad forense del detector. **No** repite la capa de seguridad/infra/performance
> de `AUDITORIA_TECNICA.md` (en su mayoría ya corregida). Fundamentado en lectura directa de
> los 8 plugins, los 11 módulos de `app/engine/` y la capa Flask/Celery/Gunicorn.
>
> Fecha: 2026-06-10 · Equipo simulado: PhD IA/NLP/Stylometry/Computational Linguistics/Digital
> Forensics + Arquitecto Python/Flask + Especialista detección LLM + Revisor Scopus/WoS.

---

## DICTAMEN EJECUTIVO

> **El "ensemble forense multi-plugin" no existe como tal a nivel de decisión.** El veredicto
> (`verdict`) y la confianza (`confidence`) se fijan **exclusivamente** desde el clasificador
> neural de 3×ModernBERT (`forensic_reports.py:1031-1043`). Perplexity, reasoning, hallucination,
> watermark, stylometric y reference **no entran en ninguna fórmula de fusión** — se renderizan
> como `evidence_points` narrativos. El único override es `verdict="Hybrid"` derivado de las
> atribuciones por oración, que a su vez salen del **mismo** score neural + un léxico de palabras.

Consecuencia: científicamente es **un único clasificador transformer de 2023 envuelto en una
narrativa forense de ~11.000 líneas**. Los 7 plugins restantes aumentan la *percepción* de rigor
sin aumentar el rigor real.

Segundo hallazgo de igual gravedad: el clasificador y los heurísticos están calibrados para
**LLMs ≤2023** (`label_mapping` termina en `gpt4o`, `detector_final.py:67-78`). No hay GPT-5,
Claude 3/4, Gemini, ni DeepSeek-R1. La proxy-perplejidad Tier 1 **no es perplejidad**, y Tier 2
usa GPT-2 (2019) como modelo de referencia.

**Calificación global:** arquitectura Flask 7/10 · calidad de código 6.5/10 · precisión
científica 3/10 · precisión del detector 4/10 · explicabilidad 3.5/10 · confiabilidad 3/10 ·
escalabilidad 5.5/10 (justificación en §10).

---

## 1. AUDITORÍA DE CÓDIGO

### 1.1 Bugs y errores algorítmicos

| ID | Archivo:línea | Severidad | Hallazgo | Solución |
|----|---------------|-----------|----------|----------|
| C-01 | `detector_final.py:167-198` | 🔴 Alta | `classify_text` crea figuras con la **API global de pyplot** dentro de código llamado desde un `ThreadPoolExecutor` (`plugin_registry.py:84`, `max_workers=8`). pyplot no es thread-safe → corrupción/fugas/crash bajo concurrencia. | Sacar matplotlib del path de inferencia (`generate_plot=False`); si se necesita, `Figure()` por hilo con backend Agg. |
| C-02 | `detector_final.py:195` | 🟠 Media | `raw_scores={"human": round(human_prob), ...}` redondea una probabilidad ∈[0,1] a entero ⇒ **siempre 0/1**. `summary()` imprime 0.0/1.0 y `generate_report` busca la clave `"ai"` inexistente (rama muerta). | Guardar porcentajes `round(x*100,2)` con clave `"ai"`. |
| C-03 | `detector_final.py:151-153` | 🟡 Baja | `total_decision_prob = human_prob + ai_total_prob` es siempre ≈1.0 (softmax). Normalización no-op. | Eliminar. |
| C-04 | `detector_final.py:317-394` | 🔴 Alta | `validar_veredicto_segmento` invierte el veredicto neural por reglas duras (`score = 100 - score`) y emite "AI (Confirmed) **100.0**". Dos heurísticos débiles sobrescriben un transformer con confianza 100% — indefendible forensemente. | Convertir a señal/anotación; nunca override binario al 100%. |
| C-05 | `detector_final.py:687-708` | 🟠 Media | `analyze_fast` redondea `human_pct/ai_pct` a entero por segmento antes de ponderar y fuerza `ai_pct = 100 − human_pct`, descartando la distribución de 41 clases. | Acumular en flotante con probabilidades sin redondear. |
| C-06 | `perplexity_profiler.py:216` | 🟡 Baja | `ppl = 2.0 ** (-avg_log_prob + (1.0/L))` mezcla log-prob y término de longitud dentro del exponente — dimensionalmente incoherente. | Separar la corrección de longitud. |
| C-07 | `detector_final.py:34-61` | 🟠 Media (verificar) | Los 3 pesos se cargan con el mismo `_config(num_labels=41)`, pero dos dirs se llaman `..._3class_...`. Si fueran de 3 clases, `load_state_dict` fallaría. Nombre engañoso o bomba latente. | Verificar shape del head; renombrar; test de smoke de carga. |
| C-08 | `routes.py:332-334` | 🟠 Media | `/analyze_status` para `SUCCESS` hace `response.update(task.info)`; si el task devolvió `status:"error"` interno, mezcla `state:SUCCESS` + `status:error`. | Normalizar contrato de estado. |

### 1.2 Errores silenciosos / excepciones tragadas (categoría crítica)

| ID | Dónde | Severidad | Problema |
|----|-------|-----------|----------|
| C-09 | Wrappers de plugins + `plugin_orchestrator.py:180-283` | 🔴 Alta | Cada carga de modelo va en `except Exception: warning`. Si ModernBERT no carga, el servicio **sigue respondiendo 200**, y `run_with_result` descarta silenciosamente plugins que lanzan (`:321,358,387,405,432,444,466`). Un forense que degrada en silencio mientras emite veredicto confiado es peligroso. |
| C-10 | `routes.py:430-443` | 🟠 Media | `/ready` solo comprueba `len(registry)>0`. Un plugin registrado con su modelo caído cuenta como ready. La readiness probe miente. |
| C-11 | `forensic_reports.py:1069-1115` | 🟠 Media | Hallucination/reasoning/watermark corren con `try/except…warning`; al fallar, sus scores caen a `0.5`/`0.0` por defecto — un fallo se vuelve indistinguible de "señal neutra". |

### 1.3 Concurrencia, CPU/RAM, Gunicorn

| ID | Dónde | Severidad | Problema | Solución |
|----|-------|-----------|----------|----------|
| C-12 | `gunicorn.conf.py:28-36` + `plugin_registry.py:83` + torch intra-op | 🔴 Alta | **Sobre-suscripción de CPU**: `workers×threads×ThreadPool(8)×torch_threads`. No se llama a `torch.set_num_threads()`. Thrashing de context-switch, p99 disparado. | `torch.set_num_threads(1)`; paralelismo en una sola capa. |
| C-13 | `detector_final.py:10` | 🟡 Baja | `torch.set_grad_enabled(False)` es side-effect global en import (correcto para inferencia; documentar). |
| C-14 | `watermark_decoder.py:476-571` | 🟠 Media | Entropía con GPT-2 ventana deslizante sobre todos los tokens (hasta 500k chars) — cuello de RAM/latencia; puede exceder `time_limit=300`. Deshabilitado por defecto. | Cap duro de tokens + muestreo. |
| C-15 | `routes.py:28` vs `config.py:51` | 🟡 Baja | `_MAX_TEXT_CHARS=500_000` vs `MAX_CONTENT_LENGTH=2MB`: límites inconsistentes. |

### 1.4 Caché

| ID | Dónde | Problema |
|----|-------|----------|
| C-16 | `routes.py:32-35` | Clave `sha256(text+plugins)` **sin versión de modelo** → sirve resultados viejos 1h tras actualizar pesos. |
| C-17 | `detector_final.py:618-651` | Dos capas desacopladas: `_FAST_CACHE` (in-process/worker, TTL 5min, 20 entradas) vs Flask-Cache (Redis, 1h). Sin invalidación cruzada; baja tasa de acierto multi-worker. |

---

## 2. AUDITORÍA CIENTÍFICA POR PLUGIN

| Plugin (motor real) | Fundamento | Clasificación | Razón |
|---------------------|-----------|---------------|-------|
| **ai_detection** (3×ModernBERT) | Clasificación supervisada transformer | **Parcialmente válido / obsoleto** | (a) entrenado ≤2023; (b) "ensemble" de 3 seeds del **mismo backbone y dataset** → errores correlacionados, reducción de varianza marginal, cero reducción de sesgo; (c) confianza = max(softmax) sin calibrar. |
| **perplexity_check** | DetectGPT/Fast-DetectGPT/Binoculars/LLMDet (citados) | **Obsoleto + propenso a errores** | Tier 1 **no calcula perplejidad**: remapea hapax+TTR+repetición a una escala inventada [1,15] (`:219-262`). Citar 4 papers no implementados es insostenible. Tier 2 usa GPT-2 (2019); el texto de LLMs 2025 ya no es de baja perplejidad bajo GPT-2. |
| **stylometric_analysis** | Burstiness, MATTR, hapax, dep-distance | **Parcialmente válido** | Métricas legítimas, pero **descriptores** no clasificadores; umbrales humanos fijos sin intervalos; alta covarianza con género/idioma/longitud. Ver §13 (subsistema de autoría latente). |
| **reasoning_check** | Marcadores CoT/backtracking por regex | **Propenso a errores** | Extractor limpio (Late-Fusion honesto). Clasificador con pesos/umbrales a mano (`forensic_reports.py:238-279`). "step by step"/"therefore" marca texto técnico humano. |
| **hallucination_check** | Cohesión semántica (Jaccard ponderado), Mündler 2024 | **Parcialmente válido (honesto)** | Se autodeclara `classifier_type:"heuristic"` y "TEMPORARY… replace with trained model". Mide inconsistencia, no "IA". |
| **citation_check** (reference_validator) | Verificación CrossRef/S2/OpenAlex | **Científicamente sólido (el más fuerte)** | Único con ground-truth externo verificable; scorer CheckIfExist real (`:719-788`); detecta chimeras. Limitado por red y por `not_found→fabricated` (§13). |
| **watermark_detection** | Kirchenbauer 2023 green/red | **Especulativo (auto-declarado EXPERIMENTAL)** | Prueba 10 seeds RNG aleatorios y toma el mejor z-score (`:363-375`). El green-list real depende de la clave secreta del watermarker → ~100% FN; "mejor de 10 particiones aleatorias" es p-hacking (aunque corrija Bonferroni). |
| **zone_classifier** | Regex de estilos APA/MLA/IEEE | **Válido para su objetivo** | Clasifica zonas de cita; correcto en su dominio. |
| **forensic_report** | Agregación + HTML | **Pseudo-científico como "fusión"** | No fusiona; sub-scores decorativos. Léxico "delve=IA" (`:75-89`) es sesgo discreditado. |

---

## 3. ANÁLISIS DE FALSOS POSITIVOS (humano → marcado IA)

Causa raíz transversal: el score por oración/palabra es el score neural global **empujado por un
léxico de buzzwords** (`forensic_reports.py:116-140`): `{delve:0.95, furthermore:0.8, utilize:0.7,
moreover:0.8, robust:0.7, comprehensive:0.7, whilst:0.8, aforementioned:0.85, pertaining:0.8…}`.

| Población | Causa raíz | Prob. FP | Severidad | Mitigación |
|-----------|-----------|----------|-----------|------------|
| Estudiante excelente / redacción formal | Usa "furthermore/comprehensive"; baja burstiness por edición (`:712`) | Alta | 🔴 | Eliminar el léxico; calibrar por registro; mostrar incertidumbre. |
| Académico / paper científico | Conectores lógicos densos disparan reasoning + buzzwords | Muy alta | 🔴 | Corpus académico humano; no usar conectores como señal. |
| Abogado / texto legal | "aforementioned/pertaining/hence/thereby" en el léxico; estructura uniforme | Muy alta | 🔴 | Igual. |
| Científico/técnico | Terminología repetida → proxy-perplexity Tier 1 lo marca "predecible/IA" (`:232-247`) | Alta | 🔴 | LM moderno; normalizar por terminología. |
| Periodista / prosa pulida | Editores reducen burstiness; `burstiness<0.12` ⇒ "AI hallmark" (`:712`) | Media-alta | 🟠 | Burstiness no es discriminante post-2023; bajar peso. |
| Texto revisado por corrector | La edición humana reduce la variabilidad usada como "humana" | Alta | 🟠 | Añadir `revision_pattern` (§8-J). |
| Traducción humana / autor ESL | MT/ESL producen sintaxis regular; regex de oración asume mayúscula inicial latina | Muy alta | 🔴 | Detección idioma/MT; calibración separada. La guardia de función-words existe en perplexity (`:122-130`) pero **no** en el resto. |

---

## 4. ANÁLISIS DE FALSOS NEGATIVOS (IA → parece humana)

| Caso | Causa raíz | Prob. FN | Mitigación |
|------|-----------|----------|-----------|
| **GPT-5 / Claude 4 / Gemini 2** | El clasificador nunca vio estas distribuciones (`label_mapping` ≤ gpt4o). OOD. | Muy alta | Reentrenar con corpus 2024-2026; monitoreo de drift. |
| **DeepSeek-R1 / o1 (razonadores)** | reasoning_check los marca por CoT — pero **no afecta el veredicto**. | Alta | Integrar reasoning en la fusión real. |
| **Modelos fine-tuned / locales** | Distribución de tokens distinta; sin watermark. | Alta | Features agnósticas al modelo. |
| **Paraphrasers / humanizadores** | Diseñados para subir perplejidad y burstiness; rompen las 3 señales. | Muy alta | Detección de artefactos de parafraseo. |
| **Mezcla humano+IA** | El override puede voltear segmentos IA a "Human" (`:353-362`). | Alta | Reportar distribución; quitar el flip. |
| Texto IA **corto** (<300 tokens) | `_score_from_severity` arrastra a 0.5 (`:1056-1058`). | Media | Declarar "insuficiente" en vez de 0.5 confiado. |

---

## 5. AUDITORÍA DE ALGORITMOS (score 1-10)

| Algoritmo | Robustez | Explicab. | Mantenib. | **Score** | Comentario |
|-----------|:-:|:-:|:-:|:-:|------|
| Ensemble 3×ModernBERT | 4 | 5 | 7 | **5** | Seeds correlacionados ≠ ensemble; sin calibración; OOD 2026. |
| `validar_veredicto_segmento` | 2 | 6 | 4 | **3** | Override binario con 100% de confianza. |
| Proxy-perplexity Tier 1 | 2 | 3 | 6 | **2** | No es perplejidad; umbrales inventados. |
| Curvature Tier 2 | 4 | 4 | 5 | **4** | Aproximación ad-hoc con GPT-2; clamp [-10,10] enmascara inestabilidad. |
| Reasoning (extractor) | 7 | 7 | 8 | **7** | Limpio, O(n), honesto. |
| Reasoning (clasificador) | 3 | 6 | 5 | **4** | Pesos/umbrales a mano. |
| Hallucination | 5 | 7 | 7 | **6** | Honesto sobre ser heurístico. |
| Reference validator | 7 | 8 | 6 | **7.5** | Único con ground-truth; frágil en extracción/red. |
| Watermark green/red | 3 | 5 | 6 | **3.5** | Búsqueda sobre 10 seeds aleatorios; FN sistemático. |
| Hybrid segment | 5 | 7 | 7 | **6** | Buena idea; umbrales 70/30 arbitrarios. |
| Word/sentence attribution | 1 | 6 | 5 | **2** | "delve shibboleth"; sesgo demográfico. |

**Errores matemáticos/estadísticos:** C-03 (no-op), C-06 (exponente mal formado), `round` de
probabilidades (C-02/C-05), y max(softmax) como "confianza" sin calibrar (overconfidence conocido).

---

## 6. AUDITORÍA DE CONFIANZA / CALIBRACIÓN

- **Calibración: inexistente.** No hay temperature scaling, Platt ni isotonic.
  `confidence = round(max(human%, ai%))` (`detector_final.py:191`) se presenta como "AI confidence: X%".
  Los softmax de transformers fine-tuneados son sistemáticamente sobreconfiados.
- **Scores engañosos detectados:**
  - `PerplexityRiskClassifier.ai_score` — el código advierte "DISPLAY ONLY… NOT calibrated…
    MUST NOT be used as ML input" (`:919-931`) — **pero** se consume como umbral en
    `validar_veredicto_segmento` (`< 0.40`, `detector_final.py:355`). Contradicción metodológica.
  - `reasoning ai_score`, `hybrid global_ai_score`, `hallucination overall_risk` — en [0,1] sin
    semántica probabilística, mostrados como porcentajes.
- **Incertidumbre:** `uncertainty_zone = |ai−human|<15` (umbral fijo, no intervalo).
- **Intervalos de confianza:** ninguno (la varianza entre los 3 seeds sería gratis y útil).
- **Normalización:** min-max del reasoning con umbrales `(lo,hi)` a ojo.

**Veredicto §6:** los scores numéricos del sistema **no tienen significado estadístico calibrado**.
Son ordenables, no interpretables como probabilidades.

---

## 7. AUDITORÍA DE EVIDENCIA

| Hallazgo | Fuerza | Justificación |
|----------|--------|---------------|
| Cita verificada como inexistente (red activa, multi-DB) | **Fuerte** | Ground truth externo. |
| Cita chimérica (título✓/autores✗) | **Fuerte** | Verificable y específica. |
| Clasificación neural **dentro** de distribución 2023 | **Moderada** | Solo para LLMs vistos. |
| Clasificación neural sobre LLM 2024-2026 | **Especulativa** | OOD. |
| Hallucination / reasoning / stylometric | **Débil** | Heurísticos correlacionados con género/idioma. |
| Proxy-perplexity Tier 1 | **Especulativa** | No mide lo que dice. |
| Watermark "candidate" | **Especulativa** | Búsqueda aleatoria; experimental. |
| Confianza % | **Especulativa** | Sin calibrar. |

---

## 8. NUEVOS PLUGINS PROPUESTOS

Objetivo transversal: **mover de "un transformer + narrativa" a una fusión tardía calibrada y
model-agnóstica.**

| Plugin | Objetivo | Algoritmo | Compl. | Costo | Precisión esp. | Valor forense |
|--------|----------|-----------|:-:|:-:|:-:|:-:|
| **A. semantic_consistency** | Contradicciones internas | NLI (DeBERTa-MNLI) sobre pares de claims; grafo de entailment | O(n·k) | Medio | Alta | Alto |
| **B. source_traceability** | Trazabilidad de afirmaciones | Claims → retrieval embeddings vs corpus/web | O(n·R) | Medio-alto | Media | Alto |
| **C. factual_density** | Densidad factual | NER + entidades/números/fechas por 100 palabras | O(n) | Bajo | Media | Medio |
| **D. author_signature** | Perfil estilométrico del autor | Vector estilométrico + verificación 1-clase / consistencia intra-doc | O(n) | Bajo | Alta con baseline | **Muy alto** |
| **E. entropy_profile** | Entropía lingüística | Perplejidad real con LM **2025** (no GPT-2) | O(n) GPU | Alto | Alta | Alto |
| **F. discourse_structure** | Estructura argumentativa | RST/grafo de conectores vs distribuciones | O(n) | Medio | Media | Medio-alto |
| **G. citation_graph** | Relaciones entre citas | Grafo cita↔bib↔co-citación; clusters fabricados | O(C²) | Bajo | Alta | Alto |
| **H. knowledge_depth** | Profundidad conceptual | Profundidad ontológica (WordNet/Wikidata) | O(n) | Medio | Media | Medio |
| **I. claim_verification** | Extraer y verificar afirmaciones | Claim extraction → fact-checking (retrieval+NLI) | O(n·R) | Alto | Media-alta | **Muy alto** |
| **J. revision_pattern** | Edición humana | Historial si existe; micro-inconsistencias si no | O(n) | Bajo | Alta (señal humana) | Alto (anti-FP) |
| **K. llm_fingerprint** | Patrones de LLM | n-gramas sobre-representados **aprendidos y calibrados**, por idioma | O(n) | Bajo | Media | Medio |
| **L. uncertainty_analysis** | Hedging / certeza | Léxico de hedging + cuantificadores; ratio certeza/hedge | O(n) | Bajo | Media | Medio |

Prioridad: **E, D, I, A** (blindan el núcleo), luego G, J (anti-FP), luego el resto.

---

## 9. ROADMAP

- **Fase 1 — Correcciones críticas:** matplotlib fuera de inferencia; quitar override binario;
  `raw_scores`/redondeos; `set_num_threads(1)`; `/ready` honesto + no tragar fallos; versión de
  modelo en caché.
- **Fase 2 — Mejoras científicas:** calibración (temp. scaling + ECE); fusión tardía real;
  reentrenar con corpus 2024-2026; reemplazar perplexity Tier 1 por `entropy_profile`; eliminar
  léxico "delve".
- **Fase 3 — Optimización:** CPU oversubscription; batching; cap de tokens en watermark/entropy;
  unificar caché.
- **Fase 4 — Plugins avanzados:** D, I, A, G, J, F, H, L.
- **Fase 5 — Validación experimental:** dataset, protocolo, métricas, publicación (§11).

---

## 10. CALIFICACIÓN FINAL (0-10)

| Dimensión | Score | Justificación |
|-----------|:-:|---------------|
| **Arquitectura Flask** | **7.0** | App-factory, blueprints, registry auto-discovery, preload+CoW, Celery, SSE, rate-limit, HMAC. Resta: readiness que miente, dos cachés, CPU oversubscription. |
| **Calidad de código** | **6.5** | Documentado, dataclasses, `__slots__`, type hints, regex compiladas. Resta: imports desnudos + `sys.path` injection, excepciones tragadas, funciones deprecadas duplicadas. |
| **Precisión científica** | **3.0** | Fusión inexistente; perplexity falsa; léxico discreditado; sin calibración; papers citados no implementados. |
| **Precisión del detector** | **4.0** | Sólido solo dentro de distribución 2023; OOD 2026; vulnerable a parafraseo. |
| **Explicabilidad** | **3.5** | Mucha narrativa, pero circular (deriva del mismo score) y los % no son interpretables. |
| **Confiabilidad** | **3.0** | Degradación silenciosa + confianza no calibrada + overrides binarios al 100%. |
| **Escalabilidad** | **5.5** | CoW y async bien pensados; pero CPU oversubscription, watermark O(n) sin cap, red en reference validator. |

---

## 11. VALIDACIÓN TIPO PAPER CIENTÍFICO

**Métricas faltantes (todas):** no hay ninguna evaluación cuantitativa en el repo. Faltan
**Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, ECE**, además de:
- **TPR @ FPR fijo bajo** (p.ej. TPR@FPR=1%) — la métrica que importa cuando un FP arruina a un estudiante.
- **Equalized odds por subgrupo** (nativo vs ESL; académico/legal/técnico).
- **Robustez adversarial** (post-paráfrasis, post-humanizador).

**Datasets recomendados:** RAID, M4/M4GT, MAGE, HC3, GPABenchmark + **un set propio 2025-2026**
con GPT-5/Claude 4/Gemini 2/DeepSeek-R1 + humano nativo y ESL + textos legales/académicos/técnicos.
Held-out **temporal** (entrenar ≤2024, testear 2025-2026) para medir drift.

**Protocolo experimental reproducible:**
1. Split estratificado por modelo/dominio/longitud; held-out temporal.
2. Calibración en validación (temp. scaling); curva de fiabilidad + ECE.
3. Métricas con **IC95% bootstrap** (≥1000 resamples).
4. **Ablation** por plugin (demuestra/refuta empíricamente que la fusión es decorativa).
5. Análisis de sesgo por subgrupo con tests de significancia.
6. Adversarial (DIPPER/parafraseo).
7. Baselines públicos (Binoculars, Fast-DetectGPT real, RADAR).
8. Reproducibilidad: semillas, versiones, datasheet, model card, código + splits.

---

## 12. ENTREGA FINAL — TABLAS

### 12.1 Riesgos
| Riesgo | Prob. | Impacto | Nivel |
|--------|:-:|:-:|:-:|
| FP a estudiante ESL/académico/legal | Alta | Severo (daño a persona) | 🔴 Crítico |
| Veredicto confiado con modelos caídos | Media | Severo | 🔴 Crítico |
| FN ante GPT-5/Claude/parafraseo | Alta | Alto (inutilidad) | 🔴 Crítico |
| Confianza no calibrada usada como prueba | Alta | Severo (legal) | 🔴 Crítico |
| CPU oversubscription bajo carga | Media | Alto (p99) | 🟠 Alto |
| Caché sirve resultados de modelo viejo | Media | Medio | 🟠 Alto |

### 12.2 Bugs
Ver §1 — críticos: C-01, C-04, C-09, C-12; altos: C-10, C-11, C-14; medios: C-02, C-05, C-07,
C-08, C-16, C-17; bajos: C-03, C-06, C-13, C-15.

### 12.3 Deuda técnica
| ID | Deuda | Severidad |
|----|-------|:-:|
| D-1 | Imports desnudos + `sys.path.insert` (`engine/__init__.py:31`) | 🟠 |
| D-2 | Funciones `analyze_long_document*` deprecadas duplicadas (`:397-607`) | 🟡 |
| D-3 | Léxico AI/HUMAN hardcodeado (`forensic_reports.py:75-89`) | 🔴 |
| D-4 | Umbrales mágicos dispersos sin fuente empírica | 🟠 |
| D-5 | Papers citados no implementados (perplexity) | 🟠 |
| D-6 | Sin tests científicos (solo plumbing) | 🔴 |

### 12.4 / 12.5 Falsos positivos / negativos
Ver §3 y §4.

### 12.6 Mejoras
| Mejora | Fase | Impacto |
|--------|:-:|:-:|
| Calibración (temp. scaling + ECE) | 2 | 🔴 |
| Fusión tardía real entrenada | 2 | 🔴 |
| Reentrenar con corpus 2025-2026 | 2 | 🔴 |
| Reemplazar perplexity por LM moderno | 2 | 🔴 |
| Eliminar léxico/override binario | 1 | 🔴 |
| Propagar fallos + readiness real | 1 | 🟠 |
| Fix CPU oversubscription | 3 | 🟠 |

### 12.7 Nuevos plugins
Ver §8 (A-L).

### 12.8 Ranking de prioridad
1. 🔴 Calibración + fusión real.
2. 🔴 Reentrenar con LLMs 2025-2026.
3. 🔴 Eliminar léxico "delve" y override binario.
4. 🔴 Dejar de tragar fallos + readiness honesta.
5. 🟠 Reemplazar perplexity Tier 1 por `entropy_profile`.
6. 🟠 CPU oversubscription + matplotlib thread-safety.
7. 🟠 Plugins D, I, A, G, J.
8. 🟡 Versión de modelo en caché, limpieza de deuda, validación experimental.

---

## 13. LECTURA PROFUNDA (hallazgos adicionales)

Hallazgos surgidos de leer `stylometric_profiler.py`, `hallucination_profile.py` y
`reference_validator.py` línea a línea:

1. **Subsistema de autoría latente (stylometric).** Existe maquinaria completa de *verificación de
   autoría*: `build_profile`, `compare`, ventana deslizante, embeddings, siamese, umbrales
   adaptativos (`:1241-1644`). Pero el pipeline solo llama a `compute_stats` (`:1189-1239`), que
   devuelve **únicamente descriptores**. El `similarity_score` que `generate_report` lee
   (`forensic_reports.py:1054`) **nunca se produce** ⇒ `stylometric_score` es siempre 0.5
   (decorativo). Esta maquinaria es la base directa del plugin propuesto **D (author_signature)**
   y debe activarse (modo consistencia intra-documento cuando no hay perfil de referencia).

2. **`top_signals` por magnitud cruda (hallucination).** `classify` ordena **todas** las features
   por valor crudo (`hallucination_profile.py:982`), no por su contribución ponderada a
   `overall_risk`. Features de alta magnitud (p.ej. entropía ~10) dominan los "top signals" aunque
   tengan peso bajo. **Bug de display** que distorsiona la evidencia mostrada.

3. **`not_found → fabricated` (reference_validator).** `fabricated = status=="not_found"`
   (`:899`). Pero "not_found" puede significar "no indexado" (papers nuevos, libros, preprints,
   venues no indexados), no "inventado". Este FP alimenta directamente la regla
   `validar_veredicto_segmento` → "AI (Confirmed) **100%**". El módulo más sólido tiene aquí su
   talón de Aquiles. Mitigación: exigir miss en ≥2 bases + validez estructural y exponer
   `not_indexed_count` separado de `fabricated_count`.

4. **Truncación 512-token del veredicto global.** En la ruta de API `full_analysis`, el veredicto
   global se deriva de `classify_text` (`detector_final.py:114-199`), que tokeniza con
   `truncation=True` ⇒ solo clasifica los **primeros ~512 tokens** del documento. Para textos
   largos, el veredicto global ignora el grueso del texto. `analyze_fast` (usado por `ai_detection`/
   `/analyze_document`) sí cubre todo el texto con agregado token-weighted; conviene unificar para
   que `full_analysis` use el mismo agregado.

5. **Scorer de citas (positivo).** `_compute_confidence` (`:719-788`) es un scorer real estilo
   CheckIfExist (título 40 / autores 30 / año 15 / journal 15 + penalizaciones) con cascada
   CrossRef→S2→OpenAlex, circuit breaker y cap de refs. Es el componente más defendible del sistema
   y debe ser un pilar de la fusión (§8-G citation_graph lo extiende).
