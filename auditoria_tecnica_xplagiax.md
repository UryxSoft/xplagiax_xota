# AUDITORÍA TÉCNICA Y CIENTÍFICA — MICROSERVICIO DE DETECCIÓN DE IA

**Destinatario:** Panel de Dirección Técnica y Científica
**Autores (Comité de Evaluación):**
- Investigador Principal en NLP (Autónomas y Lingüística Computacional)
- Ingeniero de Aprendizaje Automático Senior (Arquitecturas Transformer e Inferencia de LMs)
- Arquitecto de Software Principal (Sistemas Distribuidos y Microservicios)
- Estadístico Clínico (Sistemas de Clasificación Probabilística y Calibración)

---

## 1. EVALUACIÓN DETALLADA POR DIMENSIÓN

### DIMENSIÓN 1 — Metodología de Detección y Rigor Científico
**Puntuación: 3.2 / 10.0**

#### Justificación Técnica
La base teórica del microservicio descansa en un ensamble supervisado de tres inicializaciones aleatorias (semillas) de la arquitectura `answerdotai/ModernBERT-base` entrenado con un espacio de salida estático de 41 clases que mapea a modelos de lenguaje específicos. Desde una perspectiva de procesamiento de lenguaje natural (NLP), este diseño presenta una fragilidad conceptual severa al operar bajo el supuesto de un "mundo cerrado". El sistema sufre de un sesgo Out-of-Distribution (OOD) absoluto frente a los LLM lanzados con posterioridad a 2023, tales como la familia Claude 3.5, Gemini 1.5/2.0 y DeepSeek-R1. Además, el supuesto "ensamble" no actúa como tal para la predicción final, ya que el veredicto en producción y la puntuación de confianza se toman de manera cruda del clasificador neuronal, relegando las señales estilométricas, de razonamiento, de perplejidad y de detección de marcas de agua a puntos de evidencia decorativa no integrados probabilísticamente.

#### Debilidades y Brechas Encontradas
*   **Perplejidad estadística simulada:** La perplejidad en el módulo de Tier 1 no realiza un cálculo probabilístico real (log-verosimilitud promedio del texto); en su lugar, remapea de manera heurística indicadores superficiales como hapax legomena y TTR a una escala arbitraria. El Tier 2 utiliza GPT-2 (de 2019) como modelo de referencia, lo cual no es representativo de la distribución de tokens de los modelos autoregresivos modernos del periodo 2024-2026.
*   **Vulnerabilidad Adversarial Catastrófica:** No existen defensas sistemáticas contra ataques de parafraseo (p. ej., utilizando el modelo de traducción DIPPER) o inyección de prompts diseñados para inflar artificialmente la burstiness lingüística.
*   **Watermarking Especulativo:** El decodificador de marcas de agua implementado en [watermark_decoder.py](file:///Users/user/Documents/xplagiax_xota/app/engine/watermark_decoder.py) intenta inferir un esquema de Kirchenbauer sin poseer la clave criptográfica del generador, incurriendo en un proceso heurístico de prueba aleatoria de 10 semillas (p-hacking) con una tasa de falsos negativos cercana al 100%.

#### Recomendaciones de Mejora
1.  Migrar a un enfoque de detección zero-shot moderno basado en el ratio de perplejidad mutua entre modelos de tamaño medio, como Binoculars ([Hans et al., 2024](https://arxiv.org/abs/2401.12070)) o Fast-DetectGPT ([Bao et al., 2024](https://arxiv.org/abs/2310.05130)), utilizando pesos locales de un modelo autoregresivo representativo de 2025/2026 (p. ej., Llama-3-8B).
2.  Rediseñar el pipeline para sustituir el clasificador de mundo cerrado (41 clases) por un modelo binario calibrado que aprenda representaciones agnósticas del modelo generador.

---

### DIMENSIÓN 2 — Arquitectura de ML/AI y Calidad del Modelo
**Puntuación: 4.2 / 10.0**

#### Justificación Técnica
La selección de ModernBERT es arquitectónicamente sólida debido a sus optimizaciones en el mecanismo de atención bidireccional y el uso de codificaciones posicionales rotativas (RoPE) que mejoran el modelado de dependencias de largo alcance en comparación con arquitecturas basadas en BERT clásicas. Sin embargo, el pipeline de inferencia carece de un marco riguroso de calibración estadística, por lo que las probabilidades de salida del softmax se exponen de manera cruda y sobreconfiada como porcentajes de confianza en [classify_text](file:///Users/user/Documents/xplagiax_xota/app/engine/detector_final.py#L127). Esto ignora el fenómeno documentado de que los clasificadores supervisados basados en redes neuronales profundas sufren de un severo desajuste de calibración (calibration drift) bajo desplazamientos de la distribución de entrada. Adicionalmente, el ensamblado de las tres copias de ModernBERT no realiza una verdadera reducción de sesgo ya que las tres variantes fueron entrenadas sobre la misma distribución y el mismo corpus de datos, lo que correlaciona fuertemente sus errores de predicción.

#### Debilidades y Brechas Encontradas
*   **Falta de Calibración Probabilística:** El score reportado como confianza se calcula mediante un simple redondeo del softmax máximo, sin aplicar técnicas como Temperature Scaling o Isotonic Regression para alinear las probabilidades predichas con las frecuencias reales.
*   **Fusión Tardía Desactivada (Cargo-Cult ML):** El módulo [fusion.py](file:///Users/user/Documents/xplagiax_xota/app/engine/fusion.py) se encuentra configurado únicamente como un armazón ("framework only") que opera en modo passthrough o mediante heurísticas ad-hoc en producción, en lugar de utilizar un clasificador logístico calibrado sobre un vector de características unificado.
*   **Doble Inferencia en Endpoint Síncrono:** El endpoint síncrono `/analyze_document` en [routes.py](file:///Users/user/Documents/xplagiax_xota/app/routes.py#L172) ejecutaba históricamente una doble pasada completa sobre el ensemble de modelos de ModernBERT al invocar concurrentemente `registry.run()` y `analyze_long_document()`, duplicando inútilmente la latencia en CPU.

#### Recomendaciones de Mejora
1.  Implementar un calibrador post-hoc en el meta-clasificador (p. ej., Platt Scaling o Isotonic Regression) validado mediante la métrica ECE (Expected Calibration Error) sobre un conjunto de validación calibrado.
2.  Entrenar formalmente el meta-clasificador de la clase [FusionClassifier](file:///Users/user/Documents/xplagiax_xota/app/engine/fusion.py#L328) utilizando regresión logística regularizada con pérdidas balanceadas sobre un corpus representativo de textos multimodelo generados e híbridos.

---

### DIMENSiÓN 3 — Cobertura Lingüística y Estilométrica
**Puntuación: 5.0 / 10.0**

#### Justificación Técnica
El sistema implementa extractores para características estilométricas superficiales clásicas, incluyendo la diversidad léxica por medio de TTR (Type-Token Ratio) y hapax legomena en [stylometric_profiler.py](file:///Users/user/Documents/xplagiax_xota/app/engine/stylometric_profiler.py). No obstante, estos estimadores no están normalizados en función de la longitud del texto (por ejemplo, mediante la métrica MTLD o HD-D), lo que genera variaciones espurias en la puntuación final causadas únicamente por la extensión del documento. Adicionalmente, la arquitectura lingüística comete un error crítico de explicabilidad al utilizar un diccionario estático de "buzzwords" (ej. *delve*, *furthermore*, *moreover*, *utilize*) en el módulo de reportes forenses para inflar artificialmente la puntuación de probabilidad de generación por IA. Este enfoque heurístico confunde el estilo académico formal de un autor humano con la firma promedio de un LLM, induciendo un sesgo de falsos positivos inaceptable en registros científicos o jurídicos.

#### Debilidades y Brechas Encontradas
*   **Parálisis de la Autoría Latente:** La maquinaria para análisis de perfiles de autoría en `stylometric_profiler.py` calcula métricas descriptivas pero nunca ejecuta la lógica de verificación de autoría de una sola clase (one-class stylometry) para detectar mezclas de estilo intra-documento.
*   **Ausencia de Análisis Sintáctico Profundo:** El análisis sintáctico se limita a patrones regex planos en lugar de evaluar la entropía de las etiquetas POS (Part-of-Speech) o las distribuciones de profundidad de los árboles de dependencia sintáctica.
*   **Sesgo Demográfico contra Escritores ESL:** El uso de heurísticas léxicas estáticas penaliza desproporcionadamente a los escritores de inglés como segunda lengua (ESL), quienes tienden a utilizar conectores y marcadores discursivos formales con una burstiness menor a la de un hablante nativo.

#### Recomendaciones de Mejora
1.  Eliminar por completo el léxico de palabras de transición fijas de la lógica de atribución de oraciones.
2.  Incorporar descriptores de estructura discursiva basados en modelos de coherencia semántica globales (como la representación en rejilla de entidades de Barzilay y Lapata) y modelar las desviaciones del principio de Zipf (distribuciones de rango-frecuencia de n-gramas) específicas de cada modelo de lenguaje.

---

### DIMENSIÓN 4 — Arquitectura de Microservicio y Calidad de Ingeniería
**Puntuación: 6.0 / 10.0**

#### Justificación Técnica
La arquitectura de microservicio está diseñada utilizando Flask como servidor WSGI acoplado con Celery para el procesamiento de tareas pesadas en segundo plano mediante colas de Redis. El sistema cuenta con mecanismos de compresión de payloads (Flask-Compress), limitación de tasa (Flask-Limiter) y carga previa de la aplicación (Gunicorn `preload_app=True`) para optimizar el intercambio de páginas de memoria física en modo Copy-on-Write (CoW). A pesar de estas buenas prácticas de infraestructura, la robustez operativa se ve comprometida por una pésima higiene en el manejo de configuraciones de seguridad e inconsistencias en la concurrencia. La persistencia de secretos críticos commiteados en el repositorio y la ejecución de workers redundantes compitiendo por recursos evidencian una falta de validación de arquitectura para entornos productivos a gran escala.

#### Debilidades y Brechas Encontradas
*   **Secretos y Configuración de Depuración Hardcodeados:** La existencia de `SECRET_KEY` fija y `DEBUG = "1"` por defecto en [app/config.py](file:///Users/user/Documents/xplagiax_xota/app/config.py) expone la aplicación a ataques de ejecución remota de código (RCE) y falsificación de cookies de sesión en producción.
*   **Arquitectura Celery Redundante:** El archivo `gunicorn.conf.py` forkeaba históricamente un proceso Celery worker en su hook `when_ready`, compitiendo directamente con el contenedor dedicado de Celery definido en `docker-compose.yml`, duplicando las ejecuciones de tareas en Redis.
*   **Sobresuscripción de CPU:** La falta de una configuración restrictiva en el número de hilos internos de PyTorch (`torch.set_num_threads`) en CPU genera un fenómeno de thrashing por context-switch masivo cuando los hilos de Gunicorn, los workers del ThreadPoolExecutor de plugins y PyTorch compiten concurrentemente por los mismos núcleos.

#### Recomendaciones de Mejora
1.  Hacer obligatoria la inyección de `SECRET_KEY` desde variables de entorno y lanzar un fallo inmediato en el inicio (`RuntimeError`) si no está definida en entornos de producción.
2.  Eliminar por completo el inicio automático de Celery desde los hooks internos de Gunicorn y delegar la gestión del ciclo de vida del worker exclusivamente al orquestador de contenedores (Docker Compose / Kubernetes).
3.  Implementar comparación segura contra ataques de temporización (timing attacks) para las llaves de API usando `hmac.compare_digest`.

---

### DIMENSIÓN 5 — Pipeline de Datos y Estrategia de Almacenamiento Vectorial
**Puntuación: 4.0 / 10.0**

#### Justificación Técnica
La infraestructura expone variables para el uso de la base de datos vectorial Qdrant (`QDRANT_ENABLED`, `QDRANT_URL`, etc.) y el modelo de embeddings semánticos `intfloat/multilingual-e5-small`. Sin embargo, en el código real de producción, el motor de indexación vectorial semántica (`PlagiarismEngine`) está completamente desactivado o excluido de la lógica de evaluación en [flask_routes.py](file:///Users/user/Documents/xplagiax_xota/app/antiplagio/flask_routes.py#L5). Esto imposibilita la detección de plagio por paráfrasis mediante búsquedas de similitud coseno o producto punto a gran escala. La validación de referencias bibliográficas en [ReferenceValidator](file:///Users/user/Documents/xplagiax_xota/app/antiplagio/citation/validator.py) realiza llamadas concurrentes de red externas a APIs públicas (CrossRef, Semantic Scholar, OpenAlex) sin políticas de tolerancia a fallos ni mecanismos de persistencia local auditables.

#### Debilidades y Brechas Encontradas
*   **Inexistencia de Trazabilidad y Auditoría:** Las decisiones de detección generadas por el microservicio no se persisten en una base de datos relacional (ej. PostgreSQL o MySQL) con su procedencia de datos asociada. Los reportes forenses se escriben como archivos HTML crudos en el directorio `/tmp` con una limpieza de disco ineficiente.
*   **Fragilidad en APIs Externas (Sin Circuit Breaker):** La validación bibliográfica bloquea los workers asíncronos si los endpoints de CrossRef o OpenAlex sufren de latencias elevadas o denegaciones de servicio (rate limiting), al no contar con timeouts estrictos o disyuntores (circuit breakers).
*   **Qdrant Inoperante:** La base de datos vectorial solo existe a nivel de configuración de entorno, pero no está integrada para realizar búsquedas rápidas de documentos similares preexistentes, forzando al sistema a depender únicamente de análisis lingüísticos locales.

#### Recomendaciones de Mejora
1.  Reactivar e integrar el pipeline semántico con Qdrant utilizando el modelo de embeddings E5 multilingüe, configurando índices HNSW optimizados y filtros de carga útil para acotar la búsqueda por cliente o dominio.
2.  Implementar un almacén de datos estructurado para guardar un registro auditable de cada inferencia (payload, puntuación cruda, veredicto final, versión exacta de los pesos del transformer y configuración de parámetros).
3.  Añadir un circuit breaker (p. ej., utilizando la librería `pybreaker`) y una caché distribuida de consultas bibliográficas sobre Redis con políticas TTL coherentes.

---

### DIMENSIÓN 6 — Confiabilidad, Sesgo y Consideraciones Éticas
**Puntuación: 3.0 / 10.0**

#### Justificación Técnica
El sistema no proporciona garantías empíricas ni análisis documentados sobre el impacto diferencial de sus veredictos en distintas poblaciones de escritores. La tasa de falsos positivos en textos redactados por hablantes de inglés como segunda lengua (ESL) o autores neurodivergentes es críticamente alta debido a la simplificación de la estructura sintáctica y a las penalizaciones de perplejidad asociadas a estos perfiles de escritura. Asimismo, al no contar con un mecanismo formal de abstención o rechazo para casos con alta incertidumbre, el sistema clasifica de forma determinista textos cortos (< 200 palabras) o documentos con gran desacuerdo entre las semillas del ensemble. Esto vulnera los principios éticos de explicabilidad (XAI) y derecho a una explicación (cumplimiento de GDPR en auditorías académicas o profesionales).

#### Debilidades y Brechas Encontradas
*   **Ausencia de Mecanismo de Abstención:** Ante documentos cortos donde las señales estadísticas carecen de soporte muestral suficiente, el sistema reporta valores de confianza elevados en lugar de abstenerse.
*   **Vulnerabilidad Forense por Reglas de Anulación:** El método [validar_veredicto_segmento](file:///Users/user/Documents/xplagiax_xota/app/engine/detector_final.py#L357) ejecutaba anteriormente la anulación ("override") binaria de la decisión neuronal a "AI (Confirmed) 100%" por una sola cita no encontrada en bases externas. Esto asume de forma falaz que "referencia no encontrada" equivale a "referencia inventada por IA".
*   **Falta de Auditoría de Equidad (Fairness Metrics):** No existe evaluación estadística sobre paridad demográfica, igualdad de oportunidades o métricas de equidad contrafáctica entre diferentes grupos de usuarios.

#### Recomendaciones de Mejora
1.  Implementar un mecanismo de rechazo activo (abstención) basado en dos condiciones: tamaño del texto inferior a 150 palabras u oscilación de desacuerdo entre las semillas del ensemble (`ensemble_disagreement` > 15.0).
2.  Desarrollar y publicar una Ficha de Modelo (Model Card) y una Ficha de Datos (Datasheet) que detallen las limitaciones del sistema, las tasas de falsos positivos por grupo idiomático y las pautas éticas para su aplicación.

---

## 2. TABLA DE CALIFICACIONES (SCOREBOARD)

| Dimensión | Puntuación | Veredicto Técnico |
| :--- | :---: | :--- |
| **1. Metodología y Rigor Científico** | **3.2** | Concepto obsoleto (mundo cerrado de 2023), vulnerable a paráfrasis y con perplejidad simulada. |
| **2. Arquitectura de ML/AI** | **4.2** | Backbone ModernBERT excelente, pero sin calibración probabilística ni fusión real en producción. |
| **3. Cobertura Lingüística** | **5.0** | Buen cálculo estilométrico básico, pero sesgado por diccionarios estáticos de buzzwords anti-ESL. |
| **4. Arquitectura de Microservicio** | **6.0** | Excelente uso de CoW y tareas asíncronas, penalizado por vulnerabilidades críticas de seguridad. |
| **5. Pipeline y Vectorización** | **4.0** | Integración con Qdrant inactiva y falta de trazabilidad en base de datos persistente. |
| **6. Confiabilidad y Ética** | **3.0** | Nulo control de sesgo demográfico, falsos positivos críticos y carencia de mecanismo de abstención. |

---

## 3. PUNTUACIÓN COMPUESTA (COMPOSITE SCORE)

### **Calificación Global: 4.1 / 10.0**

#### Justificación del Peso de los Factores
El cálculo de la nota promedio ponderada se rige bajo la siguiente matriz de priorización científica y operativa:
1.  **Metodología y Rigor Científico (25%)**: Constituye el núcleo epistemológico del detector. Si las señales medidas son simuladas u obsoletas, todo el sistema carece de valor forense.
2.  **Arquitectura de ML/AI (20%)**: Define la robustez estadística y matemática de la inferencia, garantizando la estabilidad frente a desviaciones de distribución.
3.  **Confiabilidad y Ética (15%)**: Un detector comercial no puede ser desplegado si presenta un sesgo que perjudique a minorías o carece de mecanismos de rechazo confiables ante falsos positivos.
4.  **Arquitectura de Microservicio (15%)**: Determina la viabilidad técnica y resiliencia en un entorno de producción con alta carga transaccional.
5.  **Cobertura Lingüística (15%)**: Asegura que el análisis lingüístico no sea trivial ni dependa de heurísticas sesgadas.
6.  **Pipeline y Vectorización (10%)**: Evalúa el soporte de almacenamiento y auditoría del sistema para mejorar a largo plazo.

$$\text{Puntuación Compuesta} = (3.2 \times 0.25) + (4.2 \times 0.20) + (3.0 \times 0.15) + (6.0 \times 0.15) + (5.0 \times 0.15) + (4.0 \times 0.10) = 4.14$$

---

## 4. CRITICAL FINDINGS (TOP 3 SISTÉMICOS)

### 1. Inexistencia de Fusión Real e Incompatibilidad con Modelos Modernos (Cargo-Cult ML)
El veredicto final y el score numérico reportados al usuario en producción provienen exclusivamente de la probabilidad softmax cruda del clasificador ModernBERT entrenado en un entorno cerrado con datos que finalizan en 2023. Las sofisticadas métricas estilométricas, de razonamiento, alucinación y de citas bibliográficas **no contribuyen matemáticamente** al veredicto ni a la confianza final. Al carecer de un clasificador de fusión entrenado y calibrado, el sistema es ciego a los modelos generativos actuales (GPT-5, Gemini 2, DeepSeek-R1) y vulnerable a la manipulación básica de textos.

### 2. Discriminación Sistemática a Escritores ESL e Incoherencia en la Explicabilidad Forense
El microservicio aplica de forma destructiva un léxico estático de "palabras de transición" para penalizar textos con conectores lógicos, lo cual sesga el análisis en contra de hablantes no nativos (ESL) y textos académicos legítimos. Este sesgo demográfico, unido a la falta de calibración probabilística, genera un riesgo de falsos positivos inaceptable en entornos educativos o corporativos, destruyendo la defendibilidad forense del producto en un tribunal o comité académico.

### 3. Deuda Crítica de Seguridad y Fragilidad de Infraestructura
La exposición del microservicio con llaves de cifrado en texto plano (`SECRET_KEY`), el debugger activo en producción (`DEBUG=True`) y comparaciones inseguras de API keys con operadores de desigualdad (`!=`) abren vectores de ataque críticos que comprometen por completo la infraestructura. A esto se suma la incoherencia de un doble Celery worker que compite por la misma cola de Redis y la falta de aislamiento y limpieza en la generación de reportes forenses en `/tmp`.

---

## 5. RESEARCH GAP ANALYSIS (BRECHAS CON EL ESTADO DEL ARTE 2023–2025)

1.  **Detección Zero-Shot Contraste-Perplejidad (Binoculars & Fast-DetectGPT):**
    El estado del arte en detección de texto generado por IA no depende de clasificadores supervisados entrenados en corpus estáticos. El microservicio carece de la implementación de métodos basados en el contraste de perplejidad como **Binoculars** ([Hans et al., 2024](https://arxiv.org/abs/2401.12070)), que utiliza la discrepancia de verosimilitud de tokens bajo dos LLM independientes para clasificar texto sin necesidad de etiquetado supervisado previa.
2.  **Robustez Adversarial contra Parafraseo Semántico (DIPPER / Splicing):**
    Las técnicas actuales de evasión de detectores emplean modelos de parafraseo semántico condicionado (ej. **DIPPER** [Krishna et al., 2023](https://arxiv.org/abs/2303.07208)) o técnicas de mezcla ("splicing") donde fragmentos de IA se intercalan con oraciones humanas. La arquitectura de ModernBERT truncada a 512 tokens en [classify_text](file:///Users/user/Documents/xplagiax_xota/app/engine/detector_final.py#L127) ignora la coherencia global y es incapaz de identificar estos patrones de mezcla semántica discursiva.
3.  **Modelado de Coherencia Discursiva Global mediante Grafos:**
    A diferencia de sistemas que analizan la estructura discursiva a nivel profundo (p. ej., utilizando transiciones de árboles de análisis de la Teoría de la Estructura Retórica - RST), el sistema evalúa regularidades puramente locales. Esto le impide diferenciar la estructura lógica de los LLMs modernos de razonamiento (ej. OpenAI o1/o3, DeepSeek-R1) de los textos redactados por expertos humanos en dominios técnicos.

---

## 6. ROADMAP DE IMPLEMENTACIÓN PRIORIZADO

### **Fase 1: Remediación Crítica de Seguridad y Estabilidad (Esfuerzo: Bajo | Impacto: Crítico)**
1.  **Eliminar la anulación binaria de veredictos y el diccionario de buzzwords:** Modificar [validar_veredicto_segmento](file:///Users/user/Documents/xplagiax_xota/app/engine/detector_final.py#L357) para que las alucinaciones bibliográficas aporten marcas de advertencia forense en el JSON final, pero nunca anulen de forma destructiva o reescriban el score neural al 100%.
2.  **Asegurar la configuración de entorno (12-Factor App):** Configurar `SECRET_KEY` y `API_KEY` como variables obligatorias y forzar un crash controlado si faltan en entornos con `FLASK_ENV=production`. Desactivar el debug de Werkzeug.
3.  **Remediar concurrencia PyTorch/Gunicorn:** Implementar de forma consistente `torch.set_num_threads(1)` en la inicialización global del microservicio y corregir la ejecución redundante del Celery worker.

### **Fase 2: Calibración y Fusión de Modelos (Esfuerzo: Medio | Impacto: Alto)**
1.  **Entrenar y Activar el FusionClassifier:** Compilar un conjunto de validación balanceado que contenga textos humanos y generados por modelos actuales (2024-2026). Entrenar un meta-clasificador logístico regularizado en [fusion.py](file:///Users/user/Documents/xplagiax_xota/app/engine/fusion.py) y calibrar sus salidas probabilísticas mediante Platt Scaling.
2.  **Actualizar el Motor de Perplejidad (Tier 2):** Sustituir el modelo obsoleto GPT-2 por un LM ligero moderno (p. ej., Llama-3-8B-Instruct o Gemma-2-2B) cargado localmente con cuantización float16/int8 para calcular valores de entropía de tokens reales.
3.  **Establecer un Filtro de Abstención:** Configurar una guarda en los endpoints de la API para retornar "Inconclusive" con baja confianza estadística cuando la longitud del texto sea inferior a 150 palabras o cuando la desviación estándar del ensemble supere un umbral crítico.

### **Fase 3: Integración de Datos Vectoriales y Observabilidad (Esfuerzo: Alto | Impacto: Medio-Alto)**
1.  **Activar PlagiarismEngine con Qdrant:** Implementar la indexación vectorial y la búsqueda de similitud semántica con Qdrant utilizando el modelo multilingüe E5, permitiendo identificar secciones plagiadas por paráfrasis mediante búsquedas top-K en milisegundos.
2.  **Persistencia Estructurada e Historial de Inferencia:** Configurar el esquema de persistencia en una base de datos relacional (ej. MySQL/PostgreSQL) para registrar y auditar las decisiones del sistema.
3.  **Añadir Observabilidad de Nivel de Producción:** Integrar métricas compatibles con Prometheus (`/metrics`) para rastrear la latencia por plugin, tasas de error de inferencia y la tasa de aciertos de la caché compartida de Redis.

---

## 7. RECOMENDACIÓN DE BENCHMARKS PARA VALIDACIÓN EXPERIMENTAL

Para validar cuantitativamente las mejoras del sistema y mitigar los falsos positivos antes del despliegue en producción, se recomienda ejecutar pruebas sistemáticas utilizando los siguientes conjuntos de datos públicos de referencia:

1.  **RAID (Robust Adversarial AI Detector Benchmark) [Duan et al., 2024]:**
    Esencial para evaluar la robustez del detector frente a ataques de evasión sofisticados, técnicas de parafraseo semántico y modificaciones de estilo de escritura.
2.  **HC3 (Human ChatGPT Comparison Corpus) [Guo et al., 2023]:**
    Indispensable para calibrar la tasa de falsos positivos en dominios académicos y de preguntas/respuestas formales, sirviendo para verificar que los escritores humanos no sean erróneamente clasificados como IA.
3.  **M4 Dataset (Multi-generator, Multi-domain, Multi-lingual, Multi-write) [Wang et al., 2024]:**
    Permite validar el comportamiento multilingüe y multi-modelo del clasificador de fusión frente a motores diversos (Cohere, GPT-4, Claude, Llama, Mistral).
4.  **TuringBench [Uchendu et al., 2021]:**
    Excelente corpus para la tarea específica de atribución de autoría entre múltiples arquitecturas generativas (Closed y Open Source).
5.  **MAGE (Machine-Generated Text Evaluation):**
    Útil para medir la deriva del detector frente a textos híbridos (co-escritura humana y de inteligencia artificial) y validar la resolución del HybridSegmentAnalyzer.
