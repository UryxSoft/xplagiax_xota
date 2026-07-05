# ESTIMACIÓN DE IMPACTO DE LAS MEJORAS PROPUESTAS

Si se implementaran las propuestas descritas (manteniendo fijos los 3 modelos base ModernBERT y el cabezal de clasificación de 41 clases), se observaría una mejora notable en la confiabilidad, la tasa de falsos positivos y la resiliencia operativa.

---

## 1. IMPACTO EN LA CALIFICACIÓN GLOBAL

### **Nueva Calificación Global Estimada: 7.1 / 10.0** *(Antes: 4.1)*

El sistema superaría la barrera del "cargo-cult ML" para convertirse en un microservicio de rango de producción respetable. No alcanza una nota superior (8.5+) porque la base neural sigue estando limitada a los datos de entrenamiento de 2023.

### Proyección de Puntuaciones por Dimensión:

| Dimensión | Nota Actual | Nota Estimada | Factor Clave del Incremento |
| :--- | :---: | :---: | :--- |
| **1. Rigor Científico** | 3.2 | **6.0** | Fusión tardía real basada en ML que integra perplejidad y razonamiento; deja de ser decorativa. |
| **2. Arquitectura ML/AI** | 4.2 | **7.5** | Calibración probabilística (Platt/Isotonic) y corrección de la doble inferencia síncrona. |
| **3. Cobertura Lingüística** | 5.0 | **7.0** | Remoción completa del sesgo anti-ESL por buzzwords y normalización por longitud. |
| **4. Arquitectura de Microservicio** | 6.0 | **8.5** | Cierre de vulnerabilidades de seguridad (`SECRET_KEY`, `DEBUG`), y fin de la redundancia de Celery. |
| **5. Pipeline y Vectorización** | 4.0 | **7.5** | Reactivación de Qdrant para plagio por paráfrasis y base de datos relacional para auditoría. |
| **6. Confiabilidad y Ética** | 3.0 | **7.0** | Filtro de abstención activo (<150 palabras) y mitigación sistemática de falsos positivos. |

$$\text{Nueva Puntuación Compuesta} = (6.0 \times 0.25) + (7.5 \times 0.20) + (7.0 \times 0.15) + (8.5 \times 0.15) + (7.0 \times 0.15) + (7.5 \times 0.10) = 7.12$$

---

## 2. IMPACTO EN LA PRECISIÓN DEL DETECTOR (AI VS HUMAN)

### **Métrica 1: Precisión Global (Precision = TP / (TP + FP))**
*   **Actual:** ~80% - 84% (en distribución de modelos de 2023). Cae drásticamente a < 50% al evaluar textos de autores no nativos (ESL) debido a falsos positivos masivos.
*   **Estimada tras mejoras:** **92% - 95%**. 
*   **Razón:** Al remover el diccionario estático de buzzwords e implementar calibración probabilística, el modelo deja de clasificar como "IA" textos humanos altamente formales o con sintaxis regularizada.

### **Métrica 2: Tasa de Falsos Positivos (FPR - Humanos marcados como IA)**
*   **Actual:** ~15% - 20% promedio (pico de >35% en textos académicos/técnicos y autores ESL).
*   **Estimada tras mejoras:** **< 1% - 2%**.
*   **Razón:** La calibración post-hoc permite ajustar el umbral de decisión para priorizar la especificidad (minimizar FPs). Además, la política de abstención marcará los casos dudosos o cortos como "Inconclusivos" en lugar de forzar un veredicto falso.

### **Métrica 3: Sensibilidad / Tasa de Verdaderos Positivos (Recall - IA detectada)**
*   **Frente a LLMs ≤ 2023 (dentro de distribución):** Se mantendría estable en **~85% - 88%**.
*   **Frente a LLMs post-2024 (OOD - GPT-5, Gemini 2, Claude 3.5, DeepSeek-R1):**
    *   *Actual:* ~30% (el modelo base ModernBERT no reconoce estas firmas y predice "Human").
    *   *Estimada tras mejoras:** **~60% - 70%**.
    *   *Razón:** Aunque ModernBERT falle al no conocer el modelo específico, el meta-clasificador capturará las señales lingüísticas universales procesadas por los plugins (desviación de la perplejidad real bajo un LM de referencia actualizado, la alta densidad en marcadores discursivos y la uniformidad discursiva general).
