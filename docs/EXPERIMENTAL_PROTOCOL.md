# Protocolo experimental reproducible — Detector XplagiaX

> Estado: **diseño listo para ejecutar**. La Fase 2 entrega el *andamiaje* (ensamblado de
> vectores en `app/engine/fusion.py`, calibración en `app/engine/calibration.py`). Este
> documento define cómo entrenar/calibrar/validar cuando exista el corpus etiquetado.
> Hasta entonces, el veredicto de producción **no** cambia y la confianza se reporta como
> **no calibrada**.

---

## 0. Objetivo

Producir un detector con (a) confianza **calibrada** (no `max(softmax)`), (b) una **fusión
tardía real** de los plugins (hoy decorativos), y (c) una evaluación publicable con métricas,
intervalos de confianza y análisis de sesgo.

## 1. Datos

### 1.1 Corpus
- **Públicos:** RAID, M4/M4GT, MAGE, HC3, GPABenchmark.
- **Propio (imprescindible):** texto 2025-2026 de **GPT-5, Claude 4, Gemini 2, DeepSeek-R1**
  + humano **nativo** y **ESL** + dominios **legal/académico/técnico/periodístico**.
- **Adversarial:** versiones parafraseadas (DIPPER) y "humanizadas" de los textos IA.

### 1.2 Etiquetas y metadatos
Cada documento: `label ∈ {human, ai}`, `model`, `domain`, `native_speaker ∈ {0,1}`,
`length_bucket`, `year`, `adversarial ∈ {none, paraphrase, humanizer}`.

### 1.3 Splits (anti-fuga)
- Estratificado por `model × domain × length_bucket`.
- **Held-out temporal:** entrenar con `year ≤ 2024`, **testear con `year ∈ {2025,2026}`**
  para medir *drift* (el fallo principal del detector actual).
- Validación separada para calibración (no reusar test).

## 2. Pipeline experimental

1. **Extracción de features de fusión.** Para cada documento, ejecutar el orquestador y
   `FusionFeatureBuilder.build(detection_result, additional_analyses)` →
   vector de `FUSION_VECTOR_DIM` (ver `fusion.FEATURE_NAMES`). Guardar `X`, `y`, metadatos.
2. **Entrenamiento de fusión.** `FusionClassifier.fit(X_train, y_train)` (logística
   balanceada; estandarizada). Alternativas a comparar: XGBoost/LightGBM.
3. **Calibración.** Sobre el set de **validación**: `TemperatureScaler().fit(val_probs, val_labels)`;
   adjuntar con `FusionClassifier.attach_calibrator(ts)`. Reportar curva de fiabilidad.
4. **Evaluación en test** (incluido el held-out temporal y el adversarial).

## 3. Métricas (todas con IC95% bootstrap, ≥1000 resamples)

| Métrica | Por qué |
|--------|--------|
| Precision, Recall, **F1** | desempeño base |
| **ROC-AUC**, **PR-AUC** | ranking; PR-AUC ante desbalance |
| **MCC** | robusta a desbalance |
| **ECE** (`calibration.compute_ece`) + curva de fiabilidad | honestidad de la confianza |
| **Brier** (`calibration.brier_score`) | calidad probabilística |
| **TPR @ FPR=1%** | *la* métrica cuando un FP daña a un estudiante |
| **Equalized odds** por subgrupo | sesgo nativo/ESL y por dominio |

## 4. Ablation (demostrar/refutar que la fusión aporta)
Entrenar y evaluar quitando, de a uno, cada grupo de features (`neural_*`, `ppl_*`, `rsn_*`,
`hal_*`, `hyb_*`, `ref_*`, `sty_*`). Reportar ΔF1 y ΔTPR@FPR=1% por grupo. Esto cuantifica el
aporte real de cada plugin (hoy decorativo).

## 5. Análisis de sesgo
- Comparar FPR entre **nativo vs ESL** y entre dominios; test de significancia.
- Reportar la brecha de FPR como métrica de primera clase (criterio de aceptación: FPR_ESL ≈ FPR_nativo).

## 6. Robustez adversarial
Medir caída de Recall sobre el subconjunto `adversarial ∈ {paraphrase, humanizer}` vs `none`.

## 7. Criterios de aceptación para activar la fusión en producción
- ECE_test ≤ 0.05 tras calibración.
- TPR@FPR=1% mejora vs el clasificador neural solo (ablation `solo neural`).
- Brecha de FPR ESL−nativo ≤ 2 puntos.
- Sin regresión de F1 en el held-out temporal 2025-2026.

## 8. Reproducibilidad
Fijar semillas; registrar versiones (`requirements.txt`), `MODEL_VERSION`, hashes de datos;
publicar datasheet, model card, código y splits. Reportar hardware y tiempos.

## 9. Mapa código ↔ protocolo
| Paso | Código |
|------|--------|
| Vector de fusión | `app/engine/fusion.py` (`FusionFeatureBuilder`, `FEATURE_NAMES`) |
| Fusión heurística (interina, model-agnóstica) | `app/engine/fusion.py` (`heuristic_fusion`, `FusionClassifier(untrained_mode="heuristic")`) |
| Meta-clasificador entrenado | `app/engine/fusion.py` (`FusionClassifier.fit/predict_proba`) |
| Desacuerdo del ensemble (incertidumbre) | `app/engine/detector_final.py` (`_classify_batch_from_ids`, `DetectionResult.ensemble_disagreement`) |
| Temperature scaling | `app/engine/calibration.py` (`TemperatureScaler`) |
| ECE / fiabilidad / Brier | `app/engine/calibration.py` (`compute_ece`, `reliability_bins`, `brier_score`) |
| Consistencia de autoría | `app/plugins/author_signature.py` |

---

## 10. Reentrenamiento contra modelos de frontera 2025-2026 (Ruta A)

> **Por qué es obligatorio.** El núcleo neural (ModernBERT, etiquetas entrenadas en 2023) es
> ciego a las distribuciones de GPT-5 / Claude 4 / Gemini 2 / DeepSeek-R1 que **nunca vio**.
> Esto es *out-of-distribution*: **ningún cambio de código lo resuelve**. La fusión
> model-agnóstica activada (Ruta B: citas fabricadas, hedging, desacuerdo del ensemble,
> perplejidad) aporta *algo* de lift y elimina la dependencia de un único modelo de 2023,
> pero **no sustituye** el reentrenamiento. El gap de falsos negativos sólo se cierra aquí.

### 10.1 Datos (frontera)
- Generar pares humano/IA por dominio (académico, legal, periodístico, técnico, ESL) con
  **GPT-5, Claude 4.x, Gemini 2.x, DeepSeek-R1/V3, Llama-4, Qwen-3**, incluyendo variantes
  con *paraphrasers/humanizadores* (Undetectable, QuillBot) y mezclas humano+IA.
- ≥ 50k documentos balanceados; **split temporal**: train ≤ 2024-Q4, test = 2025-2026
  (held-out temporal para medir generalización a modelos no vistos).
- Datasets públicos de arranque: RAID, M4/M4GT, MAGE, HC3, GhostBuster — completar con
  generación propia 2025-2026 (los públicos envejecen rápido).

### 10.2 Receta de fine-tuning
- **Base recomendada**: re-fine-tune del propio ModernBERT-large multietiqueta, o un
  encoder más reciente, con cabeza binaria humano/IA + cabeza multiclase de atribución.
- LoRA/QLoRA (r=16, α=32) para iterar barato; full-FT sólo si hay presupuesto GPU.
- `class_weight` balanceado; *focal loss* para clases-modelo minoritarias.
- **Augmentations anti-FP**: incluir texto humano de no-nativos, técnico y editado, y
  *adversarial* (paráfrasis de texto humano) etiquetado HUMANO.
- Early stopping sobre el held-out **temporal**, no sobre un split aleatorio.

### 10.3 Calibración + fusión entrenada (cierra el bucle con §1-§7)
1. Congelar el encoder; ajustar `TemperatureScaler` en el set de validación (ECE ≤ 0.05).
2. `FusionClassifier.fit(X, y)` con el corpus etiquetado → reemplaza `heuristic_fusion`
   (de `source="heuristic_fusion"` a `source="logreg"`, `calibrated=True`).
3. Reentrenar trimestralmente con texto nuevo de frontera (los modelos evolucionan).

### 10.4 Despliegue
- Bump de `MODEL_VERSION` (invalida el caché namespaced — §C-17) en cada swap de pesos.
- Promover a producción **sólo** si se cumplen los criterios de aceptación de §7.
