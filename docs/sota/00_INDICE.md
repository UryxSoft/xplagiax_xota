# Hoja de ruta SOTA — Índice

Guías paso a paso para llevar los plugins de XplagiaX a nivel estado-del-arte.
Cada guía es autocontenida: prerrequisitos, pasos numerados, puntos de
verificación ("✅ Checkpoint") y errores típicos a evitar.

| # | Guía | Esfuerzo | Requiere GPU/Colab | Requiere corpus |
|---|------|----------|--------------------|-----------------|
| A | [Fusion entrenada + calibración + conformal](A_FUSION_ENTRENADA.md) | 2-4 semanas (domina el corpus) | Solo para generar la clase IA | **Sí** |
| B | [Binoculars — perplexity SOTA](B_BINOCULARS.md) | 1-2 semanas | Recomendado (Colab sirve) | No (solo para umbral) |
| C | [Reference check con GROBID](C_REFERENCE_CHECK.md) | 1 semana | No | No |
| D | [Author signature con embeddings](D_AUTHOR_SIGNATURE.md) | 1-2 semanas | Recomendado (Colab sirve) | Opcional (plan B) |
| E | [Suite adversarial + métrica TPR@FPR](E_SUITE_ADVERSARIAL.md) | transversal | Sí para DIPPER (Colab) | Sí (reusa el de A) |

## Orden de ejecución recomendado

```
C (evidencia inmediata, sin ML)
→ B (mejor detección sin corpus)
→ A (corpus + entrenamiento + calibración: la credibilidad científica)
→ D (sustituye features a mano por embeddings)
→ E (corre desde que exista el corpus de A; es el examen final)
```

Cada mejora termina integrándose como feature del vector de fusion
(`app/engine/fusion.py`, `_FUSION_SCHEMA`), así que **A se re-entrena al
final** con B, C y D ya integrados.

## Regla de oro global

Nunca reportar "accuracy". La métrica de este dominio es **TPR@FPR=1%**
(cuántos textos IA detectas cuando solo aceptas equivocarte con 1 de cada
100 humanos). Un falso positivo = acusar falsamente a un estudiante.
