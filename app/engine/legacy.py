"""
app/engine/legacy.py — Deprecated code kept OUT of the hot path ([C9]).

These chunked-document analyzers predate analyze_fast() (single tokenization
pass, adaptive max_tokens, TTL result cache) and are 2-12x slower. They are
preserved here verbatim — deprecation warnings included — for callers that have
not migrated yet, so detector_final.py stays focused on the production path.

Do NOT add new callers. Use detector_final.analyze_fast() or
PluginOrchestrator.run() instead.
"""

import logging
import re
import warnings

from detector_final import classify_batch, tokenizer

logger = logging.getLogger(__name__)


def validar_veredicto_segmento(segmento_dict: dict) -> dict:
    """
    Analiza el forensic_analysis de un segmento para confirmar o descartar
    si realmente es IA.

    Cambios v1.3 (BUG FIX):
    ─────────────────────────────────────────────────────────────────────
    BUG #1 CORREGIDO: El criterio de confirmación por alucinación
    bibliográfica era incondicional: cualquier fabricated_count >= 1
    sobreescribía el veredicto a "AI (Confirmed) 100%", incluso cuando
    el extractor de referencias había capturado texto estructural
    (headers de sección, alt-text de imágenes) como si fueran citas.

    La corrección agrega tres condiciones de guardia:
      1. total_references >= 2   → descartar extracciones únicas/espurias
      2. fabricated_ratio >= 0.70 → al menos 70% de las refs son inválidas
      3. El modelo base YA marcaba IA (label "AI" y score > 50)

    Esto elimina los falsos positivos sin afectar los verdaderos positivos
    (textos con múltiples citas inventadas confirmadas).
    """
    analisis = segmento_dict.get("forensic_analysis")
    if not analisis or "error_forense" in analisis:
        return segmento_dict

    razonamiento = analisis.get("reasoning", {})
    perplejidad  = analisis.get("perplexity", {})
    referencias  = analisis.get("reference_check", {})

    # [C-04 FIX] This routine used to OVERRIDE the neural verdict with hard rules:
    # it flipped `score = 100 - score` / relabelled to "Human (Validated)", and forced
    # `score = 100.0` / "AI (Confirmed)" on a single fabricated-citation signal. That
    # is indefensible forensically (a weak heuristic overriding the model at 100%
    # confidence). It is now NON-DESTRUCTIVE: it never changes `dominant_label` or
    # `score`; it only attaches advisory annotations under `forensic_flags` that the
    # report can surface as *supporting evidence*, not as a verdict.
    flags = segmento_dict.setdefault("forensic_flags", [])

    # ── 1. SOPORTE "HUMANO": ausencia de señales forenses de IA ───────────
    base_score = segmento_dict.get("score", 0.0)
    base_label = segmento_dict.get("dominant_label", "")
    high_confidence_ai = "AI" in base_label and base_score >= 85.0
    if (
        razonamiento.get("ai_score", 0) < 0.25
        and perplejidad.get("ai_score", 0) < 0.40
        and not high_confidence_ai
    ):
        flags.append({
            "type": "human_supporting",
            "note": "Los plugins forenses no muestran señales de IA (razonamiento y "
                    "perplejidad bajos). Soporte débil de autoría humana — no concluyente.",
        })

    # ── 2. SOPORTE "IA": alucinaciones bibliográficas ─────────────────────
    # Guardas para evitar falsos positivos por extracción espuria de texto
    # estructural como referencias. Sigue siendo *evidencia*, no veredicto.
    feat_vals   = referencias.get("feature_values", {})
    fab_count   = feat_vals.get("fabricated_count", 0)
    total_refs  = feat_vals.get("total_references", 0)
    fab_ratio   = feat_vals.get("fabricated_ratio", 0.0)

    if (
        fab_count > 0
        and total_refs >= 2          # guardia (a): más de una referencia
        and fab_ratio >= 0.70        # guardia (b): mayoría de citas inválidas
        and "AI" in base_label       # guardia (c): el modelo base ya sospechaba IA
        and base_score > 50.0        # guardia (c): confianza mínima en IA
    ):
        flags.append({
            "type": "ai_supporting",
            "note": "Se detectaron citas bibliográficas no verificables en múltiples "
                    "referencias. Evidencia de soporte de generación por IA — no concluyente.",
        })

    return segmento_dict


def analyze_long_document(long_text: str, orchestrator=None, max_tokens: int = 512) -> dict:
    """
    Analiza un documento completo con segmentación semántica y validación forense.

    .. deprecated::
        Use analyze_fast() for speed-optimized inference without forensic overlay,
        or PluginOrchestrator.run() for the full forensic pipeline.
    """
    warnings.warn(
        "analyze_long_document() is deprecated. Use analyze_fast() or PluginOrchestrator.run().",
        DeprecationWarning,
        stacklevel=2,
    )
    if not long_text.strip():
        return {"error": "El documento está vacío."}

    # 1. División semántica
    if "\n" in long_text:
        raw_fragments = [p.strip() for p in re.split(r'\n+', long_text) if p.strip()]
    else:
        raw_fragments = [p.strip() + "." for p in re.split(r'(?<=\.)\s+', long_text) if p.strip()]

    chunks_text = []
    current_chunk = ""
    current_length = 0

    for fragment in raw_fragments:
        fragment_tokens = len(tokenizer.encode(fragment, add_special_tokens=False))
        if current_length + fragment_tokens > max_tokens and current_chunk:
            chunks_text.append(current_chunk.strip())
            current_chunk = fragment + " "
            current_length = fragment_tokens
        else:
            current_chunk += fragment + " "
            current_length += fragment_tokens

    if current_chunk.strip():
        chunks_text.append(current_chunk.strip())

    # Fusión de fragmentos cortos al final
    if len(chunks_text) > 1:
        last_tokens = len(tokenizer.encode(chunks_text[-1], add_special_tokens=False))
        if last_tokens < 50:
            chunks_text[-2] += " " + chunks_text.pop()

    results = {"overall_summary": {}, "segments": []}
    total_human_weighted = 0.0
    total_ai_weighted = 0.0
    total_tokens_processed = 0
    
    logger.debug("Iniciando análisis forense de %d segmentos...", len(chunks_text))

    # 2. Procesamiento de Segmentos por Lotes
    BATCH_SIZE = 8
    for i in range(0, len(chunks_text), BATCH_SIZE):
        batch_slice = chunks_text[i:i + BATCH_SIZE]
        batch_results = classify_batch(batch_slice)
        
        for sub_idx, (human_pct, ai_pct) in enumerate(batch_results):
            idx = i + sub_idx
            chunk_text = batch_slice[sub_idx]
            
            dominant_label = "AI" if ai_pct > human_pct else "Human"
            forensic_data = None
            
            # Ejecución de Plugins (Individual por segmento)
            if orchestrator and dominant_label == "AI":
                try:
                    analisis_full = orchestrator.run(chunk_text)
                    forensic_data = analisis_full.get("additional_analyses", {})
                except Exception as e:
                    forensic_data = {"error_forense": str(e)}

            segmento_obj = {
                "segment_id": idx + 1,
                "text": chunk_text,
                "dominant_label": dominant_label,
                "score": round(max(ai_pct, human_pct)),
                "forensic_analysis": forensic_data,
                "status_note": None
            }

            segmento_obj = validar_veredicto_segmento(segmento_obj)
            results["segments"].append(segmento_obj)
            
            # Actualización de pesos
            final_ai_score = ai_pct if "AI" in segmento_obj["dominant_label"] else (100 - human_pct)
            final_human_score = 100 - final_ai_score

            chunk_len = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            total_human_weighted += (final_human_score * chunk_len)
            total_ai_weighted += (final_ai_score * chunk_len)
            total_tokens_processed += chunk_len

    # 5. Resumen Final
    if total_tokens_processed == 0:
        return {"error": "No se pudieron procesar tokens del documento."}
    overall_human = round(total_human_weighted / total_tokens_processed)
    overall_ai = round(total_ai_weighted / total_tokens_processed)

    results["overall_summary"] = {
        "total_human_percentage": overall_human,
        "total_ai_percentage": overall_ai,
        "overall_prediction": "AI" if overall_ai > overall_human else "Human"
    }

    return results


def analyze_long_documentsd_(long_text: str, max_tokens: int = 512) -> dict:
    """
    Analiza un documento dividiéndolo de forma inteligente por párrafos u oraciones,
    evitando cortar palabras por la mitad y agrupando fragmentos cortos.

    .. deprecated::
        Use analyze_fast() — single tokenization pass, adaptive max_tokens,
        no decode/re-encode round-trip, 2-12x faster on long documents.
    """
    warnings.warn(
        "analyze_long_documentsd_() is deprecated. Use analyze_fast() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not long_text.strip():
        return {"error": "El documento está vacío."}

    # 1. Dividir el texto en párrafos usando saltos de línea
    # Si no hay saltos de línea, dividimos por puntos (oraciones)
    if "\n" in long_text:
        raw_fragments = [p.strip() for p in re.split(r'\n+', long_text) if p.strip()]
    else:
        raw_fragments = [p.strip() + "." for p in re.split(r'(?<=\.)\s+', long_text) if p.strip()]

    chunks_text = []
    current_chunk = ""
    current_length = 0

    # 2. Agrupar fragmentos inteligentemente respetando el max_tokens
    for fragment in raw_fragments:
        # Medir cuántos tokens tiene este fragmento
        fragment_tokens = len(tokenizer.encode(fragment, add_special_tokens=False))
        
        # Si el fragmento en sí mismo es más grande que el límite (caso raro),
        # lo forzamos a entrar, pero al menos no cortamos los demás.
        if current_length + fragment_tokens > max_tokens and current_chunk:
            chunks_text.append(current_chunk.strip())
            current_chunk = fragment + " "
            current_length = fragment_tokens
        else:
            current_chunk += fragment + " "
            current_length += fragment_tokens

    # Agregar el último chunk que quedó en el buffer
    if current_chunk.strip():
        chunks_text.append(current_chunk.strip())

    # 3. Fusión del fragmento huérfano (para evitar caídas de precisión)
    # Si el último chunk tiene muy pocos tokens (ej. menos de 50) y hay más de un chunk,
    # lo fusionamos con el chunk anterior para darle contexto.
    if len(chunks_text) > 1:
        last_chunk_tokens = len(tokenizer.encode(chunks_text[-1], add_special_tokens=False))
        if last_chunk_tokens < 50:
            fragment_to_merge = chunks_text.pop()
            chunks_text[-1] += " " + fragment_to_merge

    results = {"overall_summary": {}, "segments": []}
    
    total_human_weighted = 0.0
    total_ai_weighted = 0.0
    total_tokens_processed = 0
    
    logger.debug("Iniciando análisis semántico de %d segmentos...", len(chunks_text))

    # 4. Procesar por lotes
    BATCH_SIZE = 8
    for i in range(0, len(chunks_text), BATCH_SIZE):
        batch_slice = chunks_text[i:i + BATCH_SIZE]
        batch_results = classify_batch(batch_slice)
        
        for sub_idx, (human_pct, ai_pct) in enumerate(batch_results):
            idx = i + sub_idx
            chunk_text = batch_slice[sub_idx]
            
            dominant_label = "AI" if ai_pct > human_pct else "Human"
            dominant_score = max(ai_pct, human_pct)
                
            results["segments"].append({
                "segment_id": idx + 1,
                "text": chunk_text,
                "dominant_label": dominant_label,
                "score": dominant_score
            })
            
            chunk_length = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            total_human_weighted += (human_pct * chunk_length)
            total_ai_weighted += (ai_pct * chunk_length)
            total_tokens_processed += chunk_length

    # 5. Cálculo final
    if total_tokens_processed == 0:
        return {"error": "No se pudieron procesar tokens del documento."}
    overall_human = round(total_human_weighted / total_tokens_processed)
    overall_ai = round(total_ai_weighted / total_tokens_processed)

    results["overall_summary"] = {
        "total_human_percentage": overall_human,
        "total_ai_percentage": overall_ai,
        "overall_prediction": "AI" if overall_ai > overall_human else "Human"
    }

    return results


