# ── torch + device must be defined FIRST so _load_model() can always
# reference the global `device`, even if a later import fails. ──────
import torch
import os
import warnings
import logging

# Inference-only service — disable autograd globally to eliminate gradient
# tensor allocation overhead on every forward pass (~50-150 MB per worker).
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

# [C-12 FIX] Cap torch intra-op threads to avoid CPU over-subscription.
# Concurrency already comes from gunicorn gthread workers × the plugin
# ThreadPoolExecutor (plugin_registry.py) × per-document batching. Letting torch
# also spin up one BLAS thread per core multiplies into cores² runnable threads,
# causing context-switch thrashing and p99 latency blow-ups on CPU. Default 1;
# override with TORCH_NUM_THREADS when running a single-request, latency-bound box.
if device.type == "cpu":
    try:
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    except (ValueError, RuntimeError) as _thr_err:
        logger.warning("Could not set torch num_threads: %s", _thr_err)
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tokenizers.normalizers import Sequence, Replace, Strip
from tokenizers import Regex
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model1_path = os.path.join(_BASE_DIR, "modernbert.bin")
model2_path = os.path.join(_BASE_DIR, "Model_groups_3class_seed12")
model3_path = os.path.join(_BASE_DIR, "Model_groups_3class_seed22")
#model4_path = os.path.join(_BASE_DIR, "Model_groups_41class_seed44__new")

# ── Config + tokenizer: use local cache only to avoid network failures ──
from transformers import AutoConfig
_config = AutoConfig.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=41, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base", local_files_only=True
)

# ── Helper: arquitectura vacía + pesos locales, 0 descargas ──
def _load_model(weight_path):
    m = AutoModelForSequenceClassification.from_config(_config)
    m.load_state_dict(torch.load(weight_path, map_location=device))
    m.to(device).eval()
    # Pin tensors in POSIX shared memory so forked Gunicorn/Celery workers read
    # the same physical pages without triggering Copy-on-Write faults.
    # Without this, the first inference in each worker copies all ~570 MB of
    # model weights into private pages, multiplying RSS by the worker count.
    if device.type == "cpu":
        try:
            m.share_memory()
        except Exception as _shm_err:
            # /dev/shm too small (Docker default 64 MB). Model stays in anon
            # CoW memory — perfectly fine for gthread workers and preload_app.
            logger.debug("share_memory() skipped: %s", _shm_err)
    return m

model_1 = _load_model(model1_path)
model_2 = _load_model(model2_path)
model_3 = _load_model(model3_path)
#model_4 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
#model_4.load_state_dict(torch.hub.load_state_dict_from_url(model4_path, map_location=device))
#model_4.to(device).eval()


label_mapping = {
    0: "13B", 1: "30B", 2: "65B", 3: "7B", 4: "GLM130B", 5: "bloom_7b",
    6: "bloomz", 7: "cohere", 8: "davinci", 9: "dolly", 10: "dolly-v2-12b",
    11: "flan_t5_base", 12: "flan_t5_large", 13: "flan_t5_small",
    14: "flan_t5_xl", 15: "flan_t5_xxl", 16: "gemma-7b-it", 17: "gemma2-9b-it",
    18: "gpt-3.5-turbo", 19: "gpt-35", 20: "gpt4", 21: "gpt4o",
    22: "gpt_j", 23: "gpt_neox", 24: "human", 25: "llama3-70b", 26: "llama3-8b",
    27: "mixtral-8x7b", 28: "opt_1.3b", 29: "opt_125m", 30: "opt_13b",
    31: "opt_2.7b", 32: "opt_30b", 33: "opt_350m", 34: "opt_6.7b",
    35: "opt_iml_30b", 36: "opt_iml_max_1.3b", 37: "t0_11b", 38: "t0_3b",
    39: "text-davinci-002", 40: "text-davinci-003"
}


# [ADDED v1.1] DetectionResult — structured output consumed by
# ForensicReportGenerator and PluginOrchestrator.

@dataclass
class DetectionResult:
    prediction:           str
    confidence:           float
    human_percentage:     float
    ai_percentage:        float
    detected_model:       Optional[str]
    raw_scores:           Dict[str, float]
    statistical_features: Dict[str, float] = field(default_factory=dict)
    uncertainty_zone:     bool = False
    ensemble_disagreement: float = 0.0   # std (pct points) of per-seed AI prob; higher = less certain


def clean_text(text: str) -> str:
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.;:?!])', r'\1', text)
    return text

newline_to_space = Replace(Regex(r'\s*\n\s*'), " ")
join_hyphen_break = Replace(Regex(r'(\w+)[--]\s*\n\s*(\w+)'), r"\1\2")
tokenizer.backend_tokenizer.normalizer = Sequence([
    tokenizer.backend_tokenizer.normalizer,
    join_hyphen_break,
    newline_to_space,
    Strip()
])


# [MODIFIED v1.1] Returns 3-tuple: (result_message, fig, DetectionResult).
# All inference logic is IDENTICAL to the original.

def classify_text(text, generate_plot: bool = False):
    """
    Classifies the text and (optionally) generates a plot of human vs AI probability.
    Returns (result_message, fig, DetectionResult).

    [C-01 FIX] generate_plot defaults to False. The pyplot global state is NOT
    thread-safe, and this function is reached from the ThreadPoolExecutor that runs
    plugins (plugin_registry.py). Building a figure via plt.* under concurrency can
    corrupt state or crash. The API path discards `fig`, so by default we skip it.
    Set generate_plot=True only from single-threaded callers (e.g. Gradio).
    """
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        empty_result = DetectionResult(
            prediction="Unknown",
            confidence=0,
            human_percentage=50,
            ai_percentage=50,
            detected_model=None,
            raw_scores={"human": 0.0, "ai": 0.0},
            uncertainty_zone=True,
        )
        return "", None, empty_result

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits_1 = model_1(**inputs).logits
        logits_2 = model_2(**inputs).logits
        logits_3 = model_3(**inputs).logits

        softmax_1 = torch.softmax(logits_1, dim=1)
        softmax_2 = torch.softmax(logits_2, dim=1)
        softmax_3 = torch.softmax(logits_3, dim=1)

        averaged_probabilities = (softmax_1 + softmax_2 + softmax_3) / 3
        probabilities = averaged_probabilities[0]

    human_prob = probabilities[24].item()
    ai_probs_clone = probabilities.clone()
    ai_probs_clone[24] = 0
    ai_total_prob = ai_probs_clone.sum().item()

    total_decision_prob = human_prob + ai_total_prob
    human_percentage = (human_prob / total_decision_prob) * 100
    ai_percentage = (ai_total_prob / total_decision_prob) * 100

    ai_argmax_index = torch.argmax(ai_probs_clone).item()
    ai_argmax_model = label_mapping[ai_argmax_index]

    if human_percentage > ai_percentage:
        result_message = (
            f"**The text is** <span class='highlight-human'>**{human_percentage:.2f}%** likely <b>Human written</b>.</span>"
        )
    else:
        result_message = (
            f"**The text is** <span class='highlight-ai'>**{ai_percentage:.2f}%** likely <b>AI generated</b>.</span>\n\n"
        )

    # [C-02 FIX] Keep precise percentage raw_scores BEFORE rounding the display values.
    # human_prob + ai_total_prob == 1.0 (softmax), so these are the real model scores
    # on a [0,100] scale. Exposing the "ai" key fixes the dead-code path in
    # forensic_reports.generate_report (which read a non-existent "ai" key) and the
    # summary() display that printed 0.0/1.0 from round(prob).
    raw_scores = {
        "human": round(human_prob * 100, 2),
        "ai": round(ai_total_prob * 100, 2),
    }

    # [C-01 FIX] Only touch pyplot when explicitly requested by a single-threaded caller.
    fig = None
    if generate_plot and plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figure size for better layout

        categories = ['Human', 'AI']
        probabilities_for_plot = [human_percentage, ai_percentage]

        bars = ax.bar(categories, probabilities_for_plot, color=['#4CAF50', '#FF5733'], alpha=0.8)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title('Human vs AI Probability', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Add labels to the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center')

        ax.set_ylim(0, 100)
        plt.tight_layout()

    human_percentage = round(human_percentage)
    ai_percentage    = round(ai_percentage)

    det_result = DetectionResult(
        prediction="Human" if human_percentage > ai_percentage else "AI",
        confidence=round(max(human_percentage, ai_percentage)),
        human_percentage=human_percentage,
        ai_percentage=ai_percentage,
        detected_model=ai_argmax_model if ai_percentage > human_percentage else None,
        raw_scores=raw_scores,
    )

    if fig is not None:
        plt.close(fig)
    return result_message, fig, det_result

# [ADDED v1.1] Gradio wrapper — unpacks only (msg, fig) for Gradio outputs.
def _gradio_classify(text: str):
    # Gradio runs single-threaded for this call → safe to build the pyplot figure.
    msg, fig, _ = classify_text(text, generate_plot=True)
    return msg, fig


# [ADDED v1.2] Lightweight classifier for hybrid_segment_detector.
# Reuses the 4 already-loaded models — NO extra memory.
# Returns (human_percentage, ai_percentage) as a simple tuple.

@torch.inference_mode()
def classify_batch(texts: List[str]) -> List[Tuple[float, float]]:
    """
    Clasifica una lista de segmentos en un solo lote (batch).
    Es mucho más rápido que procesar uno por uno.
    """
    if not texts:
        return []
        
    cleaned_texts = [clean_text(t) for t in texts]

    # max_length makes truncation explicit — avoids silent loss of content
    inputs = tokenizer(
        cleaned_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    # Inferencia del ensamble en paralelo
    logits_1 = model_1(**inputs).logits
    logits_2 = model_2(**inputs).logits
    logits_3 = model_3(**inputs).logits
    
    # Promediar probabilidades del lote
    avg_probs = (
        torch.softmax(logits_1, dim=1)
        + torch.softmax(logits_2, dim=1)
        + torch.softmax(logits_3, dim=1)
    ) / 3
    
    results = []
    for i in range(len(texts)):
        probs = avg_probs[i]
        human_prob = probs[24].item()
        ai_prob = 1.0 - human_prob
        results.append((round(human_prob * 100), round(ai_prob * 100)))
        
    return results


@torch.inference_mode()
def classify_segment(text: str) -> Tuple[float, float]:
    """Clasifica un único segmento. Ahora usa classify_batch internamente."""
    results = classify_batch([text])
    return results[0] if results else (50.0, 50.0)


@torch.inference_mode()
def _classify_batch_from_ids(id_seqs: List[List[int]]) -> List[Tuple[float, float, Optional[str], float]]:
    """
    Inference directly on pre-tokenized ID sequences — no decode→re-encode round-trip.

    Returns List of (human_pct, ai_pct, detected_model, ensemble_disagreement) where
    detected_model is the highest-probability AI model label (or None when human wins) and
    ensemble_disagreement is the std (in percentage points) of the per-seed AI probability.

    id_seqs must NOT contain special tokens; this function adds them via
    tokenizer.build_inputs_with_special_tokens (model-agnostic).
    Used by analyze_fast to eliminate the redundant encode/decode cycle.
    """
    if not id_seqs:
        return []

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_content = tokenizer.model_max_length - 2  # 2 slots reserved for specials

    # Add CLS/SEP (or equivalent) in a model-agnostic way
    wrapped = [
        tokenizer.build_inputs_with_special_tokens(ids[:max_content])
        for ids in id_seqs
    ]
    max_len = max(len(seq) for seq in wrapped)

    input_ids = torch.tensor(
        [seq + [pad_id] * (max_len - len(seq)) for seq in wrapped],
        dtype=torch.long, device=device,
    )
    attention_mask = torch.tensor(
        [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in wrapped],
        dtype=torch.long, device=device,
    )

    logits_1 = model_1(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_2 = model_2(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_3 = model_3(input_ids=input_ids, attention_mask=attention_mask).logits

    sm1 = torch.softmax(logits_1, dim=1)
    sm2 = torch.softmax(logits_2, dim=1)
    sm3 = torch.softmax(logits_3, dim=1)
    avg_probs = (sm1 + sm2 + sm3) / 3

    # Ensemble disagreement: std of the per-seed AI probability (1 - human@24) across the
    # 3 ModernBERT seeds. This is a FREE, dataset-free uncertainty signal — high disagreement
    # means the models are unsure (e.g. out-of-distribution frontier-model text), so callers
    # can widen uncertainty / lower confidence instead of reporting a falsely crisp verdict.
    ai_stack = torch.stack([1.0 - sm1[:, 24], 1.0 - sm2[:, 24], 1.0 - sm3[:, 24]], dim=1)
    disagreement = (ai_stack.std(dim=1, unbiased=False) * 100.0)  # percentage points

    results = []
    for i in range(len(id_seqs)):
        probs = avg_probs[i]
        human_pct = round(probs[24].item() * 100)
        ai_pct = 100 - human_pct
        # Identify the specific AI model with highest probability (excluding human index 24)
        ai_clone = probs.clone()
        ai_clone[24] = 0.0
        detected_model: Optional[str] = label_mapping[int(torch.argmax(ai_clone).item())] if ai_pct > human_pct else None
        results.append((human_pct, ai_pct, detected_model, round(float(disagreement[i].item()), 2)))
    return results


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


# ── Embedding / inference result cache ───────────────────────────────────────
# Keyed by sha256(text). Prevents re-running the 3-model ensemble when the
# same text is analyzed by multiple plugins in the same request (e.g. both
# ai_detection and full_analysis requested together) or repeated shortly after.
import hashlib as _hashlib
import threading as _threading
import time as _time

_FAST_CACHE: dict = {}
_FAST_CACHE_LOCK = _threading.Lock()
_FAST_CACHE_TTL: float = 300.0   # 5 minutes — covers same-request multi-plugin calls
_FAST_CACHE_MAX: int = 20        # keep memory bounded; LRU eviction


def _cache_namespace() -> str:
    """[C-17] Namespace the result cache by model identity + version so a model/weights
    swap (or MODEL_VERSION bump) never serves stale verdicts keyed only by text hash."""
    try:
        ident = f"{getattr(model_1.config, '_name_or_path', 'm')}:{model_1.config.num_labels}"
    except Exception:
        ident = "default"
    ident += ":" + os.environ.get("MODEL_VERSION", "")
    return _hashlib.sha1(ident.encode()).hexdigest()[:10]


_CACHE_NS: str = _cache_namespace()


@torch.inference_mode()
def analyze_fast(text: str) -> dict:
    """
    Paragraph-aware document analysis matching the reference classify_text() pipeline.

    Each paragraph (split at \\n\\n then \\n) is classified as an independent
    unit — identical to how the reference runs on a single text:
      1. clean_text() normalization
      2. tokenizer(segment, truncation=True) — one forward pass per segment
      3. 3-model softmax average → human_pct / ai_pct

    This prevents human and AI sections from bleeding into the same chunk,
    which occurred with the previous token-boundary splitting approach.

    Result cache: TTL=5min, 20-entry LRU — same-text repeated calls cost 0ms.
    """
    if not text.strip():
        return {"error": "El documento está vacío."}

    # Cache on raw text (before cleaning) to preserve hit rate across callers,
    # namespaced by model version so a model swap invalidates stale entries (C-17).
    _text_hash = _CACHE_NS + ":" + _hashlib.sha256(text.encode()).hexdigest()
    _now = _time.monotonic()
    with _FAST_CACHE_LOCK:
        _entry = _FAST_CACHE.get(_text_hash)
        if _entry is not None and _now - _entry[1] < _FAST_CACHE_TTL:
            return _entry[0]
        if len(_FAST_CACHE) >= _FAST_CACHE_MAX:
            _oldest = min(_FAST_CACHE, key=lambda k: _FAST_CACHE[k][1])
            del _FAST_CACHE[_oldest]

    # 1. Split BEFORE clean_text — clean_text collapses \s{2,} to a single
    #    space, which destroys \n\n paragraph boundaries. Splitting first
    #    preserves the human/AI paragraph separation, then we clean each
    #    segment individually so the tokenizer receives normalised text.
    segments_text: List[str] = []
    for block in text.split('\n\n'):
        for line in block.split('\n'):
            line = clean_text(line).strip()
            if line:
                segments_text.append(line)
    if not segments_text:
        segments_text = [clean_text(text).strip()]

    BATCH_SIZE = 12
    # Reserve 2 slots for CLS/SEP — matches tokenizer(truncation=True) behavior
    max_content = tokenizer.model_max_length - 2

    # 2. Tokenize each segment as an independent unit (truncation=True = reference behavior)
    segment_id_seqs: List[List[int]] = [
        tokenizer.encode(seg, add_special_tokens=False, truncation=True, max_length=max_content)
        for seg in segments_text
    ]

    # 3. Ensemble inference — same 3-model softmax average as reference classify_text()
    all_pcts: List[Tuple[float, float, Optional[str], float]] = []
    for i in range(0, len(segment_id_seqs), BATCH_SIZE):
        all_pcts.extend(_classify_batch_from_ids(segment_id_seqs[i:i + BATCH_SIZE]))

    # 4. Per-segment results + token-weighted aggregate
    segments = []
    total_human_w = total_ai_w = total_len = 0
    total_disagree_w = 0.0
    detected_model_votes: Dict[str, float] = {}

    for idx, ((human_pct, ai_pct, det_model, disagree), ids, seg_text) in enumerate(
        zip(all_pcts, segment_id_seqs, segments_text)
    ):
        tok_len = len(ids)
        segments.append({
            "segment_id": idx + 1,
            "text": seg_text,
            "dominant_label": "AI" if ai_pct > human_pct else "Human",
            "score": max(ai_pct, human_pct),
            "ensemble_disagreement": disagree,
        })
        total_human_w += human_pct * tok_len
        total_ai_w += ai_pct * tok_len
        total_disagree_w += disagree * tok_len
        total_len += tok_len
        if det_model is not None:
            detected_model_votes[det_model] = detected_model_votes.get(det_model, 0.0) + tok_len

    if total_len == 0:
        return {"error": "No se pudieron procesar tokens del documento."}

    overall_human = round(total_human_w / total_len)
    overall_ai = round(total_ai_w / total_len)
    overall_disagree = round(total_disagree_w / total_len, 2)
    overall_detected = max(detected_model_votes, key=detected_model_votes.get) if detected_model_votes else None

    _result = {
        "overall_summary": {
            "total_human_percentage": overall_human,
            "total_ai_percentage": overall_ai,
            "overall_prediction": "AI" if overall_ai > overall_human else "Human",
            "detected_model": overall_detected,
            "ensemble_disagreement": overall_disagree,
        },
        "segments": segments,
    }
    with _FAST_CACHE_LOCK:
        _FAST_CACHE[_text_hash] = (_result, _time.monotonic())
    return _result


def classify_text_aggregate(text: str) -> DetectionResult:
    """
    Document-level DetectionResult covering the FULL text.

    [C-04/§13 FIX] classify_text() tokenizes with truncation=True, so for documents
    longer than the model's max length (~512 tokens) it only classifies the first
    chunk — the forensic verdict then ignores the bulk of a long document. This
    helper instead reuses analyze_fast(), whose per-segment, token-weighted aggregate
    spans the whole text, and packages it as a DetectionResult so PluginOrchestrator
    produces a verdict representative of the entire document.

    Falls back to a neutral Unknown result on empty/error input.
    """
    doc = analyze_fast(text)
    if not isinstance(doc, dict) or "error" in doc:
        return DetectionResult(
            prediction="Unknown", confidence=0,
            human_percentage=50, ai_percentage=50,
            detected_model=None, raw_scores={"human": 0.0, "ai": 0.0},
            uncertainty_zone=True,
        )
    s = doc.get("overall_summary", {})
    human = s.get("total_human_percentage", 50)
    ai = s.get("total_ai_percentage", 50)
    prediction = s.get("overall_prediction", "Human")
    disagree = float(s.get("ensemble_disagreement", 0.0))
    # Uncertain when the margin is thin OR the 3 seeds disagree markedly (OOD signal).
    uncertain = abs(ai - human) < 15 or disagree >= 12.0
    return DetectionResult(
        prediction=prediction,
        confidence=round(max(human, ai)),
        human_percentage=human,
        ai_percentage=ai,
        detected_model=s.get("detected_model") if prediction == "AI" else None,
        raw_scores={"human": float(human), "ai": float(ai)},
        uncertainty_zone=uncertain,
        ensemble_disagreement=disagree,
    )