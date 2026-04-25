# ── torch + device must be defined FIRST so _load_model() can always
# reference the global `device`, even if a later import fails. ──────
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
import re
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tokenizers import normalizers
from tokenizers.normalizers import Sequence, Replace, Strip, NFKC
from tokenizers import Regex
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it

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

def classify_text(text):
    """
    Classifies the text and generates a plot of the human vs AI probability.
    Returns both the result message and the plot figure.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return "", None

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

    # --- LO QUE FALTABA: Construir el objeto DetectionResult ---
    human_percentage = round(human_percentage)
    ai_percentage    = round(ai_percentage)

    det_result = DetectionResult(
        prediction="Human" if human_percentage > ai_percentage else "AI",
        confidence=round(max(human_percentage, ai_percentage)),
        human_percentage=human_percentage,
        ai_percentage=ai_percentage,
        detected_model=ai_argmax_model if ai_percentage > human_percentage else None,
        raw_scores={"human": round(human_prob), "ai_total": round(ai_total_prob)}
    )

    # Devolver los 3 elementos exactamente como espera el orquestador
    return result_message, fig, det_result

# [ADDED v1.1] Gradio wrapper — unpacks only (msg, fig) for Gradio outputs.
def _gradio_classify(text: str):
    msg, fig, _ = classify_text(text)
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
    
    # Tokenizar todo el lote a la vez
    inputs = tokenizer(cleaned_texts, return_tensors="pt", truncation=True, padding=True).to(device)

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
    """
    Clasifica un único segmento. Ahora usa classify_batch internamente.
    """
    results = classify_batch([text])
    return results[0] if results else (50.0, 50.0)


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

    # ── 1. CRITERIO DE DESCARTE (Falso Positivo) ──────────────────────────
    # Si el modelo dice IA, pero no hay indicadores de razonamiento y el texto
    # tiene una entropía muy alta (humana), es probable que sea un falso positivo.
    if razonamiento.get("ai_score", 0) < 0.25 and perplejidad.get("ai_score", 0) < 0.40:
        segmento_dict["dominant_label"] = "Human (Validated)"
        segmento_dict["score"] = 100 - segmento_dict["score"]
        segmento_dict["status_note"] = (
            "Descartado: Los plugins forenses confirman estructura humana natural."
        )

    # ── 2. CRITERIO DE CONFIRMACIÓN (Alucinación detectada) ───────────────
    # [FIX v1.3] Condiciones de guardia para evitar falsos positivos
    # causados por extracción espuria de texto estructural como referencias.
    #
    # Condiciones requeridas (TODAS deben cumplirse):
    #   a) Al menos 2 referencias extraídas (evita falsos positivos
    #      de una sola extracción sobre texto de sección/imagen)
    #   b) Ratio de fabricación >= 70% (mayoría clara de citas inválidas)
    #   c) El modelo base ya sospechaba IA en este segmento
    #      (label contiene "AI" y score > 50)
    feat_vals   = referencias.get("feature_values", {})
    fab_count   = feat_vals.get("fabricated_count", 0)
    total_refs  = feat_vals.get("total_references", 0)
    fab_ratio   = feat_vals.get("fabricated_ratio", 0.0)
    base_score  = segmento_dict.get("score", 0.0)
    base_label  = segmento_dict.get("dominant_label", "")

    if (
        fab_count > 0
        and total_refs >= 2                  # guardia (a): más de una referencia
        and fab_ratio >= 0.70               # guardia (b): mayoría de citas inválidas
        and "AI" in base_label              # guardia (c): modelo base ya sospechaba IA
        and base_score > 50.0              # guardia (c): confianza mínima en IA
    ):
        segmento_dict["dominant_label"] = "AI (Confirmed)"
        segmento_dict["score"] = 100.0
        segmento_dict["status_note"] = (
            "Confirmado: Se detectaron alucinaciones bibliográficas (citas inventadas)."
        )

    return segmento_dict


def analyze_long_document(long_text: str, orchestrator=None, max_tokens: int = 512) -> dict:
    """
    Analiza un documento completo con segmentación semántica y validación forense.
    """
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
    
    print(f"\nIniciando análisis forense de {len(chunks_text)} segmentos...")
    
    # 2. Procesamiento de Segmentos por Lotes (Batch size = 4)
    BATCH_SIZE = 4
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
    """
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
    
    print(f"\nIniciando análisis semántico de {len(chunks_text)} segmentos...")
    
    # 4. Procesar por lotes (Batch size = 4)
    BATCH_SIZE = 4
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
    overall_human = round(total_human_weighted / total_tokens_processed)
    overall_ai = round(total_ai_weighted / total_tokens_processed)
    
    results["overall_summary"] = {
        "total_human_percentage": overall_human,
        "total_ai_percentage": overall_ai,
        "overall_prediction": "AI" if overall_ai > overall_human else "Human"
    }
    
    return results

"""
with iface:
    text_input.change(_gradio_classify, inputs=text_input, outputs=[result_output, plot_output])
"""