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

# XPLAGIAX_EVAL_WEIGHTS: candidate-weights override used by
# scripts/retrain_pipeline.py evaluate — never set it in production serving.
model1_path = os.getenv("XPLAGIAX_EVAL_WEIGHTS") or os.path.join(_BASE_DIR, "modernbert.bin")
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

# ── Model load bookkeeping (weight versioning / fallback) ─────────
# Populated by _load_model(); surfaced via get_model_info() and /api/drift-status
# so operators can see exactly which weight files each worker is serving.
_MODEL_LOAD_INFO: List[Dict[str, object]] = []


# ── Helper: arquitectura vacía + pesos locales, 0 descargas ──
def _load_model(weight_path):
    """Build the architecture and load local weights, with optional fallback.

    If the primary weight file is missing or corrupt and MODEL_FALLBACK_DIR is
    set, the same basename is loaded from that directory instead (the "last
    known good" weights kept by scripts/retrain_pipeline.py promote step). The
    fallback is recorded so /api/drift-status can surface that the worker is
    running on old weights rather than silently serving them as current.
    """
    m = AutoModelForSequenceClassification.from_config(_config)
    loaded_from = weight_path
    used_fallback = False
    try:
        state = torch.load(weight_path, map_location=device)
    except Exception as primary_err:
        fallback_dir = os.getenv("MODEL_FALLBACK_DIR", "")
        if not fallback_dir:
            raise
        fallback_path = os.path.join(fallback_dir, os.path.basename(weight_path))
        logger.error(
            "Primary weights unusable (%s): %s — falling back to %s",
            weight_path, primary_err, fallback_path,
        )
        state = torch.load(fallback_path, map_location=device)
        loaded_from = fallback_path
        used_fallback = True
    m.load_state_dict(state)
    _MODEL_LOAD_INFO.append({
        "requested": os.path.basename(weight_path),
        "loaded_from": loaded_from,
        "fallback": used_fallback,
    })
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


def get_model_info() -> Dict[str, object]:
    """Weight provenance + version for /api/drift-status and diagnostics."""
    return {
        "version": os.getenv("MODEL_VERSION", "2026.06"),
        "device": str(device),
        "weights": list(_MODEL_LOAD_INFO),
        "fallbacks_used": [
            i["loaded_from"] for i in _MODEL_LOAD_INFO if i.get("fallback")
        ],
    }
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


# ── Special-token wrapping derived from the tokenizer itself ──────────────────
# [CRITICAL FIX] For a *fast* tokenizer (ModernBERT ships fast-only), the special
# tokens [CLS]/[SEP] are added by the post-processor during
# tokenizer(text) / encode(add_special_tokens=True) — NOT by
# build_inputs_with_special_tokens(). On transformers 4.x that method inherits the
# base identity implementation and returns the ids UNCHANGED (no specials); on 5.x
# the fast backend does not implement it at all. Building model inputs with it
# therefore drops [CLS]/[SEP], so ModernBERT pools over the wrong first token and
# produces confident but INVERTED verdicts (all seeds agree → low disagreement),
# while the reference tokenizer(text) path stays correct. We probe the tokenizer
# once to recover the exact prefix/suffix it wraps a single sequence with, and
# replicate it when building inputs from pre-tokenized ids.
def _derive_special_token_wrap():
    probe = tokenizer.encode("text", add_special_tokens=False)
    full = tokenizer("text", add_special_tokens=True)["input_ids"]
    if probe:
        for start in range(len(full) - len(probe) + 1):
            if full[start:start + len(probe)] == probe:
                return full[:start], full[start + len(probe):]
    prefix = [tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []
    suffix = [tokenizer.sep_token_id] if tokenizer.sep_token_id is not None else []
    return prefix, suffix


_SPECIAL_PREFIX, _SPECIAL_SUFFIX = _derive_special_token_wrap()
_NUM_SPECIALS = len(_SPECIAL_PREFIX) + len(_SPECIAL_SUFFIX)
logger.info(
    "Special-token wrap derived: prefix=%s suffix=%s", _SPECIAL_PREFIX, _SPECIAL_SUFFIX
)


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

# ── [C2] Segment-level inference cache ────────────────────────────
# classify_batch() is the shared entry point for the hybrid window classifier,
# classify_segment(), and any chunked caller. The SAME window/segment text is
# often scored several times per request (e.g. the segment_analysis plugin and
# the full_analysis orchestrator walking the same document), and repeatedly
# across near-identical requests. Each entry is two floats, so the cache tops
# out at a few hundred KB. Namespaced by _CACHE_NS: a model swap invalidates it.
import threading as _threading
from collections import OrderedDict as _OrderedDict

_SEG_CACHE: "_OrderedDict[str, Tuple[float, float]]" = _OrderedDict()
_SEG_CACHE_LOCK = _threading.Lock()
_SEG_CACHE_MAX = int(os.getenv("SEGMENT_CACHE_MAX", "2048"))


@torch.inference_mode()
def classify_batch(texts: List[str]) -> List[Tuple[float, float]]:
    """
    Clasifica una lista de segmentos en un solo lote (batch).
    Es mucho más rápido que procesar uno por uno.

    [C2] Segment results are memoised in a bounded LRU keyed by the cleaned
    text, so only cache MISSES reach the 3-model ensemble. Scores for misses
    are numerically identical to the uncached path (same tokenization, same
    softmax average).
    """
    if not texts:
        return []

    # [Fase-2 M-11] Cooperative checkpoint: if the caller's timeout already expired
    # (registry reported the error), stop before paying for another forward pass.
    from exec_context import check_deadline
    check_deadline()

    cleaned_texts = [clean_text(t) for t in texts]
    keys = [
        _CACHE_NS + ":" + _hashlib.sha1(t.encode("utf-8")).hexdigest()
        for t in cleaned_texts
    ]

    results: List[Optional[Tuple[float, float]]] = [None] * len(texts)
    with _SEG_CACHE_LOCK:
        for i, key in enumerate(keys):
            hit = _SEG_CACHE.get(key)
            if hit is not None:
                _SEG_CACHE.move_to_end(key)
                results[i] = hit

    miss_idx = [i for i, r in enumerate(results) if r is None]
    if miss_idx:
        miss_texts = [cleaned_texts[i] for i in miss_idx]

        # max_length makes truncation explicit — avoids silent loss of content
        inputs = tokenizer(
            miss_texts,
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

        with _SEG_CACHE_LOCK:
            for j, i in enumerate(miss_idx):
                human_prob = avg_probs[j][24].item()
                pair = (round(human_prob * 100), round((1.0 - human_prob) * 100))
                results[i] = pair
                _SEG_CACHE[keys[i]] = pair
                _SEG_CACHE.move_to_end(keys[i])
            while len(_SEG_CACHE) > _SEG_CACHE_MAX:
                _SEG_CACHE.popitem(last=False)

    return results  # every slot is filled: hit above or miss inference here


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
    max_content = tokenizer.model_max_length - _NUM_SPECIALS  # reserve slots for specials

    # Wrap each sequence with the tokenizer's REAL special tokens ([CLS] … [SEP]),
    # exactly reproducing tokenizer(text, add_special_tokens=True). We do NOT use
    # build_inputs_with_special_tokens(): on fast tokenizers it silently returns the
    # ids unchanged, dropping the specials and corrupting the verdict (see
    # _derive_special_token_wrap above).
    wrapped = [
        _SPECIAL_PREFIX + ids[:max_content] + _SPECIAL_SUFFIX
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
        # [Fase-2 M-10/C-05] Float percentages; display layers round for presentation.
        human_pct = round(probs[24].item() * 100, 2)
        ai_pct = round(100.0 - human_pct, 2)
        # Identify the specific AI model with highest probability (excluding human index 24)
        ai_clone = probs.clone()
        ai_clone[24] = 0.0
        detected_model: Optional[str] = label_mapping[int(torch.argmax(ai_clone).item())] if ai_pct > human_pct else None
        results.append((human_pct, ai_pct, detected_model, round(float(disagreement[i].item()), 2)))
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


# ── Document segmentation ────────────────────────────────────────────────────
# Splitting on EVERY newline (the previous behaviour) made the unit of analysis
# whatever the source happened to wrap at. Text extracted from a PDF carries a
# newline at every visual line break, so each ~12-word fragment — cut mid
# sentence — was classified as an independent document and rendered with its own
# AI/human verdict. The same content pasted from a PDF reader (which rejoins
# lines into paragraphs) segmented completely differently, so identical text
# produced different results depending on how it arrived.
#
# Two extra casualties of splitting on "\n" first: the tokenizer's own
# normalizer (join_hyphen_break / newline_to_space, above) can no longer rejoin
# a word hyphenated across a line break, because the halves now live in separate
# segments and it only ever sees one at a time.
#
# Rules below mirror FinderX's chunker (app/services/chunker.py), which already
# solved this for search: collapse intra-paragraph newlines, cut on paragraph
# then sentence boundaries, never mid-word, and merge undersized fragments.
# Deliberately NOT copied: chunk overlap (would double-count text in the
# token-weighted aggregate) and SimHash dedup (would drop legitimately repeated
# paragraphs).
_SEG_MIN_WORDS = int(os.getenv("SEGMENT_MIN_WORDS", "40"))
_SEG_MAX_WORDS = int(os.getenv("SEGMENT_MAX_WORDS", "400"))

_RE_HYPHEN_LINEBREAK = re.compile(r'(\w+)[-‐‑]\s*\n\s*(\w+)')
_RE_INTRA_WS = re.compile(r'\s+')
_RE_PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')
_RE_SENTENCE_FALLBACK = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """Sentence split via NLTK punkt, regex fallback if the corpus is missing."""
    try:
        from nltk.tokenize import sent_tokenize
        parts = sent_tokenize(text)
    except Exception:
        parts = _RE_SENTENCE_FALLBACK.split(text)
    return [p.strip() for p in parts if p.strip()]


def segment_document(
    text: str,
    min_words: int = _SEG_MIN_WORDS,
    max_words: int = _SEG_MAX_WORDS,
) -> List[str]:
    """Split a document into analysis units of at least `min_words`.

    1. Paragraphs = blank-line separated blocks. A lone "\\n" is intra-paragraph
       wrapping, so it collapses to a space (words hyphenated across the break
       are rejoined first) — this is what makes an uploaded PDF and the same
       text pasted by hand segment identically.
    2. A paragraph longer than `max_words` is cut on sentence boundaries, never
       mid-sentence and never mid-word.
    3. Fragments below `min_words` are merged forward until they reach it: a
       10-word line carries no statistical signal for an AI/human verdict, and
       rendering one with its own confidence badge is worse than not splitting
       there at all. Trade-off: a very short paragraph is judged together with
       its neighbour, so a lone AI sentence between human paragraphs can be
       diluted. `min_words` tunes that (lower = finer, noisier).
    """
    if not text or not text.strip():
        return []

    raw: List[str] = []
    for block in _RE_PARAGRAPH_SPLIT.split(text):
        block = _RE_HYPHEN_LINEBREAK.sub(r'\1\2', block)
        block = clean_text(_RE_INTRA_WS.sub(' ', block)).strip()
        if not block:
            continue
        if len(block.split()) <= max_words:
            raw.append(block)
            continue
        # Oversized paragraph: accumulate whole sentences up to max_words.
        current: List[str] = []
        current_words = 0
        for sentence in _split_sentences(block):
            n = len(sentence.split())
            if current and current_words + n > max_words:
                raw.append(' '.join(current))
                current, current_words = [], 0
            current.append(sentence)
            current_words += n
        if current:
            raw.append(' '.join(current))

    # Merge undersized fragments forward; a short tail folds into the previous.
    merged: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for seg in raw:
        buf.append(seg)
        buf_words += len(seg.split())
        if buf_words >= min_words:
            merged.append(' '.join(buf))
            buf, buf_words = [], 0
    if buf:
        tail = ' '.join(buf)
        if merged:
            merged[-1] = merged[-1] + ' ' + tail
        else:
            merged.append(tail)   # whole document below min_words — keep as one
    return merged


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
    #    See segment_document(): paragraph/sentence aware, with a minimum size,
    #    so the unit of analysis no longer depends on where the source wrapped.
    segments_text: List[str] = segment_document(text)
    if not segments_text:
        segments_text = [clean_text(text).strip()]

    BATCH_SIZE = 12
    # Reserve slots for the tokenizer's specials — matches tokenizer(truncation=True)
    max_content = tokenizer.model_max_length - _NUM_SPECIALS

    # 2. Tokenize each segment as an independent unit (truncation=True = reference behavior)
    segment_id_seqs: List[List[int]] = [
        tokenizer.encode(seg, add_special_tokens=False, truncation=True, max_length=max_content)
        for seg in segments_text
    ]

    # 3. Ensemble inference — same 3-model softmax average as reference classify_text()
    # Length-bucketed batching: segments are classified in ascending-length order so
    # each batch pads to a near-uniform length instead of the batch max. Mixed-length
    # documents (short + long paragraphs) waste 30-50% of forward-pass FLOPs on pad
    # tokens otherwise. Each segment is an independent forward pass, so scores are
    # bit-identical; results are written back at their original index.
    order = sorted(range(len(segment_id_seqs)), key=lambda i: len(segment_id_seqs[i]))
    all_pcts: List[Optional[Tuple[float, float, Optional[str], float]]] = [None] * len(segment_id_seqs)
    from exec_context import check_deadline
    for i in range(0, len(order), BATCH_SIZE):
        check_deadline()  # [Fase-2 M-11] abort orphaned threads at batch boundaries
        bucket = order[i:i + BATCH_SIZE]
        for seg_idx, pcts in zip(bucket, _classify_batch_from_ids([segment_id_seqs[j] for j in bucket])):
            all_pcts[seg_idx] = pcts

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

    # [Fase-2 M-10] 2-decimal floats document-wide; presentation layers may round.
    overall_human = round(total_human_w / total_len, 2)
    overall_ai = round(total_ai_w / total_len, 2)
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
        confidence=round(max(human, ai), 1),
        human_percentage=human,
        ai_percentage=ai,
        detected_model=s.get("detected_model") if prediction == "AI" else None,
        raw_scores={"human": float(human), "ai": float(ai)},
        uncertainty_zone=uncertain,
        ensemble_disagreement=disagree,
    )