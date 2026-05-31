# XplagiaX — AI Detection Microservice

Flask microservice for AI-generated text detection. Built around a 3-model ModernBERT ensemble with a modular plugin architecture. Supports per-document segmentation, forensic reports, perplexity analysis, citation verification, zone classification, streaming analysis, and more.

---

## Architecture

```
Client
    │
    │  POST /analyze              {"text": "...", "plugins": ["ai_detection", ...]}
    │  POST /analyze_document     {"text": "...", "plugins": ["ai_detection", ...]}
    │  POST /analyze_stream       {"text": "...", "plugins": [...]}  (SSE)
    │  POST /api/v2/citations/detect    {"text": "..."}
    │  POST /api/v2/citations/validate  {"text": "..."}
    ▼
┌────────────────────────────────────────────────────────────────┐
│  Gunicorn (preload_app=True + sync workers)                    │
│                                                                │
│  Master Process                                                │
│  ├── ModernBERT ensemble loaded ONCE at module level           │
│  ├── CitationDetector singleton loaded ONCE (CoW-safe)         │
│  ├── PluginRegistry auto-discovers all plugins                 │
│  └── create_app() called once                                  │
│                  │ fork (Linux CoW)                            │
│  ┌───────────────▼──┐  ┌──────────────────┐                   │
│  │    Worker 1       │  │    Worker 2  ...  │                  │
│  │  ~50 MB overhead  │  │  ~50 MB overhead  │                  │
│  └───────────────────┘  └───────────────────┘                  │
│                                                                │
│  Redis (optional)                                              │
│  ├── Rate limit counters (Flask-Limiter)                       │
│  ├── Result cache (Flask-Caching, 1-hour TTL)                  │
│  └── Celery broker + result backend                            │
└────────────────────────────────────────────────────────────────┘
```

**Key design decisions**

| Decision | Rationale |
|---|---|
| `preload_app = True` | Models loaded once in master, shared across workers via CoW |
| Sync worker class | Stable under CPU-bound ML inference; no gevent GIL interaction |
| `OMP_NUM_THREADS=1` | Prevents torch/numpy from spawning per-worker thread explosions |
| `local_files_only=True` | No HuggingFace network calls on startup — uses local cache |
| Plugin auto-discovery | Drop a file in `app/plugins/`, restart — zero core changes needed |
| Multi-stage Docker build | Builder has gcc/dev headers; runtime image is lean |
| Non-root container user | UID 1000, K8s security best practice |
| `m.share_memory()` on models | POSIX shared memory prevents CoW page faults across workers |
| CitationDetector module singleton | Instantiated at import time, shared across all workers via CoW |
| Result cache (sha256 keyed) | Same text + same plugins = 0ms from Redis cache, no re-inference |
| `analyze_fast()` everywhere | Single-pass tokenization with token-boundary splits; 2–12× faster than chunked inference |

---

## Engine — ModernBERT Ensemble

The core classifier ([app/engine/detector_final.py](app/engine/detector_final.py)) loads three fine-tuned ModernBERT models at startup and averages their softmax outputs:

| File | Labels | Role |
|---|---|---|
| `modernbert.bin` | 41 classes | Base ensemble member |
| `Model_groups_3class_seed12` | 41 classes | Ensemble member (seed 12) |
| `Model_groups_3class_seed22` | 41 classes | Ensemble member (seed 22) |

The 41 label classes include `human` (index 24) and 40 known AI generators (GPT-4, GPT-4o, Claude variants, LLaMA 3, Mixtral, Gemma, etc.). The binary Human/AI split is derived from `prob[24]` vs the sum of all remaining probabilities.

The tokenizer and config are loaded from the local HuggingFace cache (`answerdotai/ModernBERT-base`, `local_files_only=True`).

`analyze_fast()` uses adaptive `max_tokens` (capped at sequence length), single-pass tokenization, and token-boundary chunk splitting. Results are cached in a thread-safe TTL dict (`_FAST_CACHE`, 5-minute TTL, 20-entry LRU cap) — calling it again with the same text hits the cache at 0ms.

---

## Plugins

### Available plugins

| Plugin name | Description |
|---|---|
| `ai_detection` | Binary Human/AI classification — 3-model ModernBERT ensemble. ~2s CPU, ~0.3s GPU. |
| `segment_analysis` | Per-paragraph AI/Human heatmap via `HybridSegmentAnalyzer`. Returns preview (80 chars) per segment. |
| `perplexity_check` | Text predictability analysis via n-gram proxy (Tier 1, CPU) + optional GPT-2 (Tier 2, GPU). |
| `stylometric_analysis` | Detailed analysis of writing style: sentence structure, vocabulary richness, and burstiness. |
| `hallucination_check` | Detects AI fabrication risk: internal inconsistencies and factual drift. |
| `reasoning_check` | Detects reasoning-model signals (o1, DeepSeek-R1): CoT markers and causal density. |
| `citation_check` | Verifies citations against CrossRef, Semantic Scholar, and OpenAlex. Detects fabricated references. |
| `watermark_detection` | Detects statistical watermarks embedded in text by AI models. |
| `zone_classifier` | Classifies text zones (direct quotes, paraphrases, original content). Detects citation style (APA/MLA/IEEE/Chicago/Vancouver/Harvard), coverage, and consistency. No network calls. |
| `forensic_report` | Generates a full HTML forensic report via `ForensicReportGenerator`. Requires `full_analysis` pipeline. |
| `full_analysis` | Complete pipeline: detection → stylometric → hallucination → reasoning → perplexity → segment → citation → watermark → forensic report. |

### Adding a new plugin

1. Create `app/plugins/my_plugin.py`:

```python
from app.plugins.base import BasePlugin

class MyPlugin(BasePlugin):

    def name(self) -> str:
        return "my_plugin"

    def description(self) -> str:
        return "Does something useful with text."

    def analyze(self, text: str) -> dict:
        return {"result": "..."}
```

2. Add dependencies to `requirements.in` and run `make lock`.
3. Restart the service — the plugin is auto-discovered with no other changes.

---

## API

### POST /analyze

Run any combination of plugins on a text. Results are cached by `sha256(text + plugins)` for 1 hour — repeated identical requests return immediately from Redis with `"from_cache": true`.

**Request**
```json
{
    "text": "The rapid integration of AI in academic settings...",
    "plugins": ["ai_detection", "perplexity_check"]
}
```

**Response**
```json
{
    "status": "ok",
    "word_count": 124,
    "plugins_requested": ["ai_detection", "perplexity_check"],
    "from_cache": false,
    "results": {
        "ai_detection": {
            "status": "ok",
            "elapsed_ms": 1823.4,
            "data": {
                "prediction": "AI",
                "confidence": 87.32,
                "human_percentage": 12.68,
                "ai_percentage": 87.32,
                "detected_model": "gpt4o",
                "uncertainty_zone": false,
                "raw_scores": {"human": 12.68, "ai": 87.32}
            }
        },
        "perplexity_check": {
            "status": "ok",
            "elapsed_ms": 340.1,
            "data": { "ai_score": 74.5, "risk_level": "HIGH", "tier": "tier1" }
        }
    },
    "total_elapsed_ms": 2165.3
}
```

```bash
curl -X POST http://localhost:5006/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "plugins": ["ai_detection"]}'

# With API key authentication
curl -X POST http://localhost:5006/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "Your text here...", "plugins": ["ai_detection"]}'
```

---

### POST /analyze — with `zone_classifier`

**Request**
```json
{
    "text": "García & López (2021) found that AI reduces learning time by 40% ...\n\nReferencias\nGarcía, M. (2021). ...",
    "plugins": ["ai_detection", "zone_classifier"]
}
```

**Response** (zone_classifier section)
```json
{
    "status": "ok",
    "results": {
        "zone_classifier": {
            "status": "ok",
            "elapsed_ms": 12.3,
            "data": {
                "dominant_style": "APA",
                "style_consistency": 92.5,
                "citation_coverage": 75.0,
                "total_inline_citations": 4,
                "total_bibliography": 3,
                "orphan_citations": 0,
                "uncited_bibliography": 1,
                "zones": [
                    {
                        "type": "PARAPHRASE",
                        "text_preview": "García & López (2021) found that AI reduces...",
                        "start_pos": 0,
                        "end_pos": 180,
                        "has_citation": true,
                        "citation_count": 1,
                        "plagiarism_risk": 0.2
                    },
                    {
                        "type": "DIRECT_QUOTE",
                        "text_preview": "\"La adaptación del currículo mediante algoritmos...\"",
                        "start_pos": 181,
                        "end_pos": 320,
                        "has_citation": true,
                        "citation_count": 1,
                        "plagiarism_risk": 0.05
                    },
                    {
                        "type": "ORIGINAL",
                        "text_preview": "Several unexplored dimensions remain...",
                        "start_pos": 321,
                        "end_pos": 500,
                        "has_citation": false,
                        "citation_count": 0,
                        "plagiarism_risk": 0.6
                    }
                ],
                "inline_citations": [
                    {
                        "text": "(García & López, 2021)",
                        "style": "APA",
                        "author": "García & López",
                        "year": "2021",
                        "page": null,
                        "number": null,
                        "confidence": 0.95
                    }
                ],
                "bibliography": [
                    {
                        "key": "Garcia2021",
                        "style": "APA",
                        "authors": ["García, M.", "López, J."],
                        "year": "2021",
                        "title": "Transformación digital en universidades...",
                        "doi": "10.1234/res.2021.003",
                        "url": null
                    }
                ],
                "issues": {
                    "orphan_citations": [],
                    "uncited_bibliography": [
                        {"key": "Smith2020", "authors": ["Smith, A."], "year": "2020"}
                    ]
                }
            }
        }
    }
}
```

---

### POST /analyze_document

Run any combination of plugins on a long document **and** get per-paragraph segment scores. Uses `analyze_fast()` — single-pass tokenization with token-boundary splitting and 5-minute result cache.

**Request**
```json
{
    "text": "Long document text...",
    "plugins": ["ai_detection"]
}
```

`plugins` is optional — defaults to `["ai_detection"]`.

**Response**
```json
{
    "status": "ok",
    "word_count": 620,
    "plugins_requested": ["ai_detection"],
    "results": {
        "ai_detection": {
            "status": "ok",
            "elapsed_ms": 2041.7,
            "data": {
                "prediction": "AI",
                "confidence": 90.67,
                "human_percentage": 9.33,
                "ai_percentage": 90.67,
                "detected_model": "gpt4",
                "uncertainty_zone": false,
                "raw_scores": {"human": 9.33, "ai": 90.67},
                "segments": [
                    {
                        "segment_id": 1,
                        "text": "Full paragraph text here...",
                        "dominant_label": "AI",
                        "score": 90.6663,
                        "forensic_analysis": {}
                    },
                    {
                        "segment_id": 2,
                        "text": "Another paragraph...",
                        "dominant_label": "Human",
                        "score": 82.14,
                        "forensic_analysis": {}
                    }
                ]
            }
        }
    },
    "total_elapsed_ms": 5987.2
}
```

```bash
# Default — ai_detection + segments
curl -X POST http://localhost:5006/analyze_document \
  -H "Content-Type: application/json" \
  -d '{"text": "Long document text..."}'

# Multiple plugins — all run, segments injected into ai_detection
curl -X POST http://localhost:5006/analyze_document \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "plugins": ["ai_detection", "perplexity_check", "citation_check"]}'
```

---

### POST /analyze_stream

Run plugins on a text and receive results as they complete via **Server-Sent Events (SSE)**. Fast plugins (e.g., `zone_classifier`) deliver their result immediately; slow plugins (e.g., `citation_check`) stream in as they finish. No waiting for the slowest plugin.

Rate limit: **30 requests/minute**.

**Request**
```json
{
    "text": "The rapid integration of AI in academic settings...",
    "plugins": ["ai_detection", "perplexity_check", "zone_classifier"]
}
```

**SSE Event Stream**
```
data: {"type": "init", "word_count": 124, "plugins": ["ai_detection", "perplexity_check", "zone_classifier"]}

data: {"type": "result", "plugin": "zone_classifier", "result": {"status": "ok", "elapsed_ms": 14.2, "data": {...}}}

data: {"type": "result", "plugin": "perplexity_check", "result": {"status": "ok", "elapsed_ms": 312.5, "data": {...}}}

data: {"type": "result", "plugin": "ai_detection", "result": {"status": "ok", "elapsed_ms": 1823.4, "data": {...}}}

data: {"type": "done"}
```

```bash
# curl — stream events to terminal
curl -N -X POST http://localhost:5006/analyze_stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"text": "Your text here...", "plugins": ["ai_detection", "zone_classifier"]}'
```

```javascript
// JavaScript EventSource — live updates as plugins complete
const response = await fetch("http://localhost:5006/analyze_stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Your text...", plugins: ["ai_detection", "zone_classifier"] }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    const event = JSON.parse(line.slice(6));
    if (event.type === "result") {
      console.log(`[${event.plugin}]`, event.result);
    }
  }
}
```

```python
import requests

with requests.post(
    "http://localhost:5006/analyze_stream",
    json={"text": "Your text...", "plugins": ["ai_detection", "zone_classifier"]},
    stream=True,
    headers={"Accept": "text/event-stream"},
    timeout=120,
) as resp:
    for line in resp.iter_lines():
        if line and line.startswith(b"data: "):
            import json
            event = json.loads(line[6:])
            if event["type"] == "result":
                print(f"[{event['plugin']}] {event['result']['status']} — {event['result'].get('elapsed_ms')}ms")
```

---

### GET /health

Liveness probe — always 200 if the process is alive.

```bash
curl http://localhost:5006/health
# {"status": "healthy"}
```

---

### GET /ready

Readiness probe — 200 only when plugins are loaded.

```bash
curl http://localhost:5006/ready
# {"status": "ready", "plugins_loaded": 6, "plugins": [...]}
```

---

### GET /plugins

List all registered plugins with descriptions.

```bash
curl http://localhost:5006/plugins
```

---

### GET /report/\<filename\>

Serve a generated HTML forensic report from `/tmp`. Only files with a `forensic_` prefix are served (path traversal + file-type guard). Returns with strict `Content-Security-Policy` and `X-Content-Type-Options: nosniff` headers.

```bash
curl http://localhost:5006/report/forensic_abc123.html
```

---

## Antiplagio — Citation API (`/api/v2/`)

The antiplagio module adds two dedicated citation endpoints that are independent from the plugin system. They are always available (no plugin selection needed).

Rate limits: **60 req/min** for `/detect`, **10 req/min** for `/validate`.

### POST /api/v2/citations/detect

Fast citation detection — pure regex plus optional spaCy. No network calls, no ML models.

Detects APA, MLA, IEEE, Chicago, Vancouver, and Harvard inline citations and bibliography entries. Classifies text zones and computes style consistency.

**Request**
```json
{ "text": "García & López (2021) demostró que... Referencias\nGarcía, M. (2021). ..." }
```

**Response**
```json
{
    "dominant_style": "APA",
    "style_consistency": 92.5,
    "citation_coverage": 75.0,
    "inline_citations": [
        {
            "text": "(García & López, 2021)",
            "style": "APA",
            "author": "García & López",
            "year": "2021",
            "page": null,
            "number": null,
            "confidence": 95.0,
            "position": {"start": 18, "end": 40}
        }
    ],
    "bibliography": [
        {
            "key": "Garcia2021",
            "style": "APA",
            "authors": ["García, M.", "López, J."],
            "year": "2021",
            "title": "Transformación digital en universidades latinoamericanas",
            "doi": "10.1234/res.2021.003",
            "url": null
        }
    ],
    "zones": [
        {
            "type": "PARAPHRASE",
            "text_preview": "García & López (2021) demostró que...",
            "has_citation": true,
            "citation_count": 1,
            "plagiarism_risk": 0.2
        },
        {
            "type": "ORIGINAL",
            "text_preview": "Several unexplored dimensions remain in this area...",
            "has_citation": false,
            "citation_count": 0,
            "plagiarism_risk": 0.6
        }
    ],
    "issues": {
        "orphan_citations": 0,
        "uncited_bibliography": 1
    }
}
```

**Zone types**

| Zone type | Meaning |
|---|---|
| `DIRECT_QUOTE` | Text inside quotation marks (`"..."` or `«...»`) |
| `BLOCK_QUOTE` | Indented block quotation (≥4 spaces) |
| `PARAPHRASE` | Paragraph with an inline citation but no quote marks |
| `ORIGINAL` | Paragraph with no citation markers — highest plagiarism risk |
| `BIBLIOGRAPHY` | Reference list section (excluded from zone output) |

```bash
curl -X POST http://localhost:5006/api/v2/citations/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Smith et al. (2020) demonstrated that... References\n[1] Smith, J. et al., Nature, 2020."}'
```

**Supported citation styles**

| Style | Example |
|---|---|
| APA | `(García & López, 2021)` / `García (2021, p. 45)` |
| APA multi | `(García, 2021; López, 2020; Martínez, 2019)` |
| MLA | `(García 45)` |
| IEEE | `[1]` / `[2,3]` / `[1-4]` |
| Chicago | `(García 2021, 45)` |
| Vancouver | `^1^` / `(1,2,3)` |
| Harvard | `García, 2021` |

---

### POST /api/v2/citations/validate

Asynchronous bibliography validation. Queries CrossRef, OpenAlex, and Semantic Scholar in parallel for each reference. Requires network access and `aiohttp`. SSRF-protected — only the three academic API hosts are reachable.

**Request — from full text**
```json
{ "text": "... full document with bibliography section ..." }
```

**Request — from raw reference list**
```json
{
    "bibliography": [
        "García, M., & López, J. (2021). Transformación digital en universidades. Revista de Educación Superior, 15(3), 45-67. https://doi.org/10.1234/res.2021.003",
        "Smith, A. (2020). Machine learning in higher education. Journal of Educational Technology, 8(2), 123-145."
    ]
}
```

**Response**
```json
{
    "total": 2,
    "valid": 1,
    "partial": 0,
    "not_found": 1,
    "unverifiable": 0,
    "results": [
        {
            "key": "Garcia2021",
            "raw": "García, M., & López, J. (2021). Transformación digital...",
            "validation": {
                "status": "valid",
                "confidence": 94.0,
                "source_api": "crossref",
                "found_title": "Transformación digital en universidades latinoamericanas",
                "found_doi": "10.1234/res.2021.003",
                "discrepancies": []
            }
        },
        {
            "key": "Smith2020",
            "raw": "Smith, A. (2020). Machine learning in higher education...",
            "validation": {
                "status": "not_found",
                "confidence": 0.0,
                "source_api": null,
                "found_title": null,
                "found_doi": null,
                "discrepancies": ["Title not found in CrossRef, OpenAlex, or Semantic Scholar"]
            }
        }
    ]
}
```

**Validation statuses**

| Status | Meaning |
|---|---|
| `valid` | Found in at least one academic API with high confidence (≥0.8) |
| `partial` | Found but with minor discrepancies (year mismatch, abbreviated title) |
| `not_found` | Not found in any of the three APIs after all cascade strategies |
| `unverifiable` | Only a URL — no title or DOI to query |
| `error` | `aiohttp` not installed or network error |

```bash
# Validate all references found in a document
curl -X POST http://localhost:5006/api/v2/citations/validate \
  -H "Content-Type: application/json" \
  -d '{"text": "Full document text with references section..."}'

# Validate a raw reference list
curl -X POST http://localhost:5006/api/v2/citations/validate \
  -H "Content-Type: application/json" \
  -d '{
    "bibliography": [
      "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436-444.",
      "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS."
    ]
  }'
```

---

## Usage Examples

All examples assume the server is running on `http://localhost:5006`.

---

### curl

#### Detect AI in a short text
```bash
curl -X POST http://localhost:5006/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The rapid integration of artificial intelligence in academic settings has sparked a profound ethical debate. Institutions worldwide are grappling with questions of authenticity, intellectual integrity, and the boundaries of acceptable AI assistance.",
    "plugins": ["ai_detection"]
  }'
```

#### Run multiple plugins at once
```bash
curl -X POST http://localhost:5006/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "plugins": ["ai_detection", "perplexity_check", "zone_classifier"]
  }'
```

#### Stream results as plugins complete (SSE)
```bash
curl -N -X POST http://localhost:5006/analyze_stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "text": "Your text here...",
    "plugins": ["ai_detection", "perplexity_check", "citation_check"]
  }'
```

#### Analyze a full document with per-segment breakdown
```bash
curl -X POST http://localhost:5006/analyze_document \
  -H "Content-Type: application/json" \
  -d '{
    "text": "First paragraph of the document. It introduces the topic and sets the context for subsequent discussion.\n\nSecond paragraph provides supporting evidence and examples drawn from recent literature.\n\nThe conclusion synthesizes the findings and proposes directions for future research.",
    "plugins": ["ai_detection"]
  }'
```

#### Full forensic pipeline
```bash
curl -X POST http://localhost:5006/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "plugins": ["full_analysis"]
  }'
```

#### Detect citation style and zone classification
```bash
curl -X POST http://localhost:5006/api/v2/citations/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Recent advances [1] have shown significant improvements. The BERT model [2] achieves 94% accuracy.\n\nReferences\n[1] LeCun, Y., et al. Deep learning. Nature, 2015.\n[2] Devlin, J., et al. BERT. NAACL, 2019."
  }'
```

#### Validate bibliography against academic APIs
```bash
curl -X POST http://localhost:5006/api/v2/citations/validate \
  -H "Content-Type: application/json" \
  -d '{
    "bibliography": [
      "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS, pp. 5998-6008.",
      "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436-444."
    ]
  }'
```

#### Check readiness before sending requests
```bash
curl -s http://localhost:5006/ready | python3 -m json.tool
```

#### List all available plugins
```bash
curl -s http://localhost:5006/plugins | python3 -m json.tool
```

---

### Python — `requests`

#### Basic AI detection
```python
import requests

response = requests.post(
    "http://localhost:5006/analyze",
    json={
        "text": (
            "The rapid integration of artificial intelligence in academic settings "
            "has sparked a profound ethical debate. Institutions worldwide are "
            "grappling with questions of authenticity and intellectual integrity."
        ),
        "plugins": ["ai_detection"],
    },
)
data = response.json()

result = data["results"]["ai_detection"]["data"]
print(f"Prediction:  {result['prediction']}")
print(f"Confidence:  {result['confidence']:.2f}%")
print(f"Human:       {result['human_percentage']:.2f}%")
print(f"AI:          {result['ai_percentage']:.2f}%")
print(f"Model hint:  {result['detected_model']}")
print(f"Uncertain:   {result['uncertainty_zone']}")
print(f"From cache:  {data.get('from_cache', False)}")
```

#### Streaming — receive plugin results as they complete
```python
import json
import requests

with requests.post(
    "http://localhost:5006/analyze_stream",
    json={"text": "Your text...", "plugins": ["ai_detection", "perplexity_check", "zone_classifier"]},
    stream=True,
    headers={"Accept": "text/event-stream"},
    timeout=120,
) as resp:
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:])
        if event["type"] == "init":
            print(f"Analyzing {event['word_count']} words with {event['plugins']}")
        elif event["type"] == "result":
            r = event["result"]
            print(f"[{event['plugin']}] {r['status']} — {r.get('elapsed_ms')}ms")
        elif event["type"] == "done":
            print("All plugins complete.")
```

#### Analyze document with segments
```python
import requests

text = """
The emergence of large language models has fundamentally changed the way
humans interact with information systems.

These models, trained on billions of tokens, demonstrate remarkable fluency
across domains ranging from legal analysis to creative writing.

However, the implications for education and academic integrity remain
a subject of active debate among researchers and institutional policymakers.
"""

response = requests.post(
    "http://localhost:5006/analyze_document",
    json={"text": text, "plugins": ["ai_detection"]},
)
data = response.json()

ai = data["results"]["ai_detection"]["data"]
print(f"Global prediction: {ai['prediction']} ({ai['confidence']:.2f}%)")
print(f"Segments analyzed: {len(ai.get('segments', []))}")

for seg in ai.get("segments", []):
    print(f"  [{seg['segment_id']}] {seg['dominant_label']:7s}  {seg['score']:.2f}%  {seg['text'][:60]}...")
```

#### Detect citation zones (zone_classifier plugin)
```python
import requests

text = """
La inteligencia artificial ha transformado radicalmente los métodos de enseñanza
en la educación superior (García & López, 2021). Según Smith et al. (2020), el
aprendizaje automático permite personalizar el contenido educativo.

Como señala Johnson (2019, p. 45): "La adaptación del currículo mediante
algoritmos reduce en un 40% el tiempo de aprendizaje".

Referencias
García, M., & López, J. (2021). Transformación digital. Revista de Educación, 15(3), 45-67.
Smith, A. et al. (2020). Machine learning in education. J. Ed. Technology, 8(2), 123-145.
Johnson, R. (2019). Adaptive learning systems. Oxford University Press.
"""

response = requests.post(
    "http://localhost:5006/analyze",
    json={"text": text, "plugins": ["zone_classifier"]},
)
data = response.json()
zc = data["results"]["zone_classifier"]["data"]

print(f"Style:       {zc['dominant_style']}")
print(f"Consistency: {zc['style_consistency']}%")
print(f"Coverage:    {zc['citation_coverage']}%")
print(f"Citations:   {zc['total_inline_citations']} inline, {zc['total_bibliography']} in bibliography")

for zone in zc["zones"]:
    risk = "HIGH" if zone["plagiarism_risk"] > 0.5 else "LOW"
    cited = "cited" if zone["has_citation"] else "UNCITED"
    print(f"  [{zone['type']:15s}] {cited:7s}  risk={risk}  {zone['text_preview'][:60]}...")
```

#### Validate citations against academic APIs
```python
import requests

response = requests.post(
    "http://localhost:5006/api/v2/citations/validate",
    json={
        "bibliography": [
            "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436-444.",
            "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS, pp. 5998-6008.",
        ]
    },
    timeout=60,
)
data = response.json()

print(f"Total: {data['total']}  Valid: {data['valid']}  Not found: {data['not_found']}")
for entry in data["results"]:
    v = entry["validation"]
    print(f"  [{v['status']:12s}] {v['confidence']}%  via {v['source_api']}  {entry['raw'][:60]}...")
```

#### Run multiple plugins and handle errors
```python
import requests

def analyze(text: str, plugins: list[str], base_url: str = "http://localhost:5006") -> dict:
    resp = requests.post(
        f"{base_url}/analyze",
        json={"text": text, "plugins": plugins},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

result = analyze(
    text="Your text here...",
    plugins=["ai_detection", "perplexity_check", "zone_classifier"],
)

for plugin_name, plugin_result in result["results"].items():
    if plugin_result["status"] == "ok":
        print(f"{plugin_name}: {plugin_result['data']}")
    else:
        print(f"{plugin_name}: ERROR — {plugin_result.get('error')}")
```

---

### Python — `httpx` (async)

```python
import asyncio
import httpx

async def analyze_async(text: str, plugins: list[str]) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "http://localhost:5006/analyze",
            json={"text": text, "plugins": plugins},
        )
        response.raise_for_status()
        return response.json()

async def main():
    texts = [
        "First document to analyze...",
        "Second document to analyze...",
        "Third document to analyze...",
    ]

    tasks = [analyze_async(text, ["ai_detection"]) for text in texts]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        ai = result["results"]["ai_detection"]["data"]
        print(f"Doc {i+1}: {ai['prediction']}  {ai['confidence']:.2f}%")

asyncio.run(main())
```

---

### JavaScript / Node.js — `fetch`

```js
const BASE_URL = "http://localhost:5006";

async function analyzeText(text, plugins = ["ai_detection"]) {
  const response = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, plugins }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}

async function analyzeStream(text, plugins, onResult) {
  const response = await fetch(`${BASE_URL}/analyze_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, plugins }),
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));
      onResult(event);
    }
  }
}

async function detectCitations(text) {
  const response = await fetch(`${BASE_URL}/api/v2/citations/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}

// Usage
const text = `García & López (2021) found that AI reduces learning time by 40%.
Smith et al. (2020) corroborate these findings.

Referencias
García, M., & López, J. (2021). Transformación digital. Rev. Ed. Superior, 15(3), 45-67.`;

// AI detection (cached on second call)
const aiResult = await analyzeText(text, ["ai_detection"]);
const ai = aiResult.results.ai_detection.data;
console.log(`${ai.prediction} — ${ai.confidence.toFixed(2)}%`);

// Streaming
await analyzeStream(text, ["ai_detection", "zone_classifier"], (event) => {
  if (event.type === "result") console.log(`[${event.plugin}]`, event.result.status);
});

// Citation detection
const citeResult = await detectCitations(text);
console.log(`Style: ${citeResult.dominant_style}, Citations: ${citeResult.inline_citations.length}`);
citeResult.zones.forEach(z => {
  console.log(`  [${z.type}] risk=${z.plagiarism_risk}  cited=${z.has_citation}  ${z.text_preview.slice(0, 60)}...`);
});
```

---

### Error responses

All endpoints return a JSON error body with an appropriate HTTP status code.

| Status | Cause | Body |
|---|---|---|
| `400` | Missing or invalid `text` field | `{"error": "'text' field is required and must be a non-empty string"}` |
| `400` | Invalid JSON body | `{"error": "Invalid JSON body"}` |
| `400` | `plugins` is not a non-empty list | `{"error": "'plugins' must be a non-empty list"}` |
| `400` | `/api/v2/citations/detect` missing `text` | `{"error": "Campo 'text' requerido"}` |
| `400` | `/api/v2/citations/validate` missing both `text` and `bibliography` | `{"error": "Se requiere 'text' o 'bibliography'"}` |
| `401` | Missing or invalid `X-API-Key` header (when `API_KEY` is set) | `{"error": "Unauthorized"}` |
| `415` | Missing `Content-Type: application/json` | `{"error": "Content-Type must be application/json"}` |
| `429` | Rate limit exceeded | `{"error": "Too Many Requests"}` |
| `503` | `ai_detection` plugin not loaded | `{"error": "ai_detection plugin not available"}` |

When a plugin runs but fails internally, the response is still `200` and the individual plugin entry will have `"status": "error"`:

```json
{
    "status": "ok",
    "results": {
        "ai_detection": {
            "status": "error",
            "error": "ModernBERT models not loaded. Check model paths.",
            "elapsed_ms": 0.3
        }
    }
}
```

---

## Quick Start

### Local development

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies (locked + hash-verified)
make install
# or: pip install -r requirements.txt

# 3. Place model files in app/engine/
#    app/engine/modernbert.bin
#    app/engine/Model_groups_3class_seed12
#    app/engine/Model_groups_3class_seed22

# 4. Ensure the ModernBERT tokenizer is cached locally
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')"

# 5. Start the dev server
python app.py
```

### Dependency management

```bash
# Update requirements.in with abstract deps (no version pins), then:
make lock      # pip-compile --generate-hashes → requirements.txt with SHA-256 hashes
make install   # pip install --require-hashes --no-deps -r requirements.txt

# Run quality gates
make test      # pytest tests/ -v --tb=short
make lint      # ruff check app/ tests/
make typecheck # mypy app/ --ignore-missing-imports
```

`requirements.in` is the source of truth for abstract dependencies. `requirements.txt` is the compiled, fully-pinned, hash-verified lockfile — commit both files together.

### Production (gunicorn)

```bash
gunicorn --preload -c gunicorn.conf.py "app:create_app()"
```

### Docker

#### Build

```bash
docker build -t xplagiax_xota:latest .
```

#### Run — standard (single container, no Redis)

```bash
docker run -d \
  --name xplagiax-xota \
  -p 5006:5006 \
  -e WEB_CONCURRENCY=2 \
  -e LOG_LEVEL=info \
  xplagiax_xota:latest
```

#### Run — production (with Redis for rate limiting, caching + async tasks)

```bash
# 1. Create network (if it doesn't exist)
docker network create xplagiax-net

# 2. Start Redis (with optional password)
docker run -d \
  --name redis \
  --network xplagiax-net \
  -p 6379:6379 \
  -e REDIS_PASSWORD=your-redis-password \
  redis:7-alpine \
  sh -c 'if [ -n "$REDIS_PASSWORD" ]; then redis-server --requirepass "$REDIS_PASSWORD"; else redis-server; fi'

# 3. Stop any existing container
docker stop xplagiax-xota 2>/dev/null || true
docker rm   xplagiax-xota 2>/dev/null || true

# 4. Run the service
docker run -d \
  --name xplagiax-xota \
  --network xplagiax-net \
  --restart unless-stopped \
  -p 5006:5006 \
  -e WEB_CONCURRENCY=2 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-secret-key \
  -e API_KEY=your-api-key \
  -e REDIS_URL="redis://:your-redis-password@redis:6379" \
  -e REDIS_PASSWORD=your-redis-password \
  -e REDIS_MAX_CONNECTIONS=10 \
  -e CELERY_BROKER_URL="redis://:your-redis-password@redis:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://:your-redis-password@redis:6379/1" \
  -e CROSSREF_EMAIL="your@institution.edu" \
  -e LOG_LEVEL=info \
  xplagiax_xota:latest

# 5. Verify health
curl http://localhost:5006/health
curl http://localhost:5006/ready
```

#### Run — with mounted models (avoids embedding ~1.7 GB in the image)

```bash
docker run -d \
  --name xplagiax-xota \
  --network xplagiax-net \
  --restart unless-stopped \
  -p 5006:5006 \
  -v /path/to/models:/app/app/engine \
  -e WEB_CONCURRENCY=2 \
  -e REDIS_URL="redis://redis:6379" \
  -e CROSSREF_EMAIL="your@institution.edu" \
  xplagiax_xota:latest
```

> **Note**: `preload_app=True` ensures the three ModernBERT models (~1.7 GB total) are loaded once in the Gunicorn master process and shared across all workers via Linux Copy-on-Write. Each worker adds only ~50 MB overhead instead of another full model copy.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | Set to `production` to enable production guards (requires SECRET_KEY, API_KEY) |
| `SECRET_KEY` | _(auto-generated)_ | Flask session secret. **Required in production.** Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `API_KEY` | _(empty)_ | API key for `X-API-Key` header authentication. Empty = auth disabled. **Required in production.** |
| `DEBUG` | `0` | Set to `1` to enable Flask debug mode. Blocked in production (Werkzeug RCE risk). |
| `WEB_CONCURRENCY` | `2×CPU (max 8)` | Number of gunicorn workers |
| `GUNICORN_TIMEOUT` | `120` | Hard worker kill timeout (seconds) |
| `GRACEFUL_TIMEOUT` | `30` | Soft shutdown window (seconds) |
| `MAX_CONTENT_MB` | `16` | Max request body size in MB |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warning`, `error`) |
| `PLUGIN_TIMEOUT` | `30` | Max seconds per plugin execution |
| `PERPLEXITY_TIER2` | `1` | Enable GPT-2 Tier 2 perplexity (`0` to disable) |
| `PERPLEXITY_DICT_PATH` | _(none)_ | Path to custom n-gram frequency dictionary |
| `REFERENCE_NETWORK` | `1` | Enable live citation network calls (`0` to disable) |
| `ENABLE_REFERENCE_CHECK` | `0` | Include citation check in `full_analysis` pipeline |
| `ENABLE_WATERMARK` | `0` | Include watermark detection in `full_analysis` pipeline |
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL (rate limiter + cache + Celery) |
| `REDIS_PASSWORD` | _(empty)_ | Redis password — appended to connection URL when set |
| `REDIS_MAX_CONNECTIONS` | `10` | Redis connection pool cap per process |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Celery broker URL |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/1` | Celery result backend URL |
| `CROSSREF_EMAIL` | `antiplagio@example.com` | Email sent to CrossRef Polite Pool. **Set to a real institutional address in production.** |
| `GUNICORN_SPAWN_CELERY` | `1` | Spawn an internal Celery worker from the gunicorn master (`0` to disable for separate worker deployment) |

---

## Project Structure

```
xplagiax_xota/
├── app.py                        # Entry point — gunicorn target
├── gunicorn.conf.py              # Production server config
├── requirements.in               # Abstract (unpinned) dependencies — source of truth
├── requirements.txt              # Compiled lockfile with SHA-256 hashes (pip-compile output)
├── Makefile                      # lock / install / test / lint / typecheck targets
├── Dockerfile                    # Multi-stage container build
├── docker-compose.yml            # Full stack: web + celery + redis
│
├── app/
│   ├── __init__.py               # create_app() factory + security guards
│   ├── config.py                 # All config via env vars (12-factor)
│   ├── routes.py                 # API blueprints (/analyze, /analyze_document, /analyze_stream, ...)
│   ├── tasks.py                  # Celery background tasks
│   ├── celery_app.py             # Celery worker entry point
│   ├── plugin_registry.py        # Auto-discovery, parallel dispatch, run_stream() SSE
│   │
│   ├── antiplagio/               # Citation detection and validation package
│   │   ├── __init__.py
│   │   ├── flask_routes.py       # Blueprint /api/v2/ (detect + validate, rate-limited)
│   │   └── citation/
│   │       ├── __init__.py
│   │       ├── detector.py       # CitationDetector — regex + spaCy fallback
│   │       └── validator.py      # CitationValidator — async CrossRef/OpenAlex/SemanticScholar
│   │
│   ├── plugins/
│   │   ├── base.py               # BasePlugin interface
│   │   ├── ai_detection.py       # Binary AI/Human classification
│   │   ├── segment_analysis.py   # Per-paragraph heatmap (preview)
│   │   ├── perplexity_check.py   # Text predictability analysis
│   │   ├── stylometric_analysis.py # Writing style fingerprinting
│   │   ├── hallucination_check.py # AI fabrication risk detection
│   │   ├── reasoning_check.py     # Reasoning-model detection (o1/R1)
│   │   ├── citation_check.py     # Reference existence verification
│   │   ├── watermark_detection.py # Digital watermark detection
│   │   ├── zone_classifier.py    # Citation zone detection plugin
│   │   ├── forensic_report.py    # HTML forensic report (decoupled via get_orchestrator())
│   │   └── full_analysis.py      # Complete forensic pipeline (orchestrator singleton)
│   │
│   └── engine/                   # XplagiaX core engine
│       ├── __init__.py           # sys.path setup + torch/transformers patch
│       ├── detector_final.py     # 3-model ModernBERT ensemble + analyze_fast() + TTL cache
│       ├── hybrid_segment_detector.py   # Sliding-window segment classifier
│       ├── perplexity_profiler.py       # n-gram + GPT-2 perplexity
│       ├── stylometric_profiler.py      # Writing style fingerprinting
│       ├── hallucination_profile.py     # Fabrication risk detection
│       ├── reasoning_profiler.py        # Reasoning-model detection
│       ├── reference_validator.py       # Citation verification (SSRF-protected)
│       ├── watermark_decoder.py         # Digital watermark detection
│       ├── forensic_reports.py          # HTML/JSON report generator (v3.9)
│       ├── plugin_orchestrator.py       # Pipeline coordinator + singleton factory
│       ├── modernbert.bin               # Model weights (~600 MB)
│       ├── Model_groups_3class_seed12   # Model weights (~600 MB)
│       └── Model_groups_3class_seed22   # Model weights (~600 MB)
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_citation_system.py   # pytest tests for CitationDetector
```

---

## Security Hardening (May 2026)

| ID | Severity | Module | Fix Applied |
|---|---|---|---|
| S-01 | **HIGH** | `app/routes.py` | `hmac.compare_digest()` for timing-safe API key comparison — prevents timing-attack key enumeration. |
| S-02 | **HIGH** | `app/__init__.py` | Startup `RuntimeError` if `DEBUG=True` and `FLASK_ENV=production` — blocks Werkzeug interactive debugger (remote code execution risk). |
| S-03 | **HIGH** | `app/routes.py` | `X-Request-ID` sanitized — only alphanumeric + hyphens accepted from caller header; generated UUID otherwise. |
| S-04 | **MEDIUM** | `app/routes.py` | Content-type enforcement — all POST endpoints return `415` if `Content-Type` is not `application/json`. |
| S-05 | **HIGH** | `app/__init__.py` | Startup `RuntimeError` if `API_KEY` is empty in production; `WARNING` log in development so the misconfiguration is visible. |
| S-06 | **HIGH** | `app/engine/reference_validator.py` | SSRF protection — `_ALLOWED_API_HOSTS` allowlist (CrossRef, OpenAlex, Semantic Scholar) + `_NoRedirectHandler` blocks redirect chains before any HTTP request is made. |
| S-07 | **MEDIUM** | `app/antiplagio/citation/validator.py` | External URL validation before aiohttp requests — scheme must be `https`, host must be in the academic API allowlist. |
| S-08 | **MEDIUM** | `app/routes.py` | `serve_report()` validates `forensic_` prefix on filename before serving from `/tmp`. Adds strict `Content-Security-Policy` and `X-Content-Type-Options: nosniff` response headers. |
| S-09 | **MEDIUM** | `docker-compose.yml` | Redis `requirepass` is conditional — runs `redis-server --requirepass $REDIS_PASSWORD` only when the env var is set, safe for dev without a password. |
| S-10 | **LOW** | `app/routes.py` | `word_count` capped at 200,000 words before processing — prevents memory exhaustion on pathologically large inputs. |
| S-11 | **MEDIUM** | `Dockerfile` | Build-stage `sed` narrowed to `/^torch/d` only — previously removed `numpy`, `transformers`, `spacy` version pins, leaving them unconstrained in the runtime image. |
| S-12 | **LOW** | `app/__init__.py` | Startup `WARNING` if `CROSSREF_EMAIL` ends with `@example.com` — CrossRef Polite Pool throttles or blocks example.com addresses. |

---

## Technical Debt Resolved (May 2026)

| ID | Severity | Module | Problem | Fix |
|---|---|---|---|---|
| DT-01 | **CRITICAL** | `app/tasks.py` | Double inference: task ran full ML pipeline after plugin already ran it. | Reuse plugin `segments`; fall back only when `ai_detection` not requested. |
| DT-02 | **CRITICAL** | `app/engine/detector_final.py` | CoW violation: PyTorch ref-counting duplicated 1.71 GB of model weights per worker (3 workers = 5.1 GB). | `m.share_memory()` after each model load → POSIX shared memory, no CoW faults. |
| DT-03 | **HIGH** | `app/tasks.py` | Redis result bloat: base64 chart fields (300 KB–1 MB each) serialized into Celery results with 1h TTL. | `_strip_base64()` helper strips all `*_b64` keys before Redis serialization. |
| DT-04 | **HIGH** | `app/celery_app.py` | Celery result TTL 3600s allowed 1h of large results to accumulate. | Reduced `result_expires` to 600s (10 min). |
| DT-05 | **HIGH** | `app/antiplagio/flask_routes.py` | No rate limits on `/api/v2/citations/` routes. | `@limiter.limit("60/minute")` on `detect_citations`; `@limiter.limit("10/minute")` on `validate_citations`. |
| DT-06 | **HIGH** | `app/antiplagio/citation/detector.py` | Multi-citation parentheticals `(A, 2021; B, 2020)` silently dropped — `APA_INLINE` only matched single-author pairs. | Added `APA_MULTI_PAREN` regex; preprocessing splits semicolons into individual `CitationMarker` objects. |
| DT-07 | **HIGH** | `app/antiplagio/citation/detector.py` | `import spacy` crash if spaCy not installed — entire citation module failed to load. | Separated import from `spacy.load()`. `ImportError`/`OSError` sets `_SPACY_AVAILABLE=False`, falls back to rule-based segmentation. |
| DT-08 | **MEDIUM** | `app/antiplagio/citation/validator.py` | Unconditional `import aiohttp` — module failed to import if `aiohttp` not installed. | `try/except ImportError` → `_AIOHTTP_AVAILABLE=False`; `validate_all()` returns `ERROR` results gracefully. |
| DT-09 | **MEDIUM** | `app/antiplagio/flask_routes.py` | `async_route` decorator leaked event loop into gevent global state — no `asyncio.set_event_loop(None)` after `loop.close()`. | Added `asyncio.set_event_loop(None)` in `finally` block. |
| DT-10 | **MEDIUM** | `app/antiplagio/flask_routes.py` | Dead code in `validate_citations`: `_split_bibliography()` result discarded after computation. | Removed unused call; `bibliography` obtained directly from `analysis.bibliography`. |
| DT-11 | **LOW** | `app/plugins/zone_classifier.py` | `CitationDetector()` instantiated per-request, rebuilding all compiled regex patterns on every call. | Moved to module level — singleton shared across workers via CoW. |
| DT-12 | **MEDIUM** | `app/plugins/forensic_report.py` | Tight coupling to `full_analysis._orchestrator` (private attribute import) — circular dependency risk. | `forensic_report.py` now calls `get_orchestrator()` from `app.engine.plugin_orchestrator`. No private attribute access. |
| DT-13 | **LOW** | `app/__init__.py` | No request correlation ID — log lines from parallel requests were impossible to correlate. | `X-Request-ID` set in `before_request` (from caller header or generated UUID); echoed in `after_request`. |
| DT-14 | **MEDIUM** | `app/engine/plugin_orchestrator.py` | No public API for `ForensicReportGenerator.export_html` — callers accessed `_forensic_generator` directly. | Added `export_html(self, forensic_report, output_path)` public method + module-level singleton factory (`initialize_orchestrator` / `get_orchestrator`). |
| DT-15 | **MEDIUM** | `app/routes.py` | `analyze_document` called deprecated `analyze_long_document()` — repeated tokenization per chunk, no embedding reuse, 2–12× slower than `analyze_fast()`. | Replaced with `analyze_fast(text)` — single-pass tokenization, adaptive `max_tokens`, token-boundary splits, TTL result cache. |

---

## Performance Improvements (May 2026)

| ID | Module | Improvement |
|---|---|---|
| P-01 | `app/plugin_registry.py` | `ThreadPoolExecutor` max_workers raised to 8 — ML plugins release GIL during C-level inference, so threads run truly in parallel on multi-core machines. |
| P-02 | `app/plugin_registry.py` | Per-plugin individual timeout in `future.result(timeout=...)` — each plugin gets its full budget; a slow plugin can't starve others checked later. |
| P-03 | `app/plugin_registry.py` | `run_stream()` generator using `as_completed()` — clients receive fast plugin results immediately without waiting for the slowest plugin. |
| P-04 | `app/routes.py` | Result cache: `sha256(text + plugins)` keyed in Redis/SimpleCache with 1h TTL. Repeat requests: 0ms inference, immediate response, `"from_cache": true`. |
| P-05 | `app/routes.py` | `analyze_document` replaced deprecated chunked `analyze_long_document` with `analyze_fast()` — single tokenization pass, 2–12× faster for long documents. |
| P-06 | `app/engine/detector_final.py` | `analyze_fast()` uses adaptive `max_tokens` (capped at actual sequence length) — no padding waste for short texts. |
| P-07 | `app/engine/detector_final.py` | Thread-safe TTL cache for `analyze_fast()` results (`_FAST_CACHE`, 5-min TTL, 20-entry LRU) — multiple plugins requesting same text in one request hit cache on second call. |
| P-08 | `app/config.py` | Redis connection pool cap via `CACHE_OPTIONS = {"max_connections": N}` — prevents runaway connections under burst traffic. |
| P-09 | `app/__init__.py` | Rate limiter `storage_options = {"max_connections": N}` — same pool cap for the limiter's Redis client. |
| P-10 | `requirements.in` / `Makefile` | `pip-compile --generate-hashes` infrastructure — supply-chain-safe SHA-256 hash pinning without manual maintenance. |

---

## Bug Fix & Improvement Table (May 2026)

| # | Severity | Module | Bug / Problem | Fix Applied |
|---|---|---|---|---|
| 1 | **CRITICAL** | `app/tasks.py` | Redundant double inference: `analyze_document_async` ran `analyze_long_document()` **after** `ai_detection` plugin had already performed identical inference. Every request ran model inference twice. | Reuse `segments` already computed by the plugin. Fall back to direct inference only when `ai_detection` was not requested. |
| 2 | **CRITICAL** | `app/engine/detector_final.py` | Copy-on-Write violation: PyTorch's internal reference counting wrote to the same virtual pages as model weights on first inference, causing all 1.71 GB of weights to be duplicated per worker. With 2 web + 1 Celery worker = 5.13 GB extra. | Added `m.share_memory()` after each model load. Moves tensor storage to POSIX shared memory — read-only across all forked processes, no CoW faults. |
| 3 | **HIGH** | `app/tasks.py` | Redis result bloat: Celery serialized full task results including `heatmap_b64`, `confidence_chart_b64`, `comparison_chart_b64` (300 KB–1 MB each). With 1-hour TTL, accumulated hundreds of MB under load. | Added `_strip_base64()` helper to remove all `*_b64` keys before Redis serialization. |
| 4 | **HIGH** | `app/celery_app.py` | Celery result TTL of 3600s allowed 1 hour of large results to accumulate in Redis simultaneously under moderate load. | Reduced `result_expires` from 3600s to 600s (10 minutes). |
| 5 | **HIGH** | `app/antiplagio/citation/detector.py` | Multi-citation parentheticals `(García, 2021; López, 2020; Martínez, 2019)` were not detected — `APA_INLINE` requires a single author-year pair per parenthetical. | Added `APA_MULTI_PAREN` regex pattern. Processing block splits semicolon-separated citations into individual `CitationMarker` objects. |
| 6 | **HIGH** | `app/antiplagio/citation/detector.py` | spaCy import failure would crash the entire module if `spacy` was not installed. | Separated `import spacy` from `spacy.load()`. `ImportError` sets `_SPACY_AVAILABLE = False` and falls back to rule-based sentence segmentation. |
| 7 | **MEDIUM** | `app/antiplagio/citation/validator.py` | `aiohttp` dependency was unconditional — if not installed, the entire validator module failed to import. | Wrapped in `try/except ImportError`. `_AIOHTTP_AVAILABLE = False` when missing. |
| 8 | **MEDIUM** | `app/antiplagio/flask_routes.py` | `async_route` decorator leaked event loop — no `asyncio.set_event_loop(None)` after `loop.close()`, risking interference between concurrent requests. | Added `asyncio.set_event_loop(None)` in the `finally` block. |
| 9 | **MEDIUM** | `app/antiplagio/flask_routes.py` | Dead code in `validate_citations`: `_split_bibliography()` result computed and discarded, wasting CPU. | Removed unused call. `bibliography` obtained directly from `analysis.bibliography`. |
| 10 | **LOW** | `app/plugins/zone_classifier.py` | `CitationDetector()` instantiated inside every `analyze()` call — rebuilt all compiled regex patterns per request. | Moved to module level — singleton instantiated once at gunicorn preload, shared via CoW. |
| 11 | **HIGH** | `app/engine/reference_validator.py` | No SSRF protection — any URL could be passed to the citation validator, allowing internal network probing. | Added `_ALLOWED_API_HOSTS` allowlist + `_NoRedirectHandler` that blocks HTTP redirects before the first request lands. |
| 12 | **MEDIUM** | `app/routes.py` | `serve_report()` served any file from `/tmp` by basename — no file-type guard, no security headers. | Added `forensic_` prefix validation; added `Content-Security-Policy` and `X-Content-Type-Options: nosniff` headers. |

---

## Memory & Performance Optimizations (May 2026)

To operate efficiently on memory-constrained Virtual Private Servers (VPS), several architectural optimizations have been implemented:

### 1. Single-Container CoW Architecture
Running a separate Celery worker container duplicates the ~1.7 GB ModernBERT models in memory. Instead, the application uses a **single container** where Gunicorn's master process loads the models once (`preload_app=True`) and internally forks the Celery worker (`when_ready` hook).
* **Result:** Total memory usage drops from ~6.8 GB to ~2.8 GB because the models are shared via Linux Copy-on-Write (CoW).

### 2. Gunicorn Worker Limits
ML workloads are memory-intensive (each worker adds ~200MB+ overhead). The Gunicorn config limits workers to `2` by default instead of `2 * CPU_count`.

### 3. Celery Memory Management
Celery tasks are configured to prevent memory leaks and zombie processes during long document analysis:
* **Garbage Collection:** Explicit `gc.collect()` and `torch.cuda.empty_cache()` are called in the `finally` block of every task (`tasks.py`).
* **Timeouts:** Tasks have a hard `time_limit=300s` to kill runaway processes before they exhaust memory.
* **Result TTL:** `result_expires = 600` prevents the Redis backend from accumulating stale task results indefinitely.
* **Prefetch Limits:** `worker_prefetch_multiplier = 1` ensures the worker only pulls one heavy ML task at a time.
* **Auto-Restart:** A `child_exit` hook monitors the internal Celery worker and automatically restarts it if it crashes due to an OOM or segfault.

### 4. Concurrency & Capacity Limits
* **Memory Leak Prevention:** The explicit `gc.collect()` after each analysis ensures no "residue" memory or orphaned tensors remain.
* **Simultaneous Users:** The system is strictly capped to process **3 concurrent heavy analyses** at exactly the same time (2 Web Workers + 1 Celery Worker).
* **Queuing:** If 50–100 users submit simultaneously, the system will not crash. It will process 3 immediately while safely queuing the remaining requests in Redis (async) or Gunicorn's backlog.

### Deploying the Optimized Container

```bash
# 1. Stop and remove any separate celery worker to free up RAM
docker stop xplagiax_xota_worker 2>/dev/null || true
docker rm xplagiax_xota_worker 2>/dev/null || true

# 2. Rebuild the image
docker build -t xplagiax_xota:latest .

# 3. Launch the single container (Web + Internal Celery Worker)
docker stop xplagiax-xota 2>/dev/null || true
docker run -d \
  --name xplagiax-xota \
  --network xplagiax-net \
  --restart unless-stopped \
  -p 5006:5006 \
  -e WEB_CONCURRENCY=2 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-secret-key \
  -e API_KEY=your-api-key \
  -e REDIS_URL="redis://redis:6379" \
  -e CELERY_BROKER_URL="redis://redis:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://redis:6379/1" \
  -e CROSSREF_EMAIL="your@institution.edu" \
  xplagiax_xota:latest



docker run -d \
  --name xplagiax-xota \
  --network xplagiax-net \
  --restart unless-stopped \
  -p 5006:5006 \
  -e WEB_CONCURRENCY=2 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-secret-key \
  -e API_KEY=your-api-key \
  -e REDIS_URL="redis://redis:6379" \
  -e CELERY_BROKER_URL="redis://redis:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://redis:6379/1" \
  -e CROSSREF_EMAIL="rgonzalez@uryxtech.com" \
  xplagiax_xota:latest

# 4. Verify the internal Celery worker is running alongside Gunicorn
docker exec xplagiax-xota ps aux | grep -E "gunicorn|celery"



curl -X POST http://localhost:5006/analyze_document_async -H "Content-Type: application/json" -H "X-API-Key: 7d9a2c4f8e1b3d5a6f7c9e2b4a1d8c3f" -d '{"text":" In modern workplaces, EI is often considered as important as technical expertise. Employees with high emotional intelligence tend to work better in teams. They know how to manage conflicts, stay calm under pressure, and build strong professional relationships. This emotional stability contributes to a healthier work atmosphere where productivity naturally increases. Leaders who possess emotional intelligence are especially effective. Instead of managing through fear or authority, they inspire trust and motivation. Their empathy helps them understand their team'\''s needs and align individual goals with organizational objectives","plugins":["ai_detection"]}'

docker run -d   --name xplagiax-xota   --network xplagiax-net   --restart unless-stopped   -p 5006:5006   -e WEB_CONCURRENCY=2   -e FLASK_ENV=production   -e SECRET_KEY="a3f7d9c4e8b1f6a2c5d7e9f1a3b5c7d9e2f4a6b8c1d3e5f7a9b1c3d5e7f9a2b"   -e API_KEY="7d9a2c4f8e1b3d5a6f7c9e2b4a1d8c3f"   -e REDIS_URL="redis://redis:6379"   -e CELERY_BROKER_URL="redis://redis:6379/0"   -e CELERY_RESULT_BACKEND="redis://redis:6379/1"   -e CROSSREF_EMAIL="rgonzalez@uryxtech.com"   xplagiax_xota:latest



```

### Troubleshooting

**1. Celery Crash: `ImproperlyConfigured: Cannot mix new and old setting keys`**
* **Cause:** Celery 5.x throws this error if you push old uppercase Flask configs into `celery.conf` while also using new lowercase config keys.
* **Fix:** The codebase was updated to remove the `celery.conf.update(app.config)` call in `app/celery_app.py`. Rebuild the image.

**2. Docker Error: `Conflict. The container name "/xplagiax-xota" is already in use`**
* **Cause:** A stopped container with the same name still exists.
* **Fix:**
  ```bash
  docker stop xplagiax-xota
  docker rm xplagiax-xota
  ```

**3. MarkTrack Error: `NameResolutionError: Failed to resolve 'xplagiax-xota'`**
* **Cause:** The MarkTrack container cached the old container's internal IP after recreation.
* **Fix:** `docker restart marktrack_app`

**4. Citation validation returns `error` status**
* **Cause:** `aiohttp` is not installed, or the container has no outbound network access.
* **Fix:** Verify `aiohttp>=3.9.0` is in `requirements.in` and run `make lock && make install`. Check Docker network policy with `docker inspect xplagiax-xota`.

**5. Startup error: `API_KEY must be set in production`**
* **Cause:** `FLASK_ENV=production` is set but `API_KEY` env var is missing.
* **Fix:** Generate a key and pass it via `docker run -e API_KEY=...` or your `.env` file.

**6. Startup error: `DEBUG=True is forbidden in production`**
* **Cause:** `DEBUG=1` and `FLASK_ENV=production` are set simultaneously.
* **Fix:** Remove `DEBUG=1` from the production environment.

---

## Memory Leak Analysis — Root Causes & Fixes (May 2026)

Deep forensic analysis of why the container consumed **2.7 GB at baseline** and why memory **grew further after each `analyze_document_async` call**.

### Baseline 2.7 GB — Decomposition

| Component | Size | Origin |
|---|---|---|
| `modernbert.bin` (model_1) | ~570 MB | Loaded at module import in `detector_final.py` |
| `Model_groups_3class_seed12` (model_2) | ~570 MB | Loaded at module import in `detector_final.py` |
| `Model_groups_3class_seed22` (model_3) | ~570 MB | Loaded at module import in `detector_final.py` |
| PyTorch runtime + libs | ~350 MB | Imported with the models |
| Transformers + Tokenizers | ~100 MB | HuggingFace local cache |
| spaCy + NLTK | ~80 MB | Pre-downloaded in Dockerfile |
| Flask + Celery + packages | ~100 MB | `requirements.txt` |
| Docker OS + Python 3.12 slim | ~200 MB | Base image |
| **Total baseline** | **~2.5–2.7 GB** | — |

### Memory Growth After `analyze_document_async` — Root Cause Table

| # | Severity | Category | Root Cause | Location | Fix Applied |
|---|---|---|---|---|---|
| 1 | **CRITICAL** | Redundant inference | `tasks.py` ran model inference twice on the same text (plugin + task). | `app/tasks.py:47–48` | Reuse plugin segments; fall back only when `ai_detection` not requested. |
| 2 | **CRITICAL** | CoW violation | PyTorch ref-counting triggered CoW page faults, duplicating 1.71 GB per worker (3 workers = 5.13 GB). | `detector_final.py:_load_model()` | `m.share_memory()` after load — POSIX shared memory, no CoW faults. |
| 3 | **HIGH** | Redis result bloat | Base64 charts (300 KB–1 MB each) serialized into Redis with 1-hour TTL. | `tasks.py` return + `celery_app.py` | Strip `*_b64` keys before Redis; reduce TTL to 600s. |
| 4 | **HIGH** | GPT-2 lazy-load spike | PerplexityProfiler Tier 2 loads GPT-2 (~500 MB) on first use, permanent spike. | `perplexity_profiler.py` | Set `PERPLEXITY_TIER2=0` in production if Tier 2 is not required. |
| 5 | **MEDIUM** | `/tmp` accumulation | `ForensicReportGenerator` creates 1–3 MB HTML files, cleanup only after 1 hour. | `full_analysis.py` | Mount `/tmp` as `tmpfs` in Docker to isolate from overlay filesystem. |
| 6 | **MEDIUM** | Celery result TTL | 3600s TTL allows 1 hour of large results to accumulate in Redis. | `celery_app.py:result_expires` | Reduced to 600s. |
| 7 | **LOW** | `gc.collect()` scope | GC handles Python cycles but cannot free live PyTorch tensors. | `tasks.py:finally` | No change — call is correct; documented scope limitation. |

### Memory Budget After Fixes

| Scenario | Before Fixes | After Fixes |
|---|---|---|
| Baseline (idle, before first request) | ~2.7 GB | ~2.7 GB (unchanged) |
| After first async request (all 3 processes touched) | +5.1 GB (CoW violation × 3 workers) | ~+50 MB (shared memory) |
| Redis after 100 async requests (1-hour window) | +500 MB–1 GB | ~+10–50 MB |
| **Steady-state under load** | **~8–10 GB+** | **~3–3.5 GB** |

---

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xplagiax
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: xplagiax
          image: xplagiax:latest
          ports:
            - containerPort: 5006
          env:
            - name: WEB_CONCURRENCY
              value: "2"
            - name: FLASK_ENV
              value: "production"
            - name: LOG_LEVEL
              value: "info"
            - name: CROSSREF_EMAIL
              value: "your@institution.edu"
            - name: SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: xplagiax-secrets
                  key: secret-key
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: xplagiax-secrets
                  key: api-key
          resources:
            requests:
              memory: "3Gi"
              cpu: "1000m"
            limits:
              memory: "5Gi"
              cpu: "4000m"
          volumeMounts:
            - name: models
              mountPath: /app/app/engine
          livenessProbe:
            httpGet:
              path: /health
              port: 5006
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 5006
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: xplagiax-models-pvc
```

> `initialDelaySeconds` is set high because the three ModernBERT model files take ~30–60s to load on first startup.
