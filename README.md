# XplagiaX — AI Detection Microservice

Flask microservice for AI-generated text detection. Built around a 4-model ModernBERT ensemble with a modular plugin architecture. Supports per-document segmentation, forensic reports, perplexity analysis, citation verification, zone classification, and more.

---

## Architecture

```
Client
    │
    │  POST /analyze              {"text": "...", "plugins": ["ai_detection", ...]}
    │  POST /analyze_document     {"text": "...", "plugins": ["ai_detection", ...]}
    │  POST /api/v2/citations/detect    {"text": "..."}
    │  POST /api/v2/citations/validate  {"text": "..."}
    ▼
┌────────────────────────────────────────────────────────────────┐
│  Gunicorn (preload_app=True + gevent workers)                  │
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
└────────────────────────────────────────────────────────────────┘
```

**Key design decisions**

| Decision | Rationale |
|---|---|
| `preload_app = True` | Models loaded once in master, shared across workers via CoW |
| `gevent` worker class | Cooperative I/O concurrency, handles hundreds of connections per worker |
| `OMP_NUM_THREADS=1` | Prevents torch/numpy from spawning per-worker thread explosions |
| `local_files_only=True` | No HuggingFace network calls on startup — uses local cache |
| Plugin auto-discovery | Drop a file in `app/plugins/`, restart — zero core changes needed |
| Multi-stage Docker build | Builder has gcc/dev headers; runtime image is lean |
| Non-root container user | UID 1000, K8s security best practice |
| `m.share_memory()` on models | POSIX shared memory prevents CoW page faults across workers |
| CitationDetector module singleton | Instantiated at import time, shared across all workers via CoW |

---

## Engine — ModernBERT Ensemble

The core classifier (`app/engine/detector_final.py`) loads three fine-tuned ModernBERT models at startup and averages their softmax outputs:

| File | Labels | Role |
|---|---|---|
| `modernbert.bin` | 41 classes | Base ensemble member |
| `Model_groups_3class_seed12` | 41 classes | Ensemble member (seed 12) |
| `Model_groups_3class_seed22` | 41 classes | Ensemble member (seed 22) |

The 41 label classes include `human` (index 24) and 40 known AI generators (GPT-4, GPT-4o, Claude variants, LLaMA 3, Mixtral, Gemma, etc.). The binary Human/AI split is derived from `prob[24]` vs the sum of all remaining probabilities.

The tokenizer and config are loaded from the local HuggingFace cache (`answerdotai/ModernBERT-base`, `local_files_only=True`).

---

## Plugins

### Available plugins

| Plugin name | Description |
|---|---|
| `ai_detection` | Binary Human/AI classification — 4-model ModernBERT ensemble. ~2s CPU, ~0.3s GPU. |
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

2. Add dependencies to `requirements.txt` if needed.
3. Restart the service — the plugin is auto-discovered with no other changes.

---

## API

### POST /analyze

Run any combination of plugins on a text.

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

Run any combination of plugins on a long document **and** get per-paragraph segment scores. Runs all requested plugins through the same registry as `/analyze`, then always appends per-segment results from `HybridSegmentAnalyzer` into the `ai_detection` result (if `ai_detection` was requested).

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

Serve a generated HTML forensic report from `/tmp`.

```bash
curl http://localhost:5006/report/forensic_abc123.html
```

---

## Antiplagio — Citation API (`/api/v2/`)

The antiplagio module adds two dedicated citation endpoints that are independent from the plugin system. They are always available (no plugin selection needed).

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

Asynchronous bibliography validation. Queries CrossRef, OpenAlex, and Semantic Scholar in parallel for each reference. Requires network access and `aiohttp`.

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

// AI detection
const aiResult = await analyzeText(text, ["ai_detection"]);
const ai = aiResult.results.ai_detection.data;
console.log(`${ai.prediction} — ${ai.confidence.toFixed(2)}%`);

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
| `415` | Missing `Content-Type: application/json` | `{"error": "Content-Type must be application/json"}` |
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

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model files in app/engine/
#    app/engine/modernbert.bin
#    app/engine/Model_groups_3class_seed12
#    app/engine/Model_groups_3class_seed22

# 4. Ensure the ModernBERT tokenizer is cached locally
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')"

# 5. Start the dev server
python app.py
```

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

#### Run — production (with Redis for async tasks + citation validation)

```bash
# 1. Create network (if it doesn't exist)
docker network create xplagiax-net

# 2. Start Redis
docker run -d \
  --name redis \
  --network xplagiax-net \
  -p 6379:6379 \
  redis:7-alpine

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
  -e REDIS_URL="redis://redis:6379" \
  -e CELERY_BROKER_URL="redis://redis:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://redis:6379/1" \
  -e CROSSREF_EMAIL="your@email.com" \
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
  -e CROSSREF_EMAIL="your@email.com" \
  xplagiax_xota:latest
```

> **Note**: `preload_app=True` ensures the three ModernBERT models (~1.7 GB total) are loaded once in the Gunicorn master process and shared across all workers via Linux Copy-on-Write. Each worker adds only ~50 MB overhead instead of another full model copy.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
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
| `CACHE_TYPE` | `SimpleCache` | Flask-Caching backend |
| `CACHE_TIMEOUT` | `300` | Cache TTL in seconds |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker URL (optional async queue) |
| `CROSSREF_EMAIL` | `antiplagio@example.com` | Email sent to CrossRef Polite Pool API for citation validation |

---

## Project Structure

```
xplagiax_xota/
├── app.py                        # Entry point — gunicorn target
├── gunicorn.conf.py              # Production server config
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Multi-stage container build
│
├── app/
│   ├── __init__.py               # create_app() factory
│   ├── config.py                 # All config via env vars
│   ├── routes.py                 # API blueprints (/analyze, /analyze_document, ...)
│   ├── tasks.py                  # Celery background tasks
│   ├── celery_app.py             # Celery worker entry point
│   ├── plugin_registry.py        # Auto-discovery and dispatch
│   │
│   ├── antiplagio/               # Citation detection and validation package
│   │   ├── __init__.py
│   │   ├── flask_routes.py       # Blueprint /api/v2/ (detect + validate endpoints)
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
│   │   ├── zone_classifier.py    # Citation zone detection plugin (NEW)
│   │   ├── forensic_report.py    # HTML forensic report generation
│   │   └── full_analysis.py      # Complete forensic pipeline
│   │
│   └── engine/                   # XplagiaX core — unmodified engine files
│       ├── __init__.py           # sys.path setup + torch/transformers patch
│       ├── detector_final.py     # 4-model ModernBERT ensemble
│       ├── hybrid_segment_detector.py   # Sliding-window segment classifier
│       ├── perplexity_profiler.py       # n-gram + GPT-2 perplexity
│       ├── stylometric_profiler.py      # Writing style fingerprinting
│       ├── hallucination_profile.py     # Fabrication risk detection
│       ├── reasoning_profiler.py        # Reasoning-model detection
│       ├── reference_validator.py       # Citation verification
│       ├── watermark_decoder.py         # Digital watermark detection
│       ├── forensic_reports.py          # HTML/JSON report generator (v3.9)
│       ├── plugin_orchestrator.py       # Full pipeline coordinator
│       ├── modernbert.bin               # Model weights (~600 MB)
│       ├── Model_groups_3class_seed12   # Model weights (~600 MB)
│       └── Model_groups_3class_seed22   # Model weights (~600 MB)
│
└── tests/
    ├── __init__.py
    └── test_citation_system.py   # 24 pytest tests for CitationDetector
```

---

## Bug Fix & Improvement Table (May 2026)

| # | Severity | Module | Bug / Problem | Fix Applied |
|---|---|---|---|---|
| 1 | **CRITICAL** | `app/tasks.py` | Redundant double inference: `analyze_document_async` ran `analyze_long_document()` **after** `ai_detection` plugin had already performed identical inference via `analyze_long_documentsd_()`. Every request ran model inference twice. | Reuse `segments` already computed by the plugin. Fall back to `analyze_long_documentsd_()` only when `ai_detection` was not requested. |
| 2 | **CRITICAL** | `app/engine/detector_final.py` | Copy-on-Write violation: PyTorch's internal reference counting wrote to the same virtual pages as model weights on first inference, causing all 1.71 GB of weights to be duplicated per worker. With 2 web + 1 Celery worker = 5.13 GB extra. | Added `m.share_memory()` after each model load. Moves tensor storage to POSIX shared memory — read-only across all forked processes, no CoW faults. |
| 3 | **HIGH** | `app/tasks.py` | Redis result bloat: Celery serialized full task results including `heatmap_b64`, `confidence_chart_b64`, `comparison_chart_b64` (300 KB–1 MB each). With 1-hour TTL, accumulated hundreds of MB under load. | Added `_strip_base64()` helper to remove all `*_b64` keys before Redis serialization. HTML report in `/tmp` already contains the charts. |
| 4 | **HIGH** | `app/celery_app.py` | Celery result TTL of 3600s allowed 1 hour of large results to accumulate in Redis simultaneously under moderate load. | Reduced `result_expires` from 3600s to 600s (10 minutes). |
| 5 | **HIGH** | `app/antiplagio/citation/detector.py` | Multi-citation parentheticals `(García, 2021; López, 2020; Martínez, 2019)` were not detected — `APA_INLINE` requires a single author-year pair per parenthetical, so semicolon-separated citations were silently dropped. | Added `APA_MULTI_PAREN` regex pattern. Processing block at start of `_detect_inline_citations` splits semicolon-separated parts into individual `CitationMarker` objects. Original loop skips already-processed spans. |
| 6 | **HIGH** | `app/antiplagio/citation/detector.py` | spaCy import failure would crash the entire module if `spacy` was not installed, preventing citation detection from working at all. | Separated `import spacy` from `spacy.load()`. `ImportError` sets `_SPACY_AVAILABLE = False` and falls back to rule-based sentence segmentation. `OSError` on model load (model not downloaded) also falls back gracefully. |
| 7 | **MEDIUM** | `app/antiplagio/citation/validator.py` | `aiohttp` dependency was unconditional — if not installed, the entire validator module failed to import. | Wrapped in `try/except ImportError`. `_AIOHTTP_AVAILABLE = False` when missing. `validate_all()` returns `ValidationStatus.ERROR` results instead of crashing. |
| 8 | **MEDIUM** | `app/antiplagio/flask_routes.py` | `async_route` decorator leaked event loop into gevent's global state: did not call `asyncio.set_event_loop(None)` after cleanup, risking interference between requests under concurrent load. | Added `asyncio.set_event_loop(None)` in the `finally` block after `loop.close()`. |
| 9 | **MEDIUM** | `app/antiplagio/flask_routes.py` | Dead code in `validate_citations`: unused `_, bib_text = detector.segmenter._split_bibliography(...)` called and discarded the result, wasting CPU. | Removed unused call. `bibliography` is now obtained directly from `analysis.bibliography`. |
| 10 | **LOW** | `app/plugins/zone_classifier.py` | CitationDetector was instantiated inside every `analyze()` call — each request rebuilt all compiled regex patterns, wasting CPU on every plugin invocation. | Moved `_detector = CitationDetector()` to module level. Singleton instantiated once at gunicorn preload, shared across workers via CoW. `warmup()` is a no-op. |

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
* **Memory Leak Prevention:** The explicit `gc.collect()` after each analysis ensures no "residue" memory or orphaned tensors remain. RAM usage will briefly spike during inference (forward pass) but will immediately drop back down, guaranteeing stable memory footprint over weeks of uptime.
* **Simultaneous Users:** The system is strictly capped to process **3 concurrent heavy analyses** at exactly the same time (2 Web Workers + 1 Celery Worker).
* **Queuing:** If 50-100 users submit a 500-word document simultaneously, the system will *not* crash. It will process 3 immediately while safely queuing the remaining requests in Redis (for async) or Gunicorn's backlog. At a rate of ~2-3 seconds per 500-word document, a single Celery worker can process ~20-30 documents per minute seamlessly in the background.

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
  -e REDIS_URL="redis://redis:6379" \
  -e CELERY_BROKER_URL="redis://redis:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://redis:6379/1" \
  -e CROSSREF_EMAIL="your@email.com" \
  xplagiax_xota:latest

# 4. Verify the internal Celery worker is running alongside Gunicorn
docker exec xplagiax-xota ps aux | grep -E "gunicorn|celery"
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
* **Fix:** Verify `aiohttp>=3.9.0` is in `requirements.txt` and rebuild. Check Docker network policy with `docker inspect xplagiax-xota`.

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
| Flask + Celery + 86 packages | ~100 MB | `requirements.txt` |
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
            - name: LOG_LEVEL
              value: "info"
            - name: CROSSREF_EMAIL
              value: "your@email.com"
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
