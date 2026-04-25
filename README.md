# XplagiaX — AI Detection Microservice

Flask microservice for AI-generated text detection. Built around a 4-model ModernBERT ensemble with a modular plugin architecture. Supports per-document segmentation, forensic reports, perplexity analysis, citation verification, and more.

---

## Architecture

```
Client
    │
    │  POST /analyze          {"text": "...", "plugins": ["ai_detection", ...]}
    │  POST /analyze_document {"text": "...", "plugins": ["ai_detection", ...]}
    ▼
┌────────────────────────────────────────────────────────────────┐
│  Gunicorn (preload_app=True + gevent workers)                  │
│                                                                │
│  Master Process                                                │
│  ├── ModernBERT ensemble loaded ONCE at module level           │
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
    json={
        "text": text,
        "plugins": ["ai_detection"],
    },
)
data = response.json()

ai = data["results"]["ai_detection"]["data"]
print(f"Global prediction: {ai['prediction']} ({ai['confidence']:.2f}%)")
print(f"Segments analyzed: {len(ai.get('segments', []))}")

for seg in ai.get("segments", []):
    print(f"  [{seg['segment_id']}] {seg['dominant_label']:7s}  {seg['score']:.2f}%  {seg['text'][:60]}...")
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
    plugins=["ai_detection", "perplexity_check"],
)

for plugin_name, plugin_result in result["results"].items():
    if plugin_result["status"] == "ok":
        print(f"{plugin_name}: {plugin_result['data']}")
    else:
        print(f"{plugin_name}: ERROR — {plugin_result.get('error')}")
```

#### Check server readiness before running inference
```python
import requests
import time

def wait_for_ready(base_url: str = "http://localhost:5006", retries: int = 10):
    for attempt in range(retries):
        try:
            r = requests.get(f"{base_url}/ready", timeout=5)
            if r.status_code == 200:
                data = r.json()
                print(f"Server ready — {data['plugins_loaded']} plugins loaded")
                return True
        except requests.ConnectionError:
            pass
        print(f"Not ready yet, retrying ({attempt + 1}/{retries})...")
        time.sleep(5)
    raise RuntimeError("Server did not become ready in time")

wait_for_ready()
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

    tasks = [
        analyze_async(text, ["ai_detection"])
        for text in texts
    ]
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

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

async function analyzeDocument(text, plugins = ["ai_detection"]) {
  const response = await fetch(`${BASE_URL}/analyze_document`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, plugins }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

// Usage
const text = `The integration of AI in academic settings has sparked debate.
Second paragraph with more context about the subject matter.`;

// Single plugin
const result = await analyzeText(text, ["ai_detection"]);
const ai = result.results.ai_detection.data;
console.log(`${ai.prediction} — ${ai.confidence.toFixed(2)}%`);

// Document with segments
const docResult = await analyzeDocument(text);
const segments = docResult.results.ai_detection.data.segments ?? [];
segments.forEach(seg => {
  console.log(`[${seg.segment_id}] ${seg.dominant_label} ${seg.score.toFixed(2)}% — ${seg.text.slice(0, 60)}...`);
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


curl -X POST http://localhost:5006/analyze_document  -H "Content-Type: application/json"  -d '{"text": "The Industrial Revolution was one of the most transformative periods in human history, marking the shift from agrarian societies to industrialized economies. Beginning in the late 18th century in Great Britain, it spread to other parts of Europe and eventually to the United States, fundamentally changing how people lived and worked. Before the Industrial Revolution, most production was done by hand in small workshops or at home. This system, often called the “cottage industry,” relied heavily on manual labor and simple tools. However, a series of technological innovations began to revolutionize production. One of the most important inventions was the steam engine, improved by James Watt. This machine allowed factories to operate more efficiently and independently of natural power sources like water. Another key development was the mechanization of the textile industry. Machines such as the spinning jenny and the power loom greatly increased the speed and volume of fabric production. As a result, factories began to replace traditional workshops, leading to the growth of urban centers. Cities expanded rapidly as people moved from rural areas in search of employment opportunities. The Industrial Revolution also had profound social and economic impacts. On one hand, it led to increased production, lower costs of goods, and improved standards of living for many people. On the other hand, it created harsh working conditions, especially in the early years. Workers, including women and children, often faced long hours, low wages, and unsafe environments in factories. In addition, the rise of industrial capitalism changed the structure of society. A new middle class of factory owners and entrepreneurs emerged, while the working class grew significantly. These changes eventually led to social reforms and the development of labor unions, as workers sought better conditions and rights. Despite its challenges, the Industrial Revolution laid the foundation for modern society. It introduced new technologies, transformed economies, and reshaped social structures in ways that continue to influence the world today.", "plugins": ["ai_detection","stylometric_analysis","citation_check"]}'





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

To build and run the microservice using Docker:

```bash
# 1. Build the image
docker build -t xplagiax:latest .

# 2. Run the container with 2 workers
# Ensure model files are in app/engine/ before building, or mount them.
docker run -d \
  --name xplagiax-service \
  -p 5006:5006 \
  -e WEB_CONCURRENCY=2 \
  xplagiax:latest
```

> **Note**: The Dockerfile is configured to use Gunicorn with `preload_app=True`, ensuring that the heavy ModernBERT models are shared across the 2 workers using Linux Copy-on-Write (CoW).

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
│   ├── plugin_registry.py        # Auto-discovery and dispatch
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
```

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
