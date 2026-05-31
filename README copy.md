# TextAnalyzer Microservice

High-performance Flask microservice for modular text analysis with a plugin architecture.

## Architecture

```
Client (Flask app)
    │
    │  POST /analyze {"text": "...", "plugins": ["sentiment", "keyphrases"]}
    ▼
┌─────────────────────────────────────────────────────┐
│  Gunicorn (--preload + gevent)                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  Master Process                               │  │
│  │  ├── Models loaded ONCE at module level       │  │
│  │  ├── PluginRegistry auto-discovers plugins    │  │
│  │  └── create_app() called once                 │  │
│  └───────────┬───────────┬───────────────────────┘  │
│         fork │      fork │      (Linux CoW)         │
│  ┌───────────▼┐  ┌──────▼──────┐                    │
│  │  Worker 1  │  │  Worker 2   │  ... Worker N      │
│  │  ~50MB RAM │  │  ~50MB RAM  │  (shared models)   │
│  └────────────┘  └─────────────┘                    │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `gunicorn --preload` | Load models once in master, share via CoW across workers |
| `gevent` worker class | Async I/O for thousands of concurrent connections |
| Module-level model loading | Ensures models live in master's memory before fork |
| `OMP_NUM_THREADS=1` | Prevents numpy/torch from spawning N threads per worker |
| Plugin auto-discovery | Add a file in `app/plugins/`, restart — zero core changes |
| Multi-stage Docker build | Builder installs gcc/dev headers, runtime copies only binaries |
| Non-root container user | K8s security best practice (UID 1000) |

## Included Plugins

| Plugin | Library | Size | Description |
|--------|---------|------|-------------|
| `word_stats` | Pure Python | 0 MB | Word/sentence/paragraph counts, lexical diversity |
| `sentiment` | TextBlob | ~2 MB | Polarity and subjectivity analysis |
| `keyphrases` | YAKE | ~1 MB | Unsupervised keyphrase extraction |
| `summarization` | NLTK | ~5 MB | Extractive TF-IDF sentence ranking |
| `readability` | Pure Python | 0 MB | Flesch-Kincaid, Fog, Coleman-Liau, ARI |
| `language_detect` | langdetect | ~1 MB | Language identification |

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

### Docker Build & Run

```bash
# Build (with BuildKit for layer caching)
DOCKER_BUILDKIT=1 docker build -t textanalyzer:latest .

# Run with resource limits
docker run -d \
  --name textanalyzer \
  --memory=512m \
  --cpus=2 \
  -p 5006:5006 \
  -e WEB_CONCURRENCY=4 \
  -e LOG_LEVEL=info \
  textanalyzer:latest
```

### Docker Compose (with Redis for Celery)

```yaml
services:
  analyzer:
    build: .
    ports: ["5006:5006"]
    environment:
      WEB_CONCURRENCY: "4"
      CELERY_BROKER_URL: "redis://redis:6379/0"
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "2.0"

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

## API Usage

### POST /analyze

```bash
curl -X POST http://localhost:5006/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The rapid integration of Artificial Intelligence in academic settings has sparked a profound ethical debate.",
    "plugins": ["sentiment", "keyphrases", "word_stats", "readability"]
  }'
```

### GET /health (liveness probe)

```bash
curl http://localhost:5006/health
# {"status": "healthy"}
```

### GET /ready (readiness probe)

```bash
curl http://localhost:5006/ready
# {"status": "ready", "plugins_loaded": 6, "plugins": [...]}
```

### GET /plugins (catalogue)

```bash
curl http://localhost:5006/plugins
```

## Adding a New Plugin

1. Create `app/plugins/my_plugin.py`:

```python
from app.plugins.base import BasePlugin

class MyPlugin(BasePlugin):

    def name(self) -> str:
        return "my_plugin"

    def description(self) -> str:
        return "Does something useful with text."

    def analyze(self, text: str) -> dict:
        # Your analysis logic here
        return {"result": "..."}
```

2. Add any new dependencies to `requirements.txt`.
3. Restart the service. The plugin is auto-discovered — no other changes needed.

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textanalyzer
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: analyzer
          image: textanalyzer:latest
          ports:
            - containerPort: 5006
          env:
            - name: WEB_CONCURRENCY
              value: "4"
          resources:
            requests:
              memory: "256Mi"
              cpu: "500m"
            limits:
              memory: "512Mi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 5006
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 5006
            initialDelaySeconds: 5
            periodSeconds: 10
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_CONCURRENCY` | `2×CPU (max 8)` | Number of gunicorn workers |
| `GUNICORN_TIMEOUT` | `120` | Worker timeout in seconds |
| `MAX_CONTENT_MB` | `16` | Max request body size in MB |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `CACHE_TYPE` | `SimpleCache` | Flask-Caching backend |
| `CACHE_TIMEOUT` | `300` | Cache TTL in seconds |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `PLUGIN_TIMEOUT` | `30` | Max seconds per plugin execution |
