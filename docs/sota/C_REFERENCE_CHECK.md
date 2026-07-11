# Guía C — Reference check en serio: GROBID + DOI + retracciones

**Objetivo:** convertir `app/engine/reference_validator.py` en tu evidencia
forense más fuerte. Citas fabricadas/quiméricas son casi smoking-gun de LLM
y son evidencia que un comité humano entiende sin confiar en una red neuronal.

**Lo que ya tienes** (no partir de cero):

- `ReferenceExtractor` — extracción por regex (el punto débil: la auditoría
  detectó extracción espuria).
- `_APIClient` — CrossRef + OpenAlex + Semantic Scholar con rate-limit y
  circuit breaker ([reference_validator.py:544](../../app/engine/reference_validator.py)).
- `_similarity_ratio`, `_extract_surnames` — matching básico.
- Guardas anti-falso-positivo en `detector_final.validar_veredicto_segmento`.

**Lo que falta:** parsing robusto (GROBID), resolución directa de DOI,
chequeo de retracciones, cache persistente, y encenderlo por defecto.

**Todo corre local — no se necesita Colab ni GPU en esta guía.**

---

## C.1 Levantar GROBID (30 minutos)

GROBID es el estándar de facto para parsear referencias bibliográficas
(machine learning entrenado en millones de referencias; entiende APA, IEEE,
Vancouver, Chicago… y referencias mal formateadas).

### Paso 1 — Añadir al docker-compose

```yaml
  grobid:
    image: lfoppiano/grobid:0.8.0
    container_name: xplagiax_grobid
    restart: unless-stopped
    networks:
      - xplagiax-net
    mem_limit: 2g            # el modo CRF (default) es ligero; NO uses la
                             # imagen deep-learning (necesita GPU)
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8070/api/isalive"]
      interval: 30s
      timeout: 5s
      retries: 3
```

```bash
docker compose up -d grobid
curl http://localhost:8070/api/isalive     # → true
```

### Paso 2 — Probar el endpoint de referencias

```bash
curl -s -X POST http://localhost:8070/api/processCitationList \
  --data-urlencode "citations=García, J. y López, M. (2021). Aprendizaje profundo en medicina. Revista de IA, 12(3), 45-67. https://doi.org/10.1234/rev.2021.045" \
  -H "Accept: application/xml"
```

Devuelve TEI-XML con `<author>`, `<title>`, `<date>`, `<idno type="DOI">`
estructurados.

**✅ Checkpoint:** prueba con 5 referencias reales de una tesis (APA español,
una mal formateada, una sin DOI). GROBID debe extraer título y autores de
todas o casi todas.

---

## C.2 Integrar GROBID como extractor primario

### Paso 1 — Nuevo método en `reference_validator.py`

Añade una clase `GrobidExtractor` con la MISMA interfaz que
`ReferenceExtractor.extract(text) -> (List[ParsedReference], List[str])`:

1. Localiza la sección de bibliografía (reusa la lógica/regex existente de
   detección de sección "Referencias/Bibliography").
2. Divide en líneas/entradas (una referencia por entrada — heurística:
   línea que empieza con apellido o [n] o •).
3. POST a `GROBID_URL/api/processCitationList` con las entradas
   (acepta varias separadas por `\n`).
4. Parsea el TEI-XML (usa `xml.etree.ElementTree`, namespace
   `{http://www.tei-c.org/ns/1.0}`) → rellena `ParsedReference` con
   `authors`, `title`, `year`, `journal`, `doi`.

Config por env: `GROBID_URL=http://grobid:8070` (docker) o
`http://localhost:8070` (local). Timeout 10 s por llamada.

### Paso 2 — Fallback en cascada

```python
def extract(self, text):
    if grobid_disponible():
        refs, inline = self._grobid_extract(text)
        if refs:                       # GROBID encontró algo
            return refs, inline
    return self._regex_extract(text)   # el extractor actual como fallback
```

**Nunca elimines el regex** — GROBID caído no puede significar "sin análisis".

**✅ Checkpoint:** con la misma tesis de prueba, compara
`len(refs_grobid)` vs `len(refs_regex)` y revisa a mano 10 referencias:
GROBID debe tener menos falsos positivos (fragmentos de texto normal
extraídos "como referencia" — el problema que señaló tu auditoría).

---

## C.3 Cascada de validación por referencia

Orden (de más barato/fiable a más caro):

### Paso 1 — Si hay DOI: resolución directa

```
GET https://api.crossref.org/works/{doi}
```

- 200 → el DOI existe. Compara título devuelto vs título citado con
  `_similarity_ratio` (ya existe). ≥ 0.85 → `verified`.
- 200 pero título NO coincide (< 0.60) → **`chimeric`** (DOI real de OTRO
  paper — huella clásica de LLM que inventa combinando).
- 404 → DOI no existe → **`fabricated`** (señal fuerte).

### Paso 2 — Sin DOI: búsqueda por título

Ya lo hace `_APIClient.search_crossref/openalex/semantic_scholar`. Endurece
el matching:

```python
match_ok = (
    similarity_ratio(titulo_citado, titulo_candidato) >= 0.90
    and primer_apellido_citado in apellidos_candidato
    and abs(anio_citado - anio_candidato) <= 1
)
```

- Match completo → `verified`.
- Título coincide pero autores/año no → `chimeric`.
- Sin candidato en las 3 APIs → `not_found` (cuidado: NO es sinónimo
  automático de fabricada — ver C.5).

### Paso 3 — Retracciones (diferenciador que nadie más tiene)

CrossRef expone retracciones en el propio registro del DOI:

```python
# En la respuesta de /works/{doi}:
updates = work.get("update-to", [])
retracted = any(u.get("type") in ("retraction", "withdrawal") for u in updates)
```

Citar un paper retractado no prueba IA, pero es bandera de calidad académica
y los LLMs lo hacen (su corpus es viejo). Añádelo como flag informativo:
`"retracted_citations": 2`.

---

## C.4 Cache persistente en Redis (obligatorio antes de encender)

Sin cache te banean de CrossRef en horas. Ya tienes Redis en el stack.

```python
# clave: "refcheck:doi:{doi}" o "refcheck:title:{sha1(titulo_normalizado)}"
# valor: JSON del resultado de validación
# TTL: 30 días (los papers no cambian; las retracciones sí → no infinito)
```

Reglas de cortesía de API (evita baneos):

1. CrossRef: incluye `mailto` REAL en el User-Agent
   (ya existe `CROSSREF_EMAIL` en config — **ponle un correo institucional
   real**, el default `example.com` te degrada al pool lento).
2. Máximo ~1 req/s por API (el `_RATE_LIMIT_MS` existente).
3. El circuit breaker existente ya corta si la API cae — bien, no lo toques.

---

## C.5 Reglas de veredicto (aquí es donde NO te puedes equivocar)

La cita fabricada es evidencia, no sentencia. Mantén y amplía las guardas
existentes de `validar_veredicto_segmento`:

| Situación | Veredicto |
|-----------|-----------|
| ≥ 3 referencias y ≥ 70% `fabricated`/`chimeric` | Evidencia FUERTE de IA (peso alto en fusion) |
| 1-2 referencias no encontradas, resto OK | Ruido — informativo, peso bajo |
| APIs caídas / circuit breaker abierto | `status: "unavailable"` — señal NEUTRA, jamás negativa |
| Documento sin bibliografía | Señal NEUTRA (muchos géneros no llevan) |
| Referencias a libros/informes locales sin DOI | `not_found` esperado — pondera por tipo: solo revistas/conferencias deberían resolverse |

**El error que destruye tu credibilidad:** marcar "fabricada" una referencia
a un libro en español de una editorial pequeña que no está en CrossRef.
Por eso: `not_found` con las 3 APIs y tipo=journal → sospechosa;
`not_found` y tipo=book/report/web → neutra.

---

## C.6 Encender en producción

1. `.env` / docker-compose del servicio web:

```
ENABLE_REFERENCE_CHECK=1
REFERENCE_NETWORK=1
GROBID_URL=http://grobid:8070
CROSSREF_EMAIL=tu-correo-institucional-real@universidad.edu
```

2. El plugin `citation_check` y el flag del orquestador ya leen estas vars
   ([full_analysis.py:62-63](../../app/plugins/full_analysis.py)).
3. Los timeouts adaptativos ya contemplan plugins lentos; aun así el
   validador debe tener SU PROPIO tope interno (p. ej. máx 50 referencias
   validadas online por documento; el resto `skipped` — una tesis puede
   tener 400 referencias y no puedes hacer 1.200 llamadas API por request).
   Prioriza: primero las que tienen DOI (baratas), luego muestreo aleatorio
   de las demás.

**✅ Checkpoint final:** tesis real con bibliografía sana → ≥ 80% `verified`,
0 `fabricated`. Texto generado por ChatGPT con "referencias" → mayoría
`fabricated`/`chimeric`. Si una tesis sana da fabricadas, revisa C.5 antes
de culpar al estudiante.

---

## Errores dummy típicos

1. **Encender sin cache** → baneo de CrossRef en el primer lote de tesis.
2. **`not_found` = fabricada** → falsos positivos con bibliografía en
   español/libros. Usa la tabla de C.5.
3. **Validar 400 referencias online por request** → timeout y DoS a las APIs.
   Tope + priorización por DOI.
4. **Eliminar el extractor regex al meter GROBID** → GROBID caído = plugin
   muerto. Cascada, no sustitución.
5. **`CROSSREF_EMAIL=antiplagio@example.com`** → CrossRef te manda al pool
   lento/te bloquea. Correo real.
6. **Tratar retracción como prueba de IA** → es bandera de calidad, no de autoría.
