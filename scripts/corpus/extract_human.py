"""
extract_human.py — [doc A, paso A.1] Build the HUMAN class (label 0) from the pre-2022 DB.

Inviolable rule: only documents published BEFORE 2022-11-01 (pre-ChatGPT) — anything
later may contain AI text and poisons the human class.

The detector's DB lives in another service, so the connection is external:

    export CORPUS_DB_DSN="mysql://user:pass@host:3306/dbname"
    .venv/bin/python scripts/corpus/extract_human.py \
        --query-file my_query.sql \
        --out dataset/human.jsonl --max-per-stratum 800

The SQL must yield columns: id, texto, idioma, disciplina, anio, autor_id (alias as
needed). Default query targets the schema sketched in docs/sota/A_FUSION_ENTRENADA.md —
override with --query-file for the real schema.

Chunks long documents into 500-3000-word units (doc A, paso A.3) and stratifies by
(idioma, disciplina) with a per-stratum cap so no domain dominates.
Requires: pip install pymysql   (only for this offline script; not a service dep)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from urllib.parse import urlparse

CHUNK_MIN_WORDS = 500
CHUNK_MAX_WORDS = 3000

DEFAULT_QUERY = """
SELECT id, texto, idioma, disciplina, anio, autor_id
FROM documentos
WHERE fecha_publicacion < '2022-11-01'
  AND longitud_palabras BETWEEN 300 AND 200000
"""


def chunk_text(text: str) -> list[str]:
    """Split at paragraph boundaries into 500-3000-word units."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, count = [], [], 0
    for p in paras:
        buf.append(p)
        count += len(p.split())
        if count >= CHUNK_MAX_WORDS:
            chunks.append("\n\n".join(buf))
            buf, count = [], 0
    tail = "\n\n".join(buf)
    if count >= CHUNK_MIN_WORDS:
        chunks.append(tail)
    elif chunks and tail:
        chunks[-1] += "\n\n" + tail
    elif tail and count >= 300:      # short single-chunk doc — keep (doc A floor)
        chunks.append(tail)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=os.getenv("CORPUS_DB_DSN", ""))
    ap.add_argument("--query-file", help="SQL file overriding the default query")
    ap.add_argument("--out", default="dataset/human.jsonl")
    ap.add_argument("--max-per-stratum", type=int, default=800,
                    help="cap of CHUNKS per (lang, domain) stratum")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="print the query and exit")
    args = ap.parse_args()

    query = open(args.query_file).read() if args.query_file else DEFAULT_QUERY
    if args.dry_run:
        print(query)
        return 0
    if not args.dsn:
        print("ERROR: set CORPUS_DB_DSN or pass --dsn", file=sys.stderr)
        return 2

    try:
        import pymysql
    except ImportError:
        print("ERROR: pip install pymysql (offline-script dependency)", file=sys.stderr)
        return 2

    u = urlparse(args.dsn)
    conn = pymysql.connect(host=u.hostname, port=u.port or 3306, user=u.username,
                           password=u.password or "", database=u.path.lstrip("/"),
                           charset="utf8mb4", cursorclass=pymysql.cursors.SSDictCursor)

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    strata: Counter = Counter()
    written = 0

    with conn, conn.cursor() as cur, open(args.out, "w", encoding="utf-8") as out:
        cur.execute(query)
        for row in cur:
            lang = str(row.get("idioma") or "unknown").lower()[:5]
            domain = str(row.get("disciplina") or "unknown").lower()[:40]
            year = row.get("anio")
            for chunk in chunk_text(str(row.get("texto") or "")):
                key = (lang, domain)
                # Reservoir-free cap with random drop keeps streaming memory flat.
                if strata[key] >= args.max_per_stratum and rng.random() > 0.05:
                    continue
                strata[key] += 1
                out.write(json.dumps({
                    "text": chunk, "label": 0, "lang": lang, "domain": domain,
                    "words": len(chunk.split()), "year": year,
                    "author_id": str(row.get("autor_id") or ""),
                    "doc_id": str(row.get("id") or ""),
                    "source": "db-pre2022",
                }, ensure_ascii=False) + "\n")
                written += 1

    print(f"{written} human chunks -> {args.out}")
    for (lang, domain), n in strata.most_common(20):
        print(f"  {lang:>5} / {domain:<30} {n}")
    if written < 10_000:
        print("WARNING: doc A minimum viable is 10,000 — corpus below target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
