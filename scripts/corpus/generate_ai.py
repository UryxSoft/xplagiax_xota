"""
generate_ai.py — [doc A, paso A.2] Build the AI class (label 1) — Anthropic share.

Generates AI counterparts for sampled human chunks using the Message Batches API
(50% of standard prices; ideal for corpus generation — no latency requirement).
Tasks rotate (continue / rewrite / expand / title) to produce PARALLEL pairs: same
topic and register as the human sample, so the fusion learns human-vs-AI, not topic.

Model basket: doc A mandates MULTIPLE generator models ("nunca un solo modelo, o el
detector solo detecta ESE modelo"). This script covers the Anthropic share of the
basket; the open-model share (Llama/Qwen/Mistral via Colab GPU) is covered by the
notebook snippets in docs/sota/A_FUSION_ENTRENADA.md §A.2 paso 3.

Auth: ANTHROPIC_API_KEY (or `ant auth login` profile). Resumable: reruns skip
custom_ids already present in the output file.

    .venv/bin/python scripts/corpus/generate_ai.py \
        --human dataset/human.jsonl --out dataset/ai_anthropic.jsonl --limit 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time

# Diversity basket (doc A): rotating tiers so the detector doesn't overfit one model.
MODELS = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]
BATCH_SIZE = 1000          # requests per batch (API max 100k; keep batches manageable)
MAX_TOKENS = 2048

TASKS = {
    "continue": {
        "en": "Continue this academic text, keeping its style and register. Write the continuation only:\n\n{head}",
        "es": "Continúa este texto académico manteniendo su estilo y registro. Escribe solo la continuación:\n\n{head}",
    },
    "rewrite": {
        "en": "Rewrite the following text in your own words, academic register, similar length. Output only the rewritten text:\n\n{text}",
        "es": "Reescribe el siguiente texto con tus propias palabras, registro académico, longitud similar. Devuelve solo el texto reescrito:\n\n{text}",
    },
    "expand": {
        "en": "Develop this summary into a full thesis section (600-1200 words). Output only the section:\n\n{head}",
        "es": "Desarrolla este resumen en una sección completa de tesis (600-1200 palabras). Devuelve solo la sección:\n\n{head}",
    },
    "title": {
        "en": "Write the introduction (600-1200 words) of an academic paper whose opening reads:\n\n{head}\n\nOutput only the introduction.",
        "es": "Escribe la introducción (600-1200 palabras) de un trabajo académico cuyo inicio dice:\n\n{head}\n\nDevuelve solo la introducción.",
    },
}


def build_prompt(rec: dict, task: str) -> str:
    lang = "es" if str(rec.get("lang", "en")).startswith("es") else "en"
    tpl = TASKS[task][lang]
    words = rec["text"].split()
    return tpl.format(text=rec["text"][:8000], head=" ".join(words[:200]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", default="dataset/human.jsonl")
    ap.add_argument("--out", default="dataset/ai_anthropic.jsonl")
    ap.add_argument("--limit", type=int, default=2000, help="max samples to generate this run")
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()

    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["custom_id"])
                except Exception:
                    pass

    task_names = list(TASKS)
    pending: list[tuple[str, dict, str, str]] = []  # (custom_id, rec, task, model)
    with open(args.human, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if len(pending) >= args.limit:
                break
            rec = json.loads(line)
            task = task_names[i % len(task_names)]
            model = MODELS[i % len(MODELS)]
            custom_id = f"ai-{rec.get('doc_id', i)}-{i}-{task}"
            if custom_id in done:
                continue
            pending.append((custom_id, rec, task, model))

    if not pending:
        print("Nothing to generate (all custom_ids already in output).")
        return 0
    print(f"Generating {len(pending)} samples across {len(MODELS)} models via Batches API…")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = {cid: (rec, task, model) for cid, rec, task, model in pending}

    with open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, len(pending), BATCH_SIZE):
            slice_ = pending[start:start + BATCH_SIZE]
            batch = client.messages.batches.create(requests=[
                Request(
                    custom_id=cid,
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=MAX_TOKENS,
                        messages=[{"role": "user", "content": build_prompt(rec, task)}],
                    ),
                )
                for cid, rec, task, model in slice_
            ])
            print(f"  batch {batch.id}: {len(slice_)} requests…")
            while True:
                b = client.messages.batches.retrieve(batch.id)
                if b.processing_status == "ended":
                    break
                time.sleep(args.poll_seconds)

            ok = err = 0
            for result in client.messages.batches.results(batch.id):
                rec, task, model = meta[result.custom_id]
                if result.result.type != "succeeded":
                    err += 1
                    continue
                msg = result.result.message
                text = next((blk.text for blk in msg.content if blk.type == "text"), "").strip()
                if len(text.split()) < 150:      # too short to be a usable sample
                    err += 1
                    continue
                out.write(json.dumps({
                    "custom_id": result.custom_id,
                    "text": text, "label": 1,
                    "lang": rec.get("lang", "en"), "domain": rec.get("domain", "unknown"),
                    "words": len(text.split()),
                    "model": model, "task": task,
                    "author_id": f"llm:{model}",       # group split treats each model as an author
                    "doc_id": rec.get("doc_id", ""),   # pairs with the human source
                    "source": "anthropic-batches",
                }, ensure_ascii=False) + "\n")
                ok += 1
            out.flush()
            print(f"  batch done: {ok} ok, {err} skipped/errored "
                  f"(succeeded={b.request_counts.succeeded} errored={b.request_counts.errored})")

    print(f"Done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
