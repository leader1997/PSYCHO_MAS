import argparse
import csv
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "metadata" / "selected_documents.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


EVAL_QUERIES = [
    ("PHQ-9 depression screening instrument for college students", {"DOC_B006", "DOC_B007", "DOC_A009"}),
    ("GAD-7 anxiety scale validation among college students", {"DOC_B002", "DOC_B003"}),
    ("loneliness scale social isolation depression among students", {"DOC_C001", "DOC_C002", "DOC_C003"}),
    ("academic stress impairment depression anxiety university students", {"DOC_D001", "DOC_D002", "DOC_D003", "DOC_D004"}),
    ("food insecurity financial stress mental health college students", {"DOC_E001", "DOC_E002", "DOC_E003"}),
    ("SHAP feature attribution explainable AI calibration probability", {"DOC_H001", "DOC_H002", "DOC_H003", "DOC_H005"}),
    ("RAG citation faithfulness hallucination evaluation retrieval augmented generation", {"DOC_I001", "DOC_I002", "DOC_I003", "DOC_I004", "DOC_I005", "DOC_I006"}),
]


def ollama_json(endpoint, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def embed(text, model):
    result = ollama_json("/api/embeddings", {"model": model, "prompt": text})
    return np.array(result["embedding"], dtype=np.float32)


def generate_context(doc, chunk, model):
    prompt = f"""Document title: {doc.get('title', '')}
Document topic: {doc.get('main_topic', '')}
Document category: {doc.get('category_code', '')}

Chunk:
{chunk[:3500]}

Write one concise sentence that states what this chunk is about and how it fits the document. Do not add unsupported facts."""
    result = ollama_json(
        "/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=240,
    )
    return " ".join(result.get("response", "").split())[:700]


def load_docs(limit_docs=None, categories=None, doc_ids=None):
    docs = []
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("download_status") != "downloaded":
                continue
            if doc_ids and doc.get("doc_id") not in doc_ids:
                continue
            if categories and doc.get("category_code") not in categories:
                continue
            path = ROOT / doc.get("local_pdf_path", "")
            if not path.exists():
                continue
            doc["abs_pdf_path"] = str(path)
            docs.append(doc)
    docs.sort(key=lambda d: d["doc_id"])
    return docs[:limit_docs] if limit_docs else docs


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def chunk_words(text, words_per_chunk=220, overlap=50):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + words_per_chunk])
        if len(chunk) >= 200:
            chunks.append(chunk)
        if start + words_per_chunk >= len(words):
            break
    return chunks


def extract_chunks(doc, max_pages, chunks_per_doc):
    reader = PdfReader(doc["abs_pdf_path"])
    chunks = []
    for page_idx, page in enumerate(reader.pages[:max_pages], start=1):
        text = clean_text(page.extract_text())
        for chunk_idx, chunk in enumerate(chunk_words(text), start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_p{page_idx}_c{chunk_idx}",
                    "doc_id": doc["doc_id"],
                    "page": page_idx,
                    "text": chunk,
                    "title": doc.get("title", ""),
                    "category_code": doc.get("category_code", ""),
                    "main_topic": doc.get("main_topic", ""),
                }
            )
            if len(chunks) >= chunks_per_doc:
                return chunks
    return chunks


def metadata_context(chunk):
    return (
        f"Document: {chunk['title']}. "
        f"Category: {chunk['category_code']}. "
        f"Topic: {chunk['main_topic']}. "
        f"Page: {chunk['page']}."
    )


def cosine_matrix(query_vec, matrix):
    query_norm = np.linalg.norm(query_vec) or 1.0
    matrix_norm = np.linalg.norm(matrix, axis=1)
    matrix_norm[matrix_norm == 0] = 1.0
    return matrix @ query_vec / (matrix_norm * query_norm)


def evaluate(chunks, vectors, embed_model, top_k=5):
    matrix = np.vstack(vectors)
    rows = []
    reciprocal_ranks = []
    recall_hits = 0
    for query, expected_docs in EVAL_QUERIES:
        query_vec = embed(query, embed_model)
        scores = cosine_matrix(query_vec, matrix)
        order = np.argsort(-scores)[:top_k]
        retrieved = [chunks[i] for i in order]
        retrieved_docs = [item["doc_id"] for item in retrieved]
        hit_positions = [
            pos + 1 for pos, doc_id in enumerate(retrieved_docs) if doc_id in expected_docs
        ]
        rr = 1.0 / hit_positions[0] if hit_positions else 0.0
        reciprocal_ranks.append(rr)
        recall_hits += 1 if hit_positions else 0
        rows.append(
            {
                "query": query,
                "expected_docs": ",".join(sorted(expected_docs)),
                "top_docs": ",".join(retrieved_docs),
                "hit_rank": hit_positions[0] if hit_positions else "",
                "reciprocal_rank": rr,
            }
        )
    return {
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "recall_at_k": recall_hits / len(EVAL_QUERIES),
        "rows": rows,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--context-model", default="qwen2.5:latest")
    parser.add_argument("--limit-docs", type=int, default=18)
    parser.add_argument(
        "--doc-ids",
        default="",
        help="Comma-separated document IDs for a focused smoke test.",
    )
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--context-mode",
        choices=["metadata", "ollama"],
        default="metadata",
        help="metadata is fast; ollama generates Claude-style per-chunk context locally.",
    )
    args = parser.parse_args()

    doc_ids = {item.strip() for item in args.doc_ids.split(",") if item.strip()}
    docs = load_docs(args.limit_docs, doc_ids=doc_ids or None)
    chunks = []
    for doc in docs:
        chunks.extend(extract_chunks(doc, args.max_pages, args.chunks_per_doc))

    if not chunks:
        raise SystemExit("No chunks extracted.")

    plain_vectors = []
    contextual_vectors = []
    contextual_examples = []
    start = time.time()
    for idx, chunk in enumerate(chunks, start=1):
        plain_text = chunk["text"]
        if args.context_mode == "ollama":
            context = generate_context(chunk, chunk["text"], args.context_model)
        else:
            context = metadata_context(chunk)
        contextual_text = f"{context}\n\n{plain_text}"
        plain_vectors.append(embed(plain_text, args.embed_model))
        contextual_vectors.append(embed(contextual_text, args.embed_model))
        if len(contextual_examples) < 5:
            contextual_examples.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "context": context,
                    "preview": chunk["text"][:220],
                }
            )
        print(f"embedded {idx}/{len(chunks)} chunks", flush=True)

    plain = evaluate(chunks, plain_vectors, args.embed_model, args.top_k)
    contextual = evaluate(chunks, contextual_vectors, args.embed_model, args.top_k)

    out = {
        "settings": vars(args),
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "elapsed_seconds": round(time.time() - start, 2),
        "plain": {k: v for k, v in plain.items() if k != "rows"},
        "contextual": {k: v for k, v in contextual.items() if k != "rows"},
        "contextual_examples": contextual_examples,
    }
    output_stem = (
        Path(__file__).resolve().parent
        / f"contextual_ollama_{args.context_mode}_{len(docs)}docs_{len(chunks)}chunks"
    )
    output_json = output_stem.with_suffix(".json")
    output_csv_plain = output_stem.with_name(output_stem.name + "_plain.csv")
    output_csv_contextual = output_stem.with_name(output_stem.name + "_contextual.csv")
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_csv(output_csv_plain, plain["rows"])
    write_csv(output_csv_contextual, contextual["rows"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
