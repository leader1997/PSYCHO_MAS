# Methods — Local Agentic XAI-RAG (Ollama llama3 only)

## Objective

Transform probabilistic depression/anxiety **screening signals** and **XAI explanations** into **structured advisory reports for human review** — not clinical diagnoses.

## Architecture

1. **Case formatter** (deterministic tool)
2. **XAI-aware query planning agent** (llama3) — internal questions + retrieval queries
3. **Hybrid retrieval**: BM25 + llama3 dense embeddings (RRF)
4. **Evidence verification agent** (llama3 structured JSON)
5. **Advisory report agent** (llama3 structured JSON)
6. **Safety rules** (regex: disclaimer, referral, diagnostic boundaries)
7. **Claim verification** (llama3 per citation)
8. **Human expert review** (external)

Fallback: fixed template queries when `pipeline.use_query_planning_agent: false` or `--templates-only`.

## Ollama configuration

- **Single model:** `llama3` for chat, structured JSON, and embeddings
- No cloud APIs, no separate judge model, no nomic-embed-text

## Evaluation (objective)

- Gold-chunk retrieval: P@5, R@5, R@10, MRR, nDCG@10
- Claim support rate (supported / partial / unsupported / contradicted)
- Regex safety pass rate
- System comparison B1 (retrieve→report) vs B2 (+evidence) vs B3 (+safety/claims)
