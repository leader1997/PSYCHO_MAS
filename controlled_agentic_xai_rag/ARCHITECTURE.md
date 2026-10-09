# Controlled Agentic XAI-RAG Architecture

This system generates safe, evidence-grounded, non-clinical screening narratives from ML predictions and XAI outputs. It does **not** diagnose or prescribe treatment.

## Components

1. **Input Layer** — de-identified case profile, XAI card, PDF knowledge corpus
2. **Step 0: Case Formatter** — deterministic JSON standardization
3. **Agent 1: Query Planning** — template slots + llama3 refinement
4. **Step 1: Query Plan Validator** — schema, mandatory slots, safety wording
5. **Step 2: Hybrid Retrieval** — BM25 + llama3 dense, RRF fusion
6. **Agent 2: Report Generation** — single advisory JSON report
7. **Step 3: Validators** — safety, citation, claim support, XAI alignment
8. **Output Layer** — `final_narratives.jsonl` + audit logs

## Ollama policy

- `stream: false` — wait for full response
- `inter_call_delay_seconds: 1` — pause before next call
- No parallel Ollama requests

## Evaluation

Component metrics + six ablation conditions: `no_rag`, `generic_rag`, `template_retrieval`, `template_plus_llm`, `planned_rag_validators`, `full_system`.
