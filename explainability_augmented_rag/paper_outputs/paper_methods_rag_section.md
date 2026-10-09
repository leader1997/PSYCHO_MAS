# RAG Methodology (Draft)

## Overview
We implemented an Explainability-Augmented RAG pipeline for **non-clinical** depression/anxiety **screening communication** using HMS 2024–2025 survey features. The system does **not** diagnose mental illness.

## Knowledge base
- Section-aware chunking (~450 tokens, overlap 75)
- Metadata: document ID, title, year, page, section, source type, domain tags
- Corpus includes peer-reviewed literature PDFs and project artifacts (XAI cards, ethics)

## Retrieval
Four strategies compared: dense (text-embedding-3-small), BM25, hybrid RRF, hybrid + cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2).

## Generation pipeline
1. Query construction from case profile + XAI factors (4 query types)
2. Evidence synthesis (structured JSON, chunk-linked claims)
3. Safe narrative generation (structured JSON, mandatory disclaimer)
4. Safety validator (8 checks + revision loop)
5. Claim-level faithfulness checker

## Evaluation
- 100-question RAG benchmark with remapped gold chunk IDs
- RAGAS six metrics (Batal et al. 2026 aligned) across systems S1–S6
- Full ablation A1–A8 on de-identified cases
- Human expert review protocol (template provided)

## LLM configuration
- Generation/judge: gpt-4o
- Embeddings: text-embedding-3-small
- Random seed: 42
