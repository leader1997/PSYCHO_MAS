# Limitations

1. **Not diagnostic** — advisory screening reports require human interpretation.
2. **Single Ollama model** — llama3 used for generation, verification, and embeddings (no separate judge).
3. **Retrieval quality** — depends on Phase 2 corpus (29 PDFs); gold-chunk recall remains limited.
4. **Claim verification** — llama3 may mark claims unsupported when evidence is partial.
5. **Expert review** — required before any real-world use; template provided in `paper_outputs/`.
6. **CrewAI** — sequential agent flow implemented; full `crewai run` optional when package installed.
