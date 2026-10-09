# Version roadmap (scientific paper)

| Version | Description | Status | Command |
|---------|-------------|--------|---------|
| **v1** | Baseline FAST-mode pipeline | Frozen | `python run_freeze_baseline.py` |
| **v2** | Full-quality training (300 trees, 35 DL epochs) | Pending | `$env:XAI_FAST="0"; python run_strengthened_pipeline.py --full-pipeline --full-quality` |
| **v3** | Without `aca_impa` (ablation E2) | In `outputs/strengthened/` | `python run_feature_ablation.py` |
| **v4** | Improved RAG retrieval (hybrid + rerank) | Planned | See roadmap §11 |
| **v5** | Larger LLM evaluation (30–100 cases) | Planned | See roadmap §12 |
| **v6** | Final paper version | TBD | — |

Snapshots live in `outputs/versions/v1_fast/`, etc.

Registry: `outputs/versions/VERSION_REGISTRY.json`
