#!/usr/bin/env python3
"""Replace notebook 04 sections 10-13 with link to notebook 07."""

import json
from pathlib import Path

NB = Path(__file__).parent / "04_llm_generation_evaluation.ipynb"

LINK = """## 10. Paper-aligned RAGAS evaluation (100-question benchmark)

The **100-question labeled benchmark** and **RAGAS evaluation** (six metrics from Batal et al., 2026) live in a dedicated citable notebook:

**[07_ragas_benchmark_evaluation.ipynb](07_ragas_benchmark_evaluation.ipynb)**

Run: `python run_nb7.py` or open notebook 07.
"""


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    start = end = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "## 10." in src and "benchmark" in src.lower():
            start = i
        if start is not None and "Notebook 4 complete" in src:
            end = i
            break
    if start is None or end is None:
        print("Could not find section boundaries; skipping patch.")
        return
    link_cell = {"cell_type": "markdown", "metadata": {}, "source": [LINK]}
    nb["cells"] = cells[:start] + [link_cell] + cells[end:]
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Patched {NB}: replaced cells {start}-{end - 1} with link to notebook 07")


if __name__ == "__main__":
    main()
