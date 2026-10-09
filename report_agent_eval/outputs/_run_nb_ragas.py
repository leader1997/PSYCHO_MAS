"""Execute RAGAS_Smoke_Test.ipynb with the registered python312 kernel."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(r"c:\Users\MossabBatal\Desktop\psychology_mas\report_agent_eval\RAGAS_Smoke_Test.ipynb")

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb,
    timeout=2400,
    kernel_name="python312",
    allow_errors=False,
)
print(f"Executing {NB.name} with kernel=python312 ...", flush=True)
client.execute()
nbformat.write(nb, NB)
print("Done. Notebook updated in place.", flush=True)
