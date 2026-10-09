"""Add src/ and sibling packages to sys.path so scripts can import cleanly."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_MAS = _ROOT.parent

_paths = [
    _SRC,
    _MAS / "knowledge_db_contextual_eval" / "src",
    _MAS / "xai_rag_depression",
]
for _p in _paths:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
