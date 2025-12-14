from __future__ import annotations
import os
from pathlib import Path

def data_root() -> Path | None:
    val = os.environ.get("ARCTICROUTE_DATA_ROOT") or os.environ.get("DATA_ROOT")
    if not val:
        return None
    p = Path(val).expanduser().resolve()
    return p if p.exists() else None

