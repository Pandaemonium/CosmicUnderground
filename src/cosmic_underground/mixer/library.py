from __future__ import annotations
import os
from typing import List
from cosmic_underground.core import config as C

def scan_inventory(inv_dir: str | None = None) -> List[str]:
    """Return list of .wav files in the inventory directory.
    Defaults to the unified config inventory path.
    """
    base = os.fspath(C.INVENTORY if inv_dir is None else inv_dir)
    try:
        return sorted(
            os.path.join(base, f) for f in os.listdir(base)
            if f.lower().endswith(".wav")
        )
    except Exception:
        return []
