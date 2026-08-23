"""Zentrale Pfade. Alles Erzeugte landet unter kbxp/data."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

KBXP_ROOT = Path(__file__).resolve().parent.parent      # .../Kickbase API/kbxp
PROJECT_ROOT = KBXP_ROOT.parent                          # .../Kickbase API

DATA = KBXP_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANUAL = DATA / "manual"

# Bestandsdateien der Fetcher-Pipeline - werden hier nur GELESEN.
# Liegen in data/ im Projektwurzelverzeichnis, weil die HTML-Seiten sie
# per relativem fetch() von dort laden (GitHub Pages liefert statisch aus).
LEGACY_DIR = PROJECT_ROOT / "data"


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, PROCESSED, MANUAL):
        d.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Any, *, indent: int = 1) -> None:
    """JSON vollstaendig neben dem Ziel schreiben und atomar ersetzen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=path.parent, text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
