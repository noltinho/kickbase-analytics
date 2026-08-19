"""Zentrale Pfade. Alles Erzeugte landet unter kbxp/data."""

from __future__ import annotations

from pathlib import Path

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
