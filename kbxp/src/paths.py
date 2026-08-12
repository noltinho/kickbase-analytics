"""Zentrale Pfade. Alles Erzeugte landet unter kbxp/data bzw. kbxp/output."""

from __future__ import annotations

from pathlib import Path

KBXP_ROOT = Path(__file__).resolve().parent.parent      # .../Kickbase API/kbxp
PROJECT_ROOT = KBXP_ROOT.parent                          # .../Kickbase API

DATA = KBXP_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANUAL = DATA / "manual"
OUTPUT = KBXP_ROOT / "output"
REPORTS = KBXP_ROOT / "reports"

# Bestandsdateien der Fetcher-Pipeline - werden hier nur GELESEN.
# Liegen in data/ im Projektwurzelverzeichnis, weil die HTML-Seiten sie
# per relativem fetch() von dort laden (GitHub Pages liefert statisch aus).
LEGACY_DIR = PROJECT_ROOT / "data"

LEGACY_PLAYER_FILES = [
    LEGACY_DIR / "data_1.json",
    LEGACY_DIR / "data_2.json",
    LEGACY_DIR / "data_1_2526.json",
    LEGACY_DIR / "data_2_2526.json",
]
LEGACY_MATCHDAY_FILES = [
    LEGACY_DIR / "matchdays_bl1.json",
    LEGACY_DIR / "matchdays_bl2.json",
    LEGACY_DIR / "matchdays_bl1_2526.json",
    LEGACY_DIR / "matchdays_bl2_2526.json",
]


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, PROCESSED, MANUAL, OUTPUT, REPORTS / "figures"):
        d.mkdir(parents=True, exist_ok=True)
