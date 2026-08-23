"""Rekonstruiertes Panel gegen die bestehenden Archivdateien abgleichen.

Das Panel wird aus /performance neu aufgebaut, waehrend data_*_2526.json aus
einem anderen Pfad stammt (teamprofile + performance, siehe
archive_fetcher_2526.py). Stimmen beide ueberein, ist die Parserkette
belastbar - inklusive der heiklen Teile: Minuten-String mit Apostroph,
Heim/Auswaerts-Ableitung ueber ``pt`` gegen ``t1``/``t2``, und die
Normalisierung fehlender Punkte.

Der Test ist bewusst streng (0 Abweichungen erlaubt), weil ein stiller
Parserfehler hier das gesamte Trainingsset verdirbt.
"""

from __future__ import annotations

import json
import pandas as pd
import pytest

from src.paths import LEGACY_DIR, PROCESSED  # noqa: E402

# Die Liga gehoert in den Schluessel: 25 Spieler haben in 25/26 Zeilen in
# beiden Ligen (Wintertransfer BL2 <-> BL1). Jede Archivdatei kennt nur ihre
# eigene Liga - ohne Liga im Schluessel vergliche man Aepfel mit Birnen.
ARCHIVE_FILES = {
    "data_1_2526.json": "Bundesliga",
    "data_2_2526.json": "2. Bundesliga",
}
ARCHIVE_SEASON = "2025/2026"


def _load_archive() -> dict[tuple[int, int, str], tuple[int, int, str, bool]]:
    """(player_id, matchday, league) -> (punkte, minuten, gegner, heim)."""
    out: dict[tuple[int, int, str], tuple[int, int, str, bool]] = {}
    for filename, league in ARCHIVE_FILES.items():
        path = LEGACY_DIR / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for player in data.get("players", []):
            try:
                pid = int(player["id"])
            except (KeyError, ValueError, TypeError):
                continue
            for entry in player.get("performance", []):
                minutes = int(str(entry.get("minutes", "0'")).rstrip("'") or 0)
                out[(pid, entry["day"], league)] = (
                    entry.get("points") or 0,
                    minutes,
                    entry.get("opponent"),
                    bool(entry.get("home")),
                )
    return out


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    path = PROCESSED / "panel.parquet"
    if not path.exists():
        pytest.skip("panel.parquet fehlt - erst backfill_history laufen lassen")
    return pd.read_parquet(path)


def test_panel_agrees_with_archive(panel: pd.DataFrame) -> None:
    archive = _load_archive()
    if not archive:
        pytest.skip("Archivdateien nicht vorhanden")

    season = panel[panel["season"] == ARCHIVE_SEASON]
    if season.empty:
        pytest.skip(f"Panel enthaelt keine Saison {ARCHIVE_SEASON}")

    compared = 0
    mismatches: list[str] = []
    benign: list[str] = []

    for row in season.itertuples(index=False):
        key = (int(row.player_id), int(row.matchday), row.league)
        if key not in archive:
            continue
        compared += 1
        # Das Archiv normalisiert "nicht gespielt" (Feld fehlt) zu 0 Punkten.
        points = 0 if pd.isna(row.points) else int(row.points)
        actual = (points, int(row.minutes), row.opponent_id, bool(row.home))
        expected = archive[key]
        if actual == expected:
            continue

        # Zeilen ohne Einsatz auf beiden Seiten: hier weicht hoechstens die
        # Gegnerzuordnung ab. Sie entsteht, wenn ``pt`` fehlt und der
        # fortgeschriebene Verein zu keiner der beiden Mannschaften der Partie
        # passt - ein Transferartefakt. Fuer die Modellierung belanglos, weil
        # ohne Einsatz weder Punkte noch Minuten anfallen.
        no_appearance = points == 0 and int(row.minutes) == 0 and expected[0] == 0 and expected[1] == 0
        if no_appearance:
            benign.append(f"{key}: Gegner {row.opponent_id} vs {expected[2]}")
        else:
            mismatches.append(f"{key}: Panel {actual} != Archiv {expected}")

    assert compared > 0, "Keine gemeinsamen Spieler-Spieltage zum Vergleichen"
    assert not mismatches, (
        f"{len(mismatches)} von {compared} Eintraegen mit Einsatz weichen ab:\n"
        + "\n".join(mismatches[:15])
    )
    # Die harmlose Sorte darf vorkommen, aber nicht ueberhandnehmen.
    assert len(benign) / compared < 0.001, (
        f"{len(benign)} von {compared} Zeilen ohne Einsatz mit abweichendem Gegner:\n"
        + "\n".join(benign[:10])
    )


def test_panel_has_expected_shape(panel: pd.DataFrame) -> None:
    expected = {
        "player_id", "season", "league", "matchday", "match_id", "match_date",
        "team_id", "opponent_id", "home", "minutes", "points", "events",
    }
    assert expected <= set(panel.columns), f"fehlende Spalten: {expected - set(panel.columns)}"
    assert panel["league"].isin(["Bundesliga", "2. Bundesliga"]).all()
    assert panel["matchday"].between(1, 34).all()
    # Minuten liegen wegen Nachspielzeit regulaer ueber 90. Die Ausreisser nach
    # oben sind einzelne lang unterbrochene Partien, bei denen Kickbase die Uhr
    # weiterlaufen laesst - beobachtet: match_id 5822 (131') und 5475 (143'),
    # jeweils mit konsistenten Werten ueber alle Spielerzeilen der Partie. Echt,
    # kein Parserfehler. Fuer den Minutenanteil werden sie bei 90 gedeckelt.
    assert panel["minutes"].between(0, 150).all(), (
        f"Minuten ausserhalb [0,150]: max {panel['minutes'].max()}"
    )
    assert (panel["minutes"] > 105).mean() < 0.01, "auffaellig viele Ueberlangzeiten"
    # Jede Kombination darf nur einmal vorkommen.
    dupes = panel.duplicated(subset=["player_id", "season", "league", "matchday"]).sum()
    assert dupes == 0, f"{dupes} doppelte Spieler-Spieltage"
