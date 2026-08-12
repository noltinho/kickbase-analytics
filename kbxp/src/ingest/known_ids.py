"""Spieler-IDs aus den vorhandenen JSON-Dateien einsammeln (nur lesend)."""

from __future__ import annotations

import json

from ..paths import LEGACY_PLAYER_FILES


def load_known_players() -> dict[int, dict]:
    """Bekannte Spieler aus data_1/2[_2526].json.

    Rückgabe: {player_id: {"name", "sources": [...], "position", "team_name"}}
    Die zuletzt gelesene Datei gewinnt bei Namenskollisionen - relevant ist hier
    nur die ID-Menge, nicht die Stammdaten.
    """
    players: dict[int, dict] = {}
    for path in LEGACY_PLAYER_FILES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("players", []):
            try:
                pid = int(p["id"])
            except (KeyError, TypeError, ValueError):
                continue
            entry = players.setdefault(pid, {"sources": []})
            entry["name"] = p.get("name", entry.get("name", "?"))
            entry["position"] = p.get("position", entry.get("position", 0))
            entry["team_name"] = p.get("team_name", entry.get("team_name", "?"))
            entry["sources"].append(path.name)
    return players


def known_id_set() -> set[int]:
    return set(load_known_players())
