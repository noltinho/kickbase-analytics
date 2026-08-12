"""Kickbase-Fetcher: laufende Saison, Marktwert-Historie und Archiv in einem Werkzeug.

    python fetch.py                        laufende Saison   (Default, jeden Spieltag)
    python fetch.py mv                     Marktwert-Historie verdichten (etwa monatlich)
    python fetch.py archive --season 2526  abgeschlossene Saison (sehr selten)

Welche Saison die laufende ist, steht in ``data/seasons.json`` - nicht im Code.
Dieselbe Datei lesen die HTML-Seiten, damit ein Saisonwechsel eine einzige
Stelle betrifft.

Der Zuschnitt folgt einer Unterscheidung, die sich an der API messen laesst:

*Nachholbar* ist alles aus ``/performance`` - der Endpunkt liefert **alle**
Saisons eines Spielers zurueck, bis ~2013/14. Punkte, Minuten, Event-Codes und
Ergebnisse gehen also nie verloren und werden nur so weit gespeichert, wie die
Seiten sie lesen.

*Fluechtig* ist der Marktwert: ``/marketValue/{tage}`` beantwortet nur ``365``
(groessere Fenster liefern eine leere Liste). Was aus diesem Fenster
herauslaeuft, ist endgueltig weg. Deshalb schreibt jeder Lauf den Tageswert in
``data/history.json`` fort - dort waechst die Historie ueber die API-Grenze
hinaus. Gleiches gilt fuer den Verletzungsstatus, den die API nur als
Momentaufnahme kennt.

Zugangsdaten via KICKBASE_EMAIL / KICKBASE_PASSWORD oder .env im Skriptverzeichnis.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any

import requests

# Windows-Konsole/Pipes nutzen sonst cp1252 und scheitern an ✓/⚠
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://api.kickbase.com"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ein-/Ausgabe immer neben dem Skript in data/, nie im aktuellen Arbeitsverzeichnis.
# Die HTML-Seiten laden von dort per relativem fetch('data/...').
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

MANIFEST_FILE = "seasons.json"
HISTORY_FILE = "history.json"

EPOCH = date(1970, 1, 1)

COMPETITIONS = [
    {"id": "1", "league": "Bundesliga"},
    {"id": "2", "league": "2. Bundesliga"},
]

# Fallback, falls data/seasons.json noch nicht existiert.
DEFAULT_MANIFEST = {
    "current": "2627",
    "seasons": [
        {"key": "2627", "title": "2026/2027", "label": "26/27", "suffix": ""},
        {"key": "2526", "title": "2025/2026", "label": "25/26", "suffix": "_2526"},
    ],
}


def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def day_number(d: date) -> int:
    """Tage seit 1970-01-01 - das Format der Marktwert-Historie."""
    return (d - EPOCH).days


def from_day_number(n: int) -> date:
    return date.fromordinal(EPOCH.toordinal() + n)


# --------------------------------------------------------------------------
# HTTP-Client
# --------------------------------------------------------------------------

# Sentinel: Anfrage lief sauber durch, die Ressource gibt es aber nicht.
# Unterscheidet sich von None (= Fehler/unklar) und von {} (= leere Antwort).
NOT_FOUND = object()

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def load_credentials() -> tuple[str, str]:
    email = os.environ.get("KICKBASE_EMAIL")
    password = os.environ.get("KICKBASE_PASSWORD")
    if not (email and password):
        env_path = os.path.join(SCRIPT_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip().strip('"').strip("'")
                    if key == "KICKBASE_EMAIL" and not email:
                        email = value
                    elif key == "KICKBASE_PASSWORD" and not password:
                        password = value
    if not (email and password):
        print("✗ Zugangsdaten fehlen: KICKBASE_EMAIL und KICKBASE_PASSWORD als")
        print("  Umgebungsvariablen setzen oder in einer .env-Datei neben dem Skript ablegen.")
        sys.exit(1)
    return email, password


def _is_not_found_body(response: requests.Response) -> bool:
    """Erkennt Kickbases als HTTP 500 verkleidetes "NotFound"."""
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and (body.get("errMsg") == "NotFound" or body.get("err") == 2)


class KickbaseClient:
    """Gedrosselter v4-Client mit gemeinsamer Rate-Limit-Bremse.

    ``delay`` gilt pro Thread vor jeder Anfrage; bei ``workers`` Threads ergibt
    das grob ``workers / delay`` Requests/s. Bewusst konservativ halten, siehe
    ToS-Hinweis in Kickbase-API.md.

    Ein 429 haelt **alle** Threads an (globales Event), statt dass jeder Thread
    fuer sich weiterlaeuft und das Limit erneut trifft.
    """

    def __init__(self, delay: float = 0.3, timeout: int = 15):
        self.delay = delay
        self.timeout = timeout
        self.token: str | None = None
        # gesetzt = freie Fahrt; geloescht = alle Threads warten
        self._go = threading.Event()
        self._go.set()
        self._backoff_lock = threading.Lock()

    def login(self) -> str:
        email, password = load_credentials()
        r = requests.post(
            f"{BASE_URL}/v4/user/login",
            json={"em": email, "pass": password, "loy": False, "rep": {}},
            headers=HEADERS,
            timeout=self.timeout,
        )
        if r.status_code != 200:
            print(f"✗ Login fehlgeschlagen: {r.status_code}")
            sys.exit(1)
        token = r.json().get("tkn")
        if not token:
            print("✗ Login lieferte kein Token (Feld 'tkn')")
            sys.exit(1)
        self.token = token
        print("✓ Login erfolgreich")
        return token

    def _pause_all(self, seconds: float) -> None:
        """Alle Threads anhalten. Nur der erste Thread setzt die Pause an."""
        if not self._backoff_lock.acquire(blocking=False):
            self._go.wait()  # ein anderer Thread pausiert bereits
            return
        try:
            self._go.clear()
            print(f"  ⚠ Rate-Limit - alle Threads pausieren {seconds:.0f}s")
            time.sleep(seconds)
        finally:
            self._go.set()
            self._backoff_lock.release()

    def get(self, path: str, retries: int = 3, quiet: bool = True) -> Any:
        """GET auf die v4-API.

        dict/list = Erfolg, NOT_FOUND = Ressource existiert nicht,
        None = Fehler, Ergebnis unklar.
        """
        if self.token is None:
            raise RuntimeError("Kein Token - erst login() aufrufen.")

        attempt = 0
        rate_hits = 0
        while attempt < retries:
            self._go.wait()  # globale Bremse respektieren
            time.sleep(self.delay)
            try:
                r = _session().get(
                    f"{BASE_URL}{path}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                attempt += 1
                if not quiet:
                    print(f"  ⚠ Netzfehler {path} ({attempt}/{retries}): {e}")
                time.sleep(2 * attempt)
                continue

            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return None

            if r.status_code == 404:
                return NOT_FOUND

            # Kickbase meldet "gibt es nicht" auch als HTTP 500 mit
            # {"err":2,"errMsg":"NotFound"}. Ohne Sonderbehandlung wuerde jede
            # tote ID dreimal mit Backoff wiederholt.
            if r.status_code >= 500 and _is_not_found_body(r):
                return NOT_FOUND

            if r.status_code == 429:
                rate_hits += 1
                if rate_hits > 5:
                    print(f"  ⚠ Dauerhaftes Rate-Limit bei {path}, gebe auf")
                    return None
                self._pause_all(10 * rate_hits)
                continue  # zaehlt nicht als regulaerer Versuch

            if r.status_code == 401:
                self.login()  # Token abgelaufen
                attempt += 1
                continue

            if r.status_code >= 500:
                attempt += 1
                time.sleep(2 * attempt)
                continue

            if not quiet:
                print(f"  ⚠ HTTP {r.status_code} bei {path}")
            return None

        return None

    def ok(self, path: str) -> Any:
        """Wie get(), aber NOT_FOUND und Fehler beide als None."""
        data = self.get(path)
        return None if data is NOT_FOUND else data

    def any_competition(self, template: str, prefer: str = "1", **kwargs) -> Any:
        """Erst die eigene Competition fragen, dann die andere.

        Die Reihenfolge ist nicht beliebig: /performance liefert fuer beide
        Competitions dieselbe *abgeschlossene* Historie, ergaenzt aber die
        laufende Saison nur bei der Competition, in der der Spieler gerade
        spielt. Wer einen Zweitligaspieler zuerst gegen Competition 1 fragt,
        bekommt eine plausible Antwort ohne die aktuelle Saison.

        Der zweite Versuch faengt Auf-/Abstieg und Winterwechsel ab.
        """
        other = "2" if prefer == "1" else "1"
        for cid in (prefer, other):
            data = self.ok(template.format(cid=cid, **kwargs))
            if data:
                return data
        return None


# --------------------------------------------------------------------------
# Saison-Manifest
# --------------------------------------------------------------------------

def load_manifest() -> dict:
    path = data_path(MANIFEST_FILE)
    if not os.path.exists(path):
        return json.loads(json.dumps(DEFAULT_MANIFEST))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(data_path(MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def season_by_key(manifest: dict, key: str) -> dict:
    for s in manifest["seasons"]:
        if s["key"] == key:
            return s
    known = ", ".join(s["key"] for s in manifest["seasons"])
    print(f"✗ Unbekannte Saison '{key}'. Bekannt: {known}")
    print(f"  Neue Saisons in {DATA_DIR}\\{MANIFEST_FILE} eintragen.")
    sys.exit(1)


def current_season(manifest: dict) -> dict:
    return season_by_key(manifest, manifest["current"])


def files_for(comp: dict, season: dict) -> tuple[str, str]:
    """Daten- und Matchday-Dateiname fuer Competition und Saison."""
    suffix = season["suffix"]
    return f"data_{comp['id']}{suffix}.json", f"matchdays_bl{comp['id']}{suffix}.json"


def competition_name(comp: dict, season: dict) -> str:
    # "2026/2027" -> "2026/27", damit das Feld so aussieht wie bisher
    title = season["title"]
    return f"{comp['league']} {title[:5]}{title[-2:]}"


# --------------------------------------------------------------------------
# data/history.json - das Fluechtige sichern
# --------------------------------------------------------------------------

def load_history() -> dict:
    path = data_path(HISTORY_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("players", {})


def save_history(players: dict) -> None:
    """Ein Spieler pro Zeile.

    json.dump(indent=2) wuerde jede der hunderttausenden Zahlen auf eine eigene
    Zeile setzen und die Datei vervielfachen; ohne Zeilenumbrueche waere jeder
    Git-Diff die ganze Datei. Eine Zeile je Spieler ist beides nicht.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = sorted(players.items(), key=lambda kv: int(kv[0]))
    with open(data_path(HISTORY_FILE), "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "generated_at": {json.dumps(datetime.now().isoformat())},\n')
        f.write('  "players": {\n')
        for i, (pid, entry) in enumerate(rows):
            comma = "," if i < len(rows) - 1 else ""
            f.write(f'    "{pid}": {json.dumps(entry, separators=(",", ":"))}{comma}\n')
        f.write("  }\n}\n")
    size = os.path.getsize(data_path(HISTORY_FILE)) / 1024
    print(f"  ✓ {HISTORY_FILE}: {len(rows)} Spieler ({size:.0f} KB)")


def history_add_value(players: dict, pid: str, dt: int, mv: float) -> bool:
    """Marktwert einsortieren. Vorhandene Tage werden nie ueberschrieben.

    /marketValue liefert die Werte als Float, teamprofile als Int. Ohne
    Vereinheitlichung stuenden in derselben Reihe 36419649 und 36334053.0.
    """
    entry = players.setdefault(str(pid), {"dt": [], "mv": [], "st": []})
    dts = entry["dt"]
    dt = int(dt)
    i = bisect.bisect_left(dts, dt)
    if i < len(dts) and dts[i] == dt:
        return False
    dts.insert(i, dt)
    entry["mv"].insert(i, int(mv))
    return True


def history_add_status(players: dict, pid: str, dt: int, status: int) -> None:
    """Verletzungsstatus als Aenderungspunkt [dt, code] - er wechselt selten."""
    entry = players.setdefault(str(pid), {"dt": [], "mv": [], "st": []})
    changes = entry.setdefault("st", [])
    if changes and changes[-1][1] == status:
        return
    if changes and changes[-1][0] == dt:
        changes[-1][1] = status
        return
    changes.append([dt, status])


def market_value_at(players: dict, pid: str, dt: int) -> int | None:
    """Letzter bekannter Marktwert bis einschliesslich ``dt``."""
    entry = players.get(str(pid))
    if not entry or not entry["dt"]:
        return None
    i = bisect.bisect_right(entry["dt"], dt) - 1
    return entry["mv"][i] if i >= 0 else None


# --------------------------------------------------------------------------
# Gemeinsame Fetch-Bausteine
# --------------------------------------------------------------------------

def extract_season(payload: dict, season: dict, league: str, seed_team: str,
                   match_results: dict | None = None) -> list[dict]:
    """Spieltagspunkte einer Saison aus der /performance-Antwort ziehen.

    Sammelt nebenbei Endergebnisse fuer die Matchday-Rekonstruktion ein, auch
    aus Eintraegen ohne Einsatz - fuer abgeschlossene Saisons liefert
    /matchdays laengst den aktuellen Spielplan.
    """
    points = []
    for entry_season in payload.get("it", []) or []:
        if season["title"] not in (entry_season.get("ti") or ""):
            continue
        # exakter Vergleich: "Bundesliga" ist Teilstring von "2. Bundesliga"
        if entry_season.get("n", "") != league:
            continue

        last_pt = seed_team
        for entry in entry_season.get("ph", []) or []:
            if match_results is not None and entry.get("mdst") == 2 and entry.get("t1g") is not None:
                key = (entry.get("day"), str(entry.get("t1")), str(entry.get("t2")))
                match_results[key] = (entry.get("t1g"), entry.get("t2g"))

            # pt kann einzeln fehlen; letzten bekannten Wert fortschreiben
            pt = str(entry.get("pt") or "") or last_pt
            last_pt = pt
            t1, t2 = str(entry.get("t1", "")), str(entry.get("t2", ""))

            if pt == t1:
                opponent, home = t2, True
            elif pt == t2:
                opponent, home = t1, False
            else:
                continue

            p = entry.get("p")
            points.append({
                "day": entry.get("day"),
                "points": p if p is not None else 0,
                "opponent": opponent,
                "home": home,
                "minutes": entry.get("mp") or "0'",
            })
    return points


def fetch_performances(client: KickbaseClient, comp: dict, season: dict,
                       players: dict, workers: int,
                       match_results: dict | None = None) -> dict:
    """/performance fuer alle Spieler. Ein Call je Spieler - es gibt keinen
    Spieltags-Filter, der Endpunkt liefert immer die ganze Historie."""
    print(f"\n  Lade Punkte für {len(players)} Spieler...")
    result: dict[str, list] = {}
    lock = threading.Lock()
    total, done = len(players), 0

    def fetch_one(pid: str):
        data = client.any_competition(
            "/v4/competitions/{cid}/players/{pid}/performance",
            prefer=comp["id"], pid=pid)
        return pid, data

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, pid): pid for pid in players}
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                print(f"    [{done}/{total}] ...")
            pid, data = fut.result()
            seed = players[pid].get("team_id", "")
            with lock:
                result[pid] = extract_season(
                    data, season, comp["league"], seed, match_results) if data else []

    ohne = sum(1 for v in result.values() if not v)
    print(f"  ✓ {total - ohne} Spieler mit Spieltagsdaten ({ohne} ohne)")
    return result


def fetch_lineups(client: KickbaseClient, matchdays: list, teams: dict,
                  workers: int, force: bool = False) -> dict:
    """Aufstellungen beendeter Spiele holen und in die Matchday-Daten schreiben.

    Idempotent: Spiele mit bereits gespeicherter Aufstellung werden nicht
    erneut abgerufen.

    Rueckgabe: {pid: {"name", "teams": Counter, "pos": Counter}} - der
    tatsaechliche Saisonkader. 'teamprofile' zeigt nur den heutigen Kader und
    laesst zudem einzelne Spieler weg; wer wirklich gespielt hat, steht nur in
    den Aufstellungen.
    """
    # Abbruchbedingung nur an t1lp: waere auch die Bank Pflicht, wuerden Spiele
    # ohne Bankdaten bei jedem Lauf erneut abgerufen. Fehlende Baenke aus alten
    # Laeufen holt stattdessen 'archive --force-lineups' einmalig nach.
    all_matches = [m for day in matchdays for m in day["matches"]]
    played = [m for m in all_matches if m.get("st") == 2]
    todo = [m for m in played if force or not m.get("t1lp")]
    print(f"\n  Aufstellungen: {len(todo)} abzurufen "
          f"({len(played) - len(todo)} vorhanden, "
          f"{len(all_matches) - len(played)} noch nicht gespielt)")

    names: dict[str, str] = {}
    lock = threading.Lock()
    done = 0

    def fetch_one(match):
        return match, client.ok(f"/v4/matches/{match['mi']}/details")

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(fetch_one, m) for m in todo]
            for fut in as_completed(futures):
                match, data = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"    [{done}/{len(todo)}] ...")
                if not data:
                    continue
                for side in ("t1", "t2"):
                    # Team-Kuerzel: fuer alte Saisons die einzige Quelle
                    if data.get(side + "sy"):
                        teams.setdefault(str(match[side]), data[side + "sy"])
                    starters = data.get(side + "lp") or []
                    bench = data.get(side + "nlp") or []
                    with lock:
                        for key, group in ((side + "lp", starters), (side + "nlp", bench)):
                            if group:
                                match[key] = [{"i": str(p["i"]), "pos": p.get("pos", 0)}
                                              for p in group]
                        for p in starters + bench:
                            if p.get("n"):
                                names[str(p["i"])] = p["n"]

    # Kader aus *allen* gespeicherten Aufstellungen bilden, nicht nur aus den
    # eben geholten: sonst verliert der zweite Lauf genau die Spieler, die nur
    # ueber Aufstellungen sichtbar sind - er ruft ja nichts mehr ab.
    roster: dict[str, dict] = {}
    for m in all_matches:
        for side in ("t1", "t2"):
            team_id = str(m[side])
            for key in (side + "lp", side + "nlp"):
                for p in m.get(key) or []:
                    e = roster.setdefault(str(p["i"]), {
                        "name": names.get(str(p["i"]), ""),
                        "teams": Counter(), "pos": Counter()})
                    e["teams"][team_id] += 1
                    if p.get("pos"):
                        e["pos"][p["pos"]] += 1

    covered = sum(1 for m in all_matches if m.get("t1lp"))
    print(f"  ✓ Aufstellungen für {covered} von {len(played)} gespielten Partien, "
          f"{len(roster)} Spieler gesehen")
    return roster


def fetch_unknown_players(client: KickbaseClient, comp: dict, roster: dict,
                          unknown: list, teams: dict, workers: int) -> dict:
    """Stammdaten fuer Spieler nachladen, die nur in Aufstellungen auftauchen."""
    print(f"\n  Lade Stammdaten für {len(unknown)} zusätzliche Spieler...")

    def fetch_one(pid: str):
        return pid, client.any_competition(
            "/v4/competitions/{cid}/players/{pid}", prefer=comp["id"], pid=pid)

    found = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, pid) for pid in unknown]
        for fut in as_completed(futures):
            pid, profile = fut.result()
            info = roster[pid]
            team_id = info["teams"].most_common(1)[0][0]
            # Position: Kickbase-Stammdaten bevorzugen. Das 'pos' aus der
            # Aufstellung ist die Rolle in der Formation und weicht bei rund
            # 20 % der Starter von der Kickbase-Position ab.
            position = (profile or {}).get("pos") or (
                info["pos"].most_common(1)[0][0] if info["pos"] else 0)
            found[pid] = {
                "name": (profile or {}).get("n") or info["name"],
                "team_id": team_id,
                "team_name": teams.get(team_id, team_id),
                "position": position,
            }
    print(f"  ✓ {len(found)} Spieler ergänzt")
    return found


def build_players(players: dict, performances: dict, market_values: dict) -> list[dict]:
    """Interne Struktur in das Ausgabeformat der data_*.json bringen."""
    out = []
    for pid, info in players.items():
        # Spieler ohne Spieltagsdaten nicht verwerfen: Neuzugaenge aus dem
        # Ausland haben zu Saisonbeginn keinen Bundesliga-Eintrag, stehen aber
        # im Kader. Wer sie hier wegwirft, hat sie nirgends.
        p = {
            "id": pid,
            "name": info["name"],
            "team_id": info["team_id"],
            "team_name": info["team_name"],
            "position": info["position"],
            "performance": performances.get(pid, []),
        }
        mv = market_values.get(pid)
        if mv:
            p["market_value"] = mv
        out.append(p)
    # Ohne feste Sortierung saehe jeder Lauf fuer Git wie eine komplett neue
    # 2,4-MB-Datei aus - kein lesbarer Diff und keine Delta-Kompression,
    # obwohl sich inhaltlich fast nichts aendert.
    out.sort(key=lambda p: int(p["id"]))
    return out


def write_data(comp: dict, season: dict, teams: dict, players: list[dict],
               matchdays: list) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    data_file, md_file = files_for(comp, season)
    name = competition_name(comp, season)
    now = datetime.now().isoformat()

    with open(data_path(data_file), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "competition": name,
                   "teams": teams, "players": players}, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ {data_file}: {len(players)} Spieler "
          f"({os.path.getsize(data_path(data_file))/1024:.0f} KB)")

    with open(data_path(md_file), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "competition": name,
                   "matchdays": matchdays}, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {md_file} ({os.path.getsize(data_path(md_file))/1024:.0f} KB)")


def load_existing(comp: dict, season: dict) -> tuple[dict, dict]:
    """Teams und Kader aus der vorhandenen data-Datei. Fehlt sie, ist das kein Fehler."""
    data_file, _ = files_for(comp, season)
    path = data_path(data_file)
    if not os.path.exists(path):
        return {}, {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    players = {}
    for p in data.get("players", []):
        players[str(p["id"])] = {
            "name": p["name"],
            "team_id": p["team_id"],
            "team_name": p["team_name"],
            "position": p["position"],
            "old_market_value": p.get("market_value"),
            "old_performance": p.get("performance") or [],
        }
    return data.get("teams", {}), players


def load_matchdays(comp: dict, season: dict) -> list:
    """Nur den Spielplan lesen - die data-Datei ist dafuer zu gross."""
    _, md_file = files_for(comp, season)
    path = data_path(md_file)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("matchdays", [])


def load_players(comp: dict, season: dict) -> tuple[dict, list]:
    """Teams und rohe Spielerliste einer Saison."""
    data_file, _ = files_for(comp, season)
    path = data_path(data_file)
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("teams", {}), data.get("players", [])


# --------------------------------------------------------------------------
# data/carryover.json - die Vorsaison, auf die Teams der laufenden gemappt
# --------------------------------------------------------------------------

CARRYOVER_FILE = "carryover.json"
POS_ORDER = [1, 2, 3, 4]

# Faktoren fuer den Ligawechsel, gelernt aus kbxp/data/processed/panel.parquet
# (Uebergaenge 2021/22 -> 2025/26, nur md_status == 2):
#
#   Wechsel            n    zugelassen   eigene
#   BL2 -> BL1 (up)     8      x1.59      x0.65
#   BL1 -> BL2 (down)  10      x0.69      x1.50
#   Liga gehalten      243     x1.00      x0.97
#   BL2-Neuling (new)  12    1.13 x Ligamittel / 0.84 x Ligamittel
#
# Nachrechnen: panel[md_status == 2] ->
#   groupby(season, league, opponent_id).points.sum() / matchday.nunique()
#     = zugelassene Punkte je Spieltag, analog ueber team_id = eigene Punkte.
#   Je Team aufeinanderfolgende Saisons paaren, Median von wert_neu / wert_alt
#   je Wechselrichtung. "new" = Median von wert / Ligamittel fuer Teams, die in
#   der Vorsaison in keiner der beiden Ligen auftauchen - 2021/22 dabei
#   ausschliessen, dort beginnt die BL2-Abdeckung und alle Teams waeren "neu".
#
# Die Faktoren stehen auch in der Datei, damit die Seiten sie anzeigen koennen
# und ein Nachjustieren keinen neuen Lauf braucht.
CARRYOVER_FACTORS = {
    "same": {"conceded": 1.00, "scored": 1.00},
    "up":   {"conceded": 1.59, "scored": 0.65},
    "down": {"conceded": 0.69, "scored": 1.50},
    "new":  {"conceded": 1.13, "scored": 0.84},
}


def previous_season(manifest: dict) -> dict | None:
    older = [s for s in manifest["seasons"] if s["key"] != manifest["current"]]
    return max(older, key=lambda s: s["key"]) if older else None


# Vier Positionsgruppen plus eine fuenfte Ablage fuer Spieler, fuer die Kickbase
# ueberhaupt keine Position fuehrt. Ohne sie waere die Vorsaison-Summe um deren
# Punkte zu niedrig - dieselbe Luecke, die in den Archivdateien behoben wurde.
UNPOS_IDX = len(POS_ORDER)


def _zero() -> list:
    return [0.0] * (len(POS_ORDER) + 1)


def season_profile(comp: dict, season: dict) -> dict:
    """Punkte je Team, Spieltag und Position einer abgeschlossenen Saison.

    Liefert pro Team eine Liste von Spieltagen mit vier Kennzahlen je Position:
    zugelassene Punkte, eigene Punkte, davon aus der Startelf, und die Zahl der
    besetzten Startelf-Plaetze. Damit koennen die Seiten dieselben Metriken auf
    die Vorsaison anwenden wie auf die laufende - ohne 2,6 MB Archiv zu laden.

    Jede Liste hat fuenf Eintraege: die vier Positionsgruppen und an letzter
    Stelle die Spieler ohne Position. Die Seiten zeigen nur die ersten vier als
    Balken, summieren aber ueber alle fuenf - sonst faellt die Vorsaison-Summe
    genau um die Punkte zu niedrig aus, die im Archiv gerade erst ergaenzt
    wurden.
    """
    teams, players = load_players(comp, season)
    matchdays = load_matchdays(comp, season)
    if not teams or not players:
        return {"teams": {}, "abbr": {}, "max_day": 0}

    # Gegner + Heim/Auswaerts bestimmen eindeutig, fuer wen ein Spieler auflief -
    # player.team_id ist nur der Kader-Snapshot und liegt bei Wintertransfers
    # innerhalb der Liga fuer die Hinrunde daneben.
    match_team, starter_team, home_of, played, lineup_days = {}, {}, {}, set(), set()
    max_day = 0
    for md in matchdays:
        for m in md["matches"]:
            day, t1, t2 = m["day"], str(m["t1"]), str(m["t2"])
            max_day = max(max_day, day)
            match_team[(day, t2, True)] = t1
            match_team[(day, t1, False)] = t2
            home_of[(t1, day)] = True
            home_of[(t2, day)] = False
            if m.get("st") != 2:
                continue
            played.add((t1, day))
            played.add((t2, day))
            for side, tid in (("t1lp", t1), ("t2lp", t2)):
                for x in m.get(side) or []:
                    starter_team[(str(x["i"]), day)] = tid
                    lineup_days.add((tid, day))

    # Rueckfall fuer Spiele ohne Aufstellung: die 11 mit den meisten Minuten.
    # Im Archiv sind 306/306 Aufstellungen da, aber die Metrik darf nicht
    # stillschweigend auf 0 Startplaetze fallen, falls doch eine fehlt.
    roster: dict = {}
    for p in players:
        for e in p.get("performance") or []:
            minutes = int(str(e.get("minutes") or "0").rstrip("'") or 0)
            if minutes <= 0:
                continue
            day = e["day"]
            team = match_team.get((day, str(e["opponent"]), bool(e.get("home")))) or p["team_id"]
            if (team, day) in lineup_days:
                continue
            roster.setdefault((team, day), []).append((minutes, str(p["id"])))
    for (team, day), rows in roster.items():
        for _, pid in sorted(rows, reverse=True)[:11]:
            starter_team[(pid, day)] = team

    out = {tid: {} for tid in teams}
    counted = set()
    for p in players:
        pos = p.get("position")
        idx = POS_ORDER.index(pos) if pos in POS_ORDER else UNPOS_IDX
        pid = str(p["id"])
        for e in p.get("performance") or []:
            day, opp = e["day"], str(e["opponent"])
            team = match_team.get((day, opp, bool(e.get("home")))) or p["team_id"]
            if team not in out or (team, day) not in played:
                continue
            pts = e.get("points") or 0

            slot = out[team].setdefault(day, {"conceded": _zero(), "scored": _zero(),
                                              "start": _zero(), "slots": _zero()})
            slot["scored"][idx] += pts
            key = (pid, day)
            if starter_team.get(key) == team and key not in counted:
                counted.add(key)
                slot["start"][idx] += pts
                slot["slots"][idx] += 1

            if opp in out:
                out[opp].setdefault(day, {"conceded": _zero(), "scored": _zero(),
                                          "start": _zero(), "slots": _zero()})
                out[opp][day]["conceded"][idx] += pts

    days_by_team = {}
    for tid, by_day in out.items():
        days_by_team[tid] = [
            {"day": day, "home": bool(home_of.get((tid, day), True)),
             "conceded": [round(v, 1) for v in d["conceded"]],
             "scored":   [round(v, 1) for v in d["scored"]],
             "start":    [round(v, 1) for v in d["start"]],
             "slots":    [int(v) for v in d["slots"]]}
            for day, d in sorted(by_day.items())
        ]
    return {"teams": {t: d for t, d in days_by_team.items() if d},
            "abbr": teams, "max_day": max_day}


def synth_days(profile: dict) -> list:
    """Spieltage fuer ein Team ohne Vorsaison-Datensatz (Aufsteiger aus Liga 3).

    Grundlage ist das Ligamittel der Vorsaison, getrennt nach Heim- und
    Auswaertsspielen, damit Spieltags- und Venue-Filter weiter greifen. Wie stark
    ein solcher Neuling davon abweicht, steckt im Faktor "new".
    """
    sums = {True: {}, False: {}}
    counts = {True: 0, False: 0}
    for days in profile["teams"].values():
        for d in days:
            side = sums[d["home"]]
            counts[d["home"]] += 1
            for field in ("conceded", "scored", "start", "slots"):
                acc = side.setdefault(field, _zero())
                for i, v in enumerate(d[field]):
                    acc[i] += v
    if not counts[True] or not counts[False]:
        return []

    mean = {h: {f: [v / counts[h] for v in vals] for f, vals in sums[h].items()}
            for h in (True, False)}
    out = []
    for day in range(1, profile["max_day"] + 1):
        home = day % 2 == 1
        m = mean[home]
        out.append({"day": day, "home": home,
                    "conceded": [round(v, 1) for v in m["conceded"]],
                    "scored":   [round(v, 1) for v in m["scored"]],
                    "start":    [round(v, 1) for v in m["start"]],
                    "slots":    [round(v) for v in m["slots"]]})
    return out


def build_carryover(manifest: dict) -> None:
    """data/carryover.json: die Vorsaison, gemappt auf die Teams der laufenden.

    Rein lokal - liest nur die schon vorhandenen data-/matchdays-Dateien.
    """
    cur = current_season(manifest)
    prev = previous_season(manifest)
    if prev is None:
        print(f"  ⚠ keine Vorsaison im Manifest — {CARRYOVER_FILE} übersprungen")
        return

    profiles = {comp["id"]: season_profile(comp, prev) for comp in COMPETITIONS}
    leagues = {}
    for comp in COMPETITIONS:
        cid = comp["id"]
        other = "2" if cid == "1" else "1"
        teams, _ = load_players(comp, cur)
        mine, theirs = profiles[cid], profiles[other]
        if not teams or not mine["teams"]:
            continue
        synth = synth_days(mine)

        out = {}
        for tid in sorted(teams, key=int):
            if tid in mine["teams"]:
                src, days, src_league = "same", mine["teams"][tid], cid
            elif tid in theirs["teams"]:
                # Liga 1 zieht Aufsteiger aus Liga 2 und umgekehrt
                src, days, src_league = ("up" if cid == "1" else "down",
                                         theirs["teams"][tid], other)
            else:
                src, days, src_league = "new", synth, None
            out[tid] = {
                "src": src,
                "src_league": src_league,
                "src_abbr": (mine if src_league == cid else theirs)["abbr"].get(tid)
                            if src_league else None,
                "days": days,
            }
        leagues[cid] = {"prev_max_day": mine["max_day"], "teams": out}

    save_carryover({
        "generated_at": datetime.now().isoformat(),
        "current": cur["key"], "prev": prev["key"],
        "pos_order": POS_ORDER,
        "factors": CARRYOVER_FACTORS,
        "leagues": leagues,
    })


def build_ratings() -> None:
    """data/ratings.json neu bauen - die gegnerbereinigte Teamstaerke.

    Rechnet nur mit den Dateien, die gerade geschrieben wurden, und braucht
    keinen Login. Der Import ist bewusst weich: der Fetcher selbst kommt mit
    requests aus, das Modell braucht numpy, pandas und scipy. Fehlen sie, faellt
    nur diese Datei aus - die Seiten rechnen dann wie frueher mit dem rohen
    Mittelwert weiter und weisen darauf hin.
    """
    kbxp = os.path.join(SCRIPT_DIR, "kbxp")
    if not os.path.isdir(kbxp):
        return
    if kbxp not in sys.path:
        sys.path.insert(0, kbxp)
    try:
        from src.model.team_strength import export
    except ImportError as e:
        print(f"  ratings.json uebersprungen: {e}")
        print("  (pip install -r kbxp/requirements.txt, dann 'python fetch.py ratings')")
        return

    payload = export()
    for liga, block in payload["leagues"].items():
        gespielt = max((t.get("n", 0) for t in block["teams"].values()), default=0)
        print(f"  ratings.json Liga {liga}: {len(block['teams'])} Teams, "
              f"Basis Vorsaison + {gespielt} ST")


def save_carryover(payload: dict) -> None:
    """Ein Team pro Zeile - wie bei history.json, aus demselben Diff-Grund."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(data_path(CARRYOVER_FILE), "w", encoding="utf-8") as f:
        f.write("{\n")
        for key in ("generated_at", "current", "prev", "pos_order", "factors"):
            f.write(f'  "{key}": {json.dumps(payload[key], separators=(",", ":"))},\n')
        f.write('  "leagues": {\n')
        ligas = list(payload["leagues"].items())
        for li, (cid, league) in enumerate(ligas):
            f.write(f'    "{cid}": {{\n')
            f.write(f'      "prev_max_day": {league["prev_max_day"]},\n')
            f.write('      "teams": {\n')
            rows = list(league["teams"].items())
            for i, (tid, entry) in enumerate(rows):
                comma = "," if i < len(rows) - 1 else ""
                f.write(f'        "{tid}": {json.dumps(entry, separators=(",", ":"))}{comma}\n')
            f.write("      }\n")
            f.write("    }" + ("," if li < len(ligas) - 1 else "") + "\n")
        f.write("  }\n}\n")

    size = os.path.getsize(data_path(CARRYOVER_FILE)) / 1024
    counts = {cid: len(l["teams"]) for cid, l in payload["leagues"].items()}
    print(f"  ✓ {CARRYOVER_FILE}: " +
          ", ".join(f"Liga {c} {n} Teams" for c, n in counts.items()) +
          f" ({size:.0f} KB)")


# --------------------------------------------------------------------------
# Modus: live
# --------------------------------------------------------------------------

def fetch_matchdays(client: KickbaseClient, comp: dict) -> tuple[dict, list]:
    """Spielplan und Teams der laufenden Saison.

    /matchdays hat keinen Saison-Parameter und liefert immer die laufende
    Saison - fuer abgeschlossene Saisons ist der Endpunkt nutzlos.
    """
    print("\n  Lade Spielplan & Teams...")
    data = client.ok(f"/v4/competitions/{comp['id']}/matchdays")
    if not data:
        return {}, []

    teams, matchdays = {}, []
    for day in data.get("it", []):
        matches = []
        for m in day.get("it", []):
            teams[str(m["t1"])] = m["t1sy"]
            teams[str(m["t2"])] = m["t2sy"]
            matches.append({
                "mi": str(m["mi"]), "day": m["day"], "dt": m["dt"],
                "t1": str(m["t1"]), "t2": str(m["t2"]),
                "t1sy": m["t1sy"], "t2sy": m["t2sy"],
                # API liefert fuer ungespielte Partien null statt 0
                "t1g": m.get("t1g") or 0, "t2g": m.get("t2g") or 0,
                "st": m.get("st", 0),
            })
        matchdays.append({"day": day["day"], "matches": matches})

    print(f"  ✓ {len(teams)} Teams, {len(matchdays)} Spieltage")
    return teams, matchdays


def fetch_squads(client: KickbaseClient, comp: dict, teams: dict, history: dict,
                 today: int) -> tuple[dict, dict]:
    """Kader je Team - inklusive Marktwert und Status.

    'teamprofile' liefert mv und mvt bereits mit; die Einzelabfrage
    /players/{id} je Spieler waere fuer beides ein zweiter, ueberfluessiger
    Durchlauf ueber den gesamten Kader.
    """
    print(f"\n  Lade Kader & Marktwerte für {len(teams)} Teams...")
    players, market_values = {}, {}

    for team_id, team_name in teams.items():
        data = client.ok(
            f"/v4/competitions/{comp['id']}/teams/{team_id}/teamprofile")
        if not data:
            print(f"    ⚠ Keine Daten für {team_name}")
            continue

        for p in data.get("it", []):
            pid = str(p.get("i", ""))
            if not pid:
                continue
            players[pid] = {
                "name": p.get("n", "Unbekannt"),
                "team_id": team_id,
                "team_name": team_name,
                "position": p.get("pos", 0),
            }
            if p.get("mv") is not None:
                market_values[pid] = {"current": p["mv"], "trend": p.get("mvt", 0)}
                history_add_value(history, pid, today, p["mv"])
            if p.get("st") is not None:
                history_add_status(history, pid, today, p["st"])

        print(f"    {team_name}: {len(data.get('it', []))} Spieler")

    print(f"  ✓ {len(players)} Spieler, {len(market_values)} Marktwerte")
    return players, market_values


def cmd_live(client: KickbaseClient, manifest: dict, workers: int) -> None:
    season = current_season(manifest)
    history = load_history()
    today = day_number(date.today())

    for comp in COMPETITIONS:
        print(f"\n{'='*56}\n  {competition_name(comp, season)} (Competition {comp['id']})\n{'='*56}")

        teams, matchdays = fetch_matchdays(client, comp)
        if not teams:
            print("  ⚠ Spielplan nicht abrufbar, überspringe Competition")
            continue

        players, market_values = fetch_squads(client, comp, teams, history, today)

        # Aufstellungen vor den Punkten: sie decken Spieler auf, die das
        # teamprofile nicht nennt - die muessen mit in den Punkte-Durchlauf.
        carry_lineups(load_matchdays(comp, season), matchdays)
        roster = fetch_lineups(client, matchdays, teams, workers)

        # Stammdaten nur fuer wirklich neue Spieler holen; wer schon in der
        # Datei steht, ist damit auch in Woche zwei noch da.
        _, previous = load_existing(comp, season)
        for pid in roster:
            if pid not in players and pid in previous:
                players[pid] = previous[pid]
        unknown = [pid for pid in roster if pid not in players]
        if unknown:
            players.update(
                fetch_unknown_players(client, comp, roster, unknown, teams, workers))

        performances = fetch_performances(client, comp, season, players, workers)
        write_data(comp, season, teams,
                   build_players(players, performances, market_values), matchdays)

    print(f"\n{'='*56}")
    save_history(history)
    save_manifest(manifest)
    build_carryover(manifest)
    build_ratings()
    print("  Fertig.")


def carry_lineups(old_matchdays: list, matchdays: list) -> None:
    """Bereits gespeicherte Aufstellungen in den frischen Spielplan uebernehmen.

    /matchdays kennt keine Aufstellungen; ohne diesen Schritt wuerde jeder Lauf
    saemtliche Spiele der Saison erneut abrufen.
    """
    seen = {m["mi"]: m for day in old_matchdays for m in day["matches"]}
    for day in matchdays:
        for m in day["matches"]:
            old = seen.get(m["mi"])
            if not old:
                continue
            for key in ("t1lp", "t2lp", "t1nlp", "t2nlp"):
                if old.get(key):
                    m[key] = old[key]


# --------------------------------------------------------------------------
# Modus: mv
# --------------------------------------------------------------------------

def player_competitions(manifest: dict) -> dict[str, dict[str, set]]:
    """{spieler_id: {saison_key: {competition_id, ...}}} aus den data-Dateien.

    Ein Wintertransfer ueber die Ligengrenze bringt einen Spieler in *beide*
    Dateien derselben Saison (25/26: 15 Faelle). Deshalb eine Menge und keine
    einzelne Liga - welche gilt, entscheidet spaeter die Datenlage.
    """
    membership: dict[str, dict[str, set]] = {}
    for season in manifest["seasons"]:
        for comp in COMPETITIONS:
            data_file, _ = files_for(comp, season)
            path = data_path(data_file)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for p in json.load(f).get("players", []):
                    seasons = membership.setdefault(str(p["id"]), {})
                    seasons.setdefault(season["key"], set()).add(comp["id"])
    return membership


def resolve_leagues(membership: dict, windows: list, series: dict) -> dict:
    """Je Saison die Liga waehlen, deren Marktwertreihe echte Daten enthaelt.

    Steht ein Spieler in beiden Ligen einer Saison, hilft die Mitgliedschaft
    allein nicht weiter. Messbar ist es trotzdem: die fuehrende Competition
    liefert im Saisonfenster eine taeglich wandernde Kurve, die andere einen
    eingefrorenen Wert. Also die mit den meisten verschiedenen Werten nehmen.
    """
    spans = {key: (a, b) for a, b, key in windows}
    choice = {}
    for key, cids in membership.items():
        if len(cids) == 1:
            choice[key] = next(iter(cids))
            continue
        start, end = spans.get(key, (0, 10**9))
        choice[key] = max(sorted(cids), key=lambda cid: len(
            {e["mv"] for e in series.get(cid, []) if start <= e.get("dt", -1) <= end}))
    return choice


def season_windows(manifest: dict) -> list[tuple[int, int, str]]:
    """(erster Spieltag, letzter Spieltag, saison_key), aufsteigend."""
    windows = []
    for season in manifest["seasons"]:
        days = []
        for comp in COMPETITIONS:
            for day in load_matchdays(comp, season):
                for m in day["matches"]:
                    dt = (m.get("dt") or "")[:10]
                    if len(dt) == 10:
                        days.append(day_number(date.fromisoformat(dt)))
        if days:
            windows.append((min(days), max(days), season["key"]))
    windows.sort()
    return windows


def competition_at(windows: list, membership: dict, day: int) -> str | None:
    """In welcher Liga stand der Spieler an diesem Tag?

    Innerhalb einer Saison ist das die Liga dieser Saison. In der Sommerpause
    zaehlt die *kommende* Saison: Kickbase fuehrt einen Wechsler dort bereits
    mit echten Werten, waehrend die alte Competition nur noch den letzten Wert
    einfriert.
    """
    for start, end, key in windows:
        if start <= day <= end and key in membership:
            return membership[key]
    for start, _, key in windows:          # aufsteigend: die naechste Saison
        if day < start and key in membership:
            return membership[key]
    for _, _, key in reversed(windows):    # nach der letzten bekannten Saison
        if key in membership:
            return membership[key]
    return None


def cmd_mv(client: KickbaseClient, manifest: dict, workers: int) -> None:
    """Marktwert-Historie auf Tagesaufloesung verdichten.

    /marketValue beantwortet nur ein Fenster von 365 Tagen - groessere Werte
    liefern eine leere Liste. Solange dieser Lauf oefter als jaehrlich
    passiert, entsteht in der Historie nie eine Luecke.

    Der Endpunkt ist competition-spezifisch, und zwar auf eine Art, die still
    falsche Zahlen liefert: jede Competition fuehrt den Marktwert nur, solange
    der Spieler in ihr steht, und fuellt den Rest des Fensters mit einem
    flachen Platzhalter (verifiziert: Seguin hat in Competition 1 an allen 365
    Tagen exakt 500.000, in Competition 2 eine echte Kurve). Deshalb wird fuer
    jeden Tag die Liga genommen, in der der Spieler damals wirklich stand.
    """
    history = load_history()
    membership = player_competitions(manifest)
    windows = season_windows(manifest)
    print("\n  Saisonfenster: " + ", ".join(
        f"{key} {from_day_number(a)}..{from_day_number(b)}" for a, b, key in windows))

    def leagues_of(pid: str) -> set:
        return {cid for cids in membership[pid].values() for cid in cids}

    both = [pid for pid in membership if len(leagues_of(pid)) > 1]
    print(f"  Lade Marktwert-Historie für {len(membership)} Spieler "
          f"({len(both)} davon in beiden Ligen, also zweimal)...")

    lock = threading.Lock()
    added = skipped = missing = done = 0

    def fetch_one(pid: str):
        series = {}
        for cid in sorted(leagues_of(pid)):
            data = client.ok(f"/v4/competitions/{cid}/players/{pid}/marketValue/365")
            series[cid] = (data or {}).get("it") or []
        return pid, series

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, pid) for pid in membership]
        for fut in as_completed(futures):
            pid, series = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"    [{done}/{len(membership)}] ...")
            if not any(series.values()):
                missing += 1
                continue
            leagues = resolve_leagues(membership[pid], windows, series)
            with lock:
                for cid, points in series.items():
                    for e in points:
                        dt, mv = e.get("dt"), e.get("mv")
                        if dt is None or mv is None:
                            continue
                        if competition_at(windows, leagues, int(dt)) != cid:
                            skipped += 1
                            continue
                        if history_add_value(history, pid, dt, mv):
                            added += 1

    print(f"  ✓ {added} Tageswerte ergänzt, {skipped} aus der falschen Liga verworfen "
          f"({missing} Spieler ohne Historie)")
    save_history(history)


# --------------------------------------------------------------------------
# Modus: archive
# --------------------------------------------------------------------------

def matchdays_from_parquet(season: dict, comp: dict) -> list:
    """Spielplan einer abgeschlossenen Saison aus kbxp/data/processed/matches.parquet.

    Die Datei entstand aus dem mi-Feld von /performance und deckt alle Saisons
    ab, fuer die es Spielerhistorien gibt. Damit laesst sich eine Saison von
    null rekonstruieren, auch ohne vorhandene data_*.json.
    """
    path = os.path.join(SCRIPT_DIR, "kbxp", "data", "processed", "matches.parquet")
    if not os.path.exists(path):
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  ⚠ pandas fehlt - nutze vorhandene Matchday-Datei")
        return []

    df = pd.read_parquet(path)
    df = df[(df["season"] == season["title"]) & (df["league"] == comp["league"])]
    if df.empty:
        return []

    matchdays = []
    for day, group in df.sort_values(["matchday", "match_id"]).groupby("matchday", sort=True):
        matches = []
        for _, r in group.iterrows():
            # Tore sind NaN, nicht None, wenn ein Spiel nicht gewertet ist
            played = r["md_status"] == 2 and pd.notna(r["home_goals"])
            matches.append({
                "mi": str(r["match_id"]), "day": int(day),
                "dt": r["match_date"].isoformat() if pd.notna(r["match_date"]) else "",
                "t1": str(r["home_team"]), "t2": str(r["away_team"]),
                "t1sy": "", "t2sy": "",   # Kuerzel liefern erst die Match-Details
                "t1g": int(r["home_goals"]) if pd.notna(r["home_goals"]) else 0,
                "t2g": int(r["away_goals"]) if pd.notna(r["away_goals"]) else 0,
                "st": 2 if played else 0,
            })
        matchdays.append({"day": int(day), "matches": matches})

    print(f"  ✓ Spielplan aus matches.parquet: {len(matchdays)} Spieltage, "
          f"{sum(len(d['matches']) for d in matchdays)} Spiele")
    return matchdays


def complete_matchdays(matchdays: list, match_results: dict, teams: dict) -> None:
    """Offene Spiele mit Ergebnissen aus den Performance-Daten fuellen."""
    updated, missing = 0, []
    for day in matchdays:
        for m in day["matches"]:
            # Kuerzel nachtragen, falls der Spielplan aus dem Parquet kam
            m["t1sy"] = m.get("t1sy") or teams.get(m["t1"], "")
            m["t2sy"] = m.get("t2sy") or teams.get(m["t2"], "")
            if m.get("st") == 2:
                continue
            res = match_results.get((m["day"], m["t1"], m["t2"]))
            if res:
                m["t1g"], m["t2g"] = res
                m["st"] = 2
                updated += 1
            else:
                missing.append(f"Tag {m['day']}: {m['t1sy'] or m['t1']}-{m['t2sy'] or m['t2']}")

    if updated:
        print(f"  ✓ {updated} offene Spiele mit Endergebnissen vervollständigt")
    if missing:
        print(f"  ⚠ {len(missing)} Spiele ohne Ergebnis:")
        for match in missing[:10]:
            print(f"      - {match}")


def season_end_daynum(matchdays: list) -> int:
    """Letzter Spieltermin der Saison - Stichtag fuer den Marktwert."""
    best = 0
    for day in matchdays:
        for m in day["matches"]:
            dt = (m.get("dt") or "")[:10]
            if len(dt) == 10:
                try:
                    best = max(best, day_number(date.fromisoformat(dt)))
                except ValueError:
                    pass
    return best or day_number(date.today())


def cmd_archive(client: KickbaseClient, manifest: dict, season_key: str,
                workers: int, force_lineups: bool = False) -> None:
    season = season_by_key(manifest, season_key)
    if season["key"] == manifest["current"]:
        print(f"✗ {season['title']} ist die laufende Saison — dafür 'python fetch.py' nutzen.")
        sys.exit(1)

    history = load_history()

    for comp in COMPETITIONS:
        print(f"\n{'='*56}\n  {competition_name(comp, season)} (Competition {comp['id']})\n{'='*56}")

        teams, players = load_existing(comp, season)
        matchdays = load_matchdays(comp, season)
        print(f"  Vorhanden: {len(players)} Spieler, {len(matchdays)} Spieltage")

        if not matchdays:
            matchdays = matchdays_from_parquet(season, comp)
        if not matchdays:
            print("  ✗ Kein Spielplan - weder data/-Datei noch matches.parquet.")
            continue

        # teamprofile und teamcenter bleiben hier aussen vor: beide liefern nur
        # den heutigen Stand. Der Saisonkader kommt ausschliesslich aus den
        # Aufstellungen - nur so tauchen Winterabgaenge ueberhaupt auf.
        roster = fetch_lineups(client, matchdays, teams, workers, force_lineups)

        unknown = [pid for pid in roster if pid not in players]
        if unknown:
            players.update(
                fetch_unknown_players(client, comp, roster, unknown, teams, workers))

        match_results: dict = {}
        performances = fetch_performances(
            client, comp, season, players, workers, match_results)
        complete_matchdays(matchdays, match_results, teams)

        # Kennt die API einen Spieler nicht mehr, gilt der Archivstand. Ohne
        # diesen Rueckfall wuerde ein einzelner Ausfall Spieler aus einer
        # abgeschlossenen Saison loeschen.
        restored = 0
        for pid, info in players.items():
            if not performances.get(pid) and info.get("old_performance"):
                performances[pid] = info["old_performance"]
                restored += 1
        if restored:
            print(f"  ⚠ {restored} Spieler nicht auflösbar, Archivdaten behalten")

        stichtag = season_end_daynum(matchdays)
        print(f"  Marktwert-Stichtag: {from_day_number(stichtag)}")

        market_values, from_history, kept, none_found = {}, 0, 0, 0
        for pid, info in players.items():
            mv = market_value_at(history, pid, stichtag)
            old = info.get("old_market_value") or {}
            if mv is not None:
                market_values[pid] = {"current": mv, "trend": old.get("trend", 0)}
                from_history += 1
            elif old.get("current") is not None:
                # Nie einen vorhandenen Wert mit None ueberschreiben: ausserhalb
                # des 365-Tage-Fensters ist er nicht wiederbeschaffbar.
                market_values[pid] = {"current": old["current"], "trend": old.get("trend", 0)}
                kept += 1
            else:
                none_found += 1

        print(f"  ✓ Marktwerte: {from_history} aus {HISTORY_FILE}, "
              f"{kept} aus der Datei behalten, {none_found} ohne Wert")

        # Nur wer nie gespielt hat, faellt raus. Eine fehlende Position ist kein
        # Grund: Kickbase fuehrt fuer einzelne Rand-Spieler ueberhaupt keine
        # ('iposl': false, 'opl': []) - weder im Profil noch in Aufstellung oder
        # Bank, auch nicht in spaeteren Saisons. Wer sie verwirft, macht die
        # Teamsumme falsch, und genau das war sie: acht Team-Spiele der Saison
        # 25/26 lagen um bis zu 34 Punkte zu niedrig.
        #
        # Die Seiten mit Positionszerlegung zaehlen sie in die Teamsumme, aber in
        # keinen Balkenabschnitt - die Abschnitte summieren dadurch knapp
        # darunter. Das ist ehrlicher als eine erfundene Position.
        out = [p for p in build_players(players, performances, market_values)
               if p["performance"]]
        ohne_pos = sum(1 for p in out if p["position"] not in (1, 2, 3, 4))
        dropped = len(players) - len(out)
        if dropped:
            print(f"  ⚠ {dropped} Spieler ohne Spieltagsdaten verworfen")
        if ohne_pos:
            print(f"  ⚠ {ohne_pos} Spieler ohne Position behalten (zaehlen in die Teamsumme)")

        write_data(comp, season, teams, out, matchdays)

    print(f"\n{'='*56}")
    save_manifest(manifest)
    print("  Fertig.")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kickbase-Fetcher. Ohne Argumente: laufende Saison.")
    ap.add_argument("--workers", type=int, default=4, help="parallele Requests (Default 4)")
    ap.add_argument("--delay", type=float, default=0.3, help="Pause je Thread in s (Default 0.3)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("live", help="laufende Saison holen (Default)")
    sub.add_parser("mv", help="Marktwert-Historie auf Tagesaufloesung verdichten")
    sub.add_parser("carryover", help="data/carryover.json neu bauen (lokal, ohne Login)")
    sub.add_parser("ratings", help="data/ratings.json neu bauen (lokal, ohne Login)")
    archive = sub.add_parser("archive", help="abgeschlossene Saison rekonstruieren")
    archive.add_argument("--season", required=True, help="Saison-Key aus data/seasons.json, z. B. 2526")
    archive.add_argument("--force-lineups", action="store_true",
                         help="alle Aufstellungen neu holen (z. B. um Baenke nachzutragen)")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.cmd == "archive":
        # Saison pruefen, bevor ein Login-Request rausgeht
        season_by_key(manifest, args.season)

    # Rechnen nur mit vorhandenen Dateien - kein Login noetig
    if args.cmd == "carryover":
        build_carryover(manifest)
        return
    if args.cmd == "ratings":
        build_ratings()
        return

    client = KickbaseClient(delay=args.delay)
    client.login()

    started = time.time()
    if args.cmd == "archive":
        cmd_archive(client, manifest, args.season, args.workers, args.force_lineups)
    elif args.cmd == "mv":
        cmd_mv(client, manifest, args.workers)
    else:
        cmd_live(client, manifest, args.workers)
    print(f"  Laufzeit: {(time.time() - started)/60:.1f} min")


if __name__ == "__main__":
    main()
