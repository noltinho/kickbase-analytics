"""Historisches Spieler-Spieltag-Panel aus /performance aufbauen.

Der Endpunkt ist ungewoehnlich ergiebig: **ein Call pro Spieler** liefert
*alle* Saisons zurueck bis ~2013/14, und zwar mit

* Punkten ``p`` und Minuten ``mp`` je Spieltag,
* ``pt`` = Team des Spielers **je Spieltag** (loest Winterwechsel auf, die in
  den bestehenden data_*.json nur als Endstand stehen),
* ``t1``/``t2``/``t1g``/``t2g`` = komplettes Spielergebnis 
* ``k`` = Event-Codes (nur vorhanden, wenn es Ereignisse gab).

Im Probe-Lauf verifiziert: ``/performance`` liefert fuer **beide** Competitions
dieselbe abgeschlossene Historie und ergaenzt lediglich die *laufende* Saison
bei der Competition, in der der Spieler gerade spielt. Deshalb reicht ein Call
gegen Competition 1, plus ein Nachschlag gegen Competition 2 fuer die Spieler,
die aktuell in der 2. Liga stehen.

Aufruf (aus kbxp/):
    python -m src.ingest.backfill_history --workers 3
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..paths import INTERIM, PROCESSED, RAW, ensure_dirs
from .kickbase_client import NOT_FOUND, KickbaseClient, enable_utf8_stdout

PLAYER_INDEX = RAW / "player_index.parquet"
DONE_LOG = INTERIM / "backfill_done.jsonl"
PANEL = PROCESSED / "panel.parquet"
MATCHES = PROCESSED / "matches.parquet"

VALID_LEAGUES = {"Bundesliga", "2. Bundesliga"}

_lock = threading.Lock()


def parse_minutes(value: object) -> int:
    """``"87'"`` -> 87. Kann >90 sein (Nachspielzeit)."""
    if value is None:
        return 0
    text = str(value).strip().rstrip("'")
    try:
        return int(text)
    except ValueError:
        return 0


def parse_player(pid: int, payload: dict) -> tuple[list[dict], list[dict]]:
    """Eine /performance-Antwort in Panel- und Spielzeilen zerlegen."""
    rows: list[dict] = []
    matches: list[dict] = []

    for season in payload.get("it", []) or []:
        league = season.get("n")
        # "Bundesliga" ist Teilstring von "2. Bundesliga" - exakt vergleichen.
        if league not in VALID_LEAGUES:
            continue
        season_title = season.get("ti")
        season_id = season.get("sid")

        # pt roh uebernehmen, ohne Fortschreibung: die Wahrheit entsteht
        # zentral in derive_teams(), gegen die Paarung des Spiels geprueft.
        # Die frueher hier praktizierte last_team-Fortschreibung hat falsche
        # Werte ueber Folgezeilen geschmiert - pt ist je Spieltag gemeint
        # und wechselt bei Transfers nachweislich korrekt mit.
        for entry in season.get("ph", []) or []:
            team = str(entry.get("pt") or "") or None
            t1 = str(entry.get("t1") or "")
            t2 = str(entry.get("t2") or "")
            if not (t1 and t2):
                continue

            if team == t1:
                opponent, home = t2, True
            elif team == t2:
                opponent, home = t1, False
            else:
                opponent, home = None, None

            # p fehlt komplett (nicht null), wenn der Spieler nicht gespielt hat.
            points = entry.get("p")
            rows.append({
                "player_id": pid,
                "season": season_title,
                "season_id": str(season_id) if season_id is not None else None,
                "league": league,
                "matchday": entry.get("day"),
                "match_id": str(entry.get("mi") or ""),
                "match_date": entry.get("md"),
                "team_id": team,
                "opponent_id": opponent,
                "home": home,
                "minutes": parse_minutes(entry.get("mp")),
                "points": points,                       # None = kein Einsatz
                "events": entry.get("k") or [],
                "md_status": entry.get("mdst"),
                "player_md_status": entry.get("st"),
                "season_avg_points": entry.get("ap"),
                "season_total_points": entry.get("tp"),
            })

            matches.append({
                "season": season_title,
                "league": league,
                "matchday": entry.get("day"),
                "match_id": str(entry.get("mi") or ""),
                "match_date": entry.get("md"),
                "home_team": t1,
                "away_team": t2,
                "home_goals": entry.get("t1g"),
                "away_goals": entry.get("t2g"),
                "md_status": entry.get("mdst"),
            })

    return rows, matches


def load_done() -> set[int]:
    """Spieler, die wirklich fertig sind - Protokoll UND Daten.

    Das Protokoll wird beim Abruf geschrieben, die Zeilen aber erst beim naechsten
    flush() (alle 200.000 Zeilen). Bricht der Lauf dazwischen ab, gilt ein Spieler
    als erledigt, obwohl seine Zeilen nie in einer Teildatei gelandet sind - und
    ein Wiederaufsetzen holt sie nie nach. Genau so sind 15 Spieler mit zusammen
    1.370 Zeilen verlorengegangen, darunter 4 Blaswich (B04) und 8 Kramer (BMG).

    Wer im Protokoll steht, aber Zeilen haben muesste und in keiner Teildatei
    auftaucht, gilt deshalb als offen. Der naechste Lauf holt ihn nach.
    """
    if not DONE_LOG.exists():
        return set()

    logged: dict[int, int] = {}
    with open(DONE_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    logged[int(rec["player_id"])] = int(rec.get("rows", 0))
                except (ValueError, KeyError, TypeError):
                    continue

    have: set[int] = set()
    for part in sorted(INTERIM.glob("panel_part_*.parquet")):
        have.update(int(p) for p in pd.read_parquet(part, columns=["player_id"])["player_id"])

    # Wer laut Protokoll keine Zeilen hat, kann auch keine in einer Teildatei
    # haben - der ist trotzdem fertig und darf nicht erneut abgerufen werden.
    return {pid for pid, rows in logged.items() if rows == 0 or pid in have}


def current_bl2_player_ids() -> set[int]:
    """Spieler, die aktuell in der 2. Liga stehen.

    Fuer sie liefert Competition 1 die laufende Saison nicht mit; sie brauchen
    einen zweiten Call gegen Competition 2.
    """
    from ..paths import LEGACY_DIR

    path = LEGACY_DIR / "data_2.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(p["id"]) for p in data.get("players", []) if str(p.get("id", "")).isdigit()}


def main() -> None:
    enable_utf8_stdout()
    ensure_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0, help="nur N Spieler (zum Testen)")
    ap.add_argument("--consolidate", action="store_true",
                    help="nur Teildateien zusammenfuehren und Teams herleiten - "
                         "offline, ohne Login")
    args = ap.parse_args()

    if args.consolidate:
        consolidate()
        return

    if not PLAYER_INDEX.exists():
        raise SystemExit(f"{PLAYER_INDEX} fehlt - erst 'python -m src.ingest.enumerate_ids' laufen lassen")

    index = pd.read_parquet(PLAYER_INDEX)
    all_ids = sorted(int(i) for i in index["id"])
    bl2_now = current_bl2_player_ids()

    done = load_done()
    todo = [i for i in all_ids if i not in done]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print("[backfill] nichts zu tun - alle Spieler bereits geholt")
        consolidate()
        return

    est = len(todo) * args.delay / args.workers / 60
    print(f"[backfill] Spieler im Index: {len(all_ids)}, offen: {len(todo)}")
    print(f"[backfill] davon aktuell 2. Liga (Zusatz-Call): {len(set(todo) & bl2_now)}")
    print(f"[backfill] {args.workers} Worker, {args.delay}s Delay -> ca. {est:.0f} min")
    print("[backfill] resumierbar - Abbruch ist unkritisch\n")

    client = KickbaseClient(delay=args.delay, verbose=True)
    client.login()

    started = time.time()
    counters = {"done": 0, "with_history": 0, "rows": 0}
    part_rows: list[dict] = []
    part_matches: list[dict] = []
    # Hinter den vorhandenen Teildateien weiterzaehlen. Startete der Zaehler bei
    # null, schrieb jeder zweite Lauf wieder panel_part_001.parquet und loeschte
    # damit die Ernte des ersten - beim Nachholen von 15 Spielern gingen so
    # 209.446 Zeilen verloren.
    part_no = max((int(p.stem.rsplit("_", 1)[-1]) for p in INTERIM.glob("panel_part_*.parquet")
                   if p.stem.rsplit("_", 1)[-1].isdigit()), default=0)

    def flush(force: bool = False) -> None:
        nonlocal part_rows, part_matches, part_no
        if not part_rows or (len(part_rows) < 200_000 and not force):
            return
        part_no += 1
        pd.DataFrame(part_rows).to_parquet(INTERIM / f"panel_part_{part_no:03d}.parquet", index=False)
        pd.DataFrame(part_matches).to_parquet(INTERIM / f"matches_part_{part_no:03d}.parquet", index=False)
        part_rows, part_matches = [], []

    with open(DONE_LOG, "a", encoding="utf-8") as log:
        def handle(pid: int) -> None:
            payload = client.performance("1", pid)
            rows: list[dict] = []
            matches: list[dict] = []
            if isinstance(payload, dict):
                rows, matches = parse_player(pid, payload)

            # Laufende Saison fuer aktuelle Zweitligaspieler nachholen
            if pid in bl2_now:
                extra = client.performance("2", pid)
                if isinstance(extra, dict):
                    r2, m2 = parse_player(pid, extra)
                    seen = {(r["season"], r["matchday"]) for r in rows}
                    rows.extend(r for r in r2 if (r["season"], r["matchday"]) not in seen)
                    matches.extend(m2)

            with _lock:
                part_rows.extend(rows)
                part_matches.extend(matches)
                counters["done"] += 1
                counters["rows"] += len(rows)
                if rows:
                    counters["with_history"] += 1
                log.write(json.dumps({"player_id": pid, "rows": len(rows)}) + "\n")

                if counters["done"] % 250 == 0:
                    log.flush()
                    flush()
                    elapsed = time.time() - started
                    rate = counters["done"] / elapsed
                    eta = (len(todo) - counters["done"]) / rate / 60 if rate else 0
                    print(f"  [{counters['done']}/{len(todo)}] "
                          f"mit Historie {counters['with_history']}, "
                          f"{counters['rows']} Zeilen, {rate:.1f}/s, ETA {eta:.0f} min")

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(handle, todo))
        except KeyboardInterrupt:
            print("\n[backfill] abgebrochen - Fortschritt gesichert")

    flush(force=True)
    print(f"\n[backfill] fertig in {(time.time() - started) / 60:.1f} min: "
          f"{counters['with_history']}/{counters['done']} Spieler mit Historie, "
          f"{counters['rows']} Zeilen")
    print(f"[backfill] Requests: ok={client.stats.ok} notfound={client.stats.not_found} "
          f"err={client.stats.errors} 429={client.stats.rate_limited}")
    consolidate()


def derive_teams(panel: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """team_id/opponent_id/home je Zeile aus dem Spielplan herleiten.

    Kickbase' ``pt`` ist meist richtig, aber nicht immer: 505 Zeilen
    abgeschlossener Saisons trugen ein Team, das an dem Spiel gar nicht
    beteiligt war (gehaeuft 2021/22 Liga 2), und die vor Saisonstart
    gecrawlte laufende Saison hatte gar keins. Der Spielplan kann beides
    reparieren, weil jede Zeile ihre ``match_id`` traegt und
    ``matches.parquet`` (aus ``t1``/``t2`` gebaut, von ``pt`` unabhaengig)
    alle Panel-Spiele abdeckt.

    Regeln in dieser Reihenfolge, jede an den Bestandsdaten vermessen:

    1. ``pt`` liegt in der Paarung -> glauben. Deckt ligainterne Wechsel je
       Zeile ab; dort wechselt ``pt`` nachweislich korrekt mit.
    2. Von den ueber Regel 1 etablierten Teams der Gruppe (Spieler, Saison,
       Liga) liegt genau eines in der Paarung -> dieses. Faengt die
       Randzeilen ligainterner Wechsler.
    3. Haeufigkeit: das Team der Paarung, das in den Paarungen der Gruppe
       mindestens doppelt so oft vorkommt wie das andere (und mindestens
       zweimal) -> dieses. Ein strenger Gruppen*schnitt* stand hier zuerst
       und fiel durch: bei Froede (1666, 21/22) machte eine einzige korrupte
       Zeile (Paarung zweier fremder Teams) den Schnitt leer und liess alle
       27 Zeilen unaufgeloest, obwohl 26 davon Rostock enthielten. Der
       2x-Abstand schuetzt zugleich den Winterwechsler: im direkten Duell
       seiner beiden Klubs liegen deren Haeufigkeiten nah beieinander, und
       die Zeile bleibt ehrlich offen, statt geraten zu werden.
    4. Einzelpaarung (alle Spiele der Gruppe gegen denselben Gegner, z. B.
       Maloney 21/22: Hin- und Rueckspiel gegen den BVB): haeufigstes
       ``pt`` der Gruppe liegt in der Paarung -> glauben.
    5. Sonst bleibt die Zeile ohne Team - ehrlich leer statt falsch.
    """
    pair: dict[str, tuple[str, str]] = {
        str(m): (str(a), str(b))
        for m, a, b in zip(matches["match_id"], matches["home_team"], matches["away_team"])
    }

    roh = [None if pd.isna(t) else str(t) for t in panel["team_id"]]
    mids = [str(m) for m in panel["match_id"]]
    keys = list(zip(panel["player_id"], panel["season"], panel["league"]))

    # Gruppenwissen in einem Vorlauf: etablierte Teams, Team-Haeufigkeit
    # ueber alle Paarungen, haeufigstes rohes pt.
    etabliert: dict[tuple, set] = {}
    team_zaehler: dict[tuple, dict[str, int]] = {}
    pt_zaehler: dict[tuple, dict[str, int]] = {}
    for key, mid, pt in zip(keys, mids, roh):
        p = pair.get(mid)
        if p is None:
            continue
        z = team_zaehler.setdefault(key, {})
        for t in p:
            z[t] = z.get(t, 0) + 1
        if pt in p:
            etabliert.setdefault(key, set()).add(pt)
        if pt:
            z = pt_zaehler.setdefault(key, {})
            z[pt] = z.get(pt, 0) + 1

    team_neu: list[str | None] = []
    stats = {"pt": 0, "etabliert": 0, "haeufigkeit": 0, "stich": 0, "offen": 0, "ohne_spiel": 0}
    for key, mid, pt in zip(keys, mids, roh):
        p = pair.get(mid)
        if p is None:
            team_neu.append(pt)
            stats["ohne_spiel"] += 1
            continue
        if pt in p:
            team_neu.append(pt)
            stats["pt"] += 1
            continue
        e = etabliert.get(key, set()) & set(p)
        if len(e) == 1:
            team_neu.append(next(iter(e)))
            stats["etabliert"] += 1
            continue
        z = team_zaehler[key]
        a, b = sorted(p, key=lambda t: z.get(t, 0), reverse=True)
        if z.get(a, 0) >= 2 and z.get(a, 0) >= 2 * z.get(b, 0):
            team_neu.append(a)
            stats["haeufigkeit"] += 1
            continue
        zp = pt_zaehler.get(key)
        haeufigst = max(zp, key=zp.get) if zp else None
        if haeufigst in p:
            team_neu.append(haeufigst)
            stats["stich"] += 1
            continue
        team_neu.append(None)
        stats["offen"] += 1

    panel = panel.copy()
    panel["team_id"] = team_neu
    panel["opponent_id"] = [
        None if t is None or pair.get(m) is None
        else (pair[m][1] if t == pair[m][0] else pair[m][0])
        for t, m in zip(team_neu, mids)
    ]
    panel["home"] = [
        None if t is None or pair.get(m) is None else t == pair[m][0]
        for t, m in zip(team_neu, mids)
    ]

    print(f"[backfill] Teams hergeleitet: pt {stats['pt']:,}, "
          f"etabliert {stats['etabliert']}, Haeufigkeit {stats['haeufigkeit']:,}, "
          f"Stichentscheid {stats['stich']}, offen {stats['offen']}"
          + (f", ohne Spielpaarung {stats['ohne_spiel']}" if stats["ohne_spiel"] else ""))
    return panel


def consolidate() -> None:
    """Teildateien zu panel.parquet und matches.parquet zusammenfuehren."""
    panel_parts = sorted(INTERIM.glob("panel_part_*.parquet"))
    match_parts = sorted(INTERIM.glob("matches_part_*.parquet"))
    if not panel_parts:
        print("[backfill] keine Teildateien gefunden")
        return

    panel = pd.concat([pd.read_parquet(p) for p in panel_parts], ignore_index=True)
    panel["match_date"] = pd.to_datetime(panel["match_date"], format="ISO8601", utc=True)

    # Ein Spieler kann fuer denselben Spieltag zwei Eintraege haben, wenn er
    # innerhalb der Liga den Verein gewechselt hat - beide Klubs bringen ihre
    # Partie in seine Historie ein (beobachtet bei 623 Knoche und 4590 Omlin).
    # Nur eine davon ist sein tatsaechlicher Einsatz. Nach Minuten sortieren
    # und die oberste behalten, sonst gewinnt die 0-Minuten-Zeile per Zufall
    # der Reihenfolge und ein realer Einsatz verschwindet.
    panel = (panel.sort_values(["player_id", "season", "league", "matchday", "minutes"])
             .drop_duplicates(subset=["player_id", "season", "league", "matchday"],
                              keep="last"))
    panel = panel.sort_values(["season", "league", "matchday", "player_id"]).reset_index(drop=True)

    # Schrumpft der Bestand, fehlen Teildateien - dann lieber gar nicht
    # schreiben als still Zeilen verlieren.
    if PANEL.exists():
        bestand = len(pd.read_parquet(PANEL, columns=["player_id"]))
        if len(panel) < bestand:
            print(f"[backfill] ABBRUCH: Teildateien ergeben {len(panel):,} Zeilen, "
                  f"Bestand hat {bestand:,} - interim/ unvollstaendig?")
            return

    matches = pd.concat([pd.read_parquet(p) for p in match_parts], ignore_index=True)
    matches["match_date"] = pd.to_datetime(matches["match_date"], format="ISO8601", utc=True)
    # Jedes Spiel taucht einmal pro beteiligtem Spieler auf - auf match_id eindampfen.
    matches = matches.drop_duplicates(subset=["season", "league", "match_id"])
    matches = matches.sort_values(["season", "league", "matchday", "match_id"]).reset_index(drop=True)

    panel = derive_teams(panel, matches)
    panel.to_parquet(PANEL, index=False)
    matches.to_parquet(MATCHES, index=False)

    print(f"\n[backfill] {PANEL}")
    print(f"           {len(panel):,} Zeilen, {panel['player_id'].nunique():,} Spieler, "
          f"{panel['season'].nunique()} Saisons")
    print(f"[backfill] {MATCHES}")
    print(f"           {len(matches):,} Spiele")
    per_season = panel.groupby(["season", "league"]).size().unstack(fill_value=0)
    print(f"\n{per_season.to_string()}")


if __name__ == "__main__":
    main()
