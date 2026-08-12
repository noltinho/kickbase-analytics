"""Einmaliger, gedrosselter Crawl ueber den Kickbase-Spieler-ID-Raum.

Warum ueberhaupt: Die aus data_*.json bekannten ~1.060 IDs sind ausschliesslich
Spieler, die *heute* in BL1/BL2 stehen. Ein Modell, das nur darauf trainiert,
sieht nie die Spieler, die aus der Liga verschwunden sind - und genau deren
Verschwinden korreliert mit Leistung. Der Probe-Lauf (probe_ids.py) hat gezeigt,
dass abgemeldete IDs sehr wohl aufloesen (z. B. 703 Admir Mehmedi, 9 Saisons),
also laesst sich der Survivorship-Bias durch einen Vollcrawl beheben.

Erkenntnisse aus dem Probe-Lauf, die hier eingebaut sind:

* ``/players/{pid}`` ist **competition-agnostisch** - eine Competition genuegt,
  das halbiert den Crawl.
* Nicht existierende IDs antworten mit **HTTP 500 + {"err":2,"errMsg":"NotFound"}**,
  nicht 404. Der Client faengt das ab und wiederholt sie nicht.

Der Lauf ist **resumierbar**: jede Antwort wird sofort als JSONL-Zeile
angehaengt. Ein Abbruch nach 20 Minuten kostet nichts, ein erneuter Start
ueberspringt alles bereits Gepruefte.

Aufruf (aus kbxp/):
    python -m src.ingest.enumerate_ids --max-id 20000 --workers 3
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..paths import RAW, ensure_dirs
from .kickbase_client import NOT_FOUND, KickbaseClient, enable_utf8_stdout

JSONL = RAW / "player_index.jsonl"
PARQUET = RAW / "player_index.parquet"

# /players/{pid} liefert fuer beide Competitions dasselbe (im Probe-Lauf
# verifiziert), also reicht eine.
CRAWL_COMPETITION = "1"

_write_lock = threading.Lock()


def load_probed_ids() -> set[int]:
    """IDs, die in einem frueheren Lauf bereits geprueft wurden."""
    if not JSONL.exists():
        return set()
    seen: set[int] = set()
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(int(json.loads(line)["id"]))
            except (ValueError, KeyError):
                continue
    return seen


def extract_player(pid: int, data: dict) -> dict:
    """Stammdaten aus der /players-Antwort.

    ``st`` ist der Fitness-/Kaderstatus; 128 bedeutet "nicht im Kader" und
    markiert damit genau die Ehemaligen, um die es beim Crawl geht.
    """
    name = data.get("n")
    if not name:
        name = f"{data.get('fn', '')} {data.get('ln', '')}".strip() or None
    return {
        "id": pid,
        "exists": True,
        "name": name,
        "first_name": data.get("fn"),
        "last_name": data.get("ln"),
        "team_id": str(data.get("tid")) if data.get("tid") is not None else None,
        "team_name": data.get("tn"),
        "position": data.get("pos"),
        "status": data.get("st"),
        "market_value": data.get("mv"),
        "shirt_number": data.get("shn"),
    }


def crawl_one(client: KickbaseClient, pid: int) -> dict:
    data = client.player(CRAWL_COMPETITION, pid)
    if isinstance(data, dict):
        return extract_player(pid, data)
    if data is NOT_FOUND:
        return {"id": pid, "exists": False}
    # None = unklar (Timeout/5xx ohne NotFound-Body). Nicht als "gibt es nicht"
    # verbuchen - sonst faellt der Spieler beim Resume durchs Raster.
    return {"id": pid, "exists": None}


def main() -> None:
    enable_utf8_stdout()
    ensure_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-id", type=int, default=1)
    ap.add_argument("--max-id", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--retry-unknown", action="store_true",
                    help="IDs mit unklarem Ergebnis aus frueheren Laeufen erneut versuchen")
    args = ap.parse_args()

    probed = load_probed_ids()
    if args.retry_unknown and JSONL.exists():
        with open(JSONL, encoding="utf-8") as f:
            unknown = {json.loads(l)["id"] for l in f if l.strip()
                       and json.loads(l).get("exists") is None}
        probed -= unknown
        print(f"[crawl] {len(unknown)} unklare IDs werden erneut versucht")

    todo = [i for i in range(args.min_id, args.max_id + 1) if i not in probed]
    if not todo:
        print("[crawl] nichts zu tun - alle IDs im Bereich bereits geprueft")
        write_parquet()
        return

    est_min = len(todo) * args.delay / args.workers / 60
    print(f"[crawl] bereits geprueft: {len(probed)}")
    print(f"[crawl] offen: {len(todo)} IDs ({args.min_id}..{args.max_id})")
    print(f"[crawl] {args.workers} Worker, {args.delay}s Delay -> ca. {est_min:.0f} min")
    print(f"[crawl] resumierbar: Abbruch mit Strg+C ist unkritisch\n")

    client = KickbaseClient(delay=args.delay, verbose=True)
    client.login()

    started = time.time()
    found = 0
    done = 0

    with open(JSONL, "a", encoding="utf-8") as sink:
        def handle(pid: int) -> None:
            nonlocal found, done
            rec = crawl_one(client, pid)
            with _write_lock:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
                if rec.get("exists"):
                    found += 1
                if done % 500 == 0:
                    sink.flush()
                    elapsed = time.time() - started
                    rate = done / elapsed
                    eta = (len(todo) - done) / rate / 60 if rate else 0
                    print(f"  [{done}/{len(todo)}] gefunden {found} "
                          f"({100 * found / done:.0f}%)  {rate:.1f} IDs/s  ETA {eta:.0f} min")

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(handle, todo))
        except KeyboardInterrupt:
            print("\n[crawl] abgebrochen - Fortschritt ist gesichert, "
                  "erneuter Start setzt fort")

    print(f"\n[crawl] fertig: {found} Spieler in {(time.time() - started) / 60:.1f} min")
    print(f"[crawl] Requests: ok={client.stats.ok} notfound={client.stats.not_found} "
          f"err={client.stats.errors} 429={client.stats.rate_limited}")
    write_parquet()


def write_parquet() -> None:
    """JSONL -> parquet, nur die existierenden Spieler."""
    import pandas as pd

    if not JSONL.exists():
        print("[crawl] keine JSONL vorhanden")
        return

    rows = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="last")
    existing = df[df["exists"] == True].drop(columns=["exists"])  # noqa: E712
    existing = existing.sort_values("id").reset_index(drop=True)
    existing.to_parquet(PARQUET, index=False)

    unknown = int((df["exists"].isna()).sum())
    print(f"[crawl] {PARQUET}: {len(existing)} Spieler "
          f"({len(df)} IDs geprueft, {unknown} unklar)")
    if unknown:
        print(f"[crawl] Hinweis: {unknown} unklare IDs - mit --retry-unknown nachholen")


if __name__ == "__main__":
    main()
