"""Machbarkeitstest vor dem ID-Crawl.

Beantwortet drei Fragen, von denen die Datenschicht abhaengt:

1. **Loesen unbekannte Spieler-IDs ueberhaupt auf?**
   Die bekannten ~1.060 IDs stammen alle aus aktuellen Kadern bzw. dem
   25/26-Archiv - also ausschliesslich Spieler, die *heute noch* in BL1/BL2
   sind. Trainiert man darauf, hat das Modell einen schweren
   Survivorship-Bias. Antwortet die API auch fuer abgemeldete IDs, laesst
   sich das durch einen einmaligen Crawl vollstaendig beheben.

2. **Liefert /performance fuer solche IDs echte Historie?**
   Ein Treffer nuetzt nur, wenn Punkte/Minuten vergangener Saisons dranhaengen.

3. **Wie weit reicht /marketValue/{tage} zurueck?**
   Laut Nutzer 365 Tage. Wird hier gegengeprueft, weil davon abhaengt, ueber
   wie viele Halbrunden sich das Mispricing-Signal backtesten laesst.

Aufruf (aus kbxp/):  python -m src.ingest.probe_ids [--sample 200]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..paths import RAW, ensure_dirs
from .kickbase_client import NOT_FOUND, KickbaseClient, enable_utf8_stdout
from .known_ids import load_known_players

# Bekannte IDs reichen von 4 bis 15205. Etwas Puffer nach oben, falls die
# Vergabe inzwischen weitergelaufen ist.
ID_MIN = 1
ID_MAX = 17000
BUCKET = 1000

COMPETITIONS = ("1", "2")


def stratified_sample(known: set[int], n: int, rng: random.Random) -> list[int]:
    """Gleichmaessig ueber 1000er-Baender ziehen, bekannte IDs ausgenommen.

    Gleichverteilt ueber den gesamten Bereich zu ziehen wuerde die Trefferquote
    nur global messen. Interessant ist aber das Profil: die Baender, in denen
    heute *keine* bekannten Spieler liegen (z. B. 5000-5999), sind genau die,
    in denen die verschwundenen Jahrgaenge zu erwarten sind.
    """
    buckets = list(range(ID_MIN, ID_MAX, BUCKET))
    per_bucket = max(1, n // len(buckets))
    out: list[int] = []
    for start in buckets:
        candidates = [i for i in range(start, min(start + BUCKET, ID_MAX)) if i not in known]
        if not candidates:
            continue
        out.extend(rng.sample(candidates, min(per_bucket, len(candidates))))
    return out


def probe_one(client: KickbaseClient, pid: int) -> dict:
    """Eine ID gegen beide Competitions testen."""
    result: dict = {"id": pid, "resolved_in": None, "name": None, "team": None}

    for cid in COMPETITIONS:
        data = client.player(cid, pid)
        if data is NOT_FOUND or data is None:
            continue
        result["resolved_in"] = cid
        result["name"] = data.get("n") or f"{data.get('fn', '')} {data.get('ln', '')}".strip()
        result["team"] = str(data.get("tid") or "")
        result["position"] = data.get("pos")
        result["market_value"] = data.get("mv")
        break

    if result["resolved_in"] is None:
        return result

    # Historie nachladen
    perf = client.performance(result["resolved_in"], pid)
    if isinstance(perf, dict):
        seasons = []
        for season in perf.get("it", []):
            entries = season.get("ph", []) or []
            played = sum(1 for e in entries if (e.get("p") is not None))
            seasons.append({
                "title": season.get("ti"),
                "league": season.get("n"),
                "matchdays": len(entries),
                "with_points": played,
            })
        result["seasons"] = seasons
        result["n_seasons"] = len(seasons)
        result["oldest_season"] = min((s["title"] for s in seasons if s["title"]), default=None)
        result["newest_season"] = max((s["title"] for s in seasons if s["title"]), default=None)
    return result


def probe_market_value_depth(client: KickbaseClient, pid: int, cid: str) -> dict:
    """Testen, welche Zeitraeume /marketValue/{tage} akzeptiert."""
    out = {}
    for days in (365, 730, 1825, 3650):
        data = client.market_value(cid, pid, days)
        if not isinstance(data, dict):
            out[days] = {"ok": False}
            continue
        items = data.get("it", []) or []
        dts = [i.get("dt") for i in items if i.get("dt") is not None]
        out[days] = {
            "ok": True,
            "points": len(items),
            "span_days": (max(dts) - min(dts)) if dts else 0,
        }
    return out


def main() -> None:
    enable_utf8_stdout()
    ensure_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200, help="Anzahl zu testender unbekannter IDs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    known_players = load_known_players()
    known = set(known_players)
    print(f"[probe] bekannte IDs: {len(known)} (min {min(known)}, max {max(known)})")

    rng = random.Random(args.seed)
    sample = stratified_sample(known, args.sample, rng)
    print(f"[probe] teste {len(sample)} unbekannte IDs, {args.workers} Worker, {args.delay}s Delay")
    print(f"[probe] geschaetzte Laufzeit: ~{len(sample) * args.delay * 2 / args.workers / 60:.1f} min\n")

    client = KickbaseClient(delay=args.delay, verbose=True)
    client.login()

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(lambda p: probe_one(client, p), sample), 1):
            results.append(res)
            if i % 25 == 0:
                hits = sum(1 for r in results if r["resolved_in"])
                print(f"  [{i}/{len(sample)}] Treffer bisher: {hits}")

    # ---- Auswertung ----------------------------------------------------
    hits = [r for r in results if r["resolved_in"]]
    print(f"\n{'=' * 62}")
    print(f"  ERGEBNIS: {len(hits)}/{len(results)} unbekannte IDs loesen auf "
          f"({100 * len(hits) / max(len(results), 1):.1f} %)")
    print(f"{'=' * 62}\n")

    # Trefferquote je 1000er-Band, gegen die Dichte bekannter IDs gestellt
    tested_by_bucket: Counter[int] = Counter(r["id"] // BUCKET * BUCKET for r in results)
    hit_by_bucket: Counter[int] = Counter(r["id"] // BUCKET * BUCKET for r in hits)
    known_by_bucket: Counter[int] = Counter(i // BUCKET * BUCKET for i in known)

    print("  Band          getestet  Treffer  Quote   bekannte IDs")
    for start in sorted(tested_by_bucket):
        t, h, k = tested_by_bucket[start], hit_by_bucket[start], known_by_bucket[start]
        print(f"  {start:5d}-{start + 999:5d}  {t:8d} {h:8d}  {100 * h / t:5.1f}%  {k:6d}")

    if hits:
        with_hist = [r for r in hits if r.get("n_seasons")]
        print(f"\n  mit /performance-Historie: {len(with_hist)}/{len(hits)}")
        if with_hist:
            counts = Counter(r["n_seasons"] for r in with_hist)
            print(f"  Saisons je Spieler: min {min(counts)}, max {max(counts)}, "
                  f"median {sorted(r['n_seasons'] for r in with_hist)[len(with_hist) // 2]}")
            oldest = Counter(r["oldest_season"] for r in with_hist if r.get("oldest_season"))
            print(f"  aelteste Saison (Top 5): {oldest.most_common(5)}")
            newest = Counter(r["newest_season"] for r in with_hist if r.get("newest_season"))
            print(f"  juengste Saison (Top 5): {newest.most_common(5)}")

            print("\n  Beispiele:")
            for r in with_hist[:8]:
                print(f"    {r['id']:>6}  {str(r['name'])[:22]:22}  comp {r['resolved_in']}  "
                      f"{r['n_seasons']:2d} Saisons  {r['oldest_season']} .. {r['newest_season']}")

        # Marktwert-Tiefe an einem Treffer testen
        probe_pid = with_hist[0]["id"] if with_hist else hits[0]["id"]
        probe_cid = (with_hist[0] if with_hist else hits[0])["resolved_in"]
        print(f"\n  /marketValue-Tiefe (Spieler {probe_pid}):")
        for days, info in probe_market_value_depth(client, probe_pid, probe_cid).items():
            if info["ok"]:
                print(f"    {days:5d} Tage angefragt -> {info['points']:4d} Punkte, "
                      f"Spanne {info['span_days']} Tage")
            else:
                print(f"    {days:5d} Tage angefragt -> Fehler/abgelehnt")

    out_path = RAW / "probe_ids.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sample": sample, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] Rohergebnis: {out_path}")
    print(f"[probe] Requests: ok={client.stats.ok} 404={client.stats.not_found} "
          f"err={client.stats.errors} 429={client.stats.rate_limited}")

    # ---- Empfehlung ----------------------------------------------------
    rate = len(hits) / max(len(results), 1)
    print(f"\n{'=' * 62}")
    if rate >= 0.15:
        est = int(rate * (ID_MAX - ID_MIN))
        print(f"  EMPFEHLUNG: Vollcrawl durchfuehren.")
        print(f"  Hochrechnung: ~{est} auffindbare Spieler-IDs gesamt")
        print(f"  (vs. {len(known)} bekannte) -> Survivorship-Bias wird behoben.")
    elif rate > 0.02:
        print(f"  EMPFEHLUNG: Vollcrawl lohnt, aber Ausbeute maessig ({100 * rate:.1f} %).")
        print(f"  Vor dem Lauf pruefen, ob die Treffer wirklich Ehemalige sind.")
    else:
        print(f"  EMPFEHLUNG: Kein Vollcrawl - unbekannte IDs loesen praktisch nicht auf.")
        print(f"  Fallback: bekannte IDs nutzen, Survivorship-Bias dokumentieren.")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
