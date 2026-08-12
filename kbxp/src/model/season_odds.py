"""Saison-Ausgangsquoten in Teamstaerken uebersetzen.

Warum ueberhaupt. Fuer Teams, die die Liga gewechselt haben, kennt das
Staerkemodell keinen brauchbaren Startwert: die Vorsaison fand in einer anderen
Liga statt, und uebersetzt wird sie heute mit einer einzigen Zahl je Richtung
(CARRYOVER_FACTORS in fetch.py). Gemessen ueber 2013/14-2025/26, Spieltage vor der
Winterpause, sieht das so aus:

    Herkunft        rho     R2    paarweise
    gleiche Liga  0,454  0,239       65,4 %
    abgestiegen   0,271  0,061       59,2 %

Alle Absteiger bekommen denselben Faktor, obwohl die Streuung innerhalb der
Gruppe (66 Punkte) so gross ist wie der mittlere Effekt selbst. 2026/27 trifft
das Wolfsburg, Heidenheim und St. Pauli gleichermassen - waehrend der Wettmarkt
Wolfsburg bei 2,6-facher Aufstiegswahrscheinlichkeit von Heidenheim sieht.

Der Weg von der Quote zur Staerke. P(Platz <= k) ist ein nichtlineares Funktional
des ganzen Staerkevektors - ein Team steigt auf, weil es besser ist als die
siebzehn anderen. Umgekehrt aufloesen laesst sich das nur numerisch: Staerken
raten, die Saison tausendfach simulieren, die simulierten Wahrscheinlichkeiten
mit dem Markt vergleichen, nachziehen, wiederholen.

Welcher Markt das ist, steht in data/season_odds.json und ist je Liga
verschieden: die 2. Bundesliga liegt als Aufstiegsmarkt vor (2 Plaetze), die
Bundesliga als Top-6-Markt (6 Plaetze). Eine Liga kann auch mehrere Maerkte
fuehren - noetig, sobald einer davon an einem Team ansteht. Bayern steht im
Top-6-Markt bei Quote 1,00, das heisst nur "sicher"; erst die Meisterquote sagt,
wie viel staerker sie sind.

Identifiziert wird dabei nur ein Skalar je Team. Die Aufteilung in Abwehr und
Angriff steckt in keiner Platzierungsquote und bleibt deshalb beim Carryover.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paths import LEGACY_DIR, PROCESSED  # noqa: E402

N_SIMS = 8000       # simulierte Saisons je Runde
ROUNDS = 60         # Anpassungsrunden; konvergiert in der Praxis nach ~30
STEP = 1.2          # Schrittweite der Korrektur
MAX_GOALS = 15      # Abschneiden der Poisson-Ziehung

LEAGUE_NAME = {"1": "Bundesliga", "2": "2. Bundesliga"}

# Basis-Torrate und Heimvorteil je Liga, log-Skala, aus matches.parquet gemessen
# (Bundesliga 3.978 Partien: 1,702 zu 1,353 Tore; 2. Bundesliga 1.530: 1,633 zu
# 1,348). Fest hinterlegt, damit der Export-Pfad ohne die Parquet-Dateien laeuft.
LEAGUE_GOALS = {"1": (0.423, 0.115), "2": (0.399, 0.096)}


def load_odds(path: Path | None = None) -> dict:
    """data/season_odds.json - von Hand gepflegt, ~18 Zahlen je Liga und Saison."""
    p = path or (LEGACY_DIR / "season_odds.json")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def implied_probs(odds: dict[str, float], spots: int, method: str = "power",
                  k: float | None = None) -> dict[str, float]:
    """Dezimalquoten in Wahrscheinlichkeiten, Overround herausgerechnet.

    Die rohen 1/Quote summieren sich auf mehr als die Zahl der Plaetze. Wie viel
    mehr, haengt am Markt: der Aufstiegsmarkt der 2. Bundesliga kommt auf 2,877
    statt 2,000 (44 % Marge), der Top-6-Markt der Bundesliga auf 6,679 statt
    6,000 (11 %). Wie man diese Marge verteilt, ist keine Nebensaechlichkeit:

    ``proportional`` teilt sie gleichmaessig auf. Das ist die naive Loesung und
    ueberschaetzt Aussenseiter, weil die Marge auf lange Quoten in Wahrheit
    hoeher liegt (Favorit-Aussenseiter-Verzerrung).

    ``power`` sucht stattdessen ein k mit ``sum (1/quote)^k = spots``. Lange
    Quoten schrumpfen dabei staerker als kurze. Fuer 2026/27 ergibt das
    k = 1,25: Wolfsburg steigt von 47,9 auf 62,8 Prozent, Osnabrueck faellt von
    2,0 auf 1,2.

    Der Unterschied schlaegt durch: simulierte Tabellen haben damit 33,1 statt
    31,4 Punkte Spannweite. Echte Zweitliga-Saisons liegen bei 40,0. Die
    verbleibende Luecke ist erwartet und soll nicht wegkalibriert werden - eine
    realisierte Tabelle enthaelt Trainerwechsel, Wintertransfers und aufgegebene
    Saisons, die ein Modell mit ueber die Saison konstanter Staerke nicht kennt.
    Eine Erwartung streut zu Recht enger als ein Ergebnis.
    """
    inv = {t: 1.0 / float(o) for t, o in odds.items()}
    if k is not None:
        return {t: float(v ** k) for t, v in inv.items()}
    total = sum(inv.values())
    if method == "power" and total > spots:
        kk = margin_exponent(inv, spots)
        if kk is not None:
            return {t: float(v ** kk) for t, v in inv.items()}
    return {t: v / total * spots for t, v in inv.items()}


def margin_exponent(inv: dict[str, float], spots: float) -> float | None:
    """Exponent k mit ``sum (1/quote)^k = spots``, oder None wenn unloesbar.

    Wird getrennt gebraucht, weil ein Teilmarkt sich nicht selbst normieren kann:
    fuer die Meisterquote eines einzelnen Teams gibt es keine Summe ueber alle
    Teams. Dort wird der Exponent des vollstaendigen Marktes derselben Liga
    weiterbenutzt - die Marge desselben Buchmachers ist aehnlich aufgebaut.
    """
    from scipy.optimize import brentq
    vals = np.array(list(inv.values()), dtype=float)
    try:
        return float(brentq(lambda k: float((vals ** k).sum()) - spots, 0.5, 5.0))
    except ValueError:
        return None


def goal_params(liga: str, matches: pd.DataFrame | None = None) -> tuple[float, float]:
    """Basis-Torrate und Heimvorteil der Liga, log-Skala.

    Ohne ``matches`` aus LEAGUE_GOALS - der Export braucht dann keine Parquet-Dateien.
    """
    if matches is not None:
        g = matches[matches.league == LEAGUE_NAME.get(liga, liga)].dropna(
            subset=["home_goals", "away_goals"])
        if len(g) >= 100:
            hg, ag = float(g.home_goals.mean()), float(g.away_goals.mean())
            return float(np.log((hg + ag) / 2)), float(np.log(hg / ag) / 2)
    return LEAGUE_GOALS.get(liga, (0.40, 0.10))


def place_probs(theta: np.ndarray, hi: np.ndarray, ai: np.ndarray,
                base: float, hfa: float, spots: list[int],
                n_sims: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Anteil der simulierten Saisons, in denen Team i unter die ersten k kommt.

    Tore beider Seiten poissonverteilt um exp(basis +/- Staerkedifferenz + Heimvorteil).
    Platziert wird nach Punkten, dann Tordifferenz, dann erzielten Toren - wie in
    der Liga selbst.

    Mehrere Grenzen ``spots`` aus einem Durchlauf, weil sich Maerkte oft
    ergaenzen: die Bundesliga liegt als Top-6-Markt vor, aber Bayern steht dort
    bei Quote 1,00. Das heisst nur "sicher" und sagt nichts darueber, wie viel
    staerker sie sind - erst die Meisterquote (1 Platz) pinnt das fest.
    """
    n_teams = len(theta)
    d = theta[hi] - theta[ai]
    lam_h = np.exp(base + d + hfa)
    lam_a = np.exp(base - d - hfa)

    gh = rng.poisson(lam_h[None, :], size=(n_sims, len(hi))).clip(0, MAX_GOALS)
    ga = rng.poisson(lam_a[None, :], size=(n_sims, len(hi))).clip(0, MAX_GOALS)

    pts = np.zeros((n_sims, n_teams), dtype=np.int32)
    gf = np.zeros((n_sims, n_teams), dtype=np.int32)
    gd = np.zeros((n_sims, n_teams), dtype=np.int32)
    hw, aw = gh > ga, ga > gh
    dr = ~hw & ~aw
    for arr, idx, val in ((pts, hi, hw * 3 + dr), (pts, ai, aw * 3 + dr),
                          (gf, hi, gh), (gf, ai, ga),
                          (gd, hi, gh - ga), (gd, ai, ga - gh)):
        np.add.at(arr.T, idx, val.T)

    # Rang: Punkte, dann Tordifferenz, dann erzielte Tore - absteigend.
    key = (pts.astype(np.int64) * 10_000 + (gd + 200) * 20 + np.minimum(gf, 19))
    order = np.argsort(-key, axis=1, kind="stable")
    out = {}
    for s in set(spots):
        up = np.zeros((n_sims, n_teams), dtype=bool)
        np.put_along_axis(up, order[:, :s], True, axis=1)
        out[s] = up.mean(axis=0)
    return out


MAX_TARGET = 0.995   # eine Quote von 1,00 ist als Ziel unerreichbar


def fit_strength(markets: list[dict], fixtures: list[tuple[str, str]],
                 base: float, hfa: float, n_sims: int = N_SIMS,
                 rounds: int = ROUNDS, seed: int = 0) -> dict[str, float] | None:
    """Staerkevektor, der die Marktquoten reproduziert. Einheit: Tore (log-Skala).

    Iterativ, weil sich P(Platz <= k) nicht geschlossen invertieren laesst: je
    Runde ``n_sims`` Saisons simulieren und die Staerken um den Fehler nachziehen.
    Rueckgabe ist auf Mittel 0 zentriert.

    ``markets`` ist eine Liste von {"spots": int, "probs": {team: p}} - mehrere,
    weil ein Markt allein oft nicht reicht. Die Bundesliga liegt als Top-6-Markt
    vor, dort steht Bayern bei Quote 1,00; das heisst nur "sicher" und laesst
    offen, wie viel staerker sie sind. Erst die Meisterquote pinnt das fest.

    ``fixtures`` ist die Liste der Paarungen (Heim-ID, Auswaerts-ID) der Saison.
    """
    teams = sorted({t for m in markets for t in m["probs"]})
    idx = {t: i for i, t in enumerate(teams)}
    fx = [(h, a) for h, a in fixtures if h in idx and a in idx]
    if len(fx) < 100 or len(teams) < 4:
        return None

    hi = np.array([idx[h] for h, _ in fx], dtype=np.intp)
    ai = np.array([idx[a] for _, a in fx], dtype=np.intp)
    spots = [int(m["spots"]) for m in markets]

    # Je Markt: Zielvektor und Maske, welche Teams er ueberhaupt nennt.
    ziele = []
    for m in markets:
        tgt = np.zeros(len(teams))
        msk = np.zeros(len(teams), dtype=bool)
        for t, p in m["probs"].items():
            tgt[idx[t]] = min(float(p), MAX_TARGET)
            msk[idx[t]] = True
        ziele.append((int(m["spots"]), tgt, msk))

    rng = np.random.default_rng(seed)
    theta = np.zeros(len(teams))
    for _ in range(rounds):
        p = place_probs(theta, hi, ai, base, hfa, spots, n_sims, rng)
        schritt = np.zeros(len(teams))
        schlimmster = 0.0
        for s, tgt, msk in ziele:
            err = np.where(msk, tgt - p[s], 0.0)
            schlimmster = max(schlimmster, float(np.abs(err).max()))
            # Der Gradient von P(Platz <= k) nach theta ist bei sehr sicheren und
            # sehr chancenlosen Teams flach; p*(1-p) daempft dort die Korrektur,
            # sonst laufen Favoriten und Aussenseiter davon.
            schritt += STEP * err / np.maximum(p[s] * (1 - p[s]), 0.02) * 0.05
        if schlimmster < 0.004:
            break
        theta += schritt / len(ziele)
        theta -= theta.mean()
    return {t: float(theta[i]) for t, i in idx.items()}


def fixtures_from_plan(path: Path) -> list[tuple[str, str]]:
    """Paarungen aus matchdays_bl{liga}.json - auch die noch ungespielten.

    Die Simulation braucht den ganzen Spielplan, nicht nur die beendeten Partien;
    genau darin unterscheidet sie sich von conceded_from_legacy.
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    return [(str(m["t1"]), str(m["t2"]))
            for md in plan.get("matchdays", []) for m in md.get("matches", [])]


def markets_from_entry(entry: dict) -> list[dict]:
    """Konfiguration in Maerkte mit bereinigten Wahrscheinlichkeiten uebersetzen.

    Zwei Formen werden gelesen: ein einzelner Markt (``spots`` + ``teams``, so
    liegt Liga 2 vor) oder eine Liste unter ``markets``. Der erste vollstaendige
    Markt liefert den Margen-Exponenten; Teilmaerkte - etwa eine Meisterquote fuer
    ein einziges Team - koennen sich nicht selbst normieren und uebernehmen ihn.
    """
    roh = entry.get("markets") or [{"name": entry.get("market", "?"),
                                    "spots": entry.get("spots", 2),
                                    "teams": entry.get("teams", {})}]
    out, k = [], None
    for m in roh:
        odds = {str(t): float(o) for t, o in (m.get("teams") or {}).items()}
        if not odds:
            continue
        spots = int(m.get("spots", 2))
        if m.get("partial"):
            # Ein Teilmarkt nennt nicht alle Teams; seine Wahrscheinlichkeiten
            # summieren sich also nicht auf die Zahl der Plaetze und lassen sich
            # nicht aus sich heraus bereinigen. Ohne Exponent aus einem
            # vollstaendigen Markt lieber weglassen als falsch normieren - sonst
            # wuerde eine einzelne Meisterquote auf 100 % hochgerechnet.
            if k is None:
                continue
            probs = implied_probs(odds, spots, k=k)
        else:
            probs = implied_probs(odds, spots)
            if k is None:
                k = margin_exponent({t: 1.0 / o for t, o in odds.items()}, spots)
        out.append({"name": m.get("name", "?"), "spots": spots, "probs": probs})
    return out


def strength_for(liga: str, fixtures: list[tuple[str, str]],
                 odds_doc: dict | None = None,
                 matches: pd.DataFrame | None = None, **kw) -> dict[str, float] | None:
    """Marktimplizierte Staerke je Team-ID fuer eine Liga, oder None ohne Quoten."""
    doc = odds_doc if odds_doc is not None else load_odds()
    entry = ((doc.get("leagues") or {}).get(liga)) if doc else None
    if not entry or not fixtures:
        return None
    markets = markets_from_entry(entry)
    if not markets:
        return None
    base, hfa = goal_params(liga, matches)
    return fit_strength(markets, fixtures, base, hfa, **kw)


def _cli() -> None:
    doc = load_odds()
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    season = sys.argv[1] if len(sys.argv) > 1 else "2026/2027"
    for liga, entry in (doc.get("leagues") or {}).items():
        fx = matches[(matches.season == season)
                     & (matches.league == LEAGUE_NAME.get(liga, liga))]
        theta = strength_for(liga, list(zip(fx.home_team, fx.away_team)), doc, matches)
        if theta is None:
            print(f"Liga {liga}: keine Quoten oder kein Spielplan fuer {season}")
            continue
        names = entry.get("names", {})
        markets = markets_from_entry(entry)
        print(f"\nLiga {liga}, {season} - {entry.get('market', '?')}")

        # Gegenprobe: reproduziert der gefundene Vektor die Quoten wieder? Das
        # ist die einzige Selbstkontrolle, die ohne historische Saisonquoten zu
        # haben ist.
        ids = sorted(theta)
        pos = {t: i for i, t in enumerate(ids)}
        paare = [(pos[h], pos[a]) for h, a in zip(fx.home_team, fx.away_team)
                 if h in pos and a in pos]
        hi = np.array([h for h, _ in paare], dtype=np.intp)
        ai = np.array([a for _, a in paare], dtype=np.intp)
        base, hfa = goal_params(liga, matches)
        sim = place_probs(np.array([theta[t] for t in ids]), hi, ai, base, hfa,
                          [m["spots"] for m in markets], 40000,
                          np.random.default_rng(11))

        for m in markets:
            print(f"\n  Markt '{m['name']}' ({m['spots']} Plaetze)")
            print(f"    {'Team':26s} {'P(Markt)':>9s} {'simuliert':>10s} {'Staerke':>8s}")
            for t in sorted(m["probs"], key=lambda x: -theta[x]):
                print(f"    {names.get(t, t):26s} {m['probs'][t]*100:8.1f}% "
                      f"{sim[m['spots']][pos[t]]*100:9.1f}% {theta[t]:+8.3f}")


if __name__ == "__main__":
    _cli()
