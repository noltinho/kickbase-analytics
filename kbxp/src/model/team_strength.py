"""Gegnerbereinigte Teamstaerke aus den zugelassenen Punkten.

Die Seiten haben die Staerke eines Teams bisher als zerfallsgewichteten
Mittelwert der zugelassenen Punkte gemessen. Diese Groesse vermengt zwei Dinge:
wie schwach die eigene Abwehr ist, und wie stark die Angriffe waren, auf die man
zufaellig getroffen ist. Wer frueh in der Saison die drei staerksten Kader
erwischt, sieht defensiv schlecht aus, ohne es zu sein.

Hier werden beide Groessen gleichzeitig geschaetzt:

    zugelassene Punkte(Team i gegen Gegner j) = mu + def_i + att_j + hfa*heim

Ridge-regularisiert, weil 2*18 Parameter aus neun Spielen je Spieltag sonst
ueberanpassen. Belegt im Walk-forward ueber 2013/14-2025/26 (10.659 Team-Spiele,
Modell je Spieltag nur aus vorangegangenen Spieltagen):

    Verfahren                                    rho      R2
    heutiger Score (-3..+3)                    0,206       -
    heutiger stetiger Mittelwert               0,273   -0,04
    bereinigt, nur laufende Saison             0,436    0,217
    bereinigt, Vorsaison mit Gewicht 0,3       0,464    0,243

Die Reichweite wurde durchgetestet: kurze Fenster verlieren deutlich (letzte 5
Spieltage rho 0,366, letzte 10 rho 0,407, letzte 17 rho 0,427), die ganze Saison
mit sanftem Zerfall gewinnt. Den groessten Einzelbeitrag liefert die Vorsaison,
vor allem bis Spieltag 8 (rho 0,359 -> 0,462).

Ein Boosting-Stack ueber alle Reichweiten bringt nichts (rho 0,408 gegen 0,411
linear, gleicher Testsplit) - der Zusammenhang ist additiv, und genau das
schaetzt die Ridge direkt.

Zwei Dinge sind seither dazugekommen, beide in modell-befunde.md ausgefuehrt:

  · ``--backtest`` misst nur noch die Spieltage **vor der Winterpause** (dort ist
    in den Kickbase-Ligen Reset). Die Zahlen oben gelten fuer die ganze Saison;
    im Vorpausen-Fenster sind es rho 0,435 und R2 0,223. Zusaetzlich wird die
    paarweise Treffsicherheit ausgewiesen und nach Herkunft der Teams
    aufgeschluesselt - gemessen wird an **rho**, nicht am R2, weil die Seite eine
    Rangfolge zeigt und keine Punktprognose.

  · ``fit_ratings`` nimmt einen Prior an (siehe dort). Er traegt die
    Marktquoten aus data/season_odds.json und schliesst damit die Luecke bei
    Auf- und Absteigern, wo das Modell sonst fast blind ist (R2 0,061 gegen
    0,239 bei durchlaufenden Teams).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model import season_odds  # noqa: E402
from src.paths import LEGACY_DIR, PROCESSED  # noqa: E402

# Aus dem Reichweiten-Experiment; beide Optima sind flach, eine Feinjustage je
# Liga oder Saison bringt nichts.
DECAY = 0.98        # je Spieltag Abstand innerhalb der laufenden Saison
PREV_WEIGHT = 0.30  # Gewicht jedes Vorsaison-Spiels gegenueber einem aktuellen
LAMBDA = 8.0        # Ridge-Strafe; ab 30 bricht die Guete ein

# Wie fest ein Prior gegen die Daten gehalten wird. LAMBDA taugt dafuer nicht:
# es ist gegen null geeicht, wo Schrumpfen immer hilft, und laesst mit 8 rund
# ein Neuntel jeder Beobachtung durch. Eine einzelne Zelle streut aber mit
# sd = 565 Punkten - nach einem Spieltag verschoebe das ein Team um ueber 100
# Punkte und wuerfe die Marktrangfolge um.
#
# Der Wert ist eine Aussage darueber, wie genau der Markt eine Teamstaerke
# trifft: lambda = (sd der Zelle / Genauigkeit des Priors)^2. Mit +/-100 Punkten
# - etwa ein Fuenftel der Spannweite einer Liga - ergibt das (565/100)^2 = 32.
# Ein Abschmelzen ueber die Saison braucht es nicht, die Ridge regelt das von
# selbst: mit jeder gespielten Partie waechst das Gewicht der Datenseite.
PRIOR_LAMBDA = 32.0
N_CLASSES = 7       # -3 .. +3
MIN_ROWS = 9        # ein voller Spieltag, bevor die laufende Saison zaehlt
PRIOR_GAMES = 6     # ab so vielen eigenen Spielen zaehlt ein Aufsteiger halb aus sich selbst

# Kickbase-Punkte je Tor Ueberlegenheit pro Spiel, aus 324 Teamsaisons gemessen
# (r = 0,98). Damit laesst sich eine Marktstaerke von der Tor- auf die
# Punkteskala bringen, ohne den Massstab ueber Standardabweichungen zu raten.
POINTS_PER_GOAL = {"1": 520.3, "2": 507.9}

# Streuung der Teamstaerke (att - def) innerhalb einer Liga, ueber dieselben 324
# Teamsaisons gemessen. Gebraucht wird sie, weil Platzierungsquoten die
# *Rangfolge* gut bestimmen, die *Spreizung* aber nur schwach: wie oft ein Team
# unter die ersten k kommt, haengt fast nur davon ab, wie es zu den anderen steht.
# Ungerechnet liefert die Simulation eine sd von 92 statt 256 - die halbe
# Spreizung. Auf einer Seite, die Klassen von -3 bis +3 zeigt, faellt das sofort
# auf: der Heimvorteil waere dann 2,1 Klassen wert und damit mehr als der
# Unterschied zwischen zwei mittleren Gegnern.
STRENGTH_SD = {"1": 373.0, "2": 256.0}


# ---------------------------------------------------------------------------
# Zugelassene Punkte je Team und Spiel
# ---------------------------------------------------------------------------

def conceded_from_legacy(players_file: Path, matchdays_file: Path) -> pd.DataFrame:
    """Team-Spiele aus den Dateien, die auch die HTML-Seiten laden.

    Rueckgabe: matchday, team, opp, at_home, conceded - wobei ``conceded`` die
    Summe der Punkte aller gegnerischen Spieler ist, also genau die Groesse, die
    matchup.html als "zugelassene Punkte" zeigt.
    """
    with open(players_file, encoding="utf-8") as f:
        data = json.load(f)
    with open(matchdays_file, encoding="utf-8") as f:
        plan = json.load(f)

    # Nur abgeschlossene Partien. Ohne diesen Filter zaehlen die noch leeren
    # Spieltage als echte Nullen mit - derselbe Fehler, den playedDays() in
    # common.js abfaengt.
    fixtures: dict[tuple[str, int], tuple[str, bool]] = {}
    for md in plan["matchdays"]:
        for m in md["matches"]:
            if m.get("st") != 2:
                continue
            fixtures[(str(m["t1"]), m["day"])] = (str(m["t2"]), True)
            fixtures[(str(m["t2"]), m["day"])] = (str(m["t1"]), False)

    points: dict[tuple[str, int], float] = {}
    for player in data.get("players", []):
        for e in player.get("performance", []):
            key = (str(e["opponent"]), e["day"])
            if key not in fixtures:
                continue
            points[key] = points.get(key, 0.0) + (e.get("points") or 0)

    rows = []
    for (team, day), (opp, at_home) in fixtures.items():
        if (team, day) not in points:
            continue
        rows.append(dict(matchday=day, team=team, opp=opp,
                         at_home=at_home, conceded=points[(team, day)]))
    cols = ["matchday", "team", "opp", "at_home", "conceded"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["matchday", "team"]).reset_index(drop=True)


def conceded_from_panel(panel: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Dasselbe aus dem rekonstruierten Panel - Grundlage des Backtests."""
    p = panel.copy()
    p["points"] = p["points"].fillna(0.0)
    p["home"] = p["home"].astype(bool)
    g = (p.groupby(["season", "league", "matchday", "opponent_id", "home"], as_index=False)["points"]
           .sum().rename(columns={"opponent_id": "team", "points": "conceded"}))
    # home des punktenden Spielers -> das kassierende Team spielte auswaerts
    g["at_home"] = ~g["home"]
    d = g.groupby(["season", "league", "matchday", "team"], as_index=False).agg(
        conceded=("conceded", "sum"), at_home=("at_home", "first"))

    a = matches[["season", "league", "matchday", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opp"})
    b = matches[["season", "league", "matchday", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp"})
    return d.merge(pd.concat([a, b]), on=["season", "league", "matchday", "team"], how="left")


# ---------------------------------------------------------------------------
# Ridge auf Team-Dummies
# ---------------------------------------------------------------------------

class Ratings:
    """mu + def_i + att_j + hfa*heim, samt der Teams, die geschaetzt wurden."""

    __slots__ = ("mu", "defense", "attack", "hfa")

    def __init__(self, mu: float, defense: dict, attack: dict, hfa: float):
        self.mu, self.defense, self.attack, self.hfa = mu, defense, attack, hfa

    def predict(self, team: str, opp: str, at_home: bool) -> float:
        """Erwartete zugelassene Punkte von ``team`` gegen ``opp``."""
        return (self.mu + self.defense.get(team, 0.0) + self.attack.get(opp, 0.0)
                + (self.hfa if at_home else 0.0))

    def teams(self) -> list[str]:
        return sorted(self.defense)


def fit_ratings(rows: pd.DataFrame, weights: np.ndarray | None = None,
                lam: float = LAMBDA,
                prior: dict[str, tuple[float, float]] | None = None,
                prior_lam: float = PRIOR_LAMBDA) -> Ratings:
    """Gewichtete Ridge ueber Team-Dummies.

    Abwehr- und Angriffswerte sind nur bis auf eine Konstante bestimmt (jede
    Verschiebung der einen laesst sich in der anderen ausgleichen). Die
    Ridge-Strafe zieht beide Bloecke gegen null und legt die Zerlegung damit
    fest; ``predict`` ist von der Wahl ohnehin unabhaengig.

    ``prior`` verschiebt das Ziel der Strafe fuer die betroffenen Koeffizienten:
    {team: (def, att)}. Wo ein Prior vorliegt, lautet sie

        (lam + prior_lam) * ||beta - beta_0||^2

    und wo keiner vorliegt wie bisher ``lam * ||beta||^2``. Entscheidend ist,
    dass die Strafe dort **vollstaendig** auf beta_0 zeigt und nicht daneben noch
    gegen null zieht: sonst landet ein Team ohne eigene Zeilen nicht bei seinem
    Prior, sondern bei lam/(lam+prior_lam) davon - mit 8 und 32 also bei 80 %.
    Genau diese stille Stauchung um ein Fuenftel hat die zweite Liga zu eng
    gemacht (Staerke-sd 206 statt der gemessenen 256).

    Dass ``prior_lam`` deutlich ueber ``lam`` liegt, ist ebenfalls kein Zufall:
    ``lam`` ist gegen null geeicht, wo Schrumpfen immer hilft. Mit lam = 8 allein
    kaeme nach einem Spieltag rund ein Neuntel einer Zelle durch - bei sd = 565
    Punkten je Zelle verschoebe ein einziges Spiel ein Team um ueber 100 Punkte.

    Die Gewichtung ueber die Saison regelt sich dann von selbst: ein Team ohne
    eigene Zeilen bekommt exakt seinen Prior, eines mit voller Vorsaison wird von
    den Daten dominiert. Es braucht also keinen Schalter, wer einen Prior bekommt.
    """
    if weights is None:
        weights = np.ones(len(rows))
    teams = sorted(set(rows.team) | set(rows.opp.dropna()))
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    X = np.zeros((len(rows), 2 * n_teams + 1))
    for r, (tm, op, ah) in enumerate(zip(rows.team.values, rows.opp.values, rows.at_home.values)):
        X[r, idx[tm]] = 1.0
        if op in idx:
            X[r, n_teams + idx[op]] = 1.0
        X[r, -1] = 1.0 if ah else 0.0

    y = rows.conceded.values.astype(float)
    mu = float(np.average(y, weights=weights))

    # Je Koeffizient ein eigenes Prior-Gewicht: nur wo ein Prior vorliegt, zieht
    # die zweite Strafe. Der Heimvorteil (letzte Spalte) bekommt nie einen.
    b0 = np.zeros(X.shape[1])
    lp = np.zeros(X.shape[1])
    if prior:
        for t, i in idx.items():
            p = prior.get(t)
            if p is not None:
                b0[i], b0[n_teams + i] = float(p[0]), float(p[1])
                lp[i] = lp[n_teams + i] = prior_lam

    # Gesamtstrafe je Koeffizient, gerichtet auf b0 (dort null, wo kein Prior
    # vorliegt - der Ausdruck faellt dann auf die alte Form zurueck).
    pen = np.full(X.shape[1], lam) + lp
    A = X.T @ (X * weights[:, None]) + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ (weights * (y - mu)) + pen * b0)
    return Ratings(mu,
                   {t: float(beta[i]) for t, i in idx.items()},
                   {t: float(beta[n_teams + i]) for t, i in idx.items()},
                   float(beta[-1]))


def fit_with_history(current: pd.DataFrame, previous: pd.DataFrame | None,
                     decay: float = DECAY, prev_weight: float = PREV_WEIGHT,
                     lam: float = LAMBDA,
                     prior: dict[str, tuple[float, float]] | None = None,
                     prior_lam: float = PRIOR_LAMBDA) -> Ratings:
    """Laufende Saison mit Zerfall, Vorsaison flach gedaempft dazu."""
    has_cur = current is not None and len(current) > 0
    has_prev = previous is not None and len(previous) > 0
    if not has_cur and not has_prev:
        raise ValueError("weder laufende noch vorige Saison verfuegbar")
    if not has_cur:
        return fit_ratings(previous, np.ones(len(previous)), lam, prior, prior_lam)

    w_cur = decay ** (current.matchday.max() - current.matchday.values)
    if not has_prev:
        return fit_ratings(current, w_cur, lam, prior, prior_lam)

    rows = pd.concat([previous, current], ignore_index=True)
    weights = np.concatenate([np.full(len(previous), prev_weight), w_cur])
    return fit_ratings(rows, weights, lam, prior, prior_lam)


# ---------------------------------------------------------------------------
# Klassengrenzen
# ---------------------------------------------------------------------------

def jenks_breaks(values, k: int = N_CLASSES) -> list[float]:
    """Fisher-Jenks per dynamischer Programmierung.

    simple-statistics rechnet im Browser dasselbe. Anders als dort laufen hier
    aber alle Paarungen der Liga in die Grenzen ein (rund 600 Werte statt 18) -
    bei 18 Werten auf 7 Klassen verschiebt schon ein einzelnes Spiel eine ganze
    Klassengrenze.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n <= k:
        return sorted(set(x.tolist()))
    pre = np.concatenate([[0.0], np.cumsum(x)])
    pre2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def sse(i: int, j: int) -> float:
        cnt = j - i
        s = pre[j] - pre[i]
        return float(pre2[j] - pre2[i] - s * s / cnt)

    inf = float("inf")
    cost = np.full((k + 1, n + 1), inf)
    split = np.zeros((k + 1, n + 1), dtype=int)
    cost[0, 0] = 0.0
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            best, best_i = inf, c - 1
            for i in range(c - 1, j):
                if cost[c - 1, i] == inf:
                    continue
                v = cost[c - 1, i] + sse(i, j)
                if v < best:
                    best, best_i = v, i
            cost[c, j], split[c, j] = best, best_i

    breaks = [float(x[-1])]
    j = n
    for c in range(k, 0, -1):
        i = split[c, j]
        breaks.append(float(x[i]))
        j = i
    return list(reversed(breaks))


def quantile_breaks(values, k: int = N_CLASSES) -> list[float]:
    """Gleich besetzte Klassen - jede bekommt rund ein Siebtel der Werte.

    Gegenstueck zu jenks_breaks, gebraucht fuer den Modus "Nur Spielplan". Jenks
    minimiert die Streuung innerhalb der Klassen und setzt seine Grenzen deshalb
    dort, wo die Werte weit auseinanderliegen. Ein Ausreisser bindet damit gleich
    zwei Grenzen: in der Bundesliga 2026/27 isoliert Jenks den FC Bayern und
    wirft die neun schwaechsten Teams gemeinsam in die Klasse +3, obwohl zwischen
    ihnen 103 Punkte liegen - also genau der Unterschied, den die Ansicht zeigen
    soll.

    Ueber 10.461 Zellen gemessen: die groesste Klasse eines Spieltags fasst mit
    Jenks im Mittel 30,1 % der Zellen (schlimmster Fall 50 %), mit Quantilen
    23,8 % (38,9 %). Die Sigma-Spanne ueber die 18 Teams steigt von 9,6 auf 11,9
    Klassenpunkte, die Rangfolge bleibt gleich (rho 0,253 gegen 0,250).
    """
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) <= k:
        return sorted(set(x.tolist()))
    qs = np.quantile(x, np.linspace(0.0, 1.0, k + 1))
    qs[0], qs[-1] = x[0], x[-1]
    return [float(v) for v in qs]


def classify(value: float, breaks: list[float], smin: int = -3) -> int:
    """Wert in -3..+3 einordnen - gleiche Regel wie classify() in common.js."""
    for i in range(len(breaks) - 2, -1, -1):
        if value >= breaks[i]:
            return i - 3
    return smin


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(conceded: pd.DataFrame, decay: float = DECAY,
             prev_weight: float = PREV_WEIGHT, lam: float = LAMBDA) -> pd.DataFrame:
    """Walk-forward: je Spieltag nur aus vorangegangenen Spieltagen schaetzen.

    Erwartet die Spalten aus ``conceded_from_panel``. Liefert je Team-Spiel den
    Ist-Wert und die Vorhersage - ohne jeden Blick in die Zukunft, also genau
    die Lage, in der die Seite benutzt wird.
    """
    seasons = sorted(conceded.season.unique())
    prev_of = {s: seasons[i - 1] for i, s in enumerate(seasons) if i > 0}
    by_sl = {k: v.sort_values("matchday") for k, v in conceded.groupby(["season", "league"])}

    out = []
    for (season, league), hist in by_sl.items():
        prev = by_sl.get((prev_of.get(season), league))
        for day in range(2, int(hist.matchday.max()) + 1):
            past = hist[hist.matchday < day]
            today = hist[hist.matchday == day]
            if len(today) == 0 or (len(past) < MIN_ROWS and prev is None):
                continue
            model = fit_with_history(past, prev, decay, prev_weight, lam)
            for row in today.itertuples(index=False):
                out.append(dict(season=season, league=league, matchday=day,
                                team=row.team, opp=row.opp, at_home=bool(row.at_home),
                                actual=float(row.conceded),
                                predicted=model.predict(row.team, row.opp, bool(row.at_home))))
    return pd.DataFrame(out)


def pairwise_accuracy(bt: pd.DataFrame, mask: np.ndarray | None = None) -> float:
    """Zwei Zellen desselben Spieltags: liegt die hoeher bewertete wirklich hoeher?

    Die Seite zeigt Klassen von -3 bis +3, also eine Rangfolge und keine
    Punktprognose. Ein Gewinn im R2, der hier nicht ankommt, ist auf der Seite
    unsichtbar - gemessen an einem zweiten Ridge-Kanal auf der Torskala, der das
    R2 von 0,240 auf 0,245 hob und diese Zahl exakt unveraendert liess.

    ``mask`` waehlt eine Teilmenge aus, ohne den Spieltag zu zerreissen: gezaehlt
    werden alle Paare, an denen mindestens eine markierte Zelle beteiligt ist.
    Wer stattdessen die Teilmenge herausfiltert und fuer sich vergleicht, misst
    Unsinn - von drei Absteigern je Spieltag bleibt oft kein einziges Paar uebrig.
    """
    hit = tot = 0
    idx = np.arange(len(bt))
    for _, g in bt.assign(_i=idx).groupby(["season", "league", "matchday"]):
        p, y = g.predicted.values, g.actual.values
        dp = p[:, None] - p[None, :]
        dy = y[:, None] - y[None, :]
        m = (np.triu(np.ones_like(dp, dtype=bool), 1)
             & (np.abs(dp) > 1e-9) & (np.abs(dy) > 1e-9))
        if mask is not None:
            sel = mask[g["_i"].values]
            m &= (sel[:, None] | sel[None, :])
        hit += int(((dp * dy) > 0)[m].sum())
        tot += int(m.sum())
    return hit / tot if tot else float("nan")


def score_metrics(bt: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    y = bt.actual.values
    p = bt.predicted.values
    return {
        "n": int(len(bt)),
        "spearman": float(spearmanr(p, y).statistic),
        "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pairwise": pairwise_accuracy(bt),
    }


# ---------------------------------------------------------------------------
# Bewertungsfenster: die Spiele vor der Winterpause
# ---------------------------------------------------------------------------

def load_splits() -> dict[tuple[str, str], int]:
    """Letzter Spieltag vor der Winterpause, je Saison und Liga.

    In den Kickbase-Ligen ist zur Winterpause Reset - die Rueckrunde ist ein
    neuer Wettbewerb. Bewertet wird deshalb nur, was davor liegt. Das ist keine
    feste Zahl: Bundesliga 13-17, 2. Bundesliga 16-18; 2026/27 sind es 14 und 16.

    Die Datei deckt alle 20 Saison/Liga-Paare ab, die es in den Daten gibt (die
    2. Bundesliga beginnt erst 2021/22). Sie aus den Terminluecken neu abzuleiten
    ist moeglich und trifft 19 von 20 - aber 2019/20 faende man so die
    Corona-Unterbrechung nach Spieltag 25 statt der Winterpause nach 17.
    """
    path = PROCESSED / "season_splits.parquet"
    if not path.exists():
        return {}
    s = pd.read_parquet(path)
    return {(r.season, r.league): int(r.split_after) for r in s.itertuples(index=False)}


def before_break(df: pd.DataFrame, splits: dict | None = None) -> pd.DataFrame:
    """Zeilen auf die Spieltage vor der Winterpause einschraenken."""
    if splits is None:
        splits = load_splits()
    if not splits or not len(df):
        return df
    fallback = int(np.median(list(splits.values())))
    limit = np.array([splits.get((s, lg), fallback)
                      for s, lg in zip(df.season, df.league)])
    return df[df.matchday.values <= limit]


def herkunft(conceded: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """Woher ein Team kommt: gleiche Liga, aufgestiegen, abgestiegen, oder neu.

    Der Pauschalfaktor in fetch.py behandelt alle Absteiger gleich. Gemessen
    unterscheiden sie sich aber stark, und an Spieltag 1-8 ist das Modell dort
    fast blind (R2 0,066 gegen 0,243 bei durchlaufenden Teams) - deshalb wird die
    Guete nach dieser Einteilung getrennt ausgewiesen.
    """
    seasons = sorted(conceded.season.unique())
    prev_of = {s: seasons[i - 1] for i, s in enumerate(seasons) if i > 0}
    where = {}
    for (s, lg), g in conceded.groupby(["season", "league"]):
        for t in g.team.unique():
            where[(s, t)] = lg

    out = {}
    for (s, lg), g in conceded.groupby(["season", "league"]):
        ps = prev_of.get(s)
        for t in g.team.unique():
            if ps is None:
                out[(s, lg, t)] = "unbekannt"
                continue
            prev_lg = where.get((ps, t))
            if prev_lg is None:
                out[(s, lg, t)] = "neu"
            elif prev_lg == lg:
                out[(s, lg, t)] = "gleich"
            else:
                out[(s, lg, t)] = "auf" if lg == "Bundesliga" else "ab"
    return out


# ---------------------------------------------------------------------------
# Export nach data/ratings.json
# ---------------------------------------------------------------------------

def _manifest() -> dict:
    with open(LEGACY_DIR / "seasons.json", encoding="utf-8") as f:
        return json.load(f)


def _suffix(manifest: dict, key: str) -> str:
    for s in manifest["seasons"]:
        if s["key"] == key:
            return s["suffix"]
    return ""


def _carryover() -> dict | None:
    path = LEGACY_DIR / "carryover.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _season_rows(liga: str, suffix: str) -> pd.DataFrame | None:
    players = LEGACY_DIR / f"data_{liga}{suffix}.json"
    plan = LEGACY_DIR / f"matchdays_bl{liga}{suffix}.json"
    if not players.exists() or not plan.exists():
        return None
    rows = conceded_from_legacy(players, plan)
    return rows if len(rows) else None


def _translate(rating: Ratings, team: str, factors: dict, target_mu: float) -> tuple[float, float]:
    """Rating aus einer anderen Liga auf das Niveau der Ziel-Liga heben.

    Die Faktoren stammen aus data/carryover.json und sind dort aus vergangenen
    Auf- und Abstiegen gelernt. Sie wirken auf das Niveau, nicht auf die
    Abweichung - deshalb erst mu addieren, dann skalieren, dann wieder abziehen.
    """
    lvl_def = (rating.mu + rating.defense.get(team, 0.0)) * factors["conceded"]
    lvl_att = (rating.mu + rating.attack.get(team, 0.0)) * factors["scored"]
    return lvl_def - target_mu, lvl_att - target_mu


def _odds_prior(base: dict[str, tuple[float, float]], theta: dict[str, float],
                liga: str) -> dict[str, tuple[float, float]]:
    """Marktstaerke (Tor-Skala) auf die def/att-Skala des Modells abbilden.

    Aus einer Platzierungsquote laesst sich genau ein Skalar je Team lesen. Er
    legt die **Differenz** att-def fest - wie viel mehr die eigenen Spieler
    punkten, als man zulaesst. Die **Summe** att+def, also wie punktereich die
    Spiele dieses Teams ueberhaupt sind, steckt nicht in der Quote und bleibt
    beim bisherigen Prior stehen.

    Die Umrechnung der Einheiten ist gemessen, nicht geschaetzt: ueber 324
    Teamsaisons haengt die Punktestaerke eines Teams fast vollstaendig an seiner
    Torueberlegenheit je Spiel (r = 0,97 in Liga 2, 0,98 in Liga 1). Die Steigung
    dieses Zusammenhangs steht in POINTS_PER_GOAL; die rohe Tordifferenz statt
    der gegnerbereinigten liefert mit 474 bis 486 dieselbe Groessenordnung.

    Danach wird die Spreizung auf STRENGTH_SD normiert. Das ist kein zweites
    Raten, sondern die Korrektur einer bekannten Schwaeche der Quelle: eine
    Platzierungsquote sagt vor allem, wie ein Team zu den anderen steht, und legt
    den Abstand nur lose fest. Die Rangfolge und die relativen Abstaende, die der
    Markt liefert, bleiben dabei unveraendert.
    """
    rate, _ = season_odds.LEAGUE_GOALS.get(liga, (0.40, 0.10))
    per_goal = POINTS_PER_GOAL.get(liga, 518.0)

    target = {t: (np.exp(rate + th) - np.exp(rate - th)) * per_goal
              for t, th in theta.items() if t in base}
    if len(target) < 4:
        return base
    sd = float(np.std(list(target.values())))
    want = STRENGTH_SD.get(liga)
    if want and sd > 1e-9:
        target = {t: v / sd * want for t, v in target.items()}

    out = dict(base)
    for t, v in target.items():
        d0, a0 = base[t]
        total = a0 + d0
        out[t] = ((total - v) / 2.0, (total + v) / 2.0)
    return out


def build_league(liga: str, manifest: dict, carry: dict | None) -> dict | None:
    """Ratings einer Liga: laufende Saison, Vorsaison, Auf- und Absteiger."""
    cur_suffix = _suffix(manifest, manifest["current"])
    older = sorted((s["key"] for s in manifest["seasons"] if s["key"] != manifest["current"]),
                   reverse=True)
    prev_key = older[0] if older else None

    current = _season_rows(liga, cur_suffix)
    previous = _season_rows(liga, _suffix(manifest, prev_key)) if prev_key else None
    if current is None and previous is None:
        return None

    model = fit_with_history(current if current is not None else pd.DataFrame(), previous)

    # Teams der laufenden Saison - auch die ohne einen einzigen Datenpunkt.
    with open(LEGACY_DIR / f"data_{liga}{cur_suffix}.json", encoding="utf-8") as f:
        teams = json.load(f)["teams"]

    factors = (carry or {}).get("factors", {})
    league_meta = ((carry or {}).get("leagues", {}) or {}).get(liga, {}) or {}
    src_meta = league_meta.get("teams", {})
    other_models: dict[str, Ratings] = {}

    in_prev = set(previous.team) if previous is not None else set()
    played = (current.groupby("team").size().to_dict()
              if current is not None and len(current) else {})

    # Basisprior je Team: eigene Vorsaison, sonst der uebersetzte Carryover.
    base_prior: dict[str, tuple[float, float]] = {}
    srcs: dict[str, str] = {}
    for tid in teams:
        src = (src_meta.get(tid) or {}).get("src", "same" if tid in in_prev else "new")
        srcs[tid] = src
        if tid in in_prev:
            base_prior[tid] = (model.defense.get(tid, 0.0), model.attack.get(tid, 0.0))
            continue

        # Auf- und Absteiger: die Ridge kennt sie nur aus den wenigen bisher
        # gespielten Partien. Die uebersetzte Vorsaison der anderen Liga ist die
        # bessere Auskunft, solange die laufende Saison kaum Belege liefert.
        fac = factors.get(src) or factors.get("same") or {"conceded": 1.0, "scored": 1.0}
        prior_d = model.mu * (fac["conceded"] - 1.0)
        prior_a = model.mu * (fac["scored"] - 1.0)

        src_league = (src_meta.get(tid) or {}).get("src_league")
        if src_league and src_league != liga and prev_key:
            if src_league not in other_models:
                rows = _season_rows(src_league, _suffix(manifest, prev_key))
                other_models[src_league] = fit_ratings(rows) if rows is not None else None
            foreign = other_models.get(src_league)
            if foreign is not None and tid in foreign.defense:
                prior_d, prior_a = _translate(foreign, tid, fac, model.mu)
        base_prior[tid] = (prior_d, prior_a)

    # Marktquoten, falls vorhanden: sie ersetzen den Pauschalfaktor als Auskunft
    # ueber die Staerke - fuer alle Teams, nicht nur fuer Ligawechsler. Ein
    # Schalter waere ueberfluessig, weil die Ridge selbst regelt, wie viel der
    # Prior gegen die vorhandenen Daten noch gilt.
    theta = season_odds.strength_for(
        liga, season_odds.fixtures_from_plan(LEGACY_DIR / f"matchdays_bl{liga}{cur_suffix}.json"))
    prior = _odds_prior(base_prior, theta, liga) if theta else None
    if prior:
        model = fit_with_history(current if current is not None else pd.DataFrame(),
                                 previous, prior=prior)

    out_teams = {}
    for tid in teams:
        n = int(played.get(tid, 0))

        if tid not in model.defense:
            # Das Team kommt in keiner Zeile vor und hat deshalb gar keinen
            # Koeffizienten: ein Aufsteiger vor dem ersten Spieltag der Saison.
            # Die Ridge kann hier nichts beitragen, es gilt unmittelbar der Prior.
            d, a = (prior or base_prior)[tid]
            out_teams[tid] = {"def": round(d, 2), "att": round(a, 2),
                              "src": srcs[tid], "n": n}
            continue

        fitted_d, fitted_a = model.defense[tid], model.attack[tid]
        if prior or tid in in_prev:
            # Mit Prior steckt das Vorwissen bereits in der Ridge - ein Team ohne
            # eigene Zeilen bekommt exakt seinen Prior zurueck, ein Team mit
            # voller Vorsaison wird von den Daten dominiert. Nochmals mischen
            # wuerde dasselbe Wissen doppelt zaehlen.
            d, a = fitted_d, fitted_a
        else:
            # Ohne Quoten der bisherige Weg: nach PRIOR_GAMES eigenen Spielen
            # zaehlen Vorgeschichte und laufende Saison gleich viel.
            prior_d, prior_a = base_prior[tid]
            w = n / (n + PRIOR_GAMES)
            d = w * fitted_d + (1 - w) * prior_d
            a = w * fitted_a + (1 - w) * prior_a
        out_teams[tid] = {"def": round(d, 2), "att": round(a, 2),
                          "src": srcs[tid], "n": n}

    # Klassengrenzen ueber alle moeglichen Paarungen der Liga, nicht ueber 18
    # Teamwerte - dadurch haengt keine Grenze an einem einzelnen Spiel.
    ids = list(out_teams)
    values = [model.mu + out_teams[i]["def"] + out_teams[j]["att"] + (model.hfa if h else 0.0)
              for i in ids for j in ids if i != j for h in (True, False)]
    breaks = jenks_breaks(values, N_CLASSES)

    # Zweite Grenzen fuer den Modus "Nur Spielplan". Dort steht statt des eigenen
    # Angriffs der Ligaschnitt, die Werte decken also nur einen Ausschnitt der
    # Paarungsverteilung ab. Mit den Paarungsgrenzen gemessen kamen +3 in 0,6 %
    # und -3 in 3,5 % der Zellen vor - zwei der sieben Farben waren praktisch
    # tot, und gerade dieser Modus soll leichte und schwere Spielplaene zeigen.
    #
    # Quantile statt Jenks, weil dieser Modus eine Rangfrage beantwortet ("wer
    # hat den leichteren Spielplan") und nicht nach absoluten Punktgrenzen fragt.
    # Jenks verbraucht seine Grenzen an Ausreissern; die Begruendung samt Zahlen
    # steht bei quantile_breaks.
    mean_att = sum(t["att"] for t in out_teams.values()) / len(out_teams)
    fixture_values = [model.mu + out_teams[j]["def"] + mean_att + (model.hfa if h else 0.0)
                      for j in ids for h in (True, False)]
    breaks_fixture = quantile_breaks(fixture_values, N_CLASSES)

    return {
        "mu": round(model.mu, 2),
        "hfa": round(model.hfa, 2),
        "breaks_fixture": [round(b, 2) for b in breaks_fixture],
        "breaks": [round(b, 2) for b in breaks],
        "teams": out_teams,
    }


def export(path: Path | None = None) -> dict:
    manifest = _manifest()
    carry = _carryover()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "ridge-v1",
        "decay": DECAY,
        "prev_weight": PREV_WEIGHT,
        "lambda": LAMBDA,
        # Nur zur Nachvollziehbarkeit - das Frontend liest davon nichts. Wer
        # spaeter eine Datei in der Hand hat, soll sehen, mit welchen Parametern
        # sie entstand; prior_lambda entscheidet mit, wie fest die Marktquoten
        # gegen die Spieldaten gehalten wurden.
        "prior_lambda": PRIOR_LAMBDA,
        "leagues": {},
    }
    for liga in ("1", "2"):
        league = build_league(liga, manifest, carry)
        if league is not None:
            payload["leagues"][liga] = league

    target = path or (LEGACY_DIR / "ratings.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload


def _line(label: str, bt: pd.DataFrame, full: pd.DataFrame | None = None,
          mask: np.ndarray | None = None) -> None:
    """Eine Auswertungszeile. ``full``/``mask`` fuer Teilmengen, damit die
    paarweise Treffsicherheit gegen den ganzen Spieltag gemessen wird."""
    if len(bt) < 30:
        return
    from scipy.stats import spearmanr
    y, p = bt.actual.values, bt.predicted.values
    rho = float(spearmanr(p, y).statistic)
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    pw = pairwise_accuracy(full if full is not None else bt, mask)
    print(f"  {label:22s} n={len(bt):6d}  rho={rho:6.3f}  "
          f"R2={r2:6.3f}  paarweise={pw*100:5.1f}%")


def _cli() -> None:
    if "--backtest" in sys.argv:
        panel = pd.read_parquet(PROCESSED / "panel.parquet")
        matches = pd.read_parquet(PROCESSED / "matches.parquet")
        panel = panel[panel.season != "2026/2027"]
        matches = matches[matches.season != "2026/2027"]
        conceded = conceded_from_panel(panel, matches)
        bt = backtest(conceded)

        # Bewertet wird nur vor der Winterpause - danach ist in den
        # Kickbase-Ligen Reset, die Rueckrunde ist ein neuer Wettbewerb.
        pre = before_break(bt)
        print(f"Vor der Winterpause ({len(pre)} von {len(bt)} Zellen)")
        _line("gesamt", pre)
        _line("Spieltag 2-8", pre[pre.matchday <= 8])

        hk = herkunft(conceded)
        pre = pre.assign(herkunft=[hk.get((s, lg, t), "unbekannt") for s, lg, t
                                   in zip(pre.season, pre.league, pre.team)]
                         ).reset_index(drop=True)
        early = pre[pre.matchday <= 8].reset_index(drop=True)
        for titel, frame in (("nach Herkunft des Teams", pre),
                             ("nach Herkunft, nur Spieltag 2-8", early)):
            print(f"\n{titel}")
            for h in ("gleich", "auf", "ab", "neu"):
                sel = (frame.herkunft == h).values
                _line(h, frame[sel], full=frame, mask=sel)
        return

    payload = export()
    for liga, L in payload["leagues"].items():
        print(f"Liga {liga}: mu={L['mu']}  Heimvorteil={L['hfa']}  "
              f"{len(L['teams'])} Teams  Grenzen {L['breaks'][1]}..{L['breaks'][-2]}")
    print(f"geschrieben: {LEGACY_DIR / 'ratings.json'}")


if __name__ == "__main__":
    _cli()
