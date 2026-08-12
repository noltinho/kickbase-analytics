"""Das gegnerbereinigte Team-Rating gegen Daten und gegen sich selbst pruefen.

Drei Sorten Test:
  · Die zugelassenen Punkte, die das Modell fuettert, muessen aus den
    Archivdateien und aus dem rekonstruierten Panel dieselbe Groesse ergeben.
    Verschiebt sich hier etwas, ist jede Kennzahl darunter wertlos.
  · Die Guete im Walk-forward muss die gemessene Marke halten. Der Test ist die
    Bremse gegen stille Verschlechterung durch spaetere Umbauten.
  · Die Zerlegung in Abwehr und Angriff ist nur bis auf eine Konstante
    bestimmt - die Vorhersage darf davon nicht abhaengen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import LEGACY_DIR, PROCESSED  # noqa: E402
from src.model.team_strength import (  # noqa: E402
    DECAY, LAMBDA, PREV_WEIGHT, PRIOR_LAMBDA, backtest, before_break, classify,
    conceded_from_legacy, conceded_from_panel, fit_ratings, herkunft,
    jenks_breaks, load_splits, pairwise_accuracy, score_metrics,
)
from src.model.season_odds import (  # noqa: E402
    fit_strength, implied_probs, place_probs,
)

ARCHIVE_SEASON = "2025/2026"
ARCHIVE = {"2": ("data_2_2526.json", "matchdays_bl2_2526.json", "2. Bundesliga"),
           "1": ("data_1_2526.json", "matchdays_bl1_2526.json", "Bundesliga")}


@pytest.fixture(scope="module")
def panel_conceded() -> pd.DataFrame:
    if not (PROCESSED / "panel.parquet").exists():
        pytest.skip("panel.parquet fehlt - erst backfill_history laufen lassen")
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    # 2026/2027 ist zum Zeitpunkt des Backfills noch ungespielt.
    panel = panel[panel.season != "2026/2027"]
    matches = matches[matches.season != "2026/2027"]
    return conceded_from_panel(panel, matches)


@pytest.mark.parametrize("liga", sorted(ARCHIVE))
def test_conceded_agrees_with_archive(panel_conceded: pd.DataFrame, liga: str) -> None:
    players, plan, league = ARCHIVE[liga]
    if not (LEGACY_DIR / players).exists() or not (LEGACY_DIR / plan).exists():
        pytest.skip(f"Archivdateien fuer Liga {liga} fehlen")

    legacy = conceded_from_legacy(LEGACY_DIR / players, LEGACY_DIR / plan)
    ref = panel_conceded[(panel_conceded.season == ARCHIVE_SEASON)
                         & (panel_conceded.league == league)]
    if ref.empty or legacy.empty:
        pytest.skip("keine gemeinsame Saison zum Vergleichen")

    merged = legacy.merge(ref, on=["matchday", "team"], suffixes=("_arch", "_panel"))
    assert len(merged) > 500, f"nur {len(merged)} gemeinsame Team-Spiele"

    # Heim/Auswaerts muss exakt stimmen - daran haengt der Heimvorteil.
    assert (merged.at_home_arch == merged.at_home_panel).all(), "Heim/Auswaerts weicht ab"
    assert (merged.opp_arch == merged.opp_panel).all(), "Gegnerzuordnung weicht ab"

    # Exakt gleich, keine Toleranz. Beide Wege bilden dieselbe Menge Spieler ab,
    # also muss jede Summe stimmen. Drei Ursachen haben das frueher verletzt und
    # sind alle behoben:
    #   · 15 Spieler fehlten im Panel, weil das Backfill-Protokoll sie als
    #     erledigt meldete, bevor ihre Zeilen geschrieben waren (load_done()).
    #   · Den Archivdateien fehlte die Ersatzbank auf allen 612 Partien, also
    #     jeder Spieler, der nie in der Startelf stand (archive --force-lineups).
    #   · Wer bei Kickbase keine Position hat, wurde ganz verworfen statt nur
    #     aus der Positionszerlegung herausgehalten (cmd_archive in fetch.py).
    diff = (merged.conceded_arch - merged.conceded_panel).abs()
    abweichend = merged[diff != 0]
    assert len(abweichend) == 0, (
        f"{len(abweichend)} von {len(merged)} Team-Spielen weichen ab, "
        f"groesste {diff.max()}:\n"
        + abweichend.assign(diff=diff[diff != 0]).head(10).to_string())


def test_walkforward_beats_measured_mark(panel_conceded: pd.DataFrame) -> None:
    """Gemessen wurden rho 0,464 und R2 0,243 ueber 10.659 Team-Spiele.

    Die Schranken liegen knapp darunter, damit ein neuer Datenstand sie nicht
    aus Zufall reisst - ein echter Rueckschritt faellt trotzdem auf.
    """
    bt = backtest(panel_conceded)
    m = score_metrics(bt)
    assert m["n"] > 10000, f"nur {m['n']} Team-Spiele im Backtest"
    assert m["spearman"] > 0.44, f"Rangkorrelation auf {m['spearman']:.4f} gefallen"
    assert m["r2"] > 0.22, f"R2 auf {m['r2']:.4f} gefallen"

    # Am Saisonstart traegt die Vorsaison die Vorhersage. Bricht das weg, ist
    # der Uebertrag kaputt - dort ist der Score am wenigsten belastbar.
    early = score_metrics(bt[bt.matchday <= 8])
    assert early["spearman"] > 0.40, f"Saisonstart auf {early['spearman']:.4f} gefallen"


def test_beats_unadjusted_average(panel_conceded: pd.DataFrame) -> None:
    """Gegen die Vergleichslinie, die das Verfahren abloest.

    Der rohe zerfallsgewichtete Mittelwert der zugelassenen Punkte - ohne jede
    Bereinigung um den Gegner - erreicht rho 0,25. Bleibt die Bereinigung nicht
    deutlich darueber, lohnt der Umbau nicht.
    """
    from scipy.stats import spearmanr

    rows = []
    for (season, league), hist in panel_conceded.groupby(["season", "league"]):
        hist = hist.sort_values("matchday")
        for day in range(2, int(hist.matchday.max()) + 1):
            past = hist[hist.matchday < day]
            today = hist[hist.matchday == day]
            if len(past) < 18 or today.empty:
                continue
            w = 0.95 ** (past.matchday.max() - past.matchday.values)
            acc = past.assign(p=past.conceded.values * w, w=w).groupby("team")[["p", "w"]].sum()
            avg = (acc.p / acc.w)
            for row in today.itertuples(index=False):
                rows.append((float(row.conceded), float(avg.get(row.team, avg.mean()))))
    y, pr = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    plain = spearmanr(pr, y).statistic

    adjusted = score_metrics(backtest(panel_conceded))["spearman"]
    assert plain < 0.30, f"Vergleichslinie unerwartet stark: {plain:.4f}"
    assert adjusted > plain + 0.15, (
        f"Bereinigung bringt nur {adjusted - plain:.4f} statt der gemessenen ~0,21")


def test_prediction_invariant_to_split() -> None:
    """Abwehr und Angriff sind nur gemeinsam bestimmt.

    Verschiebt man alle Abwehrwerte um c und alle Angriffswerte um -c, bleibt
    jede Vorhersage gleich. Wer die Zerlegung einzeln interpretiert, muss das
    wissen; ``predict`` darf es nicht stoeren.
    """
    rng = np.random.default_rng(0)
    teams = [str(i) for i in range(6)]
    rows = []
    for day in range(1, 11):
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            rows.append(dict(matchday=day, team=a, opp=b, at_home=True,
                             conceded=float(rng.normal(1000, 200))))
            rows.append(dict(matchday=day, team=b, opp=a, at_home=False,
                             conceded=float(rng.normal(1000, 200))))
    model = fit_ratings(pd.DataFrame(rows))

    before = [model.predict(a, b, True) for a in teams for b in teams if a != b]
    c = 37.5
    model.defense = {k: v + c for k, v in model.defense.items()}
    model.attack = {k: v - c for k, v in model.attack.items()}
    after = [model.predict(a, b, True) for a in teams for b in teams if a != b]
    assert np.allclose(before, after)


def test_jenks_and_classify_agree() -> None:
    """classify() muss die Grenzen so lesen, wie jenks_breaks() sie legt.

    Dieselbe Regel laeuft in common.js - weicht sie hier ab, zeigen Export und
    Seite verschiedene Klassen fuer denselben Wert.
    """
    values = list(np.linspace(600, 1600, 400))
    breaks = jenks_breaks(values, 7)
    assert len(breaks) == 8
    assert breaks == sorted(breaks)

    classes = [classify(v, breaks) for v in values]
    assert min(classes) == -3 and max(classes) == 3
    # Monoton: ein hoeherer Wert darf nie eine niedrigere Klasse bekommen.
    assert all(b >= a for a, b in zip(classes, classes[1:]))
    # Unterhalb der kleinsten Grenze bleibt es bei -3.
    assert classify(breaks[0] - 1, breaks) == -3


def test_export_shape(tmp_path: Path) -> None:
    """data/ratings.json muss alles enthalten, was common.js daraus liest."""
    from src.model.team_strength import export

    if not (LEGACY_DIR / "seasons.json").exists():
        pytest.skip("seasons.json fehlt")
    payload = export(tmp_path / "ratings.json")
    assert payload["leagues"], "keine Liga exportiert"
    assert payload["decay"] == DECAY and payload["lambda"] == LAMBDA
    assert payload["prev_weight"] == PREV_WEIGHT
    assert payload["prior_lambda"] == PRIOR_LAMBDA

    for liga, L in payload["leagues"].items():
        teams = json.load(open(LEGACY_DIR / f"data_{liga}.json", encoding="utf-8"))["teams"]
        assert set(L["teams"]) == set(teams), f"Liga {liga}: Teams weichen ab"
        assert len(L["breaks"]) == 8 and L["breaks"] == sorted(L["breaks"])
        # Eigene Grenzen fuer den Modus "Nur Spielplan" - ohne sie faellt
        # teamSideScore auf die Paarungsgrenzen zurueck, und dort kommen +3 und
        # -3 praktisch nicht vor.
        bf = L["breaks_fixture"]
        assert len(bf) == 8 and bf == sorted(bf), f"Liga {liga}: breaks_fixture"
        # Der Spielplan-Modus variiert nur ueber die Abwehr des Gegners, deckt
        # also einen engeren Bereich ab als alle Paarungen zusammen.
        assert bf[0] >= L["breaks"][0] and bf[-1] <= L["breaks"][-1], (
            f"Liga {liga}: breaks_fixture liegt ausserhalb der Paarungsspanne")
        # Heimteams lassen weniger zu - ein positiver Wert waere ein Vorzeichenfehler.
        assert L["hfa"] < 0, f"Liga {liga}: Heimvorteil {L['hfa']} hat falsches Vorzeichen"
        for tid, t in L["teams"].items():
            assert {"def", "att", "src", "n"} <= set(t), f"Liga {liga}, Team {tid}"
        # Jede Paarung muss in eine der sieben Klassen fallen.
        ids = list(L["teams"])
        for i in ids[:5]:
            for j in ids[:5]:
                if i == j:
                    continue
                v = L["mu"] + L["teams"][j]["def"] + L["teams"][i]["att"]
                assert -3 <= classify(v, L["breaks"]) <= 3


# ---------------------------------------------------------------------------
# Bewertungsfenster, Prior und Marktquoten
# ---------------------------------------------------------------------------

def test_prior_is_a_no_op_when_absent() -> None:
    """Ohne Prior muss exakt dasselbe herauskommen wie vorher.

    Der Prior ist der einzige Eingriff in die Ridge; waere er nicht neutral,
    haetten alle gemessenen Kennzahlen ihre Vergleichsbasis verloren.
    """
    rows = pd.DataFrame([
        dict(matchday=1, team="A", opp="B", at_home=True, conceded=1000.0),
        dict(matchday=1, team="B", opp="A", at_home=False, conceded=1200.0),
        dict(matchday=2, team="A", opp="C", at_home=False, conceded=900.0),
        dict(matchday=2, team="C", opp="A", at_home=True, conceded=1100.0),
    ])
    a, b = fit_ratings(rows), fit_ratings(rows, prior=None)
    assert a.defense == b.defense and a.attack == b.attack and a.hfa == b.hfa


def test_prior_carries_teams_without_own_rows() -> None:
    """Ein Team ohne eigene Zeilen bekommt genau seinen Prior zurueck.

    Das ist der Grund, warum es keinen Schalter braucht, wer einen Prior
    bekommt: wo Daten fehlen, traegt ihn die Ridge unveraendert durch, und wo
    Daten da sind, ueberstimmen sie ihn von selbst.
    """
    rows = pd.DataFrame([
        dict(matchday=1, team="A", opp="B", at_home=True, conceded=1000.0),
        dict(matchday=1, team="B", opp="A", at_home=False, conceded=1000.0),
    ])
    # D kommt in keiner Zeile vor, steht aber im Prior.
    r = fit_ratings(rows, prior={"D": (-150.0, 220.0)})
    assert r.defense.get("D") is None, "D darf ohne Zeilen nicht im Modell auftauchen"

    rows2 = pd.concat([rows, pd.DataFrame([
        dict(matchday=2, team="D", opp="A", at_home=True, conceded=1000.0)])],
        ignore_index=True)
    ohne = fit_ratings(rows2)
    mit = fit_ratings(rows2, prior={"D": (-150.0, 220.0)})
    assert mit.defense["D"] < ohne.defense["D"], "Prior zieht die Abwehr nicht"
    assert mit.attack["D"] > ohne.attack["D"], "Prior zieht den Angriff nicht"


def test_prior_is_not_quietly_shrunk_toward_zero() -> None:
    """Ohne Daten muss genau der Prior herauskommen - nicht ein Anteil davon.

    Zieht die allgemeine Strafe zusaetzlich gegen null, landet ein Team bei
    lam/(lam+prior_lam) seines Priors, mit 8 und 32 also bei 80 %. Das faellt
    nirgends auf, staucht aber die ganze Liga: die zweite Liga kam so auf eine
    Staerke-sd von 206 statt der gemessenen 256, und der Heimvorteil war
    dadurch zwei Klassen wert statt einer.
    """
    rows = pd.DataFrame([
        dict(matchday=1, team="A", opp="B", at_home=True, conceded=1000.0),
        dict(matchday=1, team="B", opp="A", at_home=False, conceded=1000.0),
    ])
    prior = {"A": (-150.0, 220.0), "B": (90.0, -60.0)}
    # Daten praktisch ohne Gewicht: uebrig bleibt allein der Prior.
    r = fit_ratings(rows, np.full(len(rows), 1e-9), prior=prior)
    for team, (d, a) in prior.items():
        assert r.defense[team] == pytest.approx(d, rel=1e-3), f"{team}: Abwehr gestaucht"
        assert r.attack[team] == pytest.approx(a, rel=1e-3), f"{team}: Angriff gestaucht"


def test_before_break_uses_the_split_file() -> None:
    """Bewertet werden nur die Spiele vor der Winterpause.

    In den Kickbase-Ligen ist dort Reset; die Rueckrunde ist ein neuer
    Wettbewerb. Die Grenze ist je Saison und Liga verschieden - 2026/27 sind es
    14 Spieltage in der Bundesliga und 16 in der zweiten.
    """
    splits = load_splits()
    if not splits:
        pytest.skip("season_splits.parquet fehlt")
    assert splits[("2026/2027", "Bundesliga")] == 14
    assert splits[("2026/2027", "2. Bundesliga")] == 16

    df = pd.DataFrame([
        dict(season="2026/2027", league="Bundesliga", matchday=md) for md in (1, 14, 15, 34)])
    kept = before_break(df, splits)
    assert list(kept.matchday) == [1, 14]


def test_pairwise_subset_is_measured_against_the_whole_matchday() -> None:
    """Eine Teilmenge fuer sich zu vergleichen misst Unsinn.

    Von drei Absteigern je Spieltag bleibt oft kein einziges Paar uebrig; die
    Kennzahl ist nur gegenueber den uebrigen Zellen desselben Spieltags definiert.
    """
    # Die ersten drei Zellen sind richtig geordnet, die vierte ist grob falsch
    # bewertet: sie laesst am meisten zu, wird aber am niedrigsten geschaetzt.
    bt = pd.DataFrame([
        dict(season="X", league="L", matchday=1, actual=a, predicted=p)
        for a, p in [(100.0, 1.0), (200.0, 2.0), (300.0, 3.0), (400.0, 0.5)]])
    assert pairwise_accuracy(bt) == pytest.approx(0.5), "3 von 6 Paaren richtig"

    # Nur die schlechte Zelle: alle drei Paare, an denen sie beteiligt ist,
    # sitzen falsch. Wer sie stattdessen herausfiltert, behielte eine einzelne
    # Zeile - und damit gar kein Paar, also keine Kennzahl.
    mask = np.array([False, False, False, True])
    assert pairwise_accuracy(bt, mask) == pytest.approx(0.0)
    assert np.isnan(pairwise_accuracy(bt[mask]))


def test_power_method_shifts_margin_onto_longshots() -> None:
    """Die Marge liegt auf langen Quoten, nicht gleichmaessig verteilt.

    Proportionales Normieren gibt Aussenseitern zu viel und staucht dadurch die
    abgeleiteten Staerken - beim Datenstand 2026/27 haelt es Wolfsburg bei
    47,9 statt 62,8 Prozent.
    """
    # Die rohen 1/Quote muessen ueber der Zahl der Plaetze liegen - sonst gibt
    # es keine Marge zu verteilen und beide Verfahren fallen zusammen.
    odds = {"fav": 1.45, "a": 3.0, "b": 4.0, "c": 6.0, "d": 10.0, "out": 20.0}
    assert sum(1 / q for q in odds.values()) > 1.0
    prop = implied_probs(odds, 1, method="proportional")
    powr = implied_probs(odds, 1, method="power")
    assert sum(prop.values()) == pytest.approx(1.0)
    assert sum(powr.values()) == pytest.approx(1.0)
    assert powr["fav"] > prop["fav"], "Favorit muss steigen"
    assert powr["out"] < prop["out"], "Aussenseiter muss fallen"


def test_odds_inversion_reproduces_the_market() -> None:
    """Die Simulation muss die Quoten treffen, aus denen sie abgeleitet wurde.

    Das ist die einzige Selbstkontrolle, die ohne historische Saisonquoten zu
    haben ist: P(Aufstieg) laesst sich nicht geschlossen invertieren, also muss
    der Rueckweg stimmen.
    """
    teams = [str(i) for i in range(8)]
    fixtures = [(h, a) for h in teams for a in teams if h != a] * 2
    odds = {t: q for t, q in zip(teams, [1.8, 3.0, 4.5, 6.0, 8.0, 12.0, 18.0, 30.0])}
    markets = [{"name": "Aufstieg", "spots": 2, "probs": implied_probs(odds, 2)}]
    theta = fit_strength(markets, fixtures, 0.40, 0.10, n_sims=4000, rounds=70, seed=1)
    assert theta is not None

    order = sorted(theta, key=lambda t: -theta[t])
    assert order == sorted(teams, key=lambda t: odds[t]), "Rangfolge weicht von den Quoten ab"

    idx = {t: i for i, t in enumerate(sorted(theta))}
    hi = np.array([idx[h] for h, _ in fixtures])
    ai = np.array([idx[a] for _, a in fixtures])
    vec = np.array([theta[t] for t in sorted(theta)])
    p = place_probs(vec, hi, ai, 0.40, 0.10, [2], 20000, np.random.default_rng(2))[2]
    target = implied_probs(odds, 2)
    worst = max(abs(p[idx[t]] - target[t]) for t in teams)
    assert worst < 0.02, f"Abweichung zum Markt bis {worst*100:.1f} Prozentpunkte"


def test_herkunft_finds_league_changers() -> None:
    """Auf- und Absteiger muessen als solche erkannt werden.

    Genau fuer sie ist das Modell blind (R2 0,061 gegen 0,239), und nur getrennt
    nach Herkunft wird ein Fortschritt dort ueberhaupt sichtbar.
    """
    con = pd.DataFrame([
        dict(season="2024/2025", league="Bundesliga", team="A", matchday=1),
        dict(season="2024/2025", league="2. Bundesliga", team="B", matchday=1),
        dict(season="2025/2026", league="Bundesliga", team="A", matchday=1),
        dict(season="2025/2026", league="Bundesliga", team="B", matchday=1),
        dict(season="2025/2026", league="2. Bundesliga", team="C", matchday=1),
    ])
    hk = herkunft(con)
    assert hk[("2025/2026", "Bundesliga", "A")] == "gleich"
    assert hk[("2025/2026", "Bundesliga", "B")] == "auf"
    assert hk[("2025/2026", "2. Bundesliga", "C")] == "neu"
