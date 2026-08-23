"""Das fallweise Spielermodell gegen Leakage, gegen Baselines und gegen sich selbst.

Vier Sorten Test:
  · **Leakage.** Die Zielgroesse darf auf keinem Weg in die Merkmale sickern.
    Faellt einer dieser Tests, ist jede Kennzahl darunter wertlos — und zwar
    unauffaellig, weil Leakage die Zahlen *verbessert*.
  · **Struktur.** Fallzuordnung, Positionsquelle und Teamstaerke-Skala. Das
    sind die Stellen, an denen dieses Modell schon einmal still danebenlag:
    Aufsteiger ohne Prognose, Burger als ZDM statt ZM, Bayern bei z 3,04.
  · **Guete im Walk-forward.** Schranken knapp unter den gemessenen Werten,
    immer gegen eine explizite Vergleichslinie — eine Kennzahl ohne
    Vergleichslinie sagt nichts.
  · **Export.** Was die Seite liest, muss vollstaendig und in sich stimmig sein.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.paths import LEGACY_DIR, MANUAL, PROCESSED  # noqa: E402
from src.model.player_avg import (  # noqa: E402
    ERSTE_ZIELSAISON, FENSTER, LIGA1, LIGA2, MERKMALE, MIN_STARTS,
    PFLICHT, RHO, SPITZE_MW, STARTER_JE_TEAM, TEIL, TEILE, TEIL_REF,
    Daten, _corr, anpassen, backtest, export, historie, kandidaten,
    merkmalszeilen, position_von, positions_quelle, starter_backtest,
    vorhersagen, z_hat_fuer, z_hat_skala, zeilen_aller_saisons,
)


def _vollstaendig() -> bool:
    return ((PROCESSED / "panel.parquet").exists()
            and (MANUAL / "tm_players.csv").exists()
            and (MANUAL / "fine_positions.csv").exists())


@pytest.fixture(scope="module")
def dat() -> Daten:
    if not _vollstaendig():
        pytest.skip("panel.parquet, tm_players.csv oder fine_positions.csv fehlt")
    return Daten()


@pytest.fixture(scope="module")
def bt_ohne(dat):
    """Derselbe Walk-forward ohne die Versatz-Korrektur — die Vergleichsmarke."""
    from src.model import player_avg as pa
    return pa.backtest(dat, mit_versatz=False)


@pytest.fixture(scope="module")
def bt() -> pd.DataFrame:
    """Der volle Walk-forward — einmal je Testlauf, er kostet gut eine Minute."""
    if not _vollstaendig():
        pytest.skip("Datengrundlage fehlt")
    return backtest(Daten())


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_merkmale_ignorieren_die_zielsaison(dat: Daten) -> None:
    """Die Merkmale einer Zielsaison duerfen sich nicht aendern, wenn man die
    Zielsaison aus dem Panel entfernt.

    Das ist der schaerfere der beiden Leakage-Tests: er prueft nicht, ob eine
    bekannte Spalte weg ist, sondern ob *irgendein* Weg von der Zielsaison in
    die Merkmale fuehrt.
    """
    j = dat.j_ziel - 1
    zg = dat.zielgruppe(j)
    kand = zg[["player_id", "team_id", "pos"]].head(120).copy()
    voll = merkmalszeilen(dat, j, kand)

    ohne = Daten.__new__(Daten)
    ohne.__dict__.update(dat.__dict__)
    ohne.ss = dat.ss[dat.ss.j != j]
    ohne.sk_team = dat.sk_team[dat.sk_team.j != j]
    ohne._zh = dict(dat._zh)
    beschnitten = merkmalszeilen(ohne, j, kand)

    for spalte in (
            "a1", "gewicht", "fall", "z_hat", "teil", "hist_z", "hist_mw",
            "hist_team_delta", "mw_delta", "team_delta", "cell_mw",
            "cell_age", "cell_n", "rel_mw", "rel_age", "hist_belegt",
            "a1_belegt"):
        a, b = voll[spalte], beschnitten[spalte]
        if a.dtype.kind == "f":
            assert np.allclose(a.fillna(-999), b.fillna(-999)), spalte
        else:
            assert (a.fillna("-") == b.fillna("-")).all(), spalte


def test_historie_endet_vor_der_zielsaison(dat: Daten) -> None:
    """``historie`` darf nur Saisons < j_ziel und nur Liga 1 lesen."""
    j = dat.j_ziel - 1
    h = historie(dat.ss, j)
    quelle = dat.ss[(dat.ss.j < j) & (dat.ss.j >= j - FENSTER)
                    & (dat.ss.league == LIGA1)]
    assert set(h.player_id) <= set(quelle.player_id)
    # Wer nur in der Zielsaison selbst spielt, hat keine Historie.
    nur_ziel = set(dat.ss[dat.ss.j == j].player_id) - set(quelle.player_id)
    assert not (set(h.player_id) & nur_ziel)


# ---------------------------------------------------------------------------
# Struktur
# ---------------------------------------------------------------------------

def test_starter_filter_braucht_keine_minutenschwelle(dat: Daten) -> None:
    """Eine Bedingung genuegt, und das ist gemessen, nicht geraten.

    Eine zweite Schwelle ueber die Einsatzlaenge lag nahe, greift aber nie:
    schon ab acht Startelfeinsaetzen liegt die kleinste gemessene Einsatzlaenge
    bei exakt 60,0 Minuten. Faellt dieser Test, ist die Annahme geplatzt und
    die Schwelle gehoert zurueck.
    """
    ss = dat.ss[dat.ss.league == LIGA1]
    s = ss[ss.starter]
    assert (s.n_start >= MIN_STARTS).all()
    assert float(s.minj_start.min()) >= 60.0
    assert float(ss[ss.n_start >= 8].minj_start.min()) >= 60.0
    # Groessenordnung wie im Moduldocstring beziffert.
    assert 2600 <= int(s.shape[0]) <= 3100


def test_aufsteiger_bekommen_eine_zahl(dat: Daten) -> None:
    """Wer letzte Saison in Liga 2 spielte, ist Fall b — nicht "keine Prognose".

    Der Fehler war real: ``a1`` war Pflichtmerkmal in Fall b, und damit fielen
    28 von 145 Kandidaten still aus dem Export. ``hat_hist`` traegt den Fall
    jetzt.
    """
    assert PFLICHT["b"] == []
    assert "hat_hist" in MERKMALE["b"]

    k = kandidaten(dat)
    rows = merkmalszeilen(dat, dat.j_ziel, k)
    ohne_hist = rows[rows.a1.isna() & (rows.fall == "b")]
    assert len(ohne_hist) > 0, "keine Aufsteiger-Zeilen zum Pruefen"

    fits = anpassen(zeilen_aller_saisons(dat), dat.j_ziel)
    xp = vorhersagen(fits, rows)
    assert xp[ohne_hist.index].notna().all()


def test_neuzugaenge_werden_relativ_zu_ihren_vorgaengern_bewertet(
        dat: Daten) -> None:
    """Fall c nutzt Verhältnisse, nicht denselben absoluten MW-Aufschlag überall."""
    assert "rel_mw" in MERKMALE["c"] and "rel_age" in MERKMALE["c"]
    assert "log2_mw" not in MERKMALE["c"] and "alter" not in MERKMALE["c"]

    rows = merkmalszeilen(dat, dat.j_ziel, kandidaten(dat))
    neu = rows[rows.fall == "c"]
    assert len(neu) > 5
    # Die Rückfallkette bis zum Ligaprior gibt jedem aktuellen Neuzugang einen
    # Vergleich; das Verhältnis darf nicht still zum absoluten Wert werden.
    assert neu.rel_mw.notna().all() and neu.rel_age.notna().all()
    assert np.allclose(neu.rel_mw, neu.log2_mw - neu.cell_mw)
    assert np.allclose(neu.rel_age, neu.alter - neu.cell_age)


def test_kaderpflege_schlaegt_transfermarkt(dat: Daten) -> None:
    """``fine_positions.csv`` gilt auch rueckwirkend fuer fruehere Saisons.

    Sonst wird die Vorsaison in einer anderen Sprache etikettiert als die
    Zielzeile: Burger steht bei TM als ZDM, in der Kaderpflege als ZM.
    """
    fein, grob = positions_quelle()
    kp = pd.read_csv(MANUAL / "fine_positions.csv")
    kp = kp[kp.player_id.notna() & kp.position_fine.notna()]
    pid = int(kp.player_id.iloc[0])
    erwartet = kp.position_fine.iloc[0]

    zeilen = pd.DataFrame({"player_id": [pid, pid],
                           "season": ["2024/2025", "2025/2026"]})
    assert (position_von(zeilen, fein, grob) == erwartet).all()


def test_teamstaerke_prognose_ist_geschrumpft(dat: Daten) -> None:
    """Eine Prognose streut weniger als die gemessene Groesse — sonst ist sie keine.

    Gemessen: z_hat sd 0,84 gegen sd 0,97 der gemessenen Staerke. Wer beide
    Skalen verwechselt, zieht die Spitze auseinander (Bayern +3,04 statt +2,64).
    """
    skala = z_hat_skala(dat.team, dat.j_ziel)
    assert 0.70 <= skala <= 0.95

    live = np.array(list(dat.z_hat(dat.j_ziel).values()))
    assert len(live) >= 18
    assert abs(float(live.std()) - skala) < 0.05
    assert abs(float(live.mean())) < 0.30


def test_teamstaerke_schlaegt_die_vorsaison_allein(dat: Daten) -> None:
    """z_att ~ z_prev + auf + z_mw erreicht r 0,839 gegen 0,782 fuer z_prev allein."""
    tab = dat.team.dropna(subset=["z_att"])
    tab = tab[tab.j >= ERSTE_ZIELSAISON]
    vorher = _corr(tab.z_prev0, tab.z_att)

    ist, hat = [], []
    for j in sorted(tab.j.unique()):
        zh = z_hat_fuer(dat.team, j)
        s = tab[tab.j == j]
        for t, y in zip(s.team_id, s.z_att):
            if t in zh:
                ist.append(y)
                hat.append(zh[t])
    assert _corr(ist, hat) > vorher + 0.03


# ---------------------------------------------------------------------------
# Guete
# ---------------------------------------------------------------------------

def test_fall_a_schlaegt_die_naive_fortschreibung(bt: pd.DataFrame) -> None:
    """Gemessen: MAE 16,7 gegen 18,5, Verzerrung +1,8 gegen +8,7.

    Der Rang-Gewinn ist klein (r 0,716 gegen 0,711) — der eigentliche Gewinn
    ist das Niveau. Der rohe Vorjahresschnitt liegt systematisch zu hoch, weil
    er die Rueckkehr zur Mitte ignoriert.
    """
    a = bt[(bt.fall == "a") & bt.pred.notna() & bt.a1.notna()]
    assert len(a) > 500

    mae_modell = float(np.abs(a.avg_alle - a.pred).mean())
    mae_naiv = float(np.abs(a.avg_alle - a.a1).mean())
    assert mae_modell < mae_naiv - 1.0

    verz_modell = float((a.pred - a.avg_alle).mean())
    verz_naiv = float((a.a1 - a.avg_alle).mean())
    assert abs(verz_modell) < 3.0
    assert abs(verz_modell) < abs(verz_naiv) - 4.0
    assert _corr(a.avg_alle, a.pred) > 0.68


def test_alle_faelle_treffen_die_gemessene_guete(bt: pd.DataFrame) -> None:
    """Schranken knapp unter den gemessenen Werten je Fall."""
    b = bt.dropna(subset=["pred", "avg_alle"])
    grenzen = {"a": (0.68, 18.0), "b": (0.50, 18.0), "c": (0.46, 20.0)}
    for fall, (r_min, mae_max) in grenzen.items():
        s = b[b.fall == fall]
        assert len(s) > 100, fall
        assert _corr(s.avg_alle, s.pred) > r_min, fall
        assert float(np.abs(s.avg_alle - s.pred).mean()) < mae_max, fall


def test_tail_kalibrierung_korrigiert_systematische_unterschaetzung(
        dat: Daten, bt: pd.DataFrame) -> None:
    """Seltene Qualitaetsausreisser duerfen nicht zur Mitte plattgedrueckt werden.

    Die Korrektur wird ausschliesslich aus frueheren OOF-Fehlern gelernt. Sie
    muss deshalb nicht nur die beiden Zielsegmente verbessern, sondern darf
    auch den Gesamtfehler nicht fuer schoene Einzelfaelle opfern.
    """
    basis = backtest(dat, mit_segmente=False)
    neu = bt.merge(basis[["j", "player_id", "pred"]],
                   on=["j", "player_id"], suffixes=("", "_basis"))
    neu = neu.dropna(subset=["pred", "pred_basis", "avg_alle"])
    assert np.abs(neu.pred - neu.avg_alle).mean() < np.abs(
        neu.pred_basis - neu.avg_alle).mean()

    segmente = {
        "Elite-Neuzugang": ((neu.fall == "c")
                            & (np.power(2.0, neu.log2_mw) >= SPITZE_MW)),
    }
    for name, maske in segmente.items():
        s = neu[maske]
        assert len(s) >= 10, name
        assert np.abs(s.pred - s.avg_alle).mean() < np.abs(
            s.pred_basis - s.avg_alle).mean(), name
        assert (s.pred - s.avg_alle).mean() > (
            s.pred_basis - s.avg_alle).mean(), name


def test_mehr_historie_erhoeht_das_historiensignal(dat: Daten) -> None:
    """Viele belegte Saisons duerfen nicht gleich stark schrumpfen wie eine."""
    rows = zeilen_aller_saisons(dat)
    fit = anpassen(rows, dat.j_ziel)
    koef = dict(zip(["const"] + fit["spalten"], fit["coef"]))
    assert "a:a1_belegt" in koef
    assert koef["a:a1_belegt"] > 0.0

    # Ableitung der Prognose nach a1: Grundgewicht plus der positive
    # Zuverlaessigkeitskanal. Bei 60 gewichteten Starts muss sie klar hoeher
    # sein als bei nur zehn.
    b0 = koef["a:a1"]
    bx = koef["a:a1_belegt"]
    niedrig = b0 + bx * (10.0 / (10.0 + 5.0))
    hoch = b0 + bx * (60.0 / (60.0 + 5.0))
    assert hoch > niedrig + 0.05


def test_mannschaftsteil_haelt_die_verzerrung_klein(bt: pd.DataFrame) -> None:
    """Ohne den Teil-Kanal war das Modell je Mannschaftsteil systematisch schief
    (Torhueter -8,5, Zentrum +7,4). Mit ihm bleibt die Spanne klein.

    Das ist der Test, den das Gesamt-r nicht leisten kann: dort heben sich die
    Fehler der Mannschaftsteile gegeneinander auf.
    """
    b = bt.dropna(subset=["pred", "avg_alle", "teil"])
    verz = {t: float((s.pred - s.avg_alle).mean())
            for t, s in b.groupby("teil") if len(s) >= 40}
    assert set(verz) >= {TEIL_REF} | set(TEILE) - {"AV"}
    assert max(verz.values()) - min(verz.values()) < 9.0
    assert abs(verz[TEIL_REF]) < 5.0


def test_juengste_zielsaison_ist_unverzerrt(bt: pd.DataFrame) -> None:
    """Nur die letzten Saisons zaehlen fuer die Live-Prognose.

    Die fruehen Zielsaisons liegen hoch, weil das Modell dort auf dem
    punktreicheren Niveau vor 2020 gefittet ist — genau dagegen wirkt die
    Rueckgewichtung. Ob sie reicht, entscheidet sich am juengsten Jahr.
    """
    b = bt.dropna(subset=["pred", "avg_alle"])
    letzte = b[b.j == b.j.max()]
    assert len(letzte) > 100
    assert abs(float((letzte.pred - letzte.avg_alle).mean())) < 4.0


def test_rueckgewichtung_senkt_die_verzerrung(dat: Daten) -> None:
    """rho < 1 ist gemessen noetig, nicht Geschmack.

    Gemessen an den Faellen b/c: rho 1,0 -> Verzerrung +5,8; rho 0,6 -> +3,1.
    """
    assert RHO < 1.0
    ohne = backtest(dat, rho=1.0)
    mit = backtest(dat, rho=RHO)

    def verz(bt_):
        s = bt_[(bt_.fall != "a") & bt_.pred.notna()]
        return abs(float((s.pred - s.avg_alle).mean()))

    assert verz(mit) < verz(ohne)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_ist_vollstaendig_und_stimmig(tmp_path: Path) -> None:
    if not _vollstaendig() or not (LEGACY_DIR / "data_1.json").exists():
        pytest.skip("Datengrundlage fehlt")
    ziel = tmp_path / "player_projections_avg.json"
    doc = export(ziel)
    assert ziel.exists()

    gelesen = json.loads(ziel.read_text(encoding="utf-8"))
    assert gelesen["model"] == "avg-je-einsatz-fallweise"
    assert gelesen["liga"] == "1"
    assert gelesen["kategorien"] == [1, 2]

    P = doc["players"]
    assert len(P) >= 110

    felder = {"name", "team_id", "verein", "pos", "teil", "kategorie", "fall",
              "xp", "hist", "zelle", "stufe", "z_hat", "streuung",
              "mw_verhaeltnis", "alter_delta", "vorgaenger_n",
              "sicherheit", "verletzt", "starter_wahrscheinlichkeit"}
    for pid, e in P.items():
        assert set(e) == felder, pid
        assert e["kategorie"] in (1, 2)
        assert e["fall"] in ("a", "b", "c")
        assert 20.0 <= e["xp"] <= 280.0, (pid, e["xp"])
        assert 0.0 < e["sicherheit"] <= 1.0
        assert e["teil"] in (TEIL_REF,) + TEILE
        assert TEIL[e["pos"]] == e["teil"]
        assert isinstance(e["verletzt"], bool)
        assert e["starter_wahrscheinlichkeit"] is None
        if e["fall"] == "c":
            assert e["mw_verhaeltnis"] is not None and e["mw_verhaeltnis"] > 0
            assert e["alter_delta"] is not None
            assert e["vorgaenger_n"] >= 1
        else:
            assert e["mw_verhaeltnis"] is None and e["alter_delta"] is None

    # Fall a ohne eigene Historie gibt es nicht — das ist die Definition.
    assert all(e["hist"] is not None for e in P.values() if e["fall"] == "a")
    # Und jeder Fall kommt vor, sonst greift die Unterscheidung nicht.
    assert set(e["fall"] for e in P.values()) == {"a", "b", "c"}

    # Ein sehr teurer gesetzter Liganeuling erhaelt den auf historischen
    # OOF-Fehlern gemessenen Qualitaetsaufschlag. Der Name waehlt die Regel
    # nicht aus; er haelt nur den aktuell wichtigsten Referenzfall fest.
    karetsas = next((e for e in P.values() if e["name"] == "Karetsas"), None)
    if karetsas is not None:
        assert karetsas["fall"] == "c"
        assert karetsas["zelle"] >= 110
        assert 105 <= karetsas["xp"] <= karetsas["zelle"] + 5
        tail = doc["params"]["tail_kalibrierung"]
        assert tail["bonus"]["neuzugang_elite"] > 10


def test_verletzte_bekommen_trotzdem_eine_zahl(tmp_path: Path) -> None:
    """Die Markierung ist fuer scatter.html, nicht fuer die Prognose.

    Das Modell rechnet unter der Annahme "gesund, spielt seine Rolle" — bei
    einem Verletzten ist das genau die interessante Zahl: was holt er nach der
    Rueckkehr. Herausgehalten wird er nur aus der Fair-Value-Regression, weil
    sein Marktwert ausfallbedingt gedrueckt ist.
    """
    if not _vollstaendig() or not (LEGACY_DIR / "history.json").exists():
        pytest.skip("Datengrundlage fehlt")
    P = export(tmp_path / "p.json")["players"]
    krank = [e for e in P.values() if e["verletzt"]]
    if not krank:
        pytest.skip("derzeit ist niemand aus den Kategorien 1-2 verletzt")
    assert all(np.isfinite(e["xp"]) and e["xp"] > 20 for e in krank)


def test_kandidaten_haben_alle_einen_verein(dat: Daten) -> None:
    """Verein und Marktwert kommen aus data_1.json, nicht aus dem Panel.

    Der Panel-Weg lieferte fuer 30 von 441 Kaderspielern keine team_id — es
    sind die Neuzugaenge, die vor Saisonstart noch keine Zeile haben.
    """
    k = kandidaten(dat)
    assert len(k) > 100
    assert k.team_id.notna().all()
    assert k.verein.notna().all()
    assert k.log2_mw.notna().mean() > 0.95

def test_versatz_nimmt_die_einzige_echte_verzerrung_heraus(bt_ohne, bt):
    """Die Verzerrung je Fall darf nicht mehr von null unterscheidbar sein.

    Gezaehlt wird die **Saison** als unabhaengige Einheit, nicht die Zeile:
    die Spieler einer Saison teilen sich ihr Kohortenglueck (sd der
    Saisonverzerrung 6,9 Punkte), und wer Zeilen zaehlt, findet ueberall
    Signifikanz. Ohne den Versatz ist Fall b mit t = 2,4 die eine echte
    Verzerrung des Modells; mit ihm liegt kein Fall mehr ueber t = 2.
    """
    import numpy as np

    def tw(b, fall):
        s = b[(b.fall == fall)].dropna(subset=["pred", "avg_alle"])
        js = s.assign(res=s.pred - s.avg_alle).groupby("j").res.mean()
        return js.mean() / (js.std(ddof=1) / np.sqrt(len(js)))

    assert abs(tw(bt_ohne, "b")) > 2.0, "Fall b war ohne Versatz nicht auffaellig"
    for fall in ("a", "b", "c"):
        assert abs(tw(bt, fall)) < 2.0, f"Fall {fall} bleibt verzerrt"


def test_versatz_greift_erst_ab_drei_saisons(bt_ohne):
    """Aus zwei Saisons geschaetzt schadet der Versatz mehr, als er nuetzt."""
    from src.model import player_avg as pa
    zwei = bt_ohne[bt_ohne.j < bt_ohne.j.min() + 2]
    assert all(v == 0.0 for v in pa.versatz_tabelle(zwei).values())
    viele = pa.versatz_tabelle(bt_ohne)
    assert any(abs(v) > 0.5 for v in viele.values())

def test_zelle_liest_die_vorgaenger_derselben_stelle(dat):
    """Die Zelle muss die Spieler nennen, aus denen sie gebildet ist.

    Karetsas erbt bei Dortmund die ZOM-Zelle aus Brandt und Beier — das ist
    die Frage, fuer die es diese Groesse ueberhaupt gibt, und ein Test haelt
    fest, dass sie sie woertlich beantwortet.
    """
    from src.model import player_avg as pa

    wert, stufe = pa.zelle(dat.zellen, 2026, "3", "ZOM")
    s = dat.sk_team
    zelle_spieler = s[(s.j == 2025) & (s.team_id == "3") & s.starter
                      & (s.pos == "ZOM")]
    assert len(zelle_spieler) == 2, "BVB-ZOM 2025/26 sollte zwei Starter tragen"
    assert stufe == "position"
    # Die Zelle ist das Mittel ihrer Spieler, verschoben um die Niveaukorrektur:
    # gespeichert wird die Abweichung vom Ligamittel *ihrer* Saison, addiert
    # wird der ueber LVL_FENSTER geglaettete Sockel der Zielposition. Der
    # Abstand zum rohen Mittel ist deshalb klein, aber nicht null.
    roh = zelle_spieler.avg_start.mean()
    assert abs(wert - roh) < 8, (wert, roh)
    assert wert > 100, "eine BVB-Offensivzelle liegt deutlich ueber dem Mittel"


def test_neuzugaenge_bleiben_unter_ihrer_zelle(bt):
    """Wer eine starke Zelle erbt, holt sie gemessen nicht ein.

    Das ist der Grund, warum die Zelle Merkmal und nicht Anker ist: im
    obersten Fuenftel der Zellen liegen die tatsaechlichen Ertraege rund 36
    Punkte darunter. Ein Modell, das die Zelle eins zu eins fortschreibt,
    waere fuer genau die interessantesten Neuzugaenge zu hoch.
    """
    c = bt[(bt.fall == "c") & bt.zelle.notna() & bt.avg_alle.notna()]
    assert len(c) > 100
    assert c.avg_alle.mean() < c.zelle.mean(), "Neuzugaenge erben ihre Zelle nicht"
    oben = c[c.zelle > c.zelle.quantile(0.8)]
    assert oben.zelle.mean() - oben.avg_alle.mean() > 20


# ---------------------------------------------------------------------------
# Liga 2
# ---------------------------------------------------------------------------

def test_liga2_uebernimmt_die_struktur_aber_fittet_ihr_eigenes_niveau() -> None:
    """Die Liga-1-Erkenntnisse sind Struktur, nicht ein kopierter Koeffizientensatz."""
    if not _vollstaendig():
        pytest.skip("Datengrundlage fehlt")
    dat2 = Daten(liga="2")
    assert dat2.league == LIGA2
    assert dat2.erste_zielsaison == 2022
    assert set(dat2.ss.loc[dat2.ss.j == 2025, "league"]) >= {LIGA1, LIGA2}

    bt2 = backtest(dat2).dropna(subset=["pred", "avg_alle"])
    assert len(bt2) > 600
    assert set(bt2.j.unique()) == {2023, 2024, 2025}
    assert _corr(bt2.avg_alle, bt2.pred) > 0.48
    assert float(np.abs(bt2.pred - bt2.avg_alle).mean()) < 17.5
    assert abs(float((bt2.pred - bt2.avg_alle).mean())) < 2.0
    assert set(bt2.fall) == {"a", "b", "c"}


def test_liga2_starterauswahl_ist_walk_forward_und_je_team_begrenzt() -> None:
    if not _vollstaendig():
        pytest.skip("Datengrundlage fehlt")
    bt2 = starter_backtest(Daten(liga="2"))
    assert set(bt2.j.unique()) == {2022, 2023, 2024, 2025}

    je_team = (bt2[bt2.ausgewaehlt]
               .groupby(["j", "team_id"]).size())
    assert (je_team == STARTER_JE_TEAM).all()
    assert len(je_team) == 4 * 18
    assert float(bt2.loc[bt2.ausgewaehlt, "starter"].mean()) > 0.60


def test_liga2_export_hat_acht_kandidaten_je_team(tmp_path: Path) -> None:
    if not _vollstaendig() or not (LEGACY_DIR / "data_2.json").exists():
        pytest.skip("Datengrundlage fehlt")
    ziel = tmp_path / "player_projections_avg_2.json"
    doc = export(ziel, liga="2")
    assert json.loads(ziel.read_text(encoding="utf-8"))["liga"] == "2"
    assert doc["params"]["auswahl"]["typ"] == "starter-logistik"
    assert doc["params"]["auswahl"]["je_team"] == STARTER_JE_TEAM
    assert len(doc["players"]) == 18 * STARTER_JE_TEAM

    p = pd.DataFrame(doc["players"].values())
    assert (p.groupby("team_id").size() == STARTER_JE_TEAM).all()
    assert p.starter_wahrscheinlichkeit.between(0.0, 1.0).all()
    assert set(p.kategorie) == {1, 2}
    assert set(p.fall) == {"a", "b", "c"}
