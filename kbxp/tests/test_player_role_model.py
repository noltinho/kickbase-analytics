"""Das Rollenmodell gegen Leakage, gegen Baselines und gegen sich selbst.

Vier Sorten Test:
  · **Leakage.** Die Zielgroesse darf auf keinem Weg in die Merkmale sickern.
    Zwei Wege waeren moeglich und sind beide zugehalten: die Panel-Spalten mit
    Saisonendstand, und die eigene Zielsaison in der Historie. Faellt einer
    dieser Tests, ist jede Kennzahl darunter wertlos - und zwar unauffaellig,
    weil Leakage die Zahlen *verbessert*.
  · **Grenzfaelle der Prior-Kette.** Ohne Historie muss exakt die Baseline
    herauskommen, mit viel Historie fast der eigene Wert. Das ist der Grund,
    warum es keine Spielerkategorien braucht.
  · **Guete im Walk-forward.** Schranken knapp unter den gemessenen Werten,
    immer gegen eine explizite Baseline - eine Kennzahl ohne Vergleichslinie
    sagt nichts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.paths import MANUAL, PROCESSED  # noqa: E402
from src.model.player_features import (  # noqa: E402
    LEAKAGE_SPALTEN, kaderkonkurrenz, lade_panel, lade_tm, spieler_saisons,
    tm_je_spieler,
)
from src.model.player_role_model import (  # noqa: E402
    DELTA, K_MINJE, K_MIX, KAPPA_BL2, MIN_EINSAETZE_ZIEL,
    MIN_ZIEL, ROLLEN, TW, _auc, _kappa, backtest_p90, baseline_fit,
    baseline_wert, kontext_tabelle, merkmalszeilen, minje_fuer, p90_mischung,
    report_oekonomie, rolle_aus_startquote,
    rollen_tabelle, streuung_fuer, streuungs_faktoren, team_niveaus,
    uebergaenge,
)
from src.model.team_strength import conceded_from_panel  # noqa: E402


def _hat_panel() -> bool:
    return (PROCESSED / "panel.parquet").exists()


def _hat_tm() -> bool:
    return (MANUAL / "tm_players.csv").exists()


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _hat_panel():
        pytest.skip("panel.parquet fehlt - erst backfill_history laufen lassen")
    return lade_panel()


@pytest.fixture(scope="module")
def bt() -> pd.DataFrame:
    """Der volle Walk-forward - einmal je Testlauf, er kostet gut eine Minute."""
    if not _hat_panel() or not (MANUAL / "tm_players.csv").exists():
        pytest.skip("panel.parquet oder tm_players.csv fehlt")
    return backtest_p90()


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_saisonendstand_verlaesst_das_panel(panel: pd.DataFrame) -> None:
    """season_avg_points/season_total_points sind der Saisonendstand.

    Kickbase liefert sie auf **jeder** Spieltagszeile, auch auf der von
    Spieltag 1 - als Merkmal waeren sie ein Blick auf die eigene Zielgroesse.
    Sie duerfen den Merkmalsraum gar nicht erst betreten.
    """
    for spalte in LEAKAGE_SPALTEN:
        assert spalte not in panel.columns, f"{spalte} ist noch da"


@pytest.mark.skipif(not _hat_tm(), reason="tm_players.csv ist nicht Teil des Repos")
def test_merkmale_ignorieren_die_zielsaison(panel: pd.DataFrame) -> None:
    """Die Zielsaison in der Historie darf die Merkmale nicht beruehren.

    Der schaerfste Leakage-Test, den es hier gibt: die Spielersaisons der
    Zielsaison werden grob verfaelscht (p90 verzehnfacht, Minuten verdoppelt).
    Aendert sich auch nur eine Merkmalszahl, liest das Modell seine eigene
    Antwort mit.
    """
    ss = spieler_saisons(panel[panel.season != "2026/2027"])
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    ph = panel[panel.season != "2026/2027"]
    kon = kontext_tabelle(
        team_niveaus(conceded_from_panel(ph, matches[matches.season != "2026/2027"])),
        sorted(ss.season.unique()))
    tm = tm_je_spieler(lade_tm())

    ziel = "2024/2025"
    kand = ss[(ss.season == ziel) & (ss.league == "Bundesliga")][
        ["player_id", "team_id"]]

    kaputt = ss.copy()
    sel = kaputt.season == ziel
    kaputt.loc[sel, "p90"] *= 10.0
    kaputt.loc[sel, "min_gesamt"] *= 2.0
    kaputt.loc[sel, "min_je_teamspieltag"] *= 2.0
    kaputt.loc[sel, "startquote"] = 1.0

    kaputt.loc[sel, "min_je_einsatz"] = 90.0
    kaputt.loc[sel, "einsaetze"] = 34
    kaputt.loc[sel, "einsatzquote"] = 1.0

    spalten = ["p90_mix", "baseline", "kontext", "mw_resid", "team_delta",
               "n_eff", "min_je_teamspieltag_v1", "startquote_v1",
               "minje_hist", "avg_hist", "n_eins_eff", "minje_rolle",
               "quote_rolle", "minje_dach_neu", "p_basis"]
    a = merkmalszeilen(ziel, kand, ss, kon, tm, DELTA, K_MIX, False, False)
    b = merkmalszeilen(ziel, kand, kaputt, kon, tm, DELTA, K_MIX, False, False)
    for s in spalten:
        assert np.allclose(a[s].fillna(-999), b[s].fillna(-999)), (
            f"Merkmal {s} haengt an der Zielsaison")


def test_minuten_werden_bei_90_gedeckelt(panel: pd.DataFrame) -> None:
    """Nachspielzeit darf pro-90 nicht verwaessern.

    Ein Sechstel aller Zeilen traegt mehr als 90 Minuten (bis 143). Ohne
    Deckel bekaeme ein Durchspieler systematisch weniger Punkte je 90 als ein
    85-Minuten-Spieler mit denselben Punkten.
    """
    assert panel.min90.max() <= 90
    assert (panel.minutes > 90).sum() > 1000, "der Deckel greift gar nicht mehr"


# ---------------------------------------------------------------------------
# Prior-Kette
# ---------------------------------------------------------------------------

def test_ohne_historie_genau_die_baseline() -> None:
    """Ein Spieler ohne Panel-Zeilen faellt exakt auf die Baseline.

    Genau das ersetzt die Spielerkategorien: es braucht keinen Schalter
    "Neuzugang", weil n_eff = 0 die Mischung von selbst aufloest.
    """
    assert p90_mischung(0.0, np.nan, 88.0) == 88.0
    assert p90_mischung(0.0, 120.0, 88.0) == 88.0
    # Und mit viel Historie bleibt fast nur der eigene Wert stehen:
    # 200 effektive Spiele gegen K_MIX = 16 Spiele Baseline.
    assert p90_mischung(200.0, 120.0, 88.0) == pytest.approx(
        (200 * 120 + K_MIX * 88) / (200 + K_MIX), abs=0.01)
    assert p90_mischung(200.0, 120.0, 88.0) > 117.0
    # Dazwischen liegt der Mix immer zwischen beiden Werten.
    mix = p90_mischung(16.0, 120.0, 88.0, k=16.0)
    assert 88.0 < mix < 120.0 and mix == pytest.approx(104.0)


def test_baseline_steigt_mit_der_teamstaerke() -> None:
    """Die Positions-Baseline muss mit dem Teamniveau steigen.

    Der gemessene Spread ist der Kern des Neuzugangs-Fixes: Stamm-IV holen bei
    Topteams rund 100 Punkte, bei schwachen 65. Eine Baseline ohne Steigung
    wuerde jeden Neuzugang auf den Ligaschnitt setzen.
    """
    rng = np.random.default_rng(0)
    n = 400
    kontext = rng.normal(0, 1, n)
    rows = pd.DataFrame({
        "pos": ["IV"] * n,
        "kontext": kontext,
        "p90": 85.0 + 12.0 * kontext + rng.normal(0, 5, n),
        "min_gesamt": np.full(n, 1800.0),
    })
    fit = baseline_fit(rows)
    assert baseline_wert(fit, "IV", 1.0) > baseline_wert(fit, "IV", -1.0) + 15
    # Unbekannte Positionen fallen auf die globale Gerade zurueck, nicht auf 0.
    assert baseline_wert(fit, "gibtsnicht", 0.0) == pytest.approx(
        baseline_wert(fit, "IV", 0.0), abs=2.0)


def test_kappa_bleibt_im_gemessenen_band(panel: pd.DataFrame) -> None:
    """Was eine Zweitliga-Saison in Erstliga-Waehrung wert ist.

    Gemessen an Spielern mit >=450 Minuten in beiden Ligen in Folge; der Wert
    liegt deutlich unter 1 (die zweite Liga bringt mehr Punkte je 90) und wird
    walk-forward je Zielsaison neu geschaetzt. Reisst dieses Band, hat sich
    entweder die Datenlage oder die Definition geaendert.
    """
    ss = spieler_saisons(panel[panel.season != "2026/2027"])
    k = _kappa(ss, "2026/2027")
    assert 0.60 <= k <= 0.75, f"kappa = {k:.3f}, gemessen waren 0,67"
    # Frueher stand hier ein geratenes 0,85 - die Messung liegt deutlich
    # darunter. Die Konstante ist nur der Notnagel und muss zur Messung passen.
    assert abs(KAPPA_BL2 - k) < 0.1, (
        f"Fallback {KAPPA_BL2} passt nicht mehr zur Messung {k:.3f}")


# ---------------------------------------------------------------------------
# Guete im Walk-forward
# ---------------------------------------------------------------------------

def test_ertrag_schlaegt_den_vorjahres_durchschnitt(bt: pd.DataFrame) -> None:
    """Die Pflicht - hieran ist v1 gescheitert.

    Auf der Zielgroesse, die beim Kaderbau zaehlt (Oe Punkte je Einsatz),
    erreichte das alte p90-x-Minuten-Modell nur r 0,548 und lag damit **unter**
    dem stumpfen Fortschreiben des Vorjahres. Gemessen fuer v2: 0,630 gegen
    0,627 auf der gemeinsamen Maske, bei deutlich besserem Bias (+0,3 gegen
    -2,5) und MAE (22,7 gegen 23,8) - und 0,606 auf der vollen Maske, wo die
    Baseline fuer rund 1.200 Spieler gar nichts sagt.
    """
    feld = bt[(bt.pos != TW) & (bt.ist_einsaetze >= MIN_EINSAETZE_ZIEL)
              & bt.ist_avg.notna()]
    beide = feld[feld.naiv_avg.notna() & (feld.naiv_einsaetze >= 10)]
    assert len(beide) > 2000, f"nur {len(beide)} vergleichbare Spielersaisons"

    r_modell = float(np.corrcoef(beide.xp_dach, beide.ist_avg)[0, 1])
    r_naiv = float(np.corrcoef(beide.naiv_avg, beide.ist_avg)[0, 1])
    assert r_naiv > 0.60, f"Baseline unerwartet schwach ({r_naiv:.3f})"
    assert r_modell >= r_naiv, (
        f"Modell {r_modell:.3f} unter der Baseline {r_naiv:.3f} - genau der "
        f"Fehler, wegen dem v1 verworfen wurde")
    mae_m = float(np.abs(beide.ist_avg - beide.xp_dach).mean())
    mae_n = float(np.abs(beide.ist_avg - beide.naiv_avg).mean())
    assert mae_m < mae_n - 0.5, f"MAE {mae_m:.1f} gegen {mae_n:.1f}"
    assert float(np.corrcoef(feld.xp_dach, feld.ist_avg)[0, 1]) > 0.58


def test_ertrag_ist_unverzerrt(bt: pd.DataFrame) -> None:
    """Kein systematischer Versatz — gegenwartsnah und zwischen den Gruppen.

    Was hier **nicht** geprueft wird, ist die ueber elf Zielsaisons gepoolte
    Verzerrung. Sie betraegt -3,6 Punkte und ist keine Modelleigenschaft,
    sondern ein Niveausprung der Liga: 2020/21 bis 2022/23 lag der Ist-Wert
    11 Punkte unter der Prognose, seit 2023/24 liegt er auf ihr (-1,5 / -2,3 /
    -0,2). Ein Mittel darueber beschreibt keine der beiden Epochen.

    Genau daran ist die fruehere Ertrags-Ridge gescheitert: sie mittelte
    diesen Rest ueber alle Vorsaisons und trug die alte Korrektur in die
    Gegenwart — die gepoolte Verzerrung sah mit +0,6 gut aus, die der
    juengsten drei Saisons lag bei +4,7. Ohne sie ist es umgekehrt.

    Geprueft wird deshalb zweierlei: das Niveau auf den juengsten Saisons (die
    Epoche, fuer die prognostiziert wird) und die Gleichbehandlung der
    Rollengruppen relativ zueinander. Letztere ist der eigentliche Anspruch —
    ein gemeinsamer Versatz ist Liganiveau, ein Versatz *zwischen* Gruppen ist
    ein Modellfehler.
    """
    feld = bt[(bt.pos != TW) & (bt.ist_einsaetze >= MIN_EINSAETZE_ZIEL)
              & bt.ist_avg.notna()]
    juengste = sorted(feld.saison.unique())[-3:]
    aktuell = float((feld[feld.saison.isin(juengste)].ist_avg
                     - feld[feld.saison.isin(juengste)].xp_dach).mean())
    assert abs(aktuell) < 2.5, f"Verzerrung der juengsten drei Saisons {aktuell:+.1f}"

    gesamt = float((feld.ist_avg - feld.xp_dach).mean())
    for label, sel in (("gesetzt", feld.startquote_v1 > 0.8),
                       ("Rotation", (feld.startquote_v1 > 0.55)
                        & (feld.startquote_v1 <= 0.8)),
                       ("geteilt/Rand", (feld.startquote_v1 > 0)
                        & (feld.startquote_v1 <= 0.55)),
                       ("ohne Vorsaison", feld.hat_v1 == 0)):
        g = feld[sel.fillna(False)]
        if len(g) < 100:
            continue
        b = float((g.ist_avg - g.xp_dach).mean()) - gesamt
        assert abs(b) < 2.5, f"Gruppe {label}: {b:+.1f} gegen den Schnitt"


def test_rollenkenntnis_ist_der_groesste_hebel(bt: pd.DataFrame) -> None:
    """Das Orakel beziffert, was die handgepflegte Rolle wert ist.

    Mit der tatsaechlichen Rolle der Zielsaison steigt das Modell von r 0,606
    auf 0,779, und die schwaechste Gruppe (Spieler ohne jede Historie) von
    0,37 auf 0,72. Das ist die Rechtfertigung fuer fine_positions.csv - und
    zugleich die einzige Zahl, an der sich die Kaderpflege spaeter messen
    laesst. Faellt dieser Abstand, wirkt der Rollenkanal nicht mehr.
    """
    orakel = backtest_p90(orakel=True)

    def _r(rows: pd.DataFrame) -> float:
        g = rows[(rows.pos != TW) & (rows.ist_einsaetze >= MIN_EINSAETZE_ZIEL)
                 & rows.ist_avg.notna()]
        return float(np.corrcoef(g.xp_dach, g.ist_avg)[0, 1])

    assert _r(orakel) > _r(bt) + 0.12, (
        f"Orakel {_r(orakel):.3f} gegen Modell {_r(bt):.3f} - der Rollenkanal "
        f"traegt nicht mehr")
    assert _r(orakel) > 0.74


@pytest.mark.skipif(not _hat_tm(), reason="tm_players.csv ist nicht Teil des Repos")
def test_rollentabelle_ist_eng_und_positionsabhaengig(panel: pd.DataFrame) -> None:
    """Die Einsatzlaenge haengt an Rolle und Position, und zwar praezise.

    Gemessen fuer gesetzte Spieler: IV 85,5 · AV 84,1 · ZDM 83,1 · ZM 81,1 ·
    ST 80,3 · ZOM 79,9 · FL 78,9 Minuten je Einsatz, mit einer Streuung von
    nur 4-6 Minuten **innerhalb** jeder Zelle. Genau deshalb ist die Tabelle
    ein besserer Schaetzer als die Historie eines einzelnen Spielers - sie ist
    praeziser und traegt zugleich die *neue* Rolle statt der alten.
    """
    ss = spieler_saisons(panel[panel.season != "2026/2027"])
    tm = tm_je_spieler(lade_tm())
    d = ss[(ss.league == "Bundesliga") & (ss.einsaetze >= 3)].merge(
        tm[["player_id", "season", "position_fine"]],
        on=["player_id", "season"], how="left")
    d = d.assign(pos=d.position_fine.fillna("UNB"),
                 rolle=rolle_aus_startquote(d.startquote))
    tab = rollen_tabelle(d)

    for pos in ("IV", "ST", "FL"):
        werte = [tab.get((pos, r)) for r in ROLLEN]
        vorhanden = [w for w in werte if w is not None]
        assert vorhanden == sorted(vorhanden), f"{pos}: Rollen nicht monoton"
    assert tab[("IV", "gesetzt")] > tab[("FL", "gesetzt")] + 3

    for (pos, _), g in d[d.rolle == "gesetzt"].groupby(["pos", "rolle"],
                                                       observed=True):
        if len(g) >= 100:
            assert float(g.min_je_einsatz.std()) < 9.0, f"{pos}: Zelle zu breit"

    # Eine leere Zelle darf nie NaN liefern - sonst verschwindet ein Spieler
    # still aus der Prognose.
    assert np.isfinite(minje_fuer(tab, "gibtsnicht", "gesetzt"))
    assert np.isfinite(minje_fuer(tab, "IV", None))


def test_uebergaenge_zeigen_rueckkehr_zur_mitte() -> None:
    """Rollen halten nicht - deshalb wird ueber sie gemittelt.

    Gemessen bleiben nur 59 % der Gesetzten gesetzt, waehrend 10 % der
    Randspieler aufsteigen. Wer die Vorjahresrolle festschreibt, ueberschaetzt
    Stammspieler um 7,1 Minuten je Einsatz und unterschaetzt Randspieler um
    16,1. Die Uebergangsmatrix ist die Antwort darauf.
    """
    rng = np.random.default_rng(0)
    n = 800
    von = rng.choice(ROLLEN, n)
    nach = [r if rng.random() < 0.6 else rng.choice(ROLLEN) for r in von]
    tab = uebergaenge(pd.DataFrame({"rolle_von": von, "rolle_nach": nach}))
    for r in ROLLEN:
        p = tab[r]
        assert abs(sum(p.values()) - 1.0) < 1e-9, f"{r}: keine Verteilung"
        assert p[r] == max(p.values()), f"{r}: haelt sich nicht am ehesten"
        assert p[r] < 0.95, f"{r}: Rolle wird als sicher behandelt"
    assert abs(sum(tab["_rand"].values()) - 1.0) < 1e-9


def test_rollenwahrscheinlichkeit_trennt_und_ist_kalibriert(bt: pd.DataFrame) -> None:
    """Setzt sich der Spieler durch? Zwei Anforderungen, beide gemessen.

    Trennschaerfe: AUC 0,794 bei einem Brier-Skill von +0,215 gegen die
    Basisrate. Kalibrierung: groesste Dezil-Abweichung 7,7 Prozentpunkte - der
    Rest ist der starke Zeittrend (der Anteil gesetzter Spieler faellt von
    35,2 % auf 16,0 %), gegen den Abklinggewichtung und Nacheichung arbeiten.

    Dass das ueberhaupt trennt, ist der Grund, warum die handgepflegten
    Kategorien nicht verfeinert werden muessen: innerhalb der Vorjahres-
    Rotation trennen Marktwert-Rang und Startquote von 9,1 % bis 33,2 %.
    """
    d = bt[bt.ist_startquote.notna() & bt.p_rolle.notna()]
    y, p = d.ist_gesetzt.values, d.p_rolle.values
    auc = _auc(y, p)
    assert auc > 0.75, f"AUC auf {auc:.3f} gefallen"

    basis = float(y.mean())
    skill = 1 - ((p - y) ** 2).mean() / ((basis - y) ** 2).mean()
    assert skill > 0.15, f"Brier-Skill nur {skill:+.3f}"

    dez = d.assign(z=pd.qcut(p, 10, labels=False, duplicates="drop"))
    t = dez.groupby("z").agg(prog=("p_rolle", "mean"), ist=("ist_gesetzt", "mean"))
    assert float((t.ist - t.prog).abs().max()) < 0.10, "Kalibrierung gerissen"
    assert float(np.corrcoef(t.prog, t.ist)[0, 1]) > 0.95


def test_streuung_trifft_die_beobachtete_abweichung(bt: pd.DataFrame) -> None:
    """Eine sd-Prognose hat kein r - sie muss die Streuung selbst treffen.

    Gemessen liegt die beobachtete Abweichung in jedem Terzil der Prognose
    innerhalb von 3 %. Die beiden Treiber sind fast unabhaengig und deshalb
    multiplikativ: Belegtheit (ohne Historie 39,3 gegen 28,5 bei viel) und
    Position (IV 26,6 bis ST 35,1).
    """
    fak = streuungs_faktoren(bt)
    d = bt[bt.ist_avg.notna() & (bt.ist_einsaetze >= MIN_EINSAETZE_ZIEL)].copy()
    d["s"] = [streuung_fuer(fak, n, p)
              for n, p in zip(d.n_eff.fillna(0.0), d.pos)]
    d["fehler"] = d.ist_avg - d.xp_dach
    for _, g in d.groupby(pd.qcut(d.s, 3, labels=False, duplicates="drop"),
                          observed=True):
        ist = float(np.sqrt((g.fehler ** 2).mean()))
        soll = float(g.s.mean())
        assert abs(ist - soll) / soll < 0.12, (
            f"Terzil: erwartet {soll:.1f}, beobachtet {ist:.1f}")
    assert fak["belegt"]["ohne"] > fak["belegt"].get("viel", 1.0) * 1.2


def test_p90_schlaegt_die_naive_fortschreibung(bt: pd.DataFrame) -> None:
    """Gemessen: Modell r=0,735 gegen naiv r=0,695 auf derselben Maske.

    Die Baseline ist "Vorjahres-p90 fortschreiben" - sie ist stark (die
    Befunde messen sie als staerksten Einzelpraediktor ueberhaupt), und nur
    gegen sie ist ein Fortschritt ueberhaupt sichtbar. Verglichen wird
    ausschliesslich auf den Spielern, fuer die die Baseline definiert ist;
    das Modell bewertet daneben rund 970 weitere.
    """
    feld = bt[(bt.pos != TW) & (bt.ist_min >= MIN_ZIEL) & bt.ist_p90.notna()]
    beide = feld[feld.naiv_p90.notna() & (feld.naiv_min >= 900)]
    assert len(beide) > 1500, f"nur {len(beide)} vergleichbare Spielersaisons"

    r_modell = float(np.corrcoef(beide.p90_dach, beide.ist_p90)[0, 1])
    r_naiv = float(np.corrcoef(beide.naiv_p90, beide.ist_p90)[0, 1])
    assert r_naiv > 0.65, f"Baseline unerwartet schwach ({r_naiv:.3f}) - falsch gebaut?"
    assert r_modell > r_naiv + 0.02, (
        f"Modell {r_modell:.3f} gegen Baseline {r_naiv:.3f} - der Abstand "
        f"(gemessen 0,04) ist weg")
    assert r_modell > 0.72, f"r auf {r_modell:.3f} gefallen"


def test_teamstaerke_repariert_die_spieler_ohne_vorsaison(bt: pd.DataFrame) -> None:
    """Der Zielverein-Fix, an der Gruppe gemessen, fuer die er gedacht ist.

    Ohne die Teamstaerke des aufnehmenden Vereins faellt die Gruppe "keine
    Bundesliga-Vorsaison" von r 0,59 auf 0,41 - das ist der groesste einzelne
    Beitrag im ganzen Modell und der Grund, warum die Baseline ueberhaupt
    zweistufig ist.
    """
    ohne = backtest_p90(ohne_teamstaerke=True)

    def _r(rows: pd.DataFrame) -> float:
        g = rows[(rows.pos != TW) & (rows.ist_min >= MIN_ZIEL)
                 & rows.ist_p90.notna()]
        g = g[g.naiv_min.isna() | (g.naiv_min == 0)]
        return float(np.corrcoef(g.p90_dach, g.ist_p90)[0, 1])

    mit_r, ohne_r = _r(bt), _r(ohne)
    assert mit_r > ohne_r + 0.10, (
        f"Teamstaerke bringt nur {mit_r - ohne_r:.3f} statt der gemessenen ~0,18")


def test_produkt_schlaegt_beide_einfachen_sorten(bt: pd.DataFrame) -> None:
    """Die Ebene, auf der die Seite benutzt wird.

    Gemessen: paarweise 76,9 % gegen 72,8 % (Vorjahrespunkte fortschreiben)
    und 71,5 % (Marktwert sortieren). Gemessen wird paarweise und nicht am
    R2, weil die Seite eine Rangfolge zeigt - dieselbe Begruendung wie beim
    Team-Rating.
    """
    from src.model.team_strength import pairwise_accuracy
    from src.model.player_role_model import _produkt_bt

    df = bt[bt.ist_punkte.notna() & bt.ist_st.notna()].copy()
    df["pred"] = df.xp_dach * df.quote_rolle * df.ist_st
    df["naiv"] = df.naiv_avg.fillna(0.0) * df.einsatzquote_v1 * df.ist_st
    df["mw"] = df.mw_vor_saison.fillna(0.0)

    pw = {k: pairwise_accuracy(_produkt_bt(df, k)) for k in ("pred", "naiv", "mw")}
    assert pw["pred"] > pw["naiv"] + 0.03, f"gegen Vorjahr nur {pw['pred']:.3f}"
    assert pw["pred"] > pw["mw"] + 0.03, f"gegen Marktwert nur {pw['pred']:.3f}"
    assert pw["pred"] > 0.74, f"paarweise auf {pw['pred']:.3f} gefallen"


def test_oekonomie_schlaegt_den_marktwert_sort(bt: pd.DataFrame) -> None:
    """Top-15-Picks im bezahlbaren Band - der Test, der Geld bedeutet.

    Gemessen: in 11 von 11 Saisons mindestens gleichauf mit dem
    Marktwert-Sort. Oben ist der Markt effizient, im Band 2-10 Mio nicht.
    """
    erg = report_oekonomie(bt)
    assert erg["siege"] >= erg["n"] - 1, (
        f"nur {erg['siege']} von {erg['n']} Saisons >= Marktwert-Sort")
    assert erg["modell"] > erg["mw"] * 1.2, "Vorsprung vor dem Marktwert-Sort weg"


def test_torhueter_nicht_schlechter_als_fortschreiben(bt: pd.DataFrame) -> None:
    """Bei Torhuetern ist wenig zu holen, aber nichts zu verlieren.

    Die Reliabilitaet des TW-Punkteschnitts betraegt laut Befunden nur 0,38 -
    die Rangfolge unter etablierten Torhuetern ist kaum vorhersagbar. Die
    Mischung gegen die Teamabwehr kalibriert dafuer das Niveau (MAE 13,6
    gegen 14,7) und traegt Torhueter ohne Bundesliga-Historie ueberhaupt erst.
    Mehr wird hier nicht versprochen.
    """
    tw = bt[(bt.pos == TW) & (bt.ist_min >= MIN_ZIEL) & bt.ist_p90.notna()]
    tw = tw[tw.naiv_p90.notna() & (tw.naiv_min >= 900)]
    if len(tw) < 50:
        pytest.skip("zu wenige Torhueter-Saisons")
    mae_modell = float(np.abs(tw.p90_dach - tw.ist_p90).mean())
    mae_naiv = float(np.abs(tw.naiv_p90 - tw.ist_p90).mean())
    assert mae_modell <= mae_naiv, (
        f"TW-Niveau schlechter getroffen als beim Fortschreiben "
        f"({mae_modell:.1f} gegen {mae_naiv:.1f})")


# ---------------------------------------------------------------------------
# Kaderkonkurrenz
# ---------------------------------------------------------------------------

def test_kaderkonkurrenz_zaehlt_nur_ernsthafte_konkurrenten() -> None:
    """Rang und Konkurrenz sind ordinal innerhalb des Kaderbereichs.

    Deshalb darf die Wertquelle wechseln (Transfermarkt historisch, spaeter
    auch anderes), solange sie innerhalb eines Kaders dieselbe ist. Ein
    unbewerteter Spieler zaehlt weder als Konkurrent noch bekommt er selbst
    eine Zaehlung - sonst waere "unbewertet" mit "konkurrenzlos" verwechselt.
    """
    df = pd.DataFrame({
        "grp": ["A"] * 4 + ["B"] * 2,
        "mw": [10e6, 8e6, 1e6, np.nan, 5e6, 5e6],
    })
    rang, konk = kaderkonkurrenz(df, ["grp"], "mw")
    assert list(rang[:3]) == [1.0, 2.0, 3.0]
    assert np.isnan(rang.iloc[3])
    # 10 Mio: nur die 8 Mio liegen ueber 60 % -> ein Konkurrent.
    assert konk.iloc[0] == 1
    # 1 Mio: beide teureren liegen darueber -> zwei.
    assert konk.iloc[2] == 2
    assert np.isnan(konk.iloc[3])
    # Gleichstand zaehlt beidseitig.
    assert konk.iloc[4] == 1 and konk.iloc[5] == 1
