"""Leakagefreier Unterbau des Spielermodells.

Hier entsteht alles, was das Spielermodell an Daten anfasst — und nur hier.
Der Grund fuer die eigene Datei ist nicht Ordnungsliebe: Backtest, Export und
Tests brauchen exakt dieselben Zeilen, und ein Leakage-Fehler, der nur in einem
der drei Pfade steckt, faellt nie auf. Deshalb betreten die beiden verseuchten
Panel-Spalten (``season_avg_points``/``season_total_points`` — Kickbase liefert
dort den **Saisonendstand**, auch fuer Spieltag 1) den Merkmalsraum gar nicht
erst: ``lade_panel`` wirft sie weg, und ein Test haelt das fest.

Alle Groessen je Spielersaison sind so definiert, dass sie fuer die Zielsaison s
ausschliesslich aus Saisons < s gelesen werden. Die einzige Information, die
aus der Zielsaison selbst stammen darf, ist die Kaderzugehoerigkeit (welches
Team nimmt den Spieler auf) — sie ist vor Saisonstart oeffentlich bekannt und
traegt keine Leistung.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model.team_strength import before_break, fit_ratings  # noqa: E402
from src.paths import MANUAL, PROCESSED  # noqa: E402

# Saisonendstaende, die Kickbase je Spieltagszeile mitliefert. Als Merkmal waeren
# sie ein Blick in die Zukunft der eigenen Zielgroesse.
LEAKAGE_SPALTEN = ("season_avg_points", "season_total_points")

# Stichtag fuer Alter und Marktwert: 1. August, derselbe wie fuer
# ``mw_vor_saison`` in transfermarkt.py — beide Merkmale meinen denselben
# Zeitpunkt "vor der Saison".
STICHTAG = (8, 1)

# Ab diesem MW-Anteil zaehlt ein Positionskollege als Konkurrent um Minuten.
# Ein 500k-Ersatz bedraengt einen 10-Mio-Stammspieler nicht; die Schwelle
# trennt Kaderfuellung von echter Konkurrenz.
KONKURRENZ_ANTEIL = 0.6


# ---------------------------------------------------------------------------
# Panel und Transfermarkt-Kader laden
# ---------------------------------------------------------------------------

def lade_panel() -> pd.DataFrame:
    """Panel ohne Leakage-Spalten, Minuten fuer pro-90 auf 90 gedeckelt.

    ``min90``: ein Sechstel aller Zeilen traegt mehr als 90 Minuten
    (Nachspielzeit, bis 143). Ungedeckelt bekaeme ein Durchspieler systematisch
    weniger pro-90-Punkte als ein 85-Minuten-Spieler mit gleichen Punkten —
    der Deckel misst beide am vollen Spiel.
    """
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel = panel.drop(columns=[c for c in LEAKAGE_SPALTEN if c in panel.columns])
    panel["min90"] = panel["minutes"].clip(upper=90)
    return panel


def lade_tm() -> pd.DataFrame:
    """Transfermarkt-Kaderzeilen mit Alter, log-MW und Kaderkonkurrenz.

    Eine Zeile je Spieler **und Verein** — Wechsler stehen zweimal je Saison
    (305 Faelle), und fuer die Kaderkonkurrenz ist das richtig so: sie
    konkurrieren in beiden Kadern um Minuten. Wer eine Zeile je Spieler
    braucht, nimmt ``tm_je_spieler``.

    Die Konkurrenz-Merkmale entstehen hier und nicht im Modell, weil sie auf
    der Kaderzeile leben: ``mw_rang_pos`` ist der MW-Rang innerhalb
    (Saison, Verein, Position), ``n_konkurrenten`` zaehlt Positionskollegen
    mit mindestens KONKURRENZ_ANTEIL des eigenen MW. Beides ist vor
    Saisonstart bekannt — TM-Kader und Vorsaison-MW sind oeffentlich.
    """
    tm = pd.read_csv(MANUAL / "tm_players.csv")
    tm = tm[tm.player_id.notna()].copy()
    tm["player_id"] = tm.player_id.astype("int64")

    geb = pd.to_datetime(tm.geburtsdatum, format="%d.%m.%Y", errors="coerce")
    stich = pd.to_datetime({
        "year": tm.season.str.slice(0, 4).astype(int),
        "month": STICHTAG[0], "day": STICHTAG[1]})
    tm["alter"] = (stich - geb).dt.days / 365.25

    mw = tm.mw_vor_saison.astype(float)
    tm["log2_mw"] = np.log2(mw.where(mw > 0))

    rang, konk = kaderkonkurrenz(tm, ["season", "verein", "position_fine"],
                                 "mw_vor_saison")
    tm["mw_rang_pos"], tm["n_konkurrenten"] = rang, konk
    return tm


def kaderkonkurrenz(df: pd.DataFrame, gruppen: list[str],
                    wert: str) -> tuple[pd.Series, pd.Series]:
    """Rang und Zahl ernsthafter Konkurrenten innerhalb eines Kaderbereichs.

    Beide Groessen sind **ordinal innerhalb der Gruppe** — die Quelle des
    Werts darf deshalb wechseln (Transfermarkt-Stichtagswert historisch,
    Kickbase-Marktwert fuer die laufende Saison), solange sie innerhalb eines
    Kaders dieselbe ist. Genau das braucht der Export: der TM-Crawl liefert
    den Stichtagswert einer Saison erst, wenn sie laeuft.
    """
    w = df[wert].astype(float)
    rang = df.assign(_w=w).groupby(gruppen)["_w"].rank(
        ascending=False, method="min")

    def _zaehlen(g: pd.DataFrame) -> pd.Series:
        v = g["_w"].values
        # Ein NaN-Wert zaehlt weder als Konkurrent noch bekommt er selbst eine
        # Zaehlung — sonst waere "unbewertet" mit "konkurrenzlos" verwechselt.
        n = ((v[None, :] >= KONKURRENZ_ANTEIL * v[:, None])
             & ~np.isnan(v)[None, :]).sum(axis=1) - 1
        return pd.Series(np.where(np.isnan(v), np.nan, n), index=g.index)

    konk = (df.assign(_w=w).groupby(gruppen, group_keys=False)[["_w"]]
              .apply(_zaehlen))
    return rang, konk


def tm_je_spieler(tm: pd.DataFrame) -> pd.DataFrame:
    """Der in CLAUDE.md festgeschriebene Join-Weg: eine Zeile je Spielersaison.

    ``drop_duplicates`` behaelt die erste Kaderzeile eines Wechslers. Welche
    das ist, entscheidet die Dateireihenfolge — fuer MW, Alter und Position
    ist das folgenlos (beide Zeilen tragen dieselbe ``tm_id``), nur die
    Konkurrenz-Merkmale koennen um den zweiten Kader daneben liegen. Das ist
    dokumentiertes Rauschen in 305 von 12.367 Faellen, kein Fehler.
    """
    return tm.drop_duplicates(["player_id", "season"])


# ---------------------------------------------------------------------------
# Spielersaisons: eine Zeile je (player_id, season, league)
# ---------------------------------------------------------------------------

def spieler_saisons(panel: pd.DataFrame, hinrunde: bool = False,
                    splits: dict | None = None) -> pd.DataFrame:
    """Die Kennzahlen einer Spielersaison, aus denen alles Weitere entsteht.

    Nur beendete Spieltage (``md_status == 2``) zaehlen — die vorgecrawlte
    Zielsaison besteht aus ungespielten Zeilen und faellt damit von selbst
    heraus. ``hinrunde=True`` schneidet zusaetzlich auf die Spieltage vor der
    Winterpause (echte Splits, nicht Spieltag 17): das ist das
    Bewertungsfenster, denn Kickbase resettet zur Winterpause.

    Quoten sind Anteile an den **Team-Spieltagen** (Zeilen des Spielers), nicht
    an den eigenen Einsaetzen: eine Startquote von 0,5 heisst "stand in der
    Haelfte der Spiele seines Teams in der Startelf" — die Groesse, die fuers
    Minutenmodell zaehlt. ``kader_fehlquote`` (Status 1, nicht im Kader) ist
    der backtestbare Ausfall-Proxy: history.json reicht nur ein Jahr zurueck,
    dieser Anteil steht fuer jede Saison seit 2013/14.
    """
    p = panel[panel.md_status == 2]
    if hinrunde:
        p = before_break(p, splits)
    p = p.sort_values("matchday")

    agg = (p.assign(
        pkt=p.points.fillna(0.0),
        eingesetzt=p.min90 > 0,
        start=p.player_md_status == 5,
        joker=p.player_md_status == 3,
        fehlt=p.player_md_status == 1,
    ).groupby(["player_id", "season", "league"], as_index=False).agg(
        punkte=("pkt", "sum"),
        min_gesamt=("min90", "sum"),
        einsaetze=("eingesetzt", "sum"),
        n_teamspieltage=("matchday", "size"),
        starts=("start", "sum"),
        joker=("joker", "sum"),
        fehlt=("fehlt", "sum"),
        # Erste Zeile der Saison = Hinrunden-Verein eines Wechslers. Das ist
        # Kaderzugehoerigkeit, keine Leistung — vor Saisonstart bekannt.
        team_id=("team_id", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    ))

    agg["p90"] = np.where(agg.min_gesamt > 0,
                          agg.punkte / agg.min_gesamt * 90.0, np.nan)
    agg["min_je_teamspieltag"] = agg.min_gesamt / agg.n_teamspieltage
    agg["min_je_einsatz"] = np.where(agg.einsaetze > 0,
                                     agg.min_gesamt / agg.einsaetze, 0.0)
    agg["startquote"] = agg.starts / agg.n_teamspieltage
    agg["jokerquote"] = agg.joker / agg.n_teamspieltage
    agg["kader_fehlquote"] = agg.fehlt / agg.n_teamspieltage
    # Anteil der Team-Spieltage mit Einsatz. Nicht dasselbe wie die
    # Startquote: ein Joker steht regelmaessig auf dem Platz, ohne je zu
    # beginnen. Fuers Ausfallrisiko ist diese Groesse die richtige.
    agg["einsatzquote"] = agg.einsaetze / agg.n_teamspieltage
    return agg.drop(columns=["starts", "joker", "fehlt"])


def start_kennzahlen(panel: pd.DataFrame, hinrunde: bool = False,
                     splits: dict | None = None) -> pd.DataFrame:
    """Zwei Punkteschnitte je Spielersaison **und Verein**: alle Einsaetze, nur Startelf.

    Der Unterschied ist der Kern des fallweisen Modells. Gemessen an 1.504
    Spielersaisons sagt der Vorjahres-Schnitt *aus Startelfeinsaetzen* den
    naechsten Gesamtschnitt besser voraus (r 0,678) als der Vorjahres-
    Gesamtschnitt selbst (0,664): Joker-Kurzeinsaetze sind im **Eingang**
    Rauschen, im **Ziel** aber echter Ertrag. Deshalb liefert diese Funktion
    beides nebeneinander, statt sich fuer eines zu entscheiden.

    Geschluesselt wird mit ``team_id``, nicht ohne — ein Winterwechsler traegt
    zu beiden Kadern bei, und genau so wird er in der Vorgaenger-Basis
    gebraucht. Wer eine Zeile je Spielersaison will, fasst nachtraeglich
    zusammen (``je_spielersaison`` im Spielermodell).
    """
    p = panel[panel.md_status == 2]
    if hinrunde:
        p = before_break(p, splits)

    p = p.assign(eingesetzt=p.min90 > 0, start=p.player_md_status == 5)
    schl = ["player_id", "season", "league", "team_id"]

    alle = (p[p.eingesetzt].groupby(schl, as_index=False)
            .agg(punkte=("points", "sum"), n_einsaetze=("points", "size")))
    start = (p[p.start].groupby(schl, as_index=False)
             .agg(punkte_start=("points", "sum"), n_start=("points", "size"),
                  min_start=("min90", "sum")))

    df = alle.merge(start, on=schl, how="outer")
    for col in ("punkte", "n_einsaetze", "punkte_start", "n_start", "min_start"):
        df[col] = df[col].fillna(0.0)
    # Summe statt Mittel, damit sich zwei Kaderzeilen eines Wechslers ohne
    # Gewichtungsfehler zu einer Spielersaison addieren lassen.
    df["minj_start"] = np.where(df.n_start > 0, df.min_start / df.n_start, np.nan)
    df["avg_alle"] = np.where(df.n_einsaetze > 0,
                              df.punkte / df.n_einsaetze, np.nan)
    df["avg_start"] = np.where(df.n_start > 0,
                               df.punkte_start / df.n_start, np.nan)
    return df


# ---------------------------------------------------------------------------
# Teamniveaus je Saison — der Kontext fuer Baseline und Neuzugangs-Merkmal
# ---------------------------------------------------------------------------

def team_niveaus(conceded: pd.DataFrame) -> pd.DataFrame:
    """Angriffs- und Abwehrniveau je (Saison, Liga, Team), gegnerbereinigt.

    ``att_lvl`` = mu + att ist die erwartete Punktesumme der eigenen Spieler je
    Spiel, ``def_lvl`` = mu + def die erwartete zugelassene — dieselbe
    Zerlegung wie in ratings.json, nur je historischer Saison komplett
    geschaetzt. ``z_att``/``z_def`` standardisieren innerhalb der Saison/Liga,
    weil das Punkteniveau ueber die Jahre driftet und das Spielermodell nur
    wissen will, wo ein Team **relativ zu seiner Liga** steht.
    """
    rows = []
    for (season, league), g in conceded.groupby(["season", "league"]):
        r = fit_ratings(g)
        for t in r.teams():
            rows.append(dict(season=season, league=league, team_id=t,
                             att_lvl=r.mu + r.attack[t],
                             def_lvl=r.mu + r.defense[t]))
    df = pd.DataFrame(rows)
    for col in ("att_lvl", "def_lvl"):
        grp = df.groupby(["season", "league"])[col]
        df["z_" + col.split("_")[0]] = (df[col] - grp.transform("mean")) / grp.transform("std")
    return df
