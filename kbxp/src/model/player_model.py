"""Spielermodell: Oe Punkte je Einsatz = p90 (Qualitaet) x Minuten je Einsatz (Rolle).

Die Zielgroesse ist der **Punkteschnitt je Einsatz** der Hinrunde — dieselbe
Skala, die Kickbase als "Oe Punkte" zeigt. Nicht Punkte je Spieltag: vergangene
Saisons spiegeln Rotation und Verletzungen bereits wider, und Ausfaelle sind
vorhersehbar und ersetzbar. Bewertet wird vor der Winterpause (echte Splits),
weil Kickbase dort zuruecksetzt.

    Oe Punkte je Einsatz = p90 x Minuten je Einsatz / 90

**Der groesste Hebel ist die Rolle, nicht die Qualitaet.** Mit der
tatsaechlichen Rolle der Zielsaison steigt das Modell von r 0,618 auf 0,779,
und die schwaechste Gruppe (ohne jede Historie) von 0,40 auf 0,72. Genau diese
Auskunft steht handgepflegt in fine_positions.csv.

Drei Kanaele:

  Qualitaet   p90 ueber die Prior-Kette
              p90_mix = (n_eff * eigene Historie + K_MIX * Baseline)/(n_eff + K_MIX)
              Ein Neuzugang ohne Historie faellt exakt auf die Baseline
              "Position x Teamstaerke des aufnehmenden Vereins" (Stamm-IV bei
              Topteams 100,2, bei schwachen 64,5), ein Etablierter wird kaum
              beruehrt. Eine **additive** Ridge korrigiert danach Alter,
              MW-Residuum und Teamwechsel — additiv, weil eine multiplikative
              die datenabhaengige Schrumpfung der Kette ein zweites Mal
              pauschal anwendete.
  Rolle       Tabelle Position x Rolle -> Minuten je Einsatz, gemischt mit der
              eigenen Einsatzlaenge (K_MINJE)
  Durchsetzung  Logistik auf P(Startquote > 0,8); sie eicht zugleich die
              Rollenverteilung, aus der die Einsatzlaenge gemittelt wird

Dass die Schrumpfung der Prior-Kette richtig dosiert ist, ist gemessen: die
partielle Regression ist_p90 ~ p90_hist + baseline liefert je n_eff-Band die
Gewichte 0,00 / 0,48 / 0,59 / 0,74 / 1,01 (Baender 0-10 / 10-25 / 25-45 /
45-80 / 80+), das Modell setzt 0,22 / 0,51 / 0,68 / 0,78 / 0,84. Keine
Abweichung erreicht zwei Standardfehler — an K_MIX ist nichts zu holen.

Die Teamstaerke einer Zielsaison ist immer die der **Vorsaison** — auch beim
Fitten der Baseline. Das ist Absicht: wer die Baseline an der tatsaechlichen
Staerke fittet, aber mit der Vorsaison-Schaetzung vorhersagt, ueberdehnt die
Steigung (errors in variables). So attenuiert sie sich von selbst auf das, was
Vorwissen leisten kann.

Trainiert wird auf dem Ganzjahreswert (doppelt so viele Einsaetze, halbes
Zielrauschen), bewertet auf der Hinrunde. Das ist zulaessig, weil der
Hinrunden-Aufschlag eines Spielers sich **nicht** wiederholt: gemessen ueber
4.082 aufeinanderfolgende Spielersaisons liegt seine Korrelation mit dem
Vorjahresaufschlag bei r = -0,017 (Steigung -0,017 +/- 0,015). Von Kanes 30
Punkten Hinrunden-Vorsprung ueberleben rechnerisch -0,5 — eine
Hinrunden-Korrektur je Spieler waere reines Rauschen.

Was verworfen wurde, mit Zahl:

  * das Vorgaengermodell p90 x erwartete Minuten je Spieltag (r 0,548 — unter
    dem stumpfen Fortschreiben mit 0,596)
  * eine additive Ridge auf dem **Ertrag** (r 0,609 gegen 0,618 ohne sie,
    Verzerrung der juengsten drei Saisons +4,7 gegen -1,4; Begruendung an der
    Loeschstelle in backtest_p90)
  * ein Knick in der Ertragskalibrierung oberhalb 100/120/140 Punkten, um die
    Spitze anzuheben (r 0,6090 -> 0,6096, Spitzen-Verzerrung wird schlechter)
  * eine Potenz n_eff**a in der Prior-Kette (a=1,3/1,6/2,0: r unveraendert in
    der vierten Stelle — die p90-Ridge nimmt die Aenderung ueber `belegt`
    wieder zurueck)
  * ein direkter Pfad ueber den historischen Schnitt avg_hist neben xp_rolle
    (Koeffizient 0,01-0,03 — er traegt nichts, was die Struktur nicht hat)
  * namentliche Rollenvorgaenger (0,682 mit wie ohne)
  * BL2-Saisons als zusaetzliche Trainingszeilen (0,678 statt 0,682)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.player_features import (  # noqa: E402
    lade_panel, lade_tm, spieler_saisons, team_niveaus, tm_je_spieler,
)
from src.model.team_strength import (  # noqa: E402
    conceded_from_panel, load_splits, pairwise_accuracy,
)
from src.paths import LEGACY_DIR, MANUAL, PROCESSED, atomic_write_json  # noqa: E402

# --- Prior-Kette ------------------------------------------------------------
#
# Alle Konstanten hier sind im Walk-forward gemessen (Zielsaisons 2015/16-
# 2025/26, Hinrunde >= 450 Min, ohne TW; Schalter siehe _cli). Ergebnisse der
# Laeufe, die die Wahl begruenden:
#
#   delta 0,4 / 0,5 / 0,6 / 0,7  r 0,694 / 0,695 / 0,696 / 0,696 -> flach, 0,7
#   k 8 / 12 / 16 / 24           r 0,691 / 0,694 / 0,696 / 0,696 -> 16
#   bl2_gewicht 1 / 0,5 / 0,25 / 0,15
#     gesamt                     r 0,682 / 0,685 / 0,694 / 0,695
#     nur BL2-Historie (n=104)   r 0,132 / 0,176 / 0,312 / 0,361 -> 0,25
#   --pool-bl2                   r 0,678 (schlechter)      -> verworfen
#   --vorgaenger                 r 0,682 (identisch)       -> verworfen; die
#     Positions-x-Staerke-Baseline enthaelt den Slot bereits, n=1-3 je Slot
#     traegt nichts hinzu
#   --ohne-teamstaerke: Gruppe "keine BL-Vorsaison" faellt r 0,59 -> 0,41 —
#     der Zielverein-Fix aus den Befunden, hier bestaetigt.
#
# Referenz auf derselben Maske: naive Vorjahres-p90-Baseline r=0,695; das
# Modell erreicht dort 0,733 und bewertet zusaetzlich die 968 Spieler ohne
# Stamm-Vorsaison, fuer die die Baseline gar nichts sagt.
#
# **Wie stark geschrumpft werden muss, haengt an der Datenmenge** - das ist
# der Grund fuer die Prior-Kette und gegen einen pauschalen Faktor. Gemessen
# ueberlebt vom Abstand zur Baseline ins Folgejahr:
#
#   n_eff  0-10   10-25   25-45   45+
#   Anteil  0,00   0,47    0,60   0,78
#
# Der Anteil n_eff/(n_eff + 16) bildet das ab (0,24 / 0,52 / 0,69 / 0,79) und
# trifft die gut belegten Spieler genau. Die pauschalen 0,64 aus den Befunden
# sind der Mittelwert darueber und waeren fuer beide Enden falsch.
#
# Volle Kanalkette (gleiche Zielsaisons):
#   Minuten je Spieltag  r 0,568 / MAE 20,8  gegen Fortschreiben 0,486 / 23,3
#   Minuten je Einsatz   r 0,476 / MAE 17,3, ueber alle Dezile kalibriert
#   Produktebene    rho 0,720, paarweise 78,2 %  gegen Vorjahres-Punkte
#                   fortschreiben 0,555 / 72,7 % und MW-Sort 0,560 / 71,5 %
#   Trefferquote ±10 je Einsatz: 26,6 % gegen naiv 22,6 % (Glaskugel laut
#                   Befunden 48 % — das Ziel selbst wuerfelt)
#   Oekonomie Top-15 im Band 2-10 Mio: Modell in 11 von 11 Saisons >= MW-Sort;
#                   die nackte Vorjahrespunkte-Sortierung liegt beim
#                   Pick-Ertrag knapp vorn (50 % vs. 46 % der Luecke zur
#                   perfekten Rueckschau) — sie pickt die Ausreisser des
#                   Vorjahres, von denen einige wiederholen, waehrend das
#                   Modell sie zur Mitte zieht. Dafuer ist das Modell in der
#                   Breite (paarweise +5,4 Punkte) und bei Spielern ohne
#                   Vorjahr klar besser. Beide Staerken kombinieren hiesse
#                   zwei Listen zeigen; die Seite zeigt eine.

# Abkling je Saison Abstand in der eigenen Historie.
DELTA = 0.7
DELTA_ROLLE = 0.5   # Abkling der Rollen-Logistik, siehe Drift-Kommentar dort

# Gewicht der Baseline in effektiven Spielen. Die Befunde messen, dass nur
# ~64 % einer Abweichung vom Schnitt ins Folgejahr ueberleben — ein Teil der
# noetigen Schrumpfung steckt hier, den Rest erledigt die Ridge (beta < 1).
K_MIX = 16.0

# BL2-Minuten zaehlen in der Historie nur ein Viertel: eine BL2-Saison sagt
# ueber BL1-p90 deutlich weniger als eine BL1-Saison — unabhaengig vom
# Niveaufaktor kappa, der nur die Hoehe verschiebt, nicht die Aussagekraft.
BL2_GEWICHT = 0.25

# Torhueter mischen gegen die Teamdefensive (-z_def als Kontext) statt gegen
# den Angriff. Gemessen (n=150 TW-Saisons mit Stamm-Vorsaison): k_tw
# 40/16/8/4/2 -> r 0,089/0,157/0,186/0,201/0,209 bei MAE 13,7-13,6; die naive
# Fortschreibung erreicht r 0,244, MAE 14,7. Die Rangfolge unter etablierten
# Torhuetern ist also auch mit Historie kaum zu schlagen (Reliabilitaet des
# TW-Schnitts laut Befunden nur 0,38, die r-Differenz liegt im Standardfehler),
# aber die Mischung kalibriert das Niveau besser (MAE) und traegt Torhueter
# ohne BL-Historie. Kleines k, damit das echte Signal der Historie nicht
# erdrueckt wird.
K_TW = 4.0

# BL2->BL1: was eine Zweitliga-Saison in Erstliga-Waehrung wert ist. kappa wird
# je Zielsaison aus den tatsaechlichen Aufsteigern geschaetzt (siehe _kappa);
# diese Konstante greift nur, solange dafuer zu wenige Uebergaenge vorliegen —
# die 2. Liga beginnt im Panel erst 2021/22. Der Wert ist der ueber alle Daten
# gemessene (0,67, Stand Aug 2026): ein Spieler holt in der Bundesliga rund
# zwei Drittel seiner Zweitliga-Punkte je 90 Minuten. Das ist ein Niveau-, kein
# Aussagekraft-Faktor — letzteres regelt BL2_GEWICHT.
KAPPA_BL2 = 0.67

# Aufsteiger-Teams ohne uebersetzbares Vorsaison-Niveau: gesetzt, nicht
# gemessen — Aufsteiger landen fast immer im unteren Drittel der Liga.
Z_AUFSTEIGER = -1.0

# Ridge-Strafen. Stufe B hat nur eine Handvoll Merkmale auf tausenden Zeilen —
# die Strafe dient der Kondition, nicht der Schrumpfung.
LAMBDA_B = 4.0

# Bewertungsmaske: unterhalb von 450 Hinrunden-Minuten dominiert der
# Standardfehler des Ziels (Spiel-zu-Spiel-SD 57 Punkte) jede Aussage.
MIN_ZIEL = 450

# Maske der Zielgroesse. Die Oe Punkte je Einsatz sind ein Mittelwert ueber
# Einsaetze — ihr Standardfehler haengt an deren Zahl, nicht am Minutenbudget
# (bei 5 Einsaetzen und SD 57 sind das ±25). Beide Masken stehen nebeneinander:
# MIN_ZIEL fuer die p90-Kennzahlen, diese fuer den Ertrag.
MIN_EINSAETZE_ZIEL = 5
# Trainings- und Baseline-Zeilen: gleiche Schwelle auf die ganze Saison.
MIN_TRAIN = 450

TW = "TW"
POS_UNBEKANNT = "UNB"

# --- Rolle und Einsatzlaenge -------------------------------------------------
#
# Die Rolle eines Spielers zu kennen ist der groesste Hebel im ganzen Modell.
# Gemessen (Walk-forward, Zielsaisons 2015/16-2025/26, Oe Punkte je Einsatz):
#
#   p90-Modell x eigener historischer Min/Einsatz          r 0,627
#   p90-Modell x Tabelle (Rolle der Vorsaison x Position)  r 0,627
#   dieselbe Tabelle mit der ECHTEN Rolle (Orakel)         r 0,759
#
# Die Rolle richtig zu treffen ist also mehr wert als jedes andere Merkmal —
# mehr als die vermeintliche Decke von 0,68, die unter der Annahme berechnet
# wurde, dass man sie *nicht* kennt. Genau diese Information steht
# handgepflegt in fine_positions.csv.
#
# Warum eine Tabelle und nicht die eigene Historie: die eigene Historie traegt
# die **alte** Rolle. Gemessen verzerrt sie entsprechend — Ex-Stammspieler um
# 5,6 Punkte nach oben, Ex-Joker um 7,0 nach unten. Die Tabellenzellen streuen
# dagegen nur um 4-6 Minuten (gesetzte Spieler: IV 85,5 · AV 84,1 · ZDM 83,1 ·
# ZM 81,1 · ST 80,3 · ZOM 79,9 · FL 78,9), sind also praeziser *und* tragen
# die neue Rolle.
ROLLEN = ("Rand", "geteilt", "Rotation", "gesetzt")
ROLLEN_GRENZEN = (0.25, 0.55, 0.80)   # auf der Startquote
MIN_ZELLE = 30                        # darunter faellt die Zelle auf die Position zurueck


# ---------------------------------------------------------------------------
# Kleine geschlossene Helfer
# ---------------------------------------------------------------------------

def _prev_of(seasons: list[str]) -> dict[str, str]:
    s = sorted(seasons)
    return {b: a for a, b in zip(s, s[1:])}


def p90_mischung(n_eff: float, p90_hist: float, baseline: float,
                 k: float = K_MIX) -> float:
    """Der Kern der Prior-Kette. n_eff = 0 liefert exakt die Baseline."""
    if n_eff <= 0 or not np.isfinite(p90_hist):
        return baseline
    return (n_eff * p90_hist + k * baseline) / (n_eff + k)


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> dict:
    """Gewichtete Ridge mit unbestraftem Achsenabschnitt, standardisiert."""
    mx = np.average(X, axis=0, weights=w)
    sx = np.sqrt(np.average((X - mx) ** 2, axis=0, weights=w))
    sx[sx < 1e-9] = 1.0
    Xs = (X - mx) / sx
    my = float(np.average(y, weights=w))
    A = Xs.T @ (Xs * w[:, None]) / w.sum() + lam / len(w) * np.eye(X.shape[1])
    beta = np.linalg.solve(A, Xs.T @ (w * (y - my)) / w.sum())
    return {"mx": mx, "sx": sx, "my": my, "beta": beta}


def _ridge_pred(fit: dict, X: np.ndarray) -> np.ndarray:
    return fit["my"] + ((X - fit["mx"]) / fit["sx"]) @ fit["beta"]


def _logit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
           lam: float = 4.0, runden: int = 25) -> dict:
    """Logistische Regression per IRLS, standardisiert, mit Ridge-Strafe.

    Handgeschrieben aus demselben Grund wie ``_ridge``: die Datei bringt
    scipy/numpy schon mit, und ein weiteres Paket fuer 20 Zeilen Newton-
    Verfahren waere die teurere Abhaengigkeit. Der Achsenabschnitt bleibt
    unbestraft, sonst verschoebe die Strafe die Basisrate.
    """
    mx = X.mean(axis=0)
    sx = X.std(axis=0)
    sx[sx < 1e-9] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mx) / sx])
    if w is None:
        w = np.ones(len(X))
    w = w / w.mean()
    strafe = np.full(Z.shape[1], lam)
    strafe[0] = 0.0
    beta = np.zeros(Z.shape[1])
    start = float(np.average(y, weights=w))
    beta[0] = np.log(max(start, 1e-6) / max(1 - start, 1e-6))
    for _ in range(runden):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -30, 30)))
        s = w * np.clip(p * (1 - p), 1e-6, None)
        A = Z.T @ (Z * s[:, None]) + np.diag(strafe)
        schritt = np.linalg.solve(A, Z.T @ (w * (y - p)) - strafe * beta)
        beta += schritt
        if np.max(np.abs(schritt)) < 1e-8:
            break
    return {"mx": mx, "sx": sx, "beta": beta}


def _logit_pred(fit: dict, X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), (X - fit["mx"]) / fit["sx"]])
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ fit["beta"], -30, 30)))


# ---------------------------------------------------------------------------
# Teamkontext: Vorsaison-Staerke des (aufnehmenden) Vereins
# ---------------------------------------------------------------------------

def kontext_tabelle(niveaus: pd.DataFrame, seasons: list[str]) -> dict:
    """(saison, team_id) -> Kontextwert je Position-Sicht.

    Fuer Feldspieler zaehlt der Angriff des Teams (z_att), fuer Torhueter die
    Abwehr — als **-z_def**, damit "groesser = besser" fuer beide gilt und
    dieselbe Baseline-Maschinerie laeuft.

    Ein Team, das in der Vorsaison zweite Liga spielte, bekommt sein
    BL2-z mit auf Z_AUFSTEIGER gedeckeltem Niveau: die z-Werte zweier Ligen
    sind nicht dieselbe Skala, und ein BL2-Meister ist in der BL kein
    +2-Angriff. Praezise uebersetzen liesse sich das erst mit vielen
    Aufstiegs-Saisons; bis dahin ist der Deckel die ehrliche Auskunft
    "im unteren Drittel".
    """
    prev = _prev_of(seasons)
    idx = {(r.season, r.league, r.team_id): (r.z_att, r.z_def)
           for r in niveaus.itertuples(index=False)}
    out = {}
    for season in seasons:
        ps = prev.get(season)
        if ps is None:
            continue
        teams = {t for (s, lg, t) in idx if s == season}
        for team in teams:
            hit = idx.get((ps, "Bundesliga", team))
            if hit is not None:
                z_att, z_def = hit
            else:
                hit2 = idx.get((ps, "2. Bundesliga", team))
                if hit2 is not None:
                    z_att = min(hit2[0], Z_AUFSTEIGER)
                    z_def = max(hit2[1], -Z_AUFSTEIGER)
                else:
                    z_att, z_def = Z_AUFSTEIGER, -Z_AUFSTEIGER
            out[(season, team)] = {"feld": z_att, TW: -z_def}
    return out


def _kontext(kon: dict, season: str, team: str, pos: str) -> float:
    e = kon.get((season, team))
    if e is None:
        return Z_AUFSTEIGER
    return e[TW] if pos == TW else e["feld"]


# ---------------------------------------------------------------------------
# Baseline: was ein Spieler dieser Position bei einem Team dieser Staerke holt
# ---------------------------------------------------------------------------

def rolle_aus_startquote(startquote) -> np.ndarray:
    """Startquote in eine der vier Rollen einteilen.

    Grenzen auf der Startquote statt auf den Minuten, weil die Rolle eine
    Aussage ueber die Aufstellung ist und nicht ueber die Ausdauer. Wer keine
    Vorsaison hat, bekommt None — fuer ihn traegt die Zielrolle (aus
    ``kategorie``) oder der Rueckfall die Auskunft.
    """
    q = np.asarray(startquote, dtype=float)
    idx = np.searchsorted(np.asarray(ROLLEN_GRENZEN), q, side="right")
    return np.where(np.isnan(q), None, np.array(ROLLEN, dtype=object)[idx])


def rollen_tabelle(train: pd.DataFrame) -> dict:
    """(Position, Rolle) -> erwartete Minuten je Einsatz.

    Erwartet Trainingszeilen mit ``pos``, ``rolle``, ``min_je_einsatz`` und
    ``einsaetze``; gemittelt wird ueber die Einsaetze, damit eine Saison mit
    drei Spielen die Zelle nicht so stark bewegt wie eine mit dreissig.
    Zellen unter MIN_ZELLE Zeilen fallen auf den Positionsschnitt zurueck,
    dieser notfalls auf den Gesamtschnitt — eine leere Zelle darf nie NaN
    liefern, sonst verschwindet der Spieler still aus der Prognose.
    """
    t = train[train.min_je_einsatz.notna() & train.rolle.notna()]
    if t.empty:
        return {"_global": 70.0}

    def _mittel(g: pd.DataFrame) -> float:
        return float(np.average(g.min_je_einsatz, weights=g.einsaetze))

    tab = {"_global": _mittel(t)}
    for pos, g in t.groupby("pos"):
        tab[("_pos", pos)] = _mittel(g) if len(g) >= MIN_ZELLE else tab["_global"]
        for rolle, gg in g.groupby("rolle"):
            if len(gg) >= MIN_ZELLE:
                tab[(pos, rolle)] = _mittel(gg)
    for rolle, g in t.groupby("rolle"):
        tab[("_rolle", rolle)] = _mittel(g)
    return tab


def minje_fuer(tab: dict, pos: str, rolle) -> float:
    """Zelle lesen, mit Rueckfall Position x Rolle -> Rolle -> Position -> global."""
    if rolle is not None and (pos, rolle) in tab:
        return tab[(pos, rolle)]
    if rolle is not None and ("_rolle", rolle) in tab:
        return tab[("_rolle", rolle)]
    return tab.get(("_pos", pos), tab["_global"])


# Rollen halten nicht. Gemessen (Zielsaisons 2015/16-2025/26, Bundesliga,
# >=5 Hinrunden-Einsaetze): von den Spielern, die in der Vorsaison gesetzt
# waren, sind es nur noch 59 % — ihre tatsaechliche Einsatzlaenge liegt 7,1
# Minuten unter dem Tabellenwert fuer "gesetzt". Umgekehrt spielen fruehere
# Randspieler 16,1 Minuten laenger als ihre alte Rolle nahelegt.
#
# Eine Rolle festzuschreiben ist deshalb falsch, auch wenn sie richtig
# gemessen ist. Stattdessen wird ueber die moeglichen Rollen gemittelt:
#
#     minje_erwartet = Summe_r P(Rolle = r) * Tabelle[Position, r]
#
# Dieselbe Logik wie in der Prior-Kette — nicht auf eine Zahl festlegen, wo
# die Daten eine Verteilung hergeben. Die Uebergangsraten werden je
# Zielsaison aus frueheren Saisons geschaetzt, nie gesetzt.
UEBERGANG_PRIOR = 20.0   # Schrumpfung der Uebergangsraten gegen die Randverteilung

# Wie stark die eigene Einsatzlaenge gegen die Rollenerwartung zaehlt, in
# eigenen Einsaetzen. Beide Quellen sagen Verschiedenes: die Rolle, wo ein
# Spieler *steht*; die Historie, wie sein Trainer ihn *behandelt* (wer immer
# nach 70 Minuten heruntergeht, tut das auch im neuen Verein). Gemessen auf
# der Punkteskala: nur Rolle 0,593 · nur Historie 0,591 · Mischung 0,610.
# Das Optimum ist flach zwischen k=20 und k=40.
K_MINJE = 30.0


def uebergaenge(train: pd.DataFrame) -> dict:
    """P(Rolle der Zielsaison | Rolle der Vorsaison), geschrumpft.

    ``train`` braucht ``rolle_von`` (kann None sein = ohne Vorsaison) und
    ``rolle_nach``. Die Schrumpfung gegen die Randverteilung haelt seltene
    Ausgangsrollen stabil, ohne die haeufigen zu verwaschen.
    """
    rand = train.rolle_nach.value_counts(normalize=True).to_dict()
    rand = {r: rand.get(r, 0.0) for r in ROLLEN}
    out = {"_rand": rand}
    for von, g in train.groupby(train.rolle_von.fillna("_ohne"), dropna=False):
        n = len(g)
        zaehl = g.rolle_nach.value_counts().to_dict()
        out[von] = {r: (zaehl.get(r, 0) + UEBERGANG_PRIOR * rand[r])
                    / (n + UEBERGANG_PRIOR) for r in ROLLEN}
    return out


def rollen_verteilung(uebg: dict, rolle_von) -> dict:
    """P(Rolle der Zielsaison) fuer eine gegebene Ausgangsrolle."""
    p = uebg.get("_ohne" if rolle_von is None else rolle_von, uebg["_rand"])
    return p if isinstance(p, dict) else uebg["_rand"]


def minje_erwartet(tab: dict, uebg: dict, pos: str, rolle_von) -> float:
    """Ueber die Rollenverteilung gemittelte Einsatzlaenge."""
    p = rollen_verteilung(uebg, rolle_von)
    return sum(p[r] * minje_fuer(tab, pos, r) for r in ROLLEN)


def verteilung_kalibriert(uebg: dict, rolle_von, p_gesetzt) -> dict:
    """Rollenverteilung, geeicht an der modellierten Durchsetzung.

    Die Uebergangsmatrix kennt nur die *alte* Rolle. Sie gibt damit jedem
    gesetzten Spieler dieselben 59 % Verbleib — dem Torjaeger wie dem
    Wackelkandidaten. Genau diese Unterscheidung leistet aber die
    Rollen-Logistik, die Marktwert-Rang, Kaderkonkurrenz und Alter sieht.

    Gemessen war das an der Spitze teuer: fuer die 26 Spielersaisons mit einem
    Vorjahres-Schnitt ueber 170 Punkten sagte die rohe Matrix 76,2 Minuten je
    Einsatz voraus, tatsaechlich waren es 80,1. Der Fehler ist klein — aber er
    multipliziert sich mit dem ebenfalls zu niedrigen p90.

    Also liefert die Matrix die **Form** der Verteilung und die Logistik ihre
    Masse auf "gesetzt"; der Rest wird proportional nachgezogen.
    """
    return verteilung_kalibriert_form(
        rollen_verteilung(uebg, rolle_von), p_gesetzt)


def verteilung_kalibriert_form(form: dict, p_gesetzt) -> dict:
    """Wie ``verteilung_kalibriert``, aber mit fertiger Form.

    Der Export kennt die Form aus KATEGORIE_ROLLEN statt aus der
    Uebergangsmatrix — die Eichung an der Logistik ist dieselbe.
    """
    if p_gesetzt is None or not np.isfinite(p_gesetzt):
        return dict(form)
    ziel = float(np.clip(p_gesetzt, 0.005, 0.995))
    rest = 1.0 - form["gesetzt"]
    skala = (1.0 - ziel) / rest if rest > 1e-9 else 0.0
    out = {r: form[r] * skala for r in ROLLEN}
    out["gesetzt"] = ziel
    return out


def minje_aus_verteilung(tab: dict, vert: dict, pos: str) -> float:
    return sum(vert[r] * minje_fuer(tab, pos, r) for r in ROLLEN)


def quote_aus_verteilung(qtab: dict, vert: dict) -> float:
    return sum(vert[r] * qtab.get(r, qtab["_global"]) for r in ROLLEN)


def quoten_tabelle(train: pd.DataFrame) -> dict:
    """Rolle -> Anteil der Team-Spieltage mit Einsatz.

    Getrennt von der Minutentabelle, weil beides Verschiedenes meint: *wie
    oft* jemand auf dem Platz steht und *wie lange* dann. Ein Joker hat eine
    hohe Einsatzquote bei kurzer Einsatzdauer.
    """
    t = train[train.einsatzquote.notna() & train.rolle.notna()]
    tab = {"_global": float(t.einsatzquote.mean()) if len(t) else 0.6}
    for rolle, g in t.groupby("rolle"):
        if len(g) >= MIN_ZELLE:
            tab[rolle] = float(g.einsatzquote.mean())
    return tab


def quote_erwartet(qtab: dict, uebg: dict, rolle_von) -> float:
    p = rollen_verteilung(uebg, rolle_von)
    return sum(p[r] * qtab.get(r, qtab["_global"]) for r in ROLLEN)


def baseline_fit(rows: pd.DataFrame) -> dict:
    """Je Position a_pos + b_pos * Kontext, minutengewichtet.

    Die Steigung ist positionsabhaengig — der IV-Spread (36 Punkte zwischen
    Top- und Kellerteam) ist doppelt so gross wie bei Stuermern, deren Wert
    staerker am Spieler haengt. Positionen unter 50 Zeilen fallen auf die
    globale Gerade zurueck, ebenso unbekannte Positionen.
    """
    w_all = rows.min_gesamt.values.astype(float)
    k_all = rows.kontext.values.astype(float)
    y_all = rows.p90.values.astype(float)

    def _gerade(sel: np.ndarray) -> tuple[float, float]:
        w, k, y = w_all[sel], k_all[sel], y_all[sel]
        mk = np.average(k, weights=w)
        my = np.average(y, weights=w)
        var = np.average((k - mk) ** 2, weights=w)
        b = 0.0 if var < 1e-9 else float(np.average((k - mk) * (y - my), weights=w) / var)
        return my - b * mk, b

    alle = np.ones(len(rows), dtype=bool)
    a0, b0 = _gerade(alle)
    fit = {POS_UNBEKANNT: (a0, b0)}
    for pos, grp in rows.groupby("pos"):
        sel = (rows.pos == pos).values
        fit[pos] = _gerade(sel) if sel.sum() >= 50 else (a0, b0)
    return fit


def baseline_wert(fit: dict, pos: str, kontext: float) -> float:
    a, b = fit.get(pos, fit[POS_UNBEKANNT])
    return a + b * kontext


# ---------------------------------------------------------------------------
# kappa: was eine BL2-Saison in BL1-Waehrung wert ist
# ---------------------------------------------------------------------------

def _kappa(ss: pd.DataFrame, bis_saison: str) -> float:
    """Verhaeltnis der gewichteten p90-Mittel von Wechslern BL2 -> BL1.

    Nur Uebergaenge, deren BL1-Saison **vor** der Zielsaison liegt — der Wert
    ist damit je Zielsaison walk-forward. Quotient der Mittel statt Mittel der
    Quotienten, weil einzelne p90 nahe null (oder negativ) jeden Einzelquotienten
    sprengen. Solange keine Uebergaenge messbar sind, gilt der Fallback.
    """
    prev = _prev_of(sorted(ss.season.unique()))
    a = ss[(ss.league == "2. Bundesliga") & (ss.min_gesamt >= MIN_TRAIN)]
    b = ss[(ss.league == "Bundesliga") & (ss.min_gesamt >= MIN_TRAIN)
           & (ss.season < bis_saison)]
    b = b.assign(season_prev=b.season.map(prev))
    m = a.merge(b, left_on=["player_id", "season"],
                right_on=["player_id", "season_prev"], suffixes=("_2", "_1"))
    if len(m) < 20:
        return KAPPA_BL2
    w = np.minimum(m.min_gesamt_1.values, m.min_gesamt_2.values).astype(float)
    oben = float(np.average(m.p90_1.values, weights=w))
    unten = float(np.average(m.p90_2.values, weights=w))
    if unten <= 1e-9:
        return KAPPA_BL2
    return float(np.clip(oben / unten, 0.6, 1.0))


# ---------------------------------------------------------------------------
# Historien-Aggregat je Spieler
# ---------------------------------------------------------------------------

def historie(hist: pd.DataFrame, saison: str, delta: float, kappa: float,
             bl2_gewicht: float = 1.0) -> pd.DataFrame:
    """p90_hist, n_eff und BL2-Anteil je Spieler aus allen Saisons < saison.

    Minutengewichtet mit Abkling je Saison Abstand; BL2-Zeilen gehen mit
    kappa-skaliertem p90 ein. n_eff zaehlt in effektiven Spielen (90er-Bloecke)
    und steuert damit direkt die Mischung gegen die Baseline. ``bl2_gewicht``
    laesst BL2-Minuten schwaecher zaehlen — eine BL2-Saison sagt weniger ueber
    BL1-p90 als eine BL1-Saison, unabhaengig vom Niveaufaktor kappa.
    """
    h = hist[hist.season < saison]
    h = h[h.min_gesamt > 0]
    if h.empty:
        return pd.DataFrame(columns=["player_id", "p90_hist", "n_eff", "anteil_bl2"])

    jahre = sorted(hist.season.unique().tolist() + [saison])
    rang = {s: i for i, s in enumerate(jahre)}
    abst = h.season.map(rang).values
    ist_bl2 = (h.league == "2. Bundesliga").values
    gew = (delta ** (rang[saison] - abst - 1) * h.min_gesamt.values
           * np.where(ist_bl2, bl2_gewicht, 1.0))
    p90 = np.where(ist_bl2, h.p90.values * kappa, h.p90.values)

    # Dieselben Gewichte tragen den zweiten Mittelwert: die eigene Einsatzlaenge
    # und der eigene Punkteschnitt. Sie stecken in derselben Funktion, damit
    # "letzte Saison zaehlt mehr" an genau einer Stelle definiert ist.
    avg = np.where(h.einsaetze.values > 0, h.punkte.values / h.einsaetze.values, np.nan)
    avg = np.where(ist_bl2, avg * kappa, avg)
    g = pd.DataFrame({"player_id": h.player_id.values, "w": gew,
                      "wp": gew * p90, "w2": np.where(ist_bl2, gew, 0.0),
                      "wm": gew * h.min_je_einsatz.values,
                      "wa": gew * np.nan_to_num(avg),
                      "wa_gew": np.where(np.isnan(avg), 0.0, gew),
                      "e": (delta ** (rang[saison] - abst - 1)) * h.einsaetze.values})
    g = g.groupby("player_id", as_index=False).sum()
    g["p90_hist"] = g.wp / g.w
    g["n_eff"] = g.w / 90.0
    g["anteil_bl2"] = g.w2 / g.w
    g["minje_hist"] = g.wm / g.w
    g["avg_hist"] = np.where(g.wa_gew > 0, g.wa / g.wa_gew.replace(0, np.nan), np.nan)
    g["n_eins_eff"] = g.e
    return g[["player_id", "p90_hist", "n_eff", "anteil_bl2",
              "minje_hist", "avg_hist", "n_eins_eff"]]


# ---------------------------------------------------------------------------
# Merkmalszeilen fuer eine Zielsaison — alles walk-forward
# ---------------------------------------------------------------------------

def merkmalszeilen(saison: str, kandidaten: pd.DataFrame, ss: pd.DataFrame,
                   kon: dict, tm: pd.DataFrame, delta: float, k_mix: float,
                   pool_bl2: bool, ohne_teamstaerke: bool,
                   vorgaenger: pd.DataFrame | None = None,
                   bl2_gewicht: float = BL2_GEWICHT,
                   k_tw: float = K_TW) -> pd.DataFrame:
    """Eine Merkmalszeile je Kandidat der Zielsaison.

    ``kandidaten`` traegt nur Kaderzugehoerigkeit (player_id, team_id) — die
    einzige erlaubte Information aus der Zielsaison selbst. Alles andere
    stammt aus Saisons davor (``ss``), dem TM-Kader der Zielsaison
    (vor Saisonstart bekannt: MW-Stichtag 1. August, Alter, Position) und dem
    Vorsaison-Teamkontext ``kon``.

    Der Schnitt auf Saisons < ``saison`` passiert **hier** und nicht beim
    Aufrufer. Das ist keine Bequemlichkeit: liegt er beim Aufrufer, waehlen
    "letzte Saison" und "Vorsaison-Rangfolge" stillschweigend die Zielsaison
    selbst, sobald jemand das ungefilterte Panel hereinreicht — und Leakage
    faellt nicht auf, sie verbessert die Zahlen.
    """
    ss = ss[ss.season < saison]
    prev = _prev_of(sorted(ss.season.unique().tolist() + [saison]))

    # Die eigene Historie umfasst immer beide Ligen — dass ein Aufsteiger
    # seine BL2-Saisons mitbringt, ist beschlossen, nicht schaltbar.
    # --pool-bl2 betrifft nur, ob BL2-Saisons auch als Trainingszeilen dienen.
    kappa = _kappa(ss, saison)
    h = historie(ss, saison, delta, kappa, bl2_gewicht)

    tmz = tm[tm.season == saison][["player_id", "position_fine", "alter",
                                   "log2_mw", "mw_vor_saison",
                                   "mw_rang_pos", "n_konkurrenten"]]
    df = kandidaten.merge(h, on="player_id", how="left")
    df = df.merge(tmz, on="player_id", how="left")
    if "pos_override" in df:
        df["pos"] = df.pos_override.fillna(df.position_fine).fillna(POS_UNBEKANNT)
    else:
        df["pos"] = df.position_fine.fillna(POS_UNBEKANNT)
    df["saison"] = saison

    if ohne_teamstaerke:
        df["kontext"] = 0.0
    else:
        df["kontext"] = [_kontext(kon, saison, t, p)
                         for t, p in zip(df.team_id, df.pos)]

    # Baseline und MW-Erwartung auf Trainingszeilen < saison fitten. Beide
    # nutzen als Kontext ebenfalls die jeweilige Vorsaison — siehe Moduldocstring.
    train = ss[(ss.season < saison) & (ss.min_gesamt >= MIN_TRAIN)]
    if not pool_bl2:
        train = train[train.league == "Bundesliga"]
    train = train.merge(tm[["player_id", "season", "position_fine", "log2_mw"]],
                        on=["player_id", "season"], how="left")
    train = train.assign(pos=train.position_fine.fillna(POS_UNBEKANNT))
    if ohne_teamstaerke:
        train = train.assign(kontext=0.0)
    else:
        train = train.assign(kontext=[
            _kontext(kon, s, t, p)
            for s, t, p in zip(train.season, train.team_id, train.pos)])
    train = train[[s in prev for s in train.season]]

    bfit = baseline_fit(train)
    df["baseline"] = [baseline_wert(bfit, p, kx)
                      for p, kx in zip(df.pos, df.kontext)]
    df["p90_mix"] = [
        p90_mischung(n if np.isfinite(n) else 0.0,
                     ph if pd.notna(ph) else np.nan,
                     b, k_tw if p == TW else k_mix)
        for n, ph, b, p in zip(df.n_eff.fillna(0.0), df.p90_hist,
                               df.baseline, df.pos)]

    # MW-Residuum: der Marktwert traegt bei bekannter Vorsaison fast nichts
    # (+0,03 R2) — nur seine Abweichung von der Positions-x-Staerke-Erwartung
    # ist noch Information (z. B. "teurer Neuzugang bei mittlerem Team").
    mwt = train[train.log2_mw.notna()]
    if len(mwt) >= 100:
        mfit = baseline_fit(mwt.assign(p90=mwt.log2_mw))
        erw = np.array([baseline_wert(mfit, p, kx)
                        for p, kx in zip(df.pos, df.kontext)])
        df["mw_resid"] = (df.log2_mw - erw).fillna(0.0)
    else:
        df["mw_resid"] = 0.0

    # Teamwechsel: Differenz des Kontexts neu gegen alt, nur wenn die
    # Vorsaison in der Bundesliga stattfand — Ligen-z sind keine gemeinsame
    # Skala, Aufsteiger-Spieler traegt stattdessen anteil_bl2.
    vor = ss[(ss.season == prev.get(saison, "")) & (ss.league == "Bundesliga")]
    vor = vor.set_index("player_id")
    delta_team = []
    for pid, team, pos, kx in zip(df.player_id, df.team_id, df.pos, df.kontext):
        alt = vor.team_id.get(pid)
        if alt is None or pd.isna(alt) or alt == team:
            delta_team.append(0.0)
        else:
            delta_team.append(kx - _kontext(kon, saison, alt, pos))
    df["team_delta"] = delta_team

    df["alter_c"] = (df.alter - 27.0).fillna(0.0)
    df["alter_fehlt"] = df.alter.isna().astype(float)
    df["neuzugang"] = (df.n_eff.fillna(0.0) <= 0).astype(float)
    df["anteil_bl2"] = df.anteil_bl2.fillna(0.0)

    # Wie gut ein Spieler belegt ist — und wie weit er ueber der Baseline
    # seiner Position und seines Teams liegt. Ohne dieses Paar schrumpft die
    # Ridge zweimal: die Prior-Kette gewichtet bereits nach Datenmenge, die
    # Ridge legt danach eine pauschale zweite Schrumpfung darauf, die fuer
    # einen Spieler mit 90 belegten Spielen genauso stark ist wie fuer einen
    # mit dreien. Gemessen kostete das die Gruppe mit langer, starker
    # Historie 12 Punkte (Modell 175,2 gegen 187,4 tatsaechlich), waehrend
    # einmalige Ausreisser voellig richtig gestutzt wurden. Erst die
    # Wechselwirkung erlaubt beides zugleich.
    n_eff = df.n_eff.fillna(0.0)
    df["belegt"] = n_eff / (n_eff + 30.0)
    df["abw_belegt"] = (df.p90_mix - df.baseline) * df.belegt

    # Minuten-Merkmale: letzte und vorletzte Saison, ligaunabhaengig — die
    # Startquote eines BL2-Stammspielers ist echte Information ueber seine
    # Rolle, anders als sein p90-Niveau. Bei zwei Zeilen (Ligawechsler im
    # Winter) zaehlt die mit mehr Minuten.
    def _vorsaison(s: str, suffix: str) -> pd.DataFrame:
        v = ss[ss.season == s].sort_values("min_gesamt").drop_duplicates(
            "player_id", keep="last")
        v = v[["player_id", "min_je_teamspieltag", "startquote", "jokerquote",
               "kader_fehlquote", "min_je_einsatz", "einsatzquote"]]
        v.columns = ["player_id"] + [c + suffix for c in v.columns[1:]]
        return v

    jahre = sorted(ss.season.unique().tolist() + [saison])
    v1 = jahre[-2] if len(jahre) >= 2 else ""
    v2 = jahre[-3] if len(jahre) >= 3 else ""
    df = df.merge(_vorsaison(v1, "_v1"), on="player_id", how="left")
    df = df.merge(_vorsaison(v2, "_v2"), on="player_id", how="left")
    df["hat_v1"] = df.min_je_teamspieltag_v1.notna().astype(float)
    for c in ("min_je_teamspieltag_v1", "startquote_v1", "jokerquote_v1",
              "kader_fehlquote_v1", "min_je_einsatz_v1",
              "min_je_teamspieltag_v2", "startquote_v2"):
        df[c] = df[c].fillna(0.0)

    # MW-Rang im Zielkader: fehlt er (kein TM-MW), wird tief im Kader
    # angenommen — ein unbewerteter Spieler ist praktisch nie ein Stammkandidat.
    df["mw_rang_c"] = df.mw_rang_pos.fillna(5.0)
    df["n_konk_c"] = df.n_konkurrenten.fillna(2.0)
    df["alter_q"] = df.alter_c ** 2
    df["neu_x_rang"] = df.neuzugang * df.mw_rang_c
    df["ist_tw"] = (df.pos == TW).astype(float)
    df["einsatzquote_v1"] = df.einsatzquote_v1.fillna(0.0)

    # Rolle und die daraus folgende Einsatzlaenge. Die Zielrolle ist hier die
    # der Vorsaison — das ist alles, was der Backtest vorab wissen kann. Der
    # Export ersetzt sie durch die handgepflegte ``kategorie``, die Transfers
    # kennt; wie viel das wert ist, beziffert der Orakel-Lauf.
    #
    # Eigene, breitere Trainingsmenge: ``train`` verlangt MIN_TRAIN Minuten und
    # kennt die Rolle "Rand" damit fast nicht. Fuer die Tabelle genuegen drei
    # Einsaetze — ab dort ist "Minuten je Einsatz" ueberhaupt eine Groesse.
    tr_rolle = ss[(ss.league == "Bundesliga") & (ss.einsaetze >= 3)].merge(
        tm[["player_id", "season", "position_fine"]],
        on=["player_id", "season"], how="left")
    tr_rolle = tr_rolle.assign(
        pos=tr_rolle.position_fine.fillna(POS_UNBEKANNT),
        rolle=rolle_aus_startquote(tr_rolle.startquote))
    tab = rollen_tabelle(tr_rolle)

    # Uebergangsraten: aus je zwei aufeinanderfolgenden Saisons desselben
    # Spielers, ausschliesslich aus Saisons vor der Zielsaison.
    tr_rolle = tr_rolle.sort_values("season")
    vorher = tr_rolle[["player_id", "season", "rolle"]].rename(
        columns={"rolle": "rolle_von"})
    jahre_alle = sorted(ss.season.unique().tolist() + [saison])
    folgt = {a: b for a, b in zip(jahre_alle, jahre_alle[1:])}
    vorher["season"] = vorher.season.map(folgt)
    uebg_train = tr_rolle[["player_id", "season", "rolle"]].rename(
        columns={"rolle": "rolle_nach"}).merge(
        vorher, on=["player_id", "season"], how="left")
    uebg = uebergaenge(uebg_train)

    df["rolle_v1"] = rolle_aus_startquote(
        df.startquote_v1.where(df.hat_v1 > 0))
    df["zielrolle"] = df.rolle_v1
    df["minje_rolle"] = [minje_erwartet(tab, uebg, p, r)
                         for p, r in zip(df.pos, df.zielrolle)]
    # Die Quotentabelle braucht **alle** Kaderspieler, auch die ohne einen
    # einzigen Einsatz — sonst waere sie auf die Spielenden bedingt und ein
    # Reservist bekaeme die Einsatzquote derer, die es doch aufs Feld
    # geschafft haben. Die Minutentabelle darf das nicht: ohne Einsatz gibt
    # es keine Einsatzlaenge.
    alle_rollen = ss[ss.league == "Bundesliga"].assign(
        rolle=rolle_aus_startquote(ss[ss.league == "Bundesliga"].startquote))
    qtab = quoten_tabelle(alle_rollen)
    df["quote_rolle"] = [quote_erwartet(qtab, uebg, r) for r in df.zielrolle]
    df.attrs["rollen_tabelle"] = tab
    df.attrs["quoten_tabelle"] = qtab
    df.attrs["uebergaenge"] = uebg

    # Rollenerwartung und eigene Einsatzlaenge mischen, gewichtet nach der
    # Zahl eigener Einsaetze — wer keine hat, bekommt reine Rollenerwartung.
    n_e = df.n_eins_eff.fillna(0.0)
    w_rolle = K_MINJE / (n_e + K_MINJE)
    df["minje_dach_neu"] = (w_rolle * df.minje_rolle
                            + (1 - w_rolle) * df.minje_hist.fillna(df.minje_rolle))

    # Basisrate fuer die Rollenwahrscheinlichkeit, direkt aus den
    # Uebergaengen; die Logistik verfeinert sie je Spieler.
    basis = np.array([
        (uebg.get("_ohne" if r is None else r, uebg["_rand"]))["gesetzt"]
        for r in df.zielrolle])
    basis = np.clip(basis, 0.01, 0.99)
    df["p_basis"] = basis
    df["p_basis_logit"] = np.log(basis / (1 - basis))
    # Einsatzlaenge bei gehaltener Zielrolle — das Aufwaertspotenzial.
    df["minje_gesetzt"] = [minje_fuer(tab, p, "gesetzt") for p in df.pos]

    df["n_eins_eff"] = df.n_eins_eff.fillna(0.0)

    if vorgaenger is not None:
        v = vorgaenger[vorgaenger.season == saison]
        df = df.merge(v[["team_id", "pos", "vorgaenger_p90"]],
                      on=["team_id", "pos"], how="left")
        df["vorgaenger_diff"] = (df.vorgaenger_p90 - df.baseline).fillna(0.0)
    return df


# Die Ridge korrigiert die Prior-Kette nur noch **additiv**: sie schaetzt den
# Rest ``ist - p90_mix``, statt p90_mix selbst zu skalieren. p90_mix steht
# deshalb nicht mehr in der Merkmalsliste.
#
# Grund, gemessen: die Kette schrumpft eine Abweichung von der Baseline
# bereits datenabhaengig, und zwar richtig. Der tatsaechlich ueberlebende
# Anteil betraegt 0,00 / 0,47 / 0,60 / 0,78 bei n_eff 0-10 / 10-25 / 25-45 /
# 45+, und n_eff/(n_eff + K_MIX) bildet genau das ab. Eine Ridge auf p90_mix
# legt darauf eine zweite, pauschale Schrumpfung - fuer Spieler mit langer
# starker Historie kostete das 8 Punkte (181,5 aus der Kette auf 173,9,
# tatsaechlich 187,4), waehrend sie zur Guete nichts beitrug (r 0,693 gegen
# 0,692 ohne sie). Als additive Korrektur bleibt ihr Nutzen (Alter,
# MW-Residuum, Teamwechsel) und der Schaden entfaellt.
MERKMALE = ["belegt", "alter_c", "alter_fehlt", "mw_resid", "team_delta",
            "neuzugang", "anteil_bl2"]

# Rollenwahrscheinlichkeit: setzt sich der Spieler durch? Die Merkmale sind
# gemessen wirksam — innerhalb der Vorjahres-Rotation trennen Marktwert-Rang
# und Vorjahres-Startquote von 9,1 % bis 33,2 % Durchsetzung (Basisrate 19 %).
# Genau deshalb muessen die handgepflegten Kategorien *nicht* verfeinert
# werden: die Feinunterscheidung innerhalb einer Kategorie leistet das Modell.
ROLLE_MERKMALE = ["p_basis_logit", "startquote_v1", "einsatzquote_v1",
                  "min_je_teamspieltag_v1", "kader_fehlquote_v1", "hat_v1",
                  "mw_rang_c", "n_konk_c", "mw_resid", "alter_c", "alter_q",
                  "neuzugang", "neu_x_rang", "ist_tw"]


def vorgaenger_tabelle(ss: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """p90 der Rollenvorgaenger: wer auf (Team, Position) der **Vorsaison**
    die Minuten hatte, minutengewichtet. Experiment hinter --vorgaenger —
    die Positions-x-Staerke-Baseline ist die robuste Verallgemeinerung,
    hier wird gemessen, ob der namentliche Slot mehr weiss (n ist 1-3)."""
    m = ss.merge(tm[["player_id", "season", "position_fine"]],
                 on=["player_id", "season"], how="left")
    m = m[m.position_fine.notna() & (m.min_gesamt > 0)]
    g = (m.assign(wp=m.p90 * m.min_gesamt)
          .groupby(["season", "league", "team_id", "position_fine"], as_index=False)
          .agg(wp=("wp", "sum"), w=("min_gesamt", "sum")))
    g["vorgaenger_p90"] = g.wp / g.w
    prev = _prev_of(sorted(ss.season.unique()))
    nxt = {a: b for b, a in prev.items()}
    g["season"] = g.season.map(nxt)
    g = g[g.season.notna()].rename(columns={"position_fine": "pos"})
    return g[["season", "team_id", "pos", "vorgaenger_p90"]]


# ---------------------------------------------------------------------------
# Walk-forward-Backtest
# ---------------------------------------------------------------------------

def backtest_p90(delta: float = DELTA, k_mix: float = K_MIX,
                 pool_bl2: bool = False, ohne_teamstaerke: bool = False,
                 mit_vorgaenger: bool = False, nur_mix: bool = False,
                 bl2_gewicht: float = BL2_GEWICHT, k_tw: float = K_TW,
                 nur_kalibrierzeilen: bool = False,
                 orakel: bool = False) -> pd.DataFrame:
    """Je Zielsaison 2015/16+ nur aus frueheren Saisons vorhersagen.

    Rueckgabe: eine Zeile je Kandidat mit Merkmalen, Vorhersage (roh und
    kalibriert), Hinrunden-Ist und Ganzjahres-Ist. ``nur_mix`` ueberspringt
    die p90-Ridge — der Abstand zwischen beiden Spalten misst, was die
    Kalibrierung ueber die Prior-Kette hinaus beitraegt.
    ``nur_kalibrierzeilen`` liefert die Merkmalszeilen ohne jede Kalibrierung;
    ``export`` fittet daraus einen einzigen Satz Koeffizienten fuer die
    Zielsaison, statt die Walk-forward-Kette zu wiederholen.
    """
    panel = lade_panel()
    panel = panel[panel.season != "2026/2027"]
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    matches = matches[matches.season != "2026/2027"]

    ss = spieler_saisons(panel)
    hin = spieler_saisons(panel, hinrunde=True, splits=load_splits())
    kon = kontext_tabelle(team_niveaus(conceded_from_panel(panel, matches)),
                          sorted(ss.season.unique()))
    tm = tm_je_spieler(lade_tm())
    vg = vorgaenger_tabelle(ss, tm) if mit_vorgaenger else None

    seasons = sorted(ss[ss.league == "Bundesliga"].season.unique())
    ziele = [s for s in seasons if s >= "2015/2016"]

    frames = []
    tabellen, uebergangs, quoten = {}, {}, {}
    for saison in ziele:
        bl = ss[(ss.season == saison) & (ss.league == "Bundesliga")]
        kandidaten = bl[["player_id", "team_id"]].copy()
        df = merkmalszeilen(saison, kandidaten, ss[ss.season < saison], kon,
                            tm, delta, k_mix, pool_bl2, ohne_teamstaerke, vg,
                            bl2_gewicht, k_tw)
        tabellen[saison] = df.attrs["rollen_tabelle"]
        uebergangs[saison] = df.attrs["uebergaenge"]
        quoten[saison] = df.attrs["quoten_tabelle"]

        ist = hin[(hin.season == saison) & (hin.league == "Bundesliga")]
        df = df.merge(ist[["player_id", "p90", "min_gesamt", "punkte",
                           "einsaetze", "min_je_einsatz", "min_je_teamspieltag",
                           "n_teamspieltage", "startquote"]]
                      .rename(columns={"p90": "ist_p90", "min_gesamt": "ist_min",
                                       "punkte": "ist_punkte",
                                       "einsaetze": "ist_einsaetze",
                                       "min_je_einsatz": "ist_minje",
                                       "min_je_teamspieltag": "ist_minjt",
                                       "n_teamspieltage": "ist_st",
                                       "startquote": "ist_startquote"}),
                      on="player_id", how="left")
        df = df.merge(bl[["player_id", "p90", "min_gesamt", "min_je_teamspieltag",
                          "min_je_einsatz", "punkte", "einsaetze"]]
                      .rename(columns={"p90": "jahr_p90", "min_gesamt": "jahr_min",
                                       "min_je_teamspieltag": "jahr_minjt",
                                       "min_je_einsatz": "jahr_minje",
                                       "punkte": "jahr_punkte",
                                       "einsaetze": "jahr_einsaetze"}),
                      on="player_id", how="left")

        # Naive Baselines aus der Vorsaison (Bundesliga, ganze Saison): der
        # p90-Wert fuer den Qualitaetskanal, der Punkteschnitt fuer die
        # eigentliche Zielgroesse. Letzterer ist die Latte, an der v1
        # gescheitert ist — stures Fortschreiben traf die Oe Punkte besser
        # als das Modell.
        prev = _prev_of(seasons)
        v = ss[(ss.season == prev.get(saison, "")) & (ss.league == "Bundesliga")].copy()
        v["naiv_avg"] = np.where(v.einsaetze > 0, v.punkte / v.einsaetze, np.nan)
        df = df.merge(v[["player_id", "p90", "min_gesamt", "naiv_avg", "einsaetze"]]
                      .rename(columns={"p90": "naiv_p90", "min_gesamt": "naiv_min",
                                       "einsaetze": "naiv_einsaetze"}),
                      on="player_id", how="left")
        frames.append(df)

    rows = pd.concat(frames, ignore_index=True)
    # Die Zielgroesse: Oe Punkte je Einsatz der Hinrunde — die Skala, die
    # Kickbase als "Oe Punkte" zeigt und auf der ein Kader gebaut wird.
    rows["ist_avg"] = np.where(rows.ist_einsaetze.fillna(0) > 0,
                               rows.ist_punkte / rows.ist_einsaetze, np.nan)

    # Ridge-Kalibrierung: fuer Zielsaison s auf allen frueheren Zielsaisons
    # trainiert, deren Merkmale ihrerseits nur noch aeltere Saisons sahen.
    merkmale = MERKMALE + (["vorgaenger_diff"] if mit_vorgaenger else [])
    rows["p90_dach"] = rows.p90_mix
    if nur_kalibrierzeilen:
        return rows
    if not nur_mix:
        feld = rows.pos != TW
        for saison in ziele[1:]:
            tr = rows[(rows.saison < saison) & feld
                      & (rows.jahr_min >= MIN_TRAIN) & rows.jahr_p90.notna()]
            if len(tr) < 200:
                continue
            fit = _ridge(tr[merkmale].values.astype(float),
                         (tr.jahr_p90 - tr.p90_mix).values.astype(float),
                         tr.jahr_min.values.astype(float), LAMBDA_B)
            sel = (rows.saison == saison) & feld
            rows.loc[sel, "p90_dach"] = rows.loc[sel, "p90_mix"] + _ridge_pred(
                fit, rows.loc[sel, merkmale].values.astype(float))

    # --- Rollenwahrscheinlichkeit -----------------------------------------
    # Wie viele Spieler sich durchsetzen, driftet stark: von 35,2 % (2015/16)
    # auf 16,0 % (2025/26) — die Kader sind gewachsen (460 auf 518 bewertete
    # Spieler) und es wird mehr rotiert. Ein Fit ueber alle Vorsaisons
    # ueberschaetzt die Gegenwart deshalb systematisch. Die Abklinggewichtung
    # ist dieselbe Antwort wie DECAY im Team-Rating.
    rows["ist_gesetzt"] = (rows.ist_startquote > ROLLEN_GRENZEN[-1]).astype(float)
    rows["p_rolle"] = rows.p_basis
    rang = {s: i for i, s in enumerate(ziele)}
    for saison in ziele[1:]:
        tr = rows[(rows.saison < saison) & rows.ist_startquote.notna()]
        if len(tr) < 300:
            continue
        w = DELTA_ROLLE ** (rang[saison] - tr.saison.map(rang).values - 1)
        fit = _logit(tr[ROLLE_MERKMALE].values.astype(float),
                     tr.ist_gesetzt.values.astype(float), w)
        # Niveau an der juengsten Trainingssaison nacheichen. Der Fit mittelt
        # ueber elf Saisons, die Quote faellt aber stetig — ohne diesen
        # Schritt sagt das Modell durchgaengig zu hohe Wahrscheinlichkeiten.
        # Verschoben wird nur der Achsenabschnitt: die Rangfolge und damit
        # AUC bleiben unberuehrt, die Kalibrierung wird besser.
        letzte = tr[tr.saison == ziele[rang[saison] - 1]]
        if len(letzte) > 100:
            p_l = _logit_pred(fit, letzte[ROLLE_MERKMALE].values.astype(float))
            ziel_rate = float(letzte.ist_gesetzt.mean())
            versatz = 0.0
            for _ in range(40):
                q = 1 / (1 + np.exp(-(np.log(p_l / (1 - p_l)) + versatz)))
                if abs(q.mean() - ziel_rate) < 1e-4:
                    break
                versatz += (ziel_rate - q.mean()) * 4.0
            fit = {**fit, "beta": fit["beta"].copy()}
            fit["beta"][0] += versatz

        sel = rows.saison == saison
        rows.loc[sel, "p_rolle"] = np.clip(_logit_pred(
            fit, rows.loc[sel, ROLLE_MERKMALE].values.astype(float)), 0.005, 0.995)

    # Die Rollenwahrscheinlichkeit eicht jetzt die Verteilung, aus der die
    # Einsatzlaenge gemittelt wird. Deshalb steht sie hier oben und nicht mehr
    # am Ende: sie ist keine Zusatzangabe fuer den Export, sondern ein Eingang
    # des Ertragspfads. Siehe verteilung_kalibriert fuer das Warum.
    vert = [verteilung_kalibriert(uebergangs[sa], r, pr)
            for sa, r, pr in zip(rows.saison, rows.zielrolle, rows.p_rolle)]
    rows["minje_rolle"] = [minje_aus_verteilung(tabellen[sa], v, p)
                           for sa, v, p in zip(rows.saison, vert, rows.pos)]
    rows["quote_rolle"] = [quote_aus_verteilung(quoten[sa], v)
                           for sa, v in zip(rows.saison, vert)]
    w_rolle = K_MINJE / (rows.n_eins_eff.fillna(0.0) + K_MINJE)
    rows["minje_dach_neu"] = (
        w_rolle * rows.minje_rolle
        + (1 - w_rolle) * rows.minje_hist.fillna(rows.minje_rolle))

    # Die Einsatzlaenge bei gehaltener Rolle darf nie unter der erwarteten
    # liegen. Seit die Verteilung an p_rolle geeicht wird, kann sie das: wer
    # laenger spielt als die Zellennorm seiner Position (Kleindienst 88 Min
    # gegen ST-gesetzt 80,3), bekaeme sonst ein "Potenzial" unter seiner
    # eigenen Erwartung. Die Zelle ist eine Norm, kein Deckel.
    rows["minje_gesetzt"] = np.maximum(rows.minje_gesetzt, rows.minje_dach_neu)

    # Der neue Ertragspfad: Qualitaet x Rolle. ``minje_rolle`` traegt die
    # Zielrolle; mit ``orakel`` wird sie durch die tatsaechliche Rolle der
    # Zielsaison ersetzt — das ist Wissen aus der Zukunft und ausschliesslich
    # dazu da, den Wert guter Kaderpflege zu beziffern (0,627 -> 0,759).
    if orakel:
        # Die Rolle ist bekannt, also entfaellt die Mittelung ueber die
        # Uebergaenge — und die eigene Historie wird nicht mehr gebraucht,
        # weil sie nur ein Ersatz fuer eben diese Auskunft war.
        rows["zielrolle"] = rolle_aus_startquote(rows.ist_startquote)
        rows["minje_dach_neu"] = [
            minje_fuer(tabellen[s], p, r)
            for s, p, r in zip(rows.saison, rows.pos, rows.zielrolle)]
    rows["xp_rolle"] = rows.p90_dach * rows.minje_dach_neu / 90.0

    # **Keine Kalibrierung mehr auf dem Ertrag.** Bis zum Umbau stand hier
    # eine additive Ridge. Sie war schon immer redundant — ihre wirksamen
    # Merkmale (Marktwert-Residuum, Alter, Teamwechsel, Belegtheit) stecken
    # bereits in der p90-Ridge, die rollenbezogenen traegt seit der Eichung
    # der Rollenverteilung p_rolle. Uebrig blieb ihre Niveauwirkung, und die
    # war schaedlich: sie mittelt den Rest ueber alle frueheren Zielsaisons,
    # in denen das Produkt zeitweise stark ueberschaetzte (2020/21 -11,3;
    # 2022/23 -11,1), und trug diese Korrektur in eine Gegenwart, in der es
    # unverzerrt ist (2023/24 -1,5; 2024/25 -2,3; 2025/26 -0,2).
    #
    # Gemessen, mit Ridge gegen ohne: r 0,609 -> 0,618, r je Zielsaison
    # 0,617 -> 0,621, MAE 23,24 -> 23,13, Verzerrung der juengsten drei
    # Saisons +4,73 -> -1,36, Verzerrung der Vorjahres-Spitze (>=170)
    # +15,9 -> +13,2 bei MAE 30,6 -> 28,8. Eine niveaufreie Variante (Rest je
    # Trainingssaison zentriert) lag dazwischen: r 0,614, MAE 23,48.
    #
    # Das Niveau kommt damit allein aus der Prior-Kette, die es je Zielsaison
    # aus frischen Daten hat, statt aus einem Mittel ueber elf Jahre.
    rows["jahr_avg"] = np.where(rows.jahr_einsaetze.fillna(0) > 0,
                                rows.jahr_punkte / rows.jahr_einsaetze, np.nan)
    rows["xp_dach"] = rows.xp_rolle

    # Ertrag bei gehaltener Zielrolle: dieselbe Korrektur, aber mit der
    # Einsatzlaenge eines gesetzten Spielers. Das ist das Aufwaertspotenzial,
    # das der Prognose-Modus "Potenzial" zeigt.
    rows["xp_gesetzt"] = rows.xp_dach + rows.p90_dach * (
        rows.minje_gesetzt - rows.minje_dach_neu) / 90.0

    return rows


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def _p90_zeile(label: str, df: pd.DataFrame, spalte: str) -> None:
    from scipy.stats import spearmanr
    y, p = df.ist_p90.values, df[spalte].values
    mae = float(np.abs(y - p).mean())
    print(f"  {label:34s} n={len(df):5d}  r={_corr(p, y):6.3f}  "
          f"rho={float(spearmanr(p, y).statistic):6.3f}  MAE={mae:5.1f}")


def report_p90(rows: pd.DataFrame) -> dict:
    """Kennzahlen des p90-Kanals gegen die naive Baseline, auf einer Maske."""
    feld = rows[(rows.pos != TW) & (rows.ist_min >= MIN_ZIEL)
                & rows.ist_p90.notna()].copy()
    print(f"\np90-Kanal, Hinrunde >= {MIN_ZIEL} Min, ohne TW")
    _p90_zeile("Modell (alle bewertbaren)", feld, "p90_dach")
    _p90_zeile("Prior-Kette ohne Ridge", feld, "p90_mix")

    beide = feld[feld.naiv_p90.notna() & (feld.naiv_min >= 900)]
    _p90_zeile("Modell   | Vorsaison >900", beide, "p90_dach")
    _p90_zeile("naiv p90 | Vorsaison >900", beide, "naiv_p90")

    print("  nach Historie:")
    gruppen = [
        ("Stamm-Vorsaison (>=900)", feld.naiv_min >= 900),
        ("Teil-Vorsaison (1-899)", (feld.naiv_min > 0) & (feld.naiv_min < 900)),
        ("keine BL-Vorsaison", feld.naiv_min.isna() | (feld.naiv_min == 0)),
        ("davon mit BL2-Historie", (feld.naiv_min.isna() | (feld.naiv_min == 0))
         & (feld.anteil_bl2 > 0)),
        ("davon ganz ohne Historie", feld.neuzugang == 1),
    ]
    for label, sel in gruppen:
        g = feld[sel]
        if len(g) >= 30:
            _p90_zeile("  " + label, g, "p90_dach")

    r_modell = _corr(beide.p90_dach.values, beide.ist_p90.values)
    r_naiv = _corr(beide.naiv_p90.values, beide.ist_p90.values)

    tw = rows[(rows.pos == TW) & (rows.ist_min >= MIN_ZIEL) & rows.ist_p90.notna()]
    tw_beide = tw[tw.naiv_p90.notna() & (tw.naiv_min >= 900)]
    if len(tw_beide) >= 30:
        print("  Torhueter:")
        _p90_zeile("  Modell   | Vorsaison >900", tw_beide, "p90_dach")
        _p90_zeile("  naiv p90 | Vorsaison >900", tw_beide, "naiv_p90")
    return {"n": len(feld), "r_modell": r_modell, "r_naiv": r_naiv}


def _ertrag_zeile(label: str, df: pd.DataFrame, spalte: str) -> dict | None:
    """Eine Zeile des Ertragsberichts — mit **Bias**.

    Der Bias ist die Kennzahl, die v1 fehlte: eine Rollenwechsel-Verzerrung
    (Ex-Joker um 7 Punkte zu niedrig, Ex-Stammspieler um 6 zu hoch) faellt in
    r und MAE kaum auf, macht die Prognose je Spieler aber unbrauchbar.
    """
    from scipy.stats import spearmanr
    g = df[df[spalte].notna() & df.ist_avg.notna()]
    if len(g) < 30:
        return None
    y, p = g.ist_avg.values, g[spalte].values
    m = {"n": len(g), "r": _corr(p, y),
         "rho": float(spearmanr(p, y).statistic),
         "mae": float(np.abs(y - p).mean()),
         "bias": float((y - p).mean())}
    print(f"  {label:38s} n={m['n']:5d}  r={m['r']:6.3f}  rho={m['rho']:6.3f}  "
          f"MAE={m['mae']:5.1f}  Bias={m['bias']:+5.1f}")
    return m


def _top_treffer(rows: pd.DataFrame, spalte: str, k: int) -> float:
    """Anteil der tatsaechlichen Top-k einer Saison, der auch prognostiziert
    unter den Top-k steht. Gemittelt ueber die Zielsaisons.

    Warum diese Kennzahl neben r: r bewertet die ganze Reihenfolge gleich
    stark, auch die Sortierung im Mittelfeld. Beim Kaderbau zaehlt, ob die
    besten zwanzig Spieler erkannt werden — ein Modell darf die Reihenfolge
    der Randspieler ruhig verfehlen.
    """
    tr = []
    for _, g in rows.groupby("saison"):
        if len(g) < k * 2:
            continue
        ist = set(g.nlargest(k, "ist_avg").index)
        prg = set(g.nlargest(k, spalte).index)
        tr.append(len(ist & prg) / k)
    return float(np.mean(tr)) if tr else float("nan")


def report_ertrag(rows: pd.DataFrame) -> dict:
    """Der Bericht, der ueber den Umbau entscheidet.

    Zielgroesse ist die Oe Punkte je Einsatz der Hinrunde. Zwei Masken werden
    auseinandergehalten, weil ihre Vermischung den Vergleich wertlos macht:
    auf der **gemeinsamen** Maske (Modell und Baseline beide definiert) wird
    der Abstand gemessen, auf der **vollen** die Reichweite — dort bewertet
    das Modell zusaetzlich rund 900 Spieler, ueber die das Vorjahr schweigt.
    """
    feld = rows[(rows.pos != TW) & (rows.ist_einsaetze >= MIN_EINSAETZE_ZIEL)
                & rows.ist_avg.notna()].copy()
    print(f"\nErtrag: Oe Punkte je Einsatz, Hinrunde, >= {MIN_EINSAETZE_ZIEL} "
          f"Einsaetze, ohne TW")
    print("  volle Maske (nur das Modell ist ueberall definiert):")
    voll = _ertrag_zeile("Qualitaet x Rolle", feld, "xp_dach")

    beide = feld[feld.naiv_avg.notna() & (feld.naiv_einsaetze >= 10)]
    print("  gemeinsame Maske (Modell gegen die Latte):")
    modell = _ertrag_zeile("Qualitaet x Rolle", beide, "xp_dach")
    naiv = _ertrag_zeile("naiv: Vorjahres-Oe fortschreiben", beide, "naiv_avg")

    print("  nach Rolle der Vorsaison (dort sitzt die Verzerrung):")
    for label, sel in (("war gesetzt (Startquote >0,8)", feld.startquote_v1 > 0.8),
                       ("war Rotation (0,55-0,8)", (feld.startquote_v1 > 0.55)
                        & (feld.startquote_v1 <= 0.8)),
                       ("war geteilt/Rand (<=0,55)", (feld.startquote_v1 > 0)
                        & (feld.startquote_v1 <= 0.55)),
                       ("ohne Vorsaison", feld.hat_v1 == 0)):
        _ertrag_zeile("  " + label, feld[sel.fillna(False)], "xp_dach")

    print("  nach Historie:")
    for label, sel in (("Stamm-Vorsaison (>=900 Min)", feld.naiv_min >= 900),
                       ("Teil-Vorsaison", (feld.naiv_min > 0) & (feld.naiv_min < 900)),
                       ("keine BL-Vorsaison", feld.naiv_min.isna() | (feld.naiv_min == 0)),
                       ("ganz ohne Historie", feld.neuzugang == 1)):
        _ertrag_zeile("  " + label, feld[sel.fillna(False)], "xp_dach")

    # --- Die Zielgruppe, getrennt von der Gesamtheit ------------------------
    # Ein Kader besteht zu rund einem Drittel aus Spielern, die normalerweise
    # nicht spielen (im Export 2026/27: 140 von 466 in Kategorie 5). Ein r
    # ueber alle bewertbaren Spielersaisons misst also zu einem grossen Teil,
    # wie gut man Randfiguren sortiert — eine Frage, die beim Kaderbau nicht
    # gestellt wird. Gekauft werden Kategorie 1 und 2, und dort vor allem der
    # obere Rand.
    #
    # ``kategorie`` gibt es nur fuer 2026/27, historisch braucht es einen
    # Stellvertreter aus vorab bekannten Groessen. Gegen die echten Kategorien
    # 2026/27 geprueft trifft "Vorjahres-Startquote > 0,70 UND Marktwert-Rang
    # <= 2 auf der Position" den Anteil (28 % gegen 31 %) bei 56 % Praezision —
    # mehr ist ohne die Handarbeit nicht zu holen, und genau das ist der Grund,
    # warum die Handarbeit etwas wert ist.
    print()
    print("  Zielgruppe statt Gesamtheit (dort wird gekauft):")
    _ertrag_zeile("alle bewertbaren", feld, "xp_dach")
    for label, sel in (
            ("ex ante Kat 1/2 (Stamm + MW-Rang<=2)",
             (feld.startquote_v1.fillna(0) > 0.70) & (feld.mw_rang_c <= 2.0)),
            ("ex post Stammspieler (Startquote>0,55)",
             feld.ist_startquote.fillna(0) > 0.55),
            ("ex post gesetzt (Startquote>0,80)",
             feld.ist_startquote.fillna(0) > 0.80)):
        g = feld[sel.fillna(False)]
        _ertrag_zeile("  " + label, g, "xp_dach")
        b = g[g.naiv_avg.notna() & (g.naiv_einsaetze >= 10)]
        _ertrag_zeile("    dieselben, naiv", b, "naiv_avg")

    # Der obere Rand: nicht wie gut die Reihenfolge insgesamt ist, sondern ob
    # die richtigen Spieler oben stehen. Das ist die Frage, die ein Kaderbauer
    # tatsaechlich stellt.
    # Oberer Rand. **Auf der gemeinsamen Maske**, sonst ist der Vergleich
    # geschenkt: das naive Verfahren darf nur unter Spielern mit Vorsaison
    # waehlen, und dort sitzt die echte Spitze ohnehin fast vollstaendig.
    top = feld[feld.naiv_avg.notna() & (feld.naiv_einsaetze >= 10)
               & feld.avg_hist.notna()]
    print(f"  Oberer Rand (gemeinsame Maske, n={len(top)}) — Anteil der echten "
          f"Top-N, der auch prognostiziert oben steht:")
    print(f"    {'':8}{'Modell':>9}{'naiv':>9}{'gew. Ø':>9}")
    for k in (5, 10, 20, 30):
        print(f"    Top {k:2d} " + "".join(
            f"{_top_treffer(top, sp, k)*100:8.0f}%"
            for sp in ("xp_dach", "naiv_avg", "avg_hist")))
    # Die Zahl, die beim Kaderbau zaehlt: was man mit der gewaehlten Elf
    # tatsaechlich holt, gegen die perfekte Rueckschau als Decke.
    print("    Ist-Schnitt der gewaehlten Top-20: " + "  ".join(
        f"{lab} {np.mean([g.nlargest(20, sp).ist_avg.mean() for _, g in top.groupby('saison') if len(g) >= 40]):.1f}"
        for sp, lab in (("xp_dach", "Modell"), ("naiv_avg", "naiv"),
                        ("avg_hist", "gew.Ø"), ("ist_avg", "perfekt"))))

    print("  Verzerrung je Zielsaison (das Niveau kommt aus der Prior-Kette):")
    for sa, g in feld.groupby("saison"):
        print(f"    {sa}  n={len(g):4d}  Prognose {g.xp_dach.mean():5.1f}  "
              f"Ist {g.ist_avg.mean():5.1f}  Bias {(g.ist_avg-g.xp_dach).mean():+5.1f}")

    return {"voll": voll, "modell": modell, "naiv": naiv}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rangbasiert (Mann-Whitney) — kein zusaetzliches Paket noetig."""
    r = pd.Series(p).rank().values
    n1, n0 = float(y.sum()), float((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report_rolle(rows: pd.DataFrame) -> dict:
    """Setzt sich der Spieler durch? Kalibrierung und Trennschaerfe.

    Eine Wahrscheinlichkeit hat kein r. Sie muss zweierlei leisten: die
    behauptete Rate treffen (sagt sie 20 %, muessen 20 % eintreten) und die
    Faelle trennen. Gemessen wird beides, und gegen die Basisrate — eine
    Kennzahl ohne Vergleichslinie sagt nichts.
    """
    d = rows[rows.ist_startquote.notna() & rows.p_rolle.notna()]
    if len(d) < 200:
        return {}
    y, p = d.ist_gesetzt.values, d.p_rolle.values
    basis = float(y.mean())
    brier = float(((p - y) ** 2).mean())
    brier_basis = float(((basis - y) ** 2).mean())
    auc = _auc(y, p)
    print(f"\nRollenwahrscheinlichkeit (setzt sich durch), n={len(d)}")
    print(f"  Basisrate {basis*100:4.1f}%   Prognose im Mittel {p.mean()*100:4.1f}%")
    print(f"  AUC {auc:.3f}   Brier {brier:.4f} gegen Basisrate {brier_basis:.4f} "
          f"(Skill {1 - brier/brier_basis:+.3f})")
    dez = d.assign(dez=pd.qcut(p, 10, labels=False, duplicates="drop"))
    t = dez.groupby("dez").agg(prognose=("p_rolle", "mean"),
                               tatsaechlich=("ist_gesetzt", "mean"),
                               n=("p_rolle", "size"))
    grob = float((t.tatsaechlich - t.prognose).abs().max())
    print("  Dezile Prognose/tatsaechlich: "
          + "  ".join(f"{a*100:.0f}/{b*100:.0f}"
                      for a, b in zip(t.prognose, t.tatsaechlich)))
    print(f"  groesste Abweichung {grob*100:.1f} Prozentpunkte")
    return {"auc": auc, "skill": 1 - brier / brier_basis, "max_dez": grob}


def streuungs_faktoren(rows: pd.DataFrame) -> dict:
    """Erwartete Streuung des Ergebnisses um die Prognose.

    Zwei fast unabhaengige Treiber, deshalb multiplikativ und je auf Mittel 1
    normiert: wie gut ein Spieler belegt ist und auf welcher Position er
    spielt. Gemessen (sd der Abweichung auf der Oe-Punkte-Skala): n_eff >45
    28,5 · 15-45 29,4 · <15 30,1 · **ohne Historie 39,3**; Position IV 26,6
    bis ST 35,1. Der Sprung "ohne Historie" ist groesser als die ganze
    Positionsspanne und bekommt deshalb eine eigene Stufe statt einer
    Interpolation.
    """
    d = rows[rows.ist_avg.notna() & rows.xp_dach.notna()
             & (rows.ist_einsaetze >= MIN_EINSAETZE_ZIEL)].copy()
    d["fehler"] = d.ist_avg - d.xp_dach
    global_sd = float(d.fehler.std())

    def _stufe(x: float) -> str:
        if x <= 0:
            return "ohne"
        return "viel" if x > 45 else ("mittel" if x >= 15 else "wenig")

    d["stufe"] = [_stufe(v) for v in d.n_eff.fillna(0.0)]
    f_bel = {k: float(g.fehler.std()) / global_sd
             for k, g in d.groupby("stufe") if len(g) >= 50}
    f_pos = {k: float(g.fehler.std()) / global_sd
             for k, g in d.groupby("pos") if len(g) >= 100}
    return {"global": global_sd, "belegt": f_bel, "pos": f_pos}


def streuung_fuer(fak: dict, n_eff: float, pos: str) -> float:
    stufe = "ohne" if n_eff <= 0 else ("viel" if n_eff > 45
                                       else ("mittel" if n_eff >= 15 else "wenig"))
    return (fak["global"] * fak["belegt"].get(stufe, 1.0)
            * fak["pos"].get(pos, 1.0))


def report_streuung(rows: pd.DataFrame) -> dict:
    """Trifft die Streuungsprognose die tatsaechliche Streuung?

    Eine sd-Prognose hat kein r — die einzige sinnvolle Pruefung ist, ob in
    jedem Terzil der Prognose die beobachtete Abweichung dazu passt.
    """
    fak = streuungs_faktoren(rows)
    d = rows[rows.ist_avg.notna() & (rows.ist_einsaetze >= MIN_EINSAETZE_ZIEL)].copy()
    d["streuung"] = [streuung_fuer(fak, n, p)
                     for n, p in zip(d.n_eff.fillna(0.0), d.pos)]
    d["fehler"] = d.ist_avg - d.xp_dach
    d["terzil"] = pd.qcut(d.streuung, 3, labels=False, duplicates="drop")
    print(f"\nStreuung (erwartete Abweichung), global sd={fak['global']:.1f}")
    grob = 0.0
    for t, g in d.groupby("terzil"):
        ist = float(np.sqrt((g.fehler ** 2).mean()))
        soll = float(g.streuung.mean())
        grob = max(grob, abs(ist - soll) / soll)
        print(f"  Terzil {int(t)+1}: erwartet {soll:5.1f}  beobachtet {ist:5.1f}"
              f"  ({(ist-soll)/soll*100:+4.1f} %)  n={len(g)}")
    return {"faktoren": fak, "max_abw": grob}


def _produkt_bt(df: pd.DataFrame, spalte: str) -> pd.DataFrame:
    """Frame im Format von pairwise_accuracy: eine "Zelle" je Spieler,
    Spieltag 0 je Zielsaison — verglichen wird nur innerhalb einer Saison."""
    return pd.DataFrame({"season": df.saison, "league": "Bundesliga",
                         "matchday": 0,
                         "actual": df.ist_punkte.values,
                         "predicted": df[spalte].values})


def report_produkt(rows: pd.DataFrame) -> dict:
    """Produktebene: erwartete Hinrundenpunkte gegen beide einfachen Sorten.

    naiv = Vorjahres-Gesamtpunkte fortschreiben (fehlend = 0, genau so sortiert
    auch ein Mensch), mw = Marktwert-Sort. Bewertet werden alle Kaderspieler
    mit gespielter Hinrunde — auch die, fuer die die Baselines nichts wissen,
    denn genau dort soll das Modell seinen Vorsprung holen.
    """
    from scipy.stats import spearmanr
    df = rows[rows.ist_punkte.notna() & rows.ist_st.notna()].copy()
    df["pred_punkte"] = df.xp_dach * df.quote_rolle * df.ist_st
    df["naiv_punkte"] = (df.naiv_avg.fillna(0.0) * df.einsatzquote_v1
                         * df.ist_st)
    df["mw_sort"] = df.mw_vor_saison.fillna(0.0)

    print("\nProduktebene (erwartete Hinrundenpunkte, alle Kaderspieler)")
    out = {}
    for label, sp in (("Modell p90 x Minuten", "pred_punkte"),
                      ("naiv: Vorjahr fortschreiben", "naiv_punkte"),
                      ("Marktwert-Sort", "mw_sort")):
        rho = float(spearmanr(df[sp], df.ist_punkte).statistic)
        pw = pairwise_accuracy(_produkt_bt(df, sp))
        print(f"  {label:34s} n={len(df):5d}  rho={rho:6.3f}  paarweise={pw*100:5.1f}%")
        out[sp] = pw

    # Trefferquote auf der Ø-Punkte-je-Einsatz-Skala — Referenzen aus den
    # Befunden: Baseline 29,7 %, Glaskugel 48 % (bei >= 5 Einsaetzen ±10).
    e = df[df.ist_einsaetze >= 5].copy()
    e["ist_avg"] = e.ist_punkte / e.ist_einsaetze
    e["pred_avg"] = e.xp_dach
    treffer = float((np.abs(e.pred_avg - e.ist_avg) <= 10).mean())
    naiv_avg = e.naiv_avg.fillna(0.0)
    treffer_naiv = float((np.abs(naiv_avg - e.ist_avg) <= 10).mean())
    print(f"  Trefferquote ±10 je Einsatz (n={len(e)}): Modell "
          f"{treffer*100:.1f}%  naiv {treffer_naiv*100:.1f}%")
    out["treffer10"] = treffer
    return out


def report_oekonomie(rows: pd.DataFrame, band: tuple[float, float] = (2e6, 1e7),
                     top: int = 15) -> dict:
    """Der Test, der zaehlt: Top-15-Picks im bezahlbaren Band 2-10 Mio.

    Die Befunde messen (Ganzjahr): Modell schliesst 47 % der Luecke zur
    perfekten Rueckschau, MW-Sort 24 %. Hier dasselbe fuer die Hinrunde.
    """
    df = rows[rows.ist_punkte.notna() & rows.mw_vor_saison.notna()].copy()
    df = df[(df.mw_vor_saison >= band[0]) & (df.mw_vor_saison <= band[1])]
    df["pred_punkte"] = df.xp_dach * df.quote_rolle * df.ist_st
    df["naiv_punkte"] = df.naiv_avg.fillna(0.0) * df.einsatzquote_v1 * df.ist_st

    ergebnisse = {"modell": [], "naiv": [], "mw": [], "perfekt": []}
    siege = 0
    for saison, g in df.groupby("saison"):
        picks = {
            "modell": g.nlargest(top, "pred_punkte"),
            "naiv": g.nlargest(top, "naiv_punkte"),
            "mw": g.nlargest(top, "mw_vor_saison"),
            "perfekt": g.nlargest(top, "ist_punkte"),
        }
        for k, p in picks.items():
            ergebnisse[k].append(float(p.ist_punkte.sum()))
        if ergebnisse["modell"][-1] >= ergebnisse["mw"][-1]:
            siege += 1

    n = len(ergebnisse["modell"])
    mittel = {k: float(np.mean(v)) for k, v in ergebnisse.items()}
    basis = min(mittel["naiv"], mittel["mw"])
    print(f"\nOekonomie: Top-{top} im Band 2-10 Mio, Ertrag = Hinrundenpunkte")
    for k in ("perfekt", "modell", "naiv", "mw"):
        anteil = ""
        if k != "perfekt" and mittel["perfekt"] > basis:
            luecke = (mittel[k] - basis) / (mittel["perfekt"] - basis)
            anteil = f"  ({luecke*100:4.0f} % der Luecke ueber der schwaecheren Sorte)"
        print(f"  {k:10s} {mittel[k]:7.0f}{anteil}")
    print(f"  Modell >= MW-Sort in {siege} von {n} Saisons")
    return {"siege": siege, "n": n, **mittel}


# ---------------------------------------------------------------------------
# Export fuer die Zielsaison — data/player_projections.json
# ---------------------------------------------------------------------------
#
# Alles in diesem Abschnitt ist im Walk-forward NICHT messbar (kategorie gibt
# es nur fuer 2026/27, history.json reicht ein Jahr zurueck). Deshalb liegt es
# getrennt vom validierten Kern, wird erst NACH der Modellrechnung angewandt
# und traegt je Spieler ein Quellen-Flag — im Export ist ablesbar, wie viel
# Prognose aus Messung und wie viel aus Setzung stammt.

# kategorie (fine_positions.csv) -> Verteilung ueber die Rollen.
#
# Die Kategorie ist eine **Rollen-Hierarchie im Kader**, keine Einsatzzeit:
# 1 = unumstrittener Stammspieler bis 5 = Reserve. Kategorie 1 bis 4 zielen
# alle auf dieselbe Rolle ("gesetzt") — das ist die Position, um die sie
# konkurrieren. Sie unterscheiden sich darin, wie wahrscheinlich sie sie
# halten, und genau das steht hier.
#
# **Diese Zahlen sind gesetzt, nicht gemessen** — kategorie gibt es nur fuer
# 2026/27, ein Walk-forward ist unmoeglich. Backtestbar ist die Struktur
# (Rolle -> Einsatzlaenge -> Punkte, mit der Vorsaison-Rolle als Eingang:
# r 0,606) und die obere Schranke bei perfekter Rollenkenntnis (r 0,779).
# Was kategorie davon einloest, ist erst nach ein paar Spieltagen messbar.
#
# Die Stufen folgen der Beschreibung der Kaderpflege: 1 und 2 sind gesetzt,
# 3 und 4 sind die Risikospieler ("entweder einer setzt sich durch, oder es
# wird abgewechselt"), 5 ist Reserve. Zum Vergleich die *gemessenen*
# Uebergaenge aus der Vorsaison-Rolle: gesetzt -> 59 % bleiben gesetzt,
# Rotation -> 34 %, geteilt -> 21 %, Rand -> 10 %. Die Kategorie ist eine
# Aussage ueber die kommende Saison und darf deshalb zuversichtlicher sein
# als eine Fortschreibung der alten Rolle.
KATEGORIE_ROLLEN = {
    1: {"gesetzt": 0.88, "Rotation": 0.09, "geteilt": 0.02, "Rand": 0.01},
    2: {"gesetzt": 0.72, "Rotation": 0.20, "geteilt": 0.06, "Rand": 0.02},
    3: {"gesetzt": 0.45, "Rotation": 0.31, "geteilt": 0.18, "Rand": 0.06},
    4: {"gesetzt": 0.20, "Rotation": 0.30, "geteilt": 0.30, "Rand": 0.20},
    5: {"gesetzt": 0.04, "Rotation": 0.10, "geteilt": 0.20, "Rand": 0.66},
}

# Verfuegbarkeit aus history.json: Beta-Schrumpfung der Verletzungsquote gegen
# einen milden Prior (~2 % bei leerer Historie), damit ein Jahr Beobachtung
# nicht ueberinterpretiert wird. Sie wirkt auf die **Rollenwahrscheinlichkeit**
# und nicht auf die Punkte: eine Verletzung sagt nichts darueber, wie viele
# Punkte jemand pro Einsatz holt, sondern ob er auf dem Platz steht.
A_VERF, B_VERF = 10.0, 100.0
VERLETZT_FAKTOR = 0.6


def _z_dict(werte: dict[str, float]) -> dict[str, float]:
    v = np.array(list(werte.values()), dtype=float)
    m, s = float(v.mean()), float(v.std())
    return {k: (x - m) / s if s > 1e-9 else 0.0 for k, x in werte.items()}


def _kon_zielsaison(liga: str = "1") -> dict:
    """Teamkontext der Zielsaison aus data/ratings.json.

    ratings.json traegt bereits alles, was vor Saisonstart ueber die Teams
    bekannt ist — Vorsaison, Carryover der Aufsteiger, Marktquoten-Prior. Die
    Skala ist dieselbe wie in team_niveaus (mu + att), z-standardisiert ist
    der Unterschied ohnehin weg.
    """
    with open(LEGACY_DIR / "ratings.json", encoding="utf-8") as f:
        teams = json.load(f)["leagues"][liga]["teams"]
    za = _z_dict({t: v["att"] for t, v in teams.items()})
    zd = _z_dict({t: v["def"] for t, v in teams.items()})
    return {t: {"feld": za[t], TW: -zd[t]} for t in teams}


def _verfuegbarkeit() -> dict[int, tuple[float, bool]]:
    """(Faktor, heute verletzt?) je Spieler aus der Verletzungs-Historie.

    ``st`` ist eine Stufenfunktion aus Wechselpunkten [tag, code]; ein Status
    gilt bis zum naechsten Eintrag bzw. bis zum Ende des beobachteten
    Fensters (der dt-Spanne der Marktwerte). Codes 1 (verletzt) und 4 (Reha)
    zaehlen als Ausfall; 2 (angeschlagen) und 3 (gesperrt) nicht — wer
    angeschlagen ist, spielt meist, und Sperren sind kurz.
    """
    pfad = LEGACY_DIR / "history.json"
    if not pfad.exists():
        return {}
    with open(pfad, encoding="utf-8") as f:
        spieler = json.load(f)["players"]
    out = {}
    for pid, e in spieler.items():
        dt, st = e.get("dt") or [], e.get("st") or []
        if not dt:
            continue
        ende = dt[-1]
        beob = max(ende - dt[0] + 1, 1)
        verletzt = 0
        for i, (tag, code) in enumerate(st):
            bis = st[i + 1][0] if i + 1 < len(st) else ende
            if code in (1, 4):
                verletzt += max(bis - tag, 0)
        quote = (verletzt + A_VERF) / (beob + A_VERF + B_VERF)
        akut = bool(st) and st[-1][1] in (1, 4)
        out[int(pid)] = (1.0 - quote, akut)
    return out


# Niveauversatz fuer die Spieler, von denen die Kaderpflege sagt, dass sie
# spielen (Kategorie 1-2). Er wird **abgezogen**, die Werte sind negativ: das
# Modell liegt fuer diese Gruppe zu tief.
#
# Der Grund ist Bauart, nicht Fehler. ``xp_erwartet`` wiegt die Rolle mit ihrer
# Wahrscheinlichkeit — ueber alle Kandidaten gemittelt ist das richtig, und auf
# der eigenen Bewertungsmenge ist das Modell in den drei juengsten Saisons
# praktisch unverzerrt (+1,5 / +2,3 / +0,2). Bedingt darauf, dass ein Spieler
# **tatsaechlich** gesetzt ist, ist es dagegen systematisch zu tief: gemessen
# -6,3 (t = -3,3) auf 1.541 Spielersaisons. Dieselbe Gruppe nach Ausgang
# getrennt zeigt, warum: wer sich durchsetzt, wird um 8 unterschaetzt, wer
# nicht, um 19 ueberschaetzt.
#
# Genau nach der ersten Gruppe fragt der Kaderbau — und die Kaderpflege nennt
# sie vorab. Deshalb greift der Versatz **nur** bei Kategorie 1-2; auf den
# uebrigen Export waere er falsch herum.
#
# Messaufbau: walk-forward, Zielsaisons 2019-2025, bewertet auf Spielern mit
# mindestens VERSATZ_MIN_ST Hinrunden-Startelfeinsaetzen und >= 60 Minuten je
# Einsatz, Versatz je Fall aus den Out-of-fold-Resten aller frueheren Saisons.
# Danach: Verzerrung +1,2 (t = +0,6) statt -6,3, r 0,628 statt 0,623,
# MAE 19,8 statt 20,0. Mit ``--versatz`` neu zu messen.
VERSATZ = {"a": -3.5, "b": -14.1, "c": -11.6}
VERSATZ_MIN_ST = 10
VERSATZ_KATEGORIEN = (1.0, 2.0)

# Ab dieser Vorsaison-Startquote gilt ein Spieler als damals gesetzt. Trennt
# Fall a von Fall b; 0,60 liegt in der flachen Mitte der Rollen-Grenzen.
VERSATZ_STARTQUOTE = 0.60


def versatz_fall(hat_v1, startquote_v1) -> np.ndarray:
    """Drei Faelle aus dem, was vor der Saison ueber einen Spieler bekannt ist.

    a: Vorsaison in der Liga und dort gesetzt. b: Vorsaison ja, gesetzt nein.
    c: keine Vorsaison. Der Versatz ist in b und c drei- bis viermal so gross
    wie in a — wer neu ist oder die Rolle wechselt, wird am staerksten
    unterschaetzt.
    """
    hat = np.asarray(pd.Series(hat_v1).fillna(0.0), dtype=float)
    sq = np.asarray(pd.Series(startquote_v1).fillna(0.0), dtype=float)
    return np.where(hat == 0, "c", np.where(sq >= VERSATZ_STARTQUOTE, "a", "b"))


def versatz_messen(rows: pd.DataFrame) -> dict:
    """VERSATZ aus Backtest-Zeilen neu bestimmen — die Konstante gegenpruefen.

    Gemittelt wird ueber **Saisons**, nicht ueber Zeilen: die Spieler einer
    Saison teilen sich ihr Kohortenglueck, und wer Zeilen zaehlt, findet
    ueberall Signifikanz.
    """
    g = rows.dropna(subset=["xp_dach", "ist_avg"])
    g = g[(g.ist_st >= VERSATZ_MIN_ST) & (g.ist_minje >= 60)].copy()
    g["fall"] = versatz_fall(g.hat_v1, g.startquote_v1)
    g["res"] = g.xp_dach - g.ist_avg
    return {f: round(float(s.groupby("saison").res.mean().mean()), 1)
            for f, s in g.groupby("fall")}


def export(path: Path | None = None) -> dict:
    """Prognose je Kaderspieler der laufenden Bundesliga-Saison schreiben.

    Bewusst NICHT in fetch.py ratings eingehaengt: CLAUDE.md garantiert, dass
    ratings.json ohne die Parquet-Dateien baubar ist — dieses Modell braucht
    panel.parquet und tm_players.csv zwingend. Eigenes Kommando, und die
    Seite behandelt die Datei als optional.
    """
    with open(LEGACY_DIR / "data_1.json", encoding="utf-8") as f:
        kader = json.load(f)

    # Die gepflegte Formation beschreibt die aktuelle Rolle genauer als die
    # TM-Nominalposition (z. B. Moreira: ZOM statt Linksaußen). Sie muss vor
    # der Qualitätsprognose greifen, nicht erst beim Kategorie-Export.
    fp = pd.read_csv(MANUAL / "fine_positions.csv")

    panel = lade_panel()
    saison = max(panel.season.unique())
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    hist_panel = panel[panel.season < saison]
    hist_matches = matches[matches.season < saison]

    ss = spieler_saisons(hist_panel)
    kon = kontext_tabelle(team_niveaus(conceded_from_panel(hist_panel, hist_matches)),
                          sorted(ss.season.unique()))
    for team, e in _kon_zielsaison("1").items():
        kon[(saison, team)] = e
    tm = tm_je_spieler(lade_tm())

    pos_override = dict(zip(
        fp.loc[(fp.season == saison) & fp.player_id.notna(), "player_id"].astype(int),
        fp.loc[(fp.season == saison) & fp.player_id.notna(), "position_fine"]))
    kandidaten = pd.DataFrame([
        {"player_id": int(p["id"]), "team_id": str(p["team_id"]),
         "kb_id": str(p["id"]), "name": p["name"], "kb_pos": p.get("position"),
         "pos_override": pos_override.get(int(p["id"]))}
        for p in kader["players"]])
    df = merkmalszeilen(
        saison, kandidaten[["player_id", "team_id", "pos_override"]], ss, kon, tm,
                        DELTA, K_MIX, False, False, None, BL2_GEWICHT, K_TW)
    tab = df.attrs["rollen_tabelle"]   # ein merge verwirft attrs
    qtab = df.attrs["quoten_tabelle"]
    uebg = df.attrs["uebergaenge"]
    df = kandidaten.merge(df, on=["player_id", "team_id"])
    # Kickbase kennt die Position auch ohne TM-Zeile — fuers TW-Handling
    # genuegt sie (Kickbase-Position 1 = Torwart).
    df.loc[(df.pos == POS_UNBEKANNT) & (df.kb_pos == 1), "pos"] = TW

    # Kalibrierung auf allen historischen Zielsaisons — dieselben Fits wie im
    # Backtest, nur ohne dessen letzte Zurueckhaltung (alle Saisons duerfen
    # ins Training, die Zielsaison liegt ohnehin in der Zukunft).
    rows = backtest_p90(nur_kalibrierzeilen=True)
    feld = rows[(rows.pos != TW) & (rows.jahr_min >= MIN_TRAIN)
                & rows.jahr_p90.notna()]
    fit_p = _ridge(feld[MERKMALE].values.astype(float),
                   (feld.jahr_p90 - feld.p90_mix).values.astype(float),
                   feld.jahr_min.values.astype(float), LAMBDA_B)
    df["p90_dach"] = df.p90_mix
    sel = (df.pos != TW).values
    df.loc[sel, "p90_dach"] = df.loc[sel, "p90_mix"] + _ridge_pred(
        fit_p, df.loc[sel, MERKMALE].values.astype(float))

    # Rollen-Logistik auf allen historischen Zielsaisons.
    voll = backtest_p90()
    tr_r = voll[voll.ist_startquote.notna()]
    zs = sorted(tr_r.saison.unique())
    rang = {x: i for i, x in enumerate(zs)}
    w_r = DELTA_ROLLE ** (len(zs) - tr_r.saison.map(rang).values - 1)
    fit_r = _logit(tr_r[ROLLE_MERKMALE].values.astype(float),
                   tr_r.ist_gesetzt.values.astype(float), w_r)
    letzte = tr_r[tr_r.saison == zs[-1]]
    p_l = _logit_pred(fit_r, letzte[ROLLE_MERKMALE].values.astype(float))
    versatz = 0.0
    for _ in range(40):
        q = 1 / (1 + np.exp(-(np.log(p_l / (1 - p_l)) + versatz)))
        if abs(q.mean() - letzte.ist_gesetzt.mean()) < 1e-4:
            break
        versatz += (letzte.ist_gesetzt.mean() - q.mean()) * 4.0
    fit_r["beta"][0] += versatz
    fak = streuungs_faktoren(voll)

    # --- Zielsaison: die handgepflegte Rolle tritt an die Stelle der
    # Vorsaison-Rolle. Das ist der Punkt, an dem die Kaderpflege wirkt —
    # gemessen ist perfekte Rollenkenntnis den Sprung von r 0,606 auf 0,779
    # wert, und die schwaechste Gruppe (ohne jede Historie) steigt von 0,37
    # auf 0,72.
    fp = fp[(fp.season == saison) & fp.player_id.notna()]
    kat = dict(zip(fp.player_id.astype(int), fp.kategorie))
    df["kategorie"] = [kat.get(pid) for pid in df.player_id]
    df["quellen"] = [["hist"] if n > 0 else ["baseline"] for n in df.n_eff.fillna(0)]

    # Schritt 1: die Kategorie setzt nur noch die **Basisrate** und damit die
    # Form der Rollenverteilung. Die Minuten folgen erst in Schritt 3, wenn
    # die Logistik gesagt hat, wie sicher *dieser* Spieler sie haelt.
    rollen_form, p_basis = [], []
    for i, pid in enumerate(df.player_id):
        k = kat.get(pid)
        vert = KATEGORIE_ROLLEN.get(int(k)) if k is not None else None
        if vert is None:
            rollen_form.append(rollen_verteilung(uebg, df.at[i, "zielrolle"]))
            p_basis.append(df.at[i, "p_basis"])
            continue
        rollen_form.append(dict(vert))
        p_basis.append(vert["gesetzt"])
        df.at[i, "quellen"] = df.at[i, "quellen"] + ["kategorie"]
    df["p_basis"] = np.clip(p_basis, 0.01, 0.99)
    df["p_basis_logit"] = np.log(df.p_basis / (1 - df.p_basis))

    # Schritt 2: Rollenwahrscheinlichkeit je Spieler.
    df["p_rolle"] = np.clip(_logit_pred(
        fit_r, df[ROLLE_MERKMALE].values.astype(float)), 0.005, 0.995)

    # Die Kategorie setzt die Basisrate, das Modell verfeinert **innerhalb**.
    # Das ist die Arbeitsteilung, die die Messung nahelegt: die Kaderpflege
    # weiss, wer um welche Rolle konkurriert (und kennt Transfers), das Modell
    # trennt innerhalb einer Kategorie die Sicheren von den Wackligen — im
    # Backtest von 9,1 % bis 33,2 % Durchsetzung. Ohne diese Eichung traegt
    # die Logistik das Niveau der historischen Vorjahres-Rollen, und ein
    # gepflegter Stammspieler erschiene mit 25 % Startplatzchance.
    for k, vert in KATEGORIE_ROLLEN.items():
        sel = (df.kategorie == k).values
        if sel.sum() < 5:
            continue
        p = df.loc[sel, "p_rolle"].values
        lo = np.log(p / (1 - p))
        versatz = 0.0
        for _ in range(60):
            q = 1 / (1 + np.exp(-(lo + versatz)))
            if abs(q.mean() - vert["gesetzt"]) < 1e-4:
                break
            versatz += (vert["gesetzt"] - q.mean()) * 4.0
        df.loc[sel, "p_rolle"] = np.clip(
            1 / (1 + np.exp(-(lo + versatz))), 0.005, 0.995)

    # Verletzungshistorie: sie senkt die Rollenwahrscheinlichkeit, nicht den
    # Ertrag je Einsatz.
    verf = _verfuegbarkeit()
    for i, pid in enumerate(df.player_id):
        v = verf.get(pid)
        if v is None:
            continue
        faktor, akut = v
        if akut:
            faktor *= VERLETZT_FAKTOR
            df.at[i, "quellen"] = df.at[i, "quellen"] + ["verletzt"]
        df.at[i, "p_rolle"] = df.at[i, "p_rolle"] * faktor

    # Schritt 3: dieselbe Eichung wie im Backtest — die Kategorie gibt die Form
    # der Verteilung, die Logistik ihre Masse auf "gesetzt". Erst dadurch
    # unterscheidet sich der sichere Kategorie-1-Spieler vom wackligen: beide
    # haetten sonst dieselbe Einsatzlaenge bekommen.
    vert_k = [verteilung_kalibriert_form(form, pr)
              for form, pr in zip(rollen_form, df.p_rolle)]
    df["minje_rolle"] = [minje_aus_verteilung(tab, v, p)
                         for v, p in zip(vert_k, df.pos)]
    df["quote_rolle"] = [quote_aus_verteilung(qtab, v) for v in vert_k]
    w_rolle = K_MINJE / (df.n_eins_eff.fillna(0.0) + K_MINJE)
    df["minje_dach_neu"] = (w_rolle * df.minje_rolle
                            + (1 - w_rolle) * df.minje_hist.fillna(df.minje_rolle))

    # Die Einsatzlaenge bei gehaltener Rolle darf nie unter der erwarteten
    # liegen. Seit die Verteilung an p_rolle geeicht wird, kann sie das: wer
    # laenger spielt als die Zellennorm seiner Position (Kleindienst 88 Min
    # gegen ST-gesetzt 80,3), bekaeme sonst ein "Potenzial" unter seiner
    # eigenen Erwartung. Die Zelle ist eine Norm, kein Deckel.
    df["minje_gesetzt"] = np.maximum(df.minje_gesetzt, df.minje_dach_neu)

    df["xp_rolle"] = df.p90_dach * df.minje_dach_neu / 90.0
    df["xp_dach"] = df.xp_rolle
    df["xp_gesetzt"] = df.xp_dach + df.p90_dach * (
        df.minje_gesetzt - df.minje_dach_neu) / 90.0

    # Eichung auf die Spieler, von denen die Kaderpflege sagt, dass sie spielen
    # (Begruendung und Messaufbau bei VERSATZ). Beide Zahlen wandern gleich
    # weit, damit ihr Abstand — das Aufwaertspotenzial — unberuehrt bleibt.
    ist_kat12 = df.kategorie.isin(VERSATZ_KATEGORIEN)
    versatz = pd.Series(versatz_fall(df.hat_v1, df.startquote_v1),
                        index=df.index).map(VERSATZ).astype(float).fillna(0.0)
    versatz = versatz.where(ist_kat12, 0.0)
    df["xp_dach"] = df.xp_dach - versatz
    df["xp_gesetzt"] = df.xp_gesetzt - versatz
    df["versatz"] = versatz

    df["streuung"] = [streuung_fuer(fak, n, p)
                      for n, p in zip(df.n_eff.fillna(0.0), df.pos)]
    sd_ref = float(min(fak["global"] * b * o
                       for b in fak["belegt"].values()
                       for o in fak["pos"].values()))
    # Sicherheit fuer die Punktgroesse: wie verlaesslich ist die Zahl? Zwei
    # dimensionslose Faktoren — wie sicher der Spieler ueberhaupt spielt, und
    # wie eng das Ergebnis um die Prognose streut.
    df["sicherheit"] = np.clip(df.quote_rolle * (sd_ref / df.streuung), 0.0, 1.0)

    splits = load_splits()
    hin_spieltage = splits.get((saison, "Bundesliga"), 17)
    # Erwartete Einsaetze aus der **Einsatzquote**, nicht aus P(gesetzt):
    # letztere fragt, ob jemand ueber 80 % der Spiele beginnt, und das ist
    # eine viel strengere Huerde als ueberhaupt zu spielen.
    df["einsaetze_erwartet"] = df.quote_rolle.clip(0.0, 1.0) * hin_spieltage

    # Kategorie 5 wird nicht prognostiziert. Das ist keine Sparsamkeit,
    # sondern die ehrliche Antwort auf eine Frage, die das Modell nicht
    # beantworten kann: ein Reservist spielt im Normalfall nicht, ein
    # verschwindend kleiner Teil bricht durch — aber *welcher*, weiss niemand
    # vorab. Ihnen allen ein paar Punkte zuzuschreiben (im Lauf davor im
    # Schnitt 51,5 bei 6,5 erwarteten Einsaetzen) erzeugt eine Zahl, wo keine
    # Information ist, und drueckt zugleich jede Kennzahl, die ueber den
    # ganzen Kader gemittelt wird: ein Drittel des Exports bestand aus ihnen.
    #
    # Wer ohne Kategorie in der Kaderpflege fehlt, bleibt drin — nicht
    # eingetragen heisst nicht "Reserve".
    vorher = len(df)
    df = df[df.kategorie.isna() | (df.kategorie != 5)].reset_index(drop=True)
    ohne_k5 = vorher - len(df)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "avg-je-einsatz-v2",
        "season": saison,
        "liga": "1",
        "hinrunden_spieltage": hin_spieltage,
        "params": {"k": K_MIX, "k_tw": K_TW, "delta": DELTA,
                   "bl2_gewicht": BL2_GEWICHT, "kappa_bl2": KAPPA_BL2,
                   "k_minje": K_MINJE, "delta_rolle": DELTA_ROLLE,
                   "lambda": LAMBDA_B, "sd_ref": round(sd_ref, 2),
                   "kategorie_rollen": KATEGORIE_ROLLEN},
        "ohne_prognose": {"kategorie_5": ohne_k5},
        "players": {
            r.kb_id: {
                "name": r.name,
                "team_id": r.team_id,
                "pos": r.pos,
                "kategorie": None if pd.isna(r.kategorie) else int(r.kategorie),
                "p90": round(float(r.p90_dach), 1),
                "minje": round(float(r.minje_dach_neu), 1),
                "xp_erwartet": round(float(r.xp_dach), 1),
                "xp_gesetzt": round(float(r.xp_gesetzt), 1),
                "p_rolle": round(float(r.p_rolle), 3),
                "einsaetze_erwartet": round(float(r.einsaetze_erwartet), 1),
                "streuung": round(float(r.streuung), 1),
                "sicherheit": round(float(r.sicherheit), 3),
                "n_eff": round(float(r.n_eff) if pd.notna(r.n_eff) else 0.0, 1),
                "quellen": list(r.quellen),
            } for r in df.itertuples(index=False)
        },
    }
    target = path or (LEGACY_DIR / "player_projections.json")
    atomic_write_json(target, payload)
    return payload


def _cli() -> None:
    # Die Windows-Konsole liefert cp1252 - an "Pavlović" scheitert sonst die
    # Ausgabe, nicht die Rechnung. Die JSON-Datei ist davon unberuehrt.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    args = sys.argv[1:]

    def _wert(name: str, default: float) -> float:
        return float(args[args.index(name) + 1]) if name in args else default

    if "--versatz" in args:
        gemessen = versatz_messen(backtest_p90())
        print("VERSATZ neu gemessen (gesetzte Spieler, walk-forward):")
        for f in ("a", "b", "c"):
            print(f"  Fall {f}: {gemessen.get(f, float('nan')):+6.1f}   "
                  f"im Code: {VERSATZ.get(f, 0.0):+6.1f}")
        return

    if "--backtest" in args:
        rows = backtest_p90(
            delta=_wert("--delta", DELTA), k_mix=_wert("--k", K_MIX),
            pool_bl2="--pool-bl2" in args,
            ohne_teamstaerke="--ohne-teamstaerke" in args,
            mit_vorgaenger="--vorgaenger" in args,
            nur_mix="--nur-mix" in args,
            bl2_gewicht=_wert("--bl2-gewicht", BL2_GEWICHT),
            k_tw=_wert("--k-tw", K_TW),
            orakel="--orakel" in args)
        report_ertrag(rows)
        report_rolle(rows)
        report_streuung(rows)
        report_p90(rows)
        report_produkt(rows)
        report_oekonomie(rows)
        return

    payload = export()
    spieler = payload["players"]
    mit_kat = sum(1 for p in spieler.values() if "kategorie" in p["quellen"])
    ohne_hist = sum(1 for p in spieler.values() if "baseline" in p["quellen"])
    print(f"{len(spieler)} Spieler, Saison {payload['season']}, "
          f"{payload['hinrunden_spieltage']} Hinrunden-Spieltage")
    print(f"  {mit_kat} mit gepflegter Rolle, {ohne_hist} ohne eigene Historie")
    k5 = payload.get("ohne_prognose", {}).get("kategorie_5", 0)
    print(f"  {k5} Spieler der Kategorie 5 ohne Prognose — wer normalerweise")
    print("  nicht spielt, bekommt keine Zahl; welcher davon durchbricht, ist")
    print("  vorab nicht bekannt.")
    print("  Rolle bekannt zu haben ist gemessen den Sprung r 0,618 -> 0,779 wert;")
    print("  wie viel die Pflege davon einloest, zeigt erst die laufende Saison.")
    top = sorted(spieler.items(), key=lambda kv: -kv[1]["xp_erwartet"])[:10]
    print("  Top 10 nach erwarteten Oe Punkten je Einsatz:")
    for _, p in top:
        print(f"    {p['name']:18s} {p['pos']:3s} Oe {p['xp_erwartet']:6.1f}  "
              f"(gesetzt {p['xp_gesetzt']:6.1f})  P(Rolle) {p['p_rolle']*100:4.0f}%"
              f"  +/-{p['streuung']:4.1f}")
    print(f"geschrieben: {LEGACY_DIR / 'player_projections.json'}")


if __name__ == "__main__":
    _cli()
