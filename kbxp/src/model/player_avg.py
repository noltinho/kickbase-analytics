"""Spielermodell nach Fallunterscheidung — Ø Punkte je Einsatz.

Der Forschungs-Challenger (``player_role_model.py``) presst alle Spieler in *ein* Modell
``p90 x Minuten je Einsatz``. Dieses hier dreht das um: es lernt
ausschliesslich auf Spielern, die eine **normale Saison als Gesetzte**
gespielt haben, und unterscheidet danach, was ueber den Spieler ueberhaupt
bekannt ist. Zielgroesse ist der Kickbase-Schnitt **Ø Punkte je Einsatz**,
modelliert unter der Annahme *keine Verletzungen*: Ausfaelle sind nicht
vorhersehbar und veraendern den Schnitt kaum, weil man den Spieler ersetzen
kann.

Drei Faelle, drei Koeffizientensaetze in **einer** Regression:

    a  Vorsaison in Liga 1 und dort Starter   -> eigene Historie
    b  Vorsaison vorhanden, aber kein Starter -> Historie als Nebenmerkmal
    c  keine verwertbare Vorsaison            -> Vorgängerzelle, relative Qualität

Dazu, ueber alle drei Faelle hinweg gemeinsam geschaetzt, der
**Mannschaftsteil** und seine Wechselwirkung mit der Teamstaerke.

Walk-forward ueber die Zielsaisons 2019-2025 (Ziel: Ø Punkte je Einsatz):

    Fall a  n=1011  r 0,709  MAE 16,4  Verzerrung +1,9
    Fall b  n=278   r 0,556  MAE 15,9  Verzerrung +0,1
    Fall c  n=282   r 0,544  MAE 18,1  Verzerrung +0,8
    alle    n=1571  r 0,689  MAE 16,6  Verzerrung +1,4

Liga 2 verwendet dieselbe fachliche Struktur, fittet ihr Niveau und ihre
Koeffizienten aber ausschliesslich auf Zweitliga-Zeilen. Das Panel beginnt
dort 2021/22; der ehrliche Walk-forward umfasst deshalb die Zielsaisons
2023-2025:

    Fall a  n=317   r 0,591  MAE 16,2  Verzerrung +1,2
    Fall b  n=151   r 0,340  MAE 16,3  Verzerrung -1,6
    Fall c  n=179   r 0,327  MAE 16,2  Verzerrung -0,2
    alle    n=647   r 0,529  MAE 16,2  Verzerrung +0,1

Anders als Liga 1 besitzt Liga 2 keine handgepflegten Kategorien. Eine kleine
vorgeschaltete Logistik ordnet deshalb nur die Zielgruppe, nicht ihre Punkte:
aus Vorsaison-Starts, -Einsaetzen, Alter und relativem Kader-Marktwert werden
die besten acht je Verein gewaehlt. Walk-forward 2022-2025 wurden 64,2 % von
ihnen tatsaechlich Starter (Recall 42,5 %). Acht je Verein entsprechen im
Umfang exakt den 144 manuell gepflegten Liga-1-Kandidaten; eine ligaweite
Schwelle wurde verworfen, weil sie kleine Vereine fast leer liess.

Die t-Werte zaehlen die **Saison** als unabhaengige Einheit, nicht die Zeile —
die Spieler einer Saison teilen sich ihr Kohortenglueck. Ohne den Versatz
(``versatz_tabelle``) stuende Fall b bei +5,3 mit t = 2,4, also als einzige
echte Verzerrung des Modells; er kostet 0,007 r und senkt den MAE um 0,1.

Die Vergleichsmarke gilt nur fuer Fall a — dort schreibt der rohe
Vorjahres-Startelfschnitt mit r 0,711 fast gleich gut fort, aber bei MAE 18,5
und einer Verzerrung von +8,7. Der Gewinn des Modells liegt im Niveau, nicht
im Rang, und darin, dass b und c ueberhaupt eine Zahl bekommen. Alle Konstanten sind gemessen, nicht gesetzt; der
Messaufbau steht jeweils dabei, damit niemand dieselbe Frage zweimal stellt.

**Startelf-Historie als Eingang, Gesamtschnitt als Ziel.** Bei gleicher
Zielgroesse sagt der Vorjahres-Schnitt aus Startelfeinsaetzen besser voraus
(r 0,678) als der Vorjahres-Gesamtschnitt (0,664), n = 1.504. Joker-
Kurzeinsaetze sind im Eingang Rauschen, im Ziel aber echter Ertrag — Rotation
gehoert zum Kickbase-Schnitt dazu. Der Jokeranteil selbst traegt nichts
(0,678 -> 0,679 bei einer Autokorrelation von nur 0,404) und ist kein Merkmal.

**Mehrere Vorsaisons schlagen die letzte allein.** Auf dem Startelf-Ziel: nur
Vorsaison r 0,717, geometrisch gewichtet (delta 0,5) 0,736, einsatzgewichtet
mit delta 0,7 ueber drei Saisons 0,740. Schrumpfung zur Ligamitte bringt
nichts (0,717).

**Wie stark die Historie zaehlt, haengt von ihrer Belegung ab.** Ein fester
Koeffizient schrumpft Kimmich mit vielen belegten Saisons genauso wie einen
Spieler mit einer einzigen Saison. Deshalb interagiert die Abweichung vom
100-Punkte-Sockel mit ``gewicht / (gewicht + 5)``. Der gelernte Grenzertrag
der Historie steigt dadurch mit der Zahl der belegten Starts; im Walk-forward
sinkt der Gesamt-MAE von 16,67 auf 16,62, bei Fall a von 16,50 auf 16,45 und
im starken Historiensegment von 23,35 auf 22,28.

**Die Teamstaerke wirkt als Niveau, nicht als Differenz.** Fall a: nur
Eigenhistorie r 0,718, plus *Differenz* der prognostizierten Teamstaerke 0,719
(also nichts), plus **Niveau** ``z_hat`` 0,746. Mit der tatsaechlichen
Teamstaerke der Zielsaison waeren es 0,790 — die Luecke ist der
unvorhersehbare Teil und die obere Schranke dieses Kanals.

**Die Teamstaerke-Prognose ist historisch backtestbar.** Der Kader-Marktwert
vor der Saison (``tm_players.csv``, je Verein summiert) sagt das Angriffsniveau
so gut voraus wie die Vorsaison (r 0,792 gegen 0,782) und zusammen mit ihr
r 0,839. Das ist das historische Gegenstueck zu ``season_odds.json``, das es
nur fuer die laufende Saison gibt.

**Die Vorgaenger-Zelle traegt Fall c — und nur ihn.** Was die Startelfspieler
desselben Vereins auf derselben Position zuletzt geholt haben, ist die
woertliche Antwort auf die Frage, die bei einem Neuzugang gestellt wird: Gadou
erbt von Anton, Schlotterbeck und Bensebaini. Sie deckt zwei Drittel der Faelle
c ab (250 von 412 auf der feinsten Stufe), der Rest faellt ueber
Mannschaftsteil und Verein bis aufs Ligamittel zurueck.

Die Qualität des Neuzugangs wird dabei immer relativ zu genau dieser Zelle
formuliert: ``rel_mw = log2(MW neu / MW Vorgänger)`` und ``rel_age = Alter neu
- Alter Vorgänger``. Fehlt die Positionszelle, fallen Punktebasis und
Vergleich gemeinsam über Mannschaftsteil und Verein bis zum Positions-
Ligaprior zurück. Damit bedeutet derselbe absolute Marktwert je nach Verein
etwas anderes.

**Wie stark sie in der breiten Mitte durchschlagen darf, ist gemessen.**
Neuzugaenge holen im Schnitt 83 % ihrer Zelle (93,7 -> 77,7), und der Abschlag
waechst mit der Zelle: im obersten Fuenftel (Zelle > 105) stehen 128,4 gegen
tatsaechlich erzielte 92,2 — 36 Punkte darunter. Bivariat ergibt das
``Ist = 0,41 x Zelle + 39,1`` bei r 0,345; neben ``z_hat`` und der relativen
Qualität bleibt davon ein Koeffizient von 0,095, weil die Teamstaerke
denselben Inhalt schon traegt.
Wer der Zelle mehr Gewicht gibt, zahlt monoton: 0,3 kostet Fall c
r 0,498 / MAE 19,0, das volle Gewicht 1,0 dann 0,397 / 24,0.

Die relative Spezifikation liegt im Walk-forward bei Fall-c r 0,544 und MAE
18,12. Absolute Werte sind beim MAE mit 17,98 minimal besser, sortieren aber
einen gleich teuren Augsburg- und Leverkusen-Neuzugang nach demselben Maßstab;
sie widersprechen damit der erklärten Modellfrage. Der Unterschied kostet im
Gesamt-MAE nur 0,02 Punkte (16,64 statt 16,62). Ein zusätzliches Teamdelta
wird nicht doppelt eingerechnet: Es verschlechtert Fall c auf r 0,534 / MAE
18,29; das aktuelle Teamniveau steht bereits als ``z_hat`` im Modell.

Konkret: Moreira erbt in Leverkusen 116,6 Punkte; seine 20 Mio. entsprechen
nahezu exakt dem Marktwert der Vorgänger (Verhältnis 1,02), auch das Alter ist
praktisch gleich. Daraus werden 98,6 Punkte. Karetsas erbt bei Dortmund die
Zelle aus Brandt und Beier mit 118,4 Punkten. Er kostet das 1,40-Fache seiner
Vorgänger und ist 7,3 Jahre jünger. Für Liganeulinge ab 35 Mio. ist außerdem
auf strikt
frueheren OOF-Prognosen eine eigene Unterschaetzung messbar: 13 Testfaelle
lagen im Mittel 25,3 Punkte zu tief. Eine ueber Saisons gemittelte und um drei
Prior-Saisons geschrumpfte Tail-Kalibrierung hebt Karetsas deshalb allgemein,
nicht namensbezogen, auf 109,9. Der Grund fuer den sonst kleinen Zellbeitrag
ist, dass sie weitgehend dasselbe misst wie die
Teamstaerke-Prognose — r 0,749 zwischen beiden, Eigenbeitrag nach ``z_hat``
r 0,051 — und dass sie duenn steht: 81 % aller Zellen tragen hoechstens zwei
Spieler, und in 37 % der Faelle besetzt derselbe Spieler sie in beiden
Saisons, misst also eine Person und keine Stelle.

**Als Anker taugt sie nicht.** Die Zielgroesse als Abweichung von der Zelle zu
modellieren — also ihr per Konstruktion Koeffizient 1 zu geben — bricht ein:
Fall c faellt auf r 0,397 bei MAE 24,0. Sie streut zu weit, um das Niveau
allein zu setzen; ein BVB-Neuzugang erbt 118 Punkte, waehrend Neuzugaenge
tatsaechlich hart zur Mitte streben. Sie gehoert als **Merkmal** neben die
Teamstaerke, nicht an ihre Stelle.

**Der Mannschaftsteil dagegen traegt, aber nur als Wechselwirkung.** Ohne ihn
ist das Modell systematisch verzerrt: Torhueter -8,5, Zentrum +7,4, Abwehr
+1,9, Offensive +3,3 Punkte. Im Gesamt-r faellt das nicht auf, weil sich die
Fehler gegeneinander aufheben — genau deshalb weist ``--backtest`` die
Verzerrung je Teil aus. Mit Teil-Dummies **und** ihrer Wechselwirkung zu
``z_hat`` steigt r von 0,670 auf 0,685, MAE faellt von 17,4 auf 17,0, und die
Torwart-Verzerrung schrumpft auf -1,8. Das dahinterliegende Gesetz ist
einfach: Teamstaerke schlaegt sich in der Offensive nieder (+6,2 Punkte je sd
gegenueber dem Torwart), im Tor gar nicht — wer dominiert, laesst seinen
Keeper ohne Paraden.

**Die vereinsspezifische Abweichung darueber hinaus ist real, aber nicht
nutzbar.** Verein x Mannschaftsteil hat, ueber die Vereinsstaerke hinaus,
sd 13,4 Punkte und eine Autokorrelation von r 0,598 (0,540 auch ohne jede
Personenueberlappung — es ist die Stelle, nicht die Person). Bayern: Offensive
+16,6, Zentrum +14,8, Abwehr -12,4, Tor -49,1. Als Merkmal aus der Historie
bringt sie im Walk-forward trotzdem nichts (r 0,680 gegen 0,682 ohne sie); das
strukturelle Gesetz oben nimmt den nutzbaren Teil bereits mit.

**Die Live-Teamstaerke muss skaliert werden.** ``z_hat`` ist historisch eine
geschrumpfte Prognose mit sd 0,84; ``ratings.json`` liefert dagegen einen auf
sd 0,97 standardisierten Wert, also die Skala der *gemessenen* Staerke. Ohne
Korrektur stuende Bayern bei +3,04 statt +2,64 und alle Bayern-Spieler zu
hoch. ``z_hat_zielsaison`` rechnet die Live-Werte deshalb auf die historische
Prognoseskala um.

**Rueckgewichtete Trainingssaisons.** Rollenwechsler und Neuzugaenge sind ab
2020 um rund 20 Punkte abgefallen (Fall b: 94,8 in 2018/19, 65,1 in 2021/22),
waehrend etablierte Starter ihr Niveau hielten. Ein ungewichteter Fit traegt
das alte Niveau in die Gegenwart: Verzerrung der Faelle b/c +7,0 statt +4,3.
Die Wahl von rho ist am juengsten Jahr entschieden, nicht am Mittel ueber alle
- die Begruendung steht bei der Konstante.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model.player_features import (  # noqa: E402
    lade_panel, lade_tm, start_kennzahlen, team_niveaus, tm_je_spieler)
from src.model.team_strength import conceded_from_panel  # noqa: E402
from src.paths import LEGACY_DIR, MANUAL, PROCESSED, atomic_write_json  # noqa: E402

LIGEN = {"1": "Bundesliga", "2": "2. Bundesliga"}
LIGA_IDS = {v: k for k, v in LIGEN.items()}
LIGA1 = LIGEN["1"]
LIGA2 = LIGEN["2"]

# Der Starter-Filter. Er entscheidet, wer ins Training kommt und wer als
# "Rolle gehalten" gilt: 2.842 Spielersaisons in Liga 1 bestehen ihn.
#
# Es ist bewusst nur *eine* Bedingung. Eine zweite ueber die Einsatzlaenge lag
# nahe (der fruehe Auswechsler, der zwar oft beginnt, aber nach einer Stunde
# geht) — sie greift nur nie: schon ab acht Startelfeinsaetzen liegt die
# kleinste gemessene Einsatzlaenge bei exakt 60,0 Minuten, und im Walk-forward
# aendert eine Schwelle von 70 nichts (r 0,685 mit wie ohne). Trainer lassen
# einen Spieler nicht fuenfzehnmal beginnen und dann regelmaessig frueh raus.
# Die Startzahl selbst ist flach: 15 -> r 0,685, 17 -> 0,688, 20 -> 0,685; 15
# haelt die Zielgruppe am groessten, ohne Guete zu kosten.
MIN_STARTS = 15

# Gewichtung der eigenen Historie: delta je Saison zurueck, zusaetzlich nach
# Startelfeinsaetzen gewichtet, Fenster drei Saisons.
DELTA = 0.7
FENSTER = 3

# Mannschaftsteile. Die Aufloesung ist gemessen: so erreicht das Modell r 0,685
# bei einer groessten Teil-Verzerrung von 4,0 — feiner wird es schlechter
# (ST einzeln 0,674; jede Position einzeln 0,672 bei Verzerrung 5,6), weil die
# Zellen zu duenn werden. IV und AV bleiben getrennt: 6,2 Punkte
# Niveauunterschied, und es kostet nichts.
TEIL = {"TW": "TW", "IV": "IV", "AV": "AV", "ZDM": "MIT", "ZM": "MIT",
        "ZOM": "OFF", "FL": "OFF", "ST": "OFF"}
TEIL_REF = "TW"
TEILE = ("IV", "AV", "MIT", "OFF")

# Rueckgewichtung der Trainingssaisons: Gewicht rho^Abstand. Das ist die
# einzige der drei freien Konstanten, die ueberhaupt wirkt (delta und fenster
# liegen ueber das ganze Raster zwischen r 0,681 und 0,686).
#
# Gemessen, und der Zielkonflikt ist echt: r steigt monoton mit rho
# (0,30 -> 0,677 ... 1,00 -> 0,692), die Verzerrung der Faelle b/c faellt
# monoton mit rho (+7,0 bei 1,00 ... +3,0 bei 0,30). Entschieden hat die
# Verzerrung auf der **juengsten** Zielsaison — die einzige, die einer
# Live-Prognose gleicht: +1,9 bei rho 1,00, +0,9 bei 0,60 und 0,70, wieder
# +1,8 bei 0,30. 0,6 ist damit kein Kompromiss, sondern das Minimum; die
# 0,007 r gegenueber dem ungewichteten Fit sind der Preis fuer die Eichung.
#
# Der Grund fuer den Effekt: Rollenwechsler und Neuzugaenge sind ab 2020 um
# rund 20 Punkte abgefallen (Fall b: 94,8 in 2018/19, 65,1 in 2021/22),
# waehrend etablierte Starter ihr Niveau hielten.
RHO = 0.6

# Erste Zielsaison des Walk-forward. Davor ist die Historie zu duenn.
ERSTE_ZIELSAISON = 2016

# Grundmerkmale je Fall. Der gemeinsame Mannschaftsteil-Block kommt in
# ``_entwurf`` dazu. Fall c beschreibt die Qualität eines Neuzugangs relativ
# zu den tatsächlichen Vorgängern seiner Zelle: ``rel_mw`` ist der Logarithmus
# des Marktwertverhältnisses, ``rel_age`` die Altersdifferenz. Die absolute
# Qualität der Stelle tragen ``zelle`` und das aktuelle Teamniveau ``z_hat``.
MERKMALE = {
    "a": ["a1", "z_hat", "log2_mw", "hist_belegt", "a1_belegt"],
    "b": ["z_hat", "log2_mw", "alter", "a1_ers", "hat_hist"],
    "c": ["zelle", "z_hat", "rel_mw", "rel_age"],
}

# Merkmale, ohne die eine Zeile keine Beobachtung ist, sondern ein Mittelwert
# mit Etikett. Sie werden nie gefuellt, die uebrigen schon.
#
# Nur Fall a hat eines: dort *ist* die eigene Historie das Modell. In Fall b
# darf sie fehlen — ein Aufsteiger-Stammspieler hat eine Vorsaison, aber keine
# in Liga 1, und genau fuer ihn ist ``hat_hist`` da. Waere sie Pflicht, fielen
# die Aufsteiger still aus dem Export (gemessen: 28 von 145 Kandidaten).
PFLICHT = {"a": ["a1"], "b": [], "c": []}

# Erwarteter mittlerer Fehler je Fall, aus dem Walk-forward (--backtest).
# Traegt die Punktgroesse in scatter.html; wer eine kleinere Zahl sehen will,
# muss sie erst messen.
STREUUNG = {"a": 16.4, "b": 15.9, "c": 18.1}
STREUUNG_LIGA = {
    "1": STREUUNG,
    # Walk-forward 2023-2025. Die Zweitliga-Stichprobe ist kuerzer, deshalb
    # werden die Werte bewusst nur auf eine Nachkommastelle als Groessenordnung
    # und nicht als scheinbar praezises Konfidenzintervall exportiert.
    "2": {"a": 16.2, "b": 16.3, "c": 16.2},
}

# Liga 2 hat keine handgepflegte Rollenliste. Die bedingte Punkteprognose darf
# aber nicht still dem ganzen Kader unterstellen, er spiele eine normale Saison
# als Gesetzter. Deshalb steht davor ein eigener, strikt vor Saisonbeginn
# beobachtbarer Auswahlkanal. Die besten acht je Verein entsprechen im Umfang
# den 144 handgepflegten Liga-1-Kategorien 1-2. Im Walk-forward 2022-2025
# wurden 64,1 % davon tatsaechlich Starter; ein Teil der uebrigen Faelle sind
# Verletzungen und damit fuer die Annahme "gesund und gesetzt" keine echten
# Fehlklassifikationen. Eine ligaweite Wahrscheinlichkeitsschwelle waere sachlich
# falsch: sie liess kleine Vereine fast leer und grosse Vereine uebervoll.
STARTER_JE_TEAM = 8
STARTER_KATEGORIE_1_JE_TEAM = 4
STARTER_MERKMALE = (
    "prev_starts", "prev_apps", "had_prev", "mw_team_pct",
    "mw_liga_z", "alter", "age2",
)

# Ueber so viele Saisons wird der ligaweite Positionsaufschlag gemittelt. Das
# Niveau einer einzelnen Saison ist verrauscht (ZM schwankt zwischen 74 und
# 110), ueber alle zu mitteln waere falsch, weil es driftet. Gemessen am Fehler
# des Sockels: Vorsaison allein 5,58 - drei Saisons 5,47 - alle 6,10.
LVL_FENSTER = 3

# Ueber so viele Saisons wird gesucht, ob ein Spieler schon einmal Starter war.
# Der Grund ist eine Luecke, die sich nicht schliessen laesst: Bank und
# Verletzung sind im Panel nicht zu unterscheiden — beide sind
# ``player_md_status == 4`` mit null Minuten, und history.json traegt den
# Verletzungsstatus erst ab dem Tag, an dem der Fetcher ihn mitzuschreiben
# begann (2026-08-10), also fuer keine vergangene Saison.
#
# Statt die Ursache zu raten, wird sie umgangen: wer in einer der letzten zwei
# Saisons Starter war, bleibt Fall a. Eine ausgefallene Saison loescht damit
# nicht, was ueber den Spieler bekannt ist. Gemessen im Walk-forward ist das
# auf jeder Kennzahl besser als nur die Vorsaison zu lesen (r 0,684 statt
# 0,678, MAE 16,7 statt 16,9), und Fall b gewinnt am meisten: Verzerrung +0,7
# statt +2,7 bei MAE 15,9 statt 16,3. Drei Saisons sind wieder schlechter
# (0,682) — dann kommen die echten Absteiger mit herein.
RUECKBLICK = 2

# Zuverlaessigkeit der eigenen Historie. ``gewicht`` zaehlt die nach
# Aktualitaet gewichteten Startelfeinsaetze. Die Wechselwirkung erlaubt der
# Regression, eine Abweichung vom 100-Punkte-Sockel bei 60 belegten Starts
# staerker zu erhalten als bei einer einzigen Saison. Im Walk-forward sinkt
# der Gesamt-MAE 16,67 -> 16,62 und der Fall-a-MAE 16,50 -> 16,45.
K_HIST_BELEGT = 5.0

# So viele Out-of-fold-Saisons braucht eine Versatz-Schaetzung mindestens.
# Darunter ist ihr Mittelwert selbst Kohortenglueck (sd der Saisonverzerrung
# 6,9 Punkte) und die Korrektur schadet mehr, als sie nuetzt.
MIN_VERSATZ_SAISONS = 3

# Die Regression trifft die breite Mitte, unterschaetzt aber sehr teure
# Liganeulinge systematisch. Das ist ein sachlich vorab definiertes Segment,
# kein Spieler-Override. Auf den strikt walk-forward erzeugten Rohprognosen
# verbessert seine Kalibrierung den Segment-MAE von 35,19 auf 30,89. Der Bonus
# wird ueber Saisons gemittelt und um drei Prior-Saisons zu null geschrumpft.
# Starke eigene Historien werden dagegen kontinuierlich ueber
# ``hist_belegt`` behandelt, nicht mehr ueber einen harten Grenzwert.
SPITZE_MW = 35_000_000.0
KALIBRIER_PRIOR_SAISONS = 3.0

# Verletzungscodes aus history.json, die als Ausfall zaehlen. 2 (angeschlagen)
# und 3 (gesperrt) nicht — wer angeschlagen ist, spielt meist, Sperren sind kurz.
AUSFALL_CODES = (1, 4)


# ---------------------------------------------------------------------------
# Kleinwerkzeug
# ---------------------------------------------------------------------------

def _jahr(season):
    """"2025/2026" -> 2025. Das Anfangsjahr ist ueberall der Schluessel."""
    if isinstance(season, str):
        return int(season[:4])
    return season.str.slice(0, 4).astype(int)


def _wols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Kleinste Quadrate mit Achsenabschnitt, optional saisonsgewichtet.

    Keine Ridge: bei drei bis elf Merkmalen und mindestens hundert Zeilen
    braucht es keine, und eine Strafe wuerde genau das Niveau schrumpfen, das
    hier die eigentliche Leistung ist.
    """
    A = np.c_[np.ones(len(X)), X]
    if w is None:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    sw = np.sqrt(w)[:, None]
    return np.linalg.lstsq(A * sw, y * np.sqrt(w), rcond=None)[0]


def _pred(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ coef


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Positionen: eine Quelle, ein Vorrang
# ---------------------------------------------------------------------------

def positions_quelle() -> tuple[dict, dict]:
    """Kaderpflege vor Transfermarkt — und die Kaderpflege gilt rueckwirkend.

    ``fine_positions.csv`` sagt, wo ein Spieler in der Formation seines
    Vereins *spielt*, Transfermarkt, was er nominell *ist*. Beide stimmen nur
    zu 86,7 % ueberein, haeufigste Abweichung ZM gegen ZDM (18 von 52 Faellen).
    Wer in der Kaderpflege steht, bekommt deren Etikett auch fuer seine
    frueheren Saisons: von den 221 Startern 2025/26 sind das 162, und bei 29
    korrigiert es die TM-Position.
    """
    kp = pd.read_csv(MANUAL / "fine_positions.csv")
    kp = kp[kp.player_id.notna() & kp.position_fine.notna()]
    fein = dict(zip(kp.player_id.astype("int64"), kp.position_fine))

    tm = tm_je_spieler(lade_tm())
    tm = tm[tm.position_fine.notna()]
    grob = {(int(p), s): v for p, s, v
            in zip(tm.player_id, tm.season, tm.position_fine)}
    return fein, grob


def position_von(df: pd.DataFrame, fein: dict, grob: dict) -> pd.Series:
    """Der Vorrang, spaltenweise auf beliebige Spielersaison-Zeilen angewandt."""
    aus_kp = df.player_id.map(fein)
    aus_tm = pd.Series(
        [grob.get((p, s)) for p, s in zip(df.player_id, df.season)],
        index=df.index, dtype="object")
    return aus_kp.astype("object").fillna(aus_tm)


# ---------------------------------------------------------------------------
# Spielersaisons
# ---------------------------------------------------------------------------

def ist_starter(df: pd.DataFrame, min_starts: int = MIN_STARTS,
                min_minuten: float = 0.0) -> pd.Series:
    """Hat der Spieler eine normale Saison als Gesetzter gespielt?

    ``min_minuten`` steht nur noch als Schalter fuer die Messung da und ist
    im Regelfall aus — die Begruendung steht bei MIN_STARTS.
    """
    ok = df.n_start >= min_starts
    return ok & (df.minj_start >= min_minuten) if min_minuten > 0 else ok


def je_spielersaison(sk: pd.DataFrame) -> pd.DataFrame:
    """Zwei Kaderzeilen eines Winterwechslers zu einer Spielersaison addieren.

    Fuer die Frage "war er Stammspieler" zaehlt die ganze Saison, nicht die
    Haelfte bei einem der beiden Vereine.
    """
    g = sk.groupby(["player_id", "season", "league"], as_index=False).agg(
        punkte=("punkte", "sum"), n_einsaetze=("n_einsaetze", "sum"),
        punkte_start=("punkte_start", "sum"), n_start=("n_start", "sum"),
        min_start=("min_start", "sum"), team_id=("team_id", "first"))
    g["avg_alle"] = np.where(g.n_einsaetze > 0, g.punkte / g.n_einsaetze, np.nan)
    g["avg_start"] = np.where(g.n_start > 0, g.punkte_start / g.n_start, np.nan)
    g["minj_start"] = np.where(g.n_start > 0, g.min_start / g.n_start, np.nan)
    g["j"] = _jahr(g.season)
    return g


# ---------------------------------------------------------------------------
# Teamstaerke
# ---------------------------------------------------------------------------

def kader_marktwerte(tm: pd.DataFrame, panel: pd.DataFrame,
                     liga: str = "1") -> pd.DataFrame:
    """Kader-Marktwert je Verein und Saison, uebersetzt auf die Kickbase-team_id.

    Die Bruecke laeuft ueber die Spieler, nicht ueber eine gepflegte
    Vereinsliste: Kickbase fuehrt Vereine unter wechselnden IDs (Rostock
    2021/22 unter 49, ab 2022/23 unter 23), eine Namensliste waere still
    falsch. Das haeufigste team_id der TM-Kaderspieler gewinnt.
    """
    gespielt = panel[panel.md_status == 2]
    bruecke = (gespielt.merge(tm[["player_id", "season", "verein"]],
                              on=["player_id", "season"])
               .groupby(["season", "verein", "team_id"], as_index=False).size()
               .sort_values("size", ascending=False)
               .drop_duplicates(["season", "verein"]))

    kad = (tm.groupby(["season", "verein"], as_index=False)
             .agg(mw=("mw_vor_saison", "sum"), liga=("liga", "first")))
    kad = kad.merge(bruecke[["season", "verein", "team_id"]],
                    on=["season", "verein"])
    kad = kad[kad.liga == int(liga)].copy()
    kad["j"] = _jahr(kad.season)
    kad["log_mw"] = np.log(kad.mw.where(kad.mw > 0))
    kad["z_mw"] = kad.groupby("j").log_mw.transform(
        lambda s: (s - s.mean()) / s.std())
    return kad[["j", "season", "team_id", "verein", "z_mw"]]


def teamstaerke_tabelle(panel: pd.DataFrame, tm: pd.DataFrame,
                        league: str = LIGA1,
                        liga: str = "1") -> pd.DataFrame:
    """Je (Saison, Team) das gemessene Angriffsniveau und seine Vorhersager.

    ``z_att`` ist die gegnerbereinigte, innerhalb der Saison standardisierte
    Punktesumme der eigenen Spieler — dieselbe Zerlegung wie in ratings.json.
    ``z_prev`` und ``z_mw`` sind vor Saisonstart bekannt, ``auf`` markiert
    Aufsteiger, fuer die es kein z_prev gibt.
    """
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    niv = team_niveaus(conceded_from_panel(panel, matches))
    niv["j"] = _jahr(niv.season)
    nz = niv[niv.league == league][["j", "team_id", "z_att"]]

    kad = kader_marktwerte(tm, panel, liga)
    vor = nz.rename(columns={"z_att": "z_prev"}).assign(j=lambda d: d.j + 1)
    tab = (kad.merge(nz, on=["j", "team_id"], how="left")
              .merge(vor, on=["j", "team_id"], how="left"))
    tab["auf"] = tab.z_prev.isna().astype(float)
    tab["z_prev0"] = tab.z_prev.fillna(0.0)
    return tab


Z_MERKMALE = ["z_prev0", "auf", "z_mw"]


def z_hat_fuer(tab: pd.DataFrame, j_ziel: int) -> dict:
    """Prognostizierte Teamstaerke der Zielsaison, walk-forward gefittet.

    Gemessen erreicht ``z_att ~ z_prev + auf + z_mw`` r 0,839 gegen 0,782 fuer
    die Vorsaison allein und 0,792 fuer den Kader-Marktwert allein. Der Rest
    ist unvorhersehbar — deshalb steht im Spielermodell ein kleinerer
    Koeffizient als der, den man mit Kenntnis der Zielsaison messen wuerde.
    """
    tr = tab[(tab.j < j_ziel) & tab.z_att.notna()]
    zt = tab[tab.j == j_ziel]
    if len(tr) < 15 or not len(zt):
        return {}
    coef = _wols(tr[Z_MERKMALE].fillna(0.0).values, tr.z_att.values)
    return dict(zip(zt.team_id, _pred(coef, zt[Z_MERKMALE].fillna(0.0).values)))


def z_hat_skala(tab: pd.DataFrame, bis_j: int) -> float:
    """Streuung, die eine ehrliche Teamstaerke-*Prognose* hat.

    Nicht 1: eine Prognose ist geschrumpft, weil sie nur das Vorhersehbare
    enthaelt. Gemessen liegt sie bei 0,84, waehrend die gemessene Staerke auf
    1,0 standardisiert ist. Wer beide Skalen verwechselt, zieht die Spitze
    auseinander.
    """
    werte = [v for j in range(ERSTE_ZIELSAISON, bis_j)
             for v in z_hat_fuer(tab, j).values()]
    return float(np.std(werte)) if werte else 1.0


def z_hat_zielsaison(tab: pd.DataFrame, j_ziel: int, liga: str = "1") -> dict:
    """Teamstaerke der laufenden Saison, bevorzugt aus ratings.json.

    Vor dem ersten Spieltag ist ratings.json der reine Quoten-Prior aus
    season_odds.json — dieselbe Groesse wie ``z_hat``, nur aus einer besseren
    Quelle als Vorsaison plus Kader-Marktwert. Weil ratings.json auf die Skala
    der gemessenen Staerke standardisiert ist, wird es auf die Prognoseskala
    zurueckgerechnet; sonst stuende Bayern bei +3,04 statt +2,64.

    Fehlt die Datei, faellt es auf die historische Regression zurueck — der
    Export soll nicht an einer optionalen Datei scheitern.
    """
    pfad = LEGACY_DIR / "ratings.json"
    if not pfad.exists():
        return z_hat_fuer(tab, j_ziel)
    teams = json.loads(pfad.read_text(encoding="utf-8"))["leagues"][liga]["teams"]
    att = pd.Series({t: v["att"] for t, v in teams.items()}, dtype=float)
    if att.std() < 1e-9:
        return z_hat_fuer(tab, j_ziel)
    z = (att - att.mean()) / att.std()
    return (z * (z_hat_skala(tab, j_ziel) / float(z.std()))).to_dict()


# ---------------------------------------------------------------------------
# Eigene Historie
# ---------------------------------------------------------------------------

def historie(ss: pd.DataFrame, j_ziel: int, delta: float = DELTA,
             fenster: int = FENSTER, league: str = LIGA1) -> pd.DataFrame:
    """Einsatzgewichteter Startelfschnitt der letzten Saisons derselben Liga.

    Gemessen: nur Vorsaison r 0,717, einsatzgewichtet ueber drei Saisons mit
    delta 0,7 dann 0,740. Das Gewicht zaehlt doppelt — nach Aktualitaet und
    nach Zahl der Startelfeinsaetze —, weil eine halbe Saison eben eine halbe
    Aussage ist.
    """
    h = ss[(ss.league == league) & (ss.j < j_ziel)
           & (ss.j >= j_ziel - fenster)]
    h = h.dropna(subset=["avg_start"])
    if not len(h):
        return pd.DataFrame({"player_id": [], "a1": [], "gewicht": []})
    w = h.n_start * (delta ** (j_ziel - h.j - 1))
    g = (h.assign(_w=w, _p=h.avg_start * w)
           .groupby("player_id", as_index=False)
           .agg(_p=("_p", "sum"), gewicht=("_w", "sum")))
    g["a1"] = g._p / g.gewicht
    return g[["player_id", "a1", "gewicht"]]


def historie_kontext(dat: "Daten", j_ziel: int, delta: float = DELTA,
                      fenster: int = FENSTER) -> pd.DataFrame:
    """Historische Teamstaerke und Marktwertskala derselben Spiele wie ``a1``.

    Diese Groessen sind keine direkten Punktetreiber. Sie beschreiben die
    Rahmenbedingungen, unter denen die eigene Historie entstanden ist, damit
    Modelle eine Veraenderung zur Zielsaison statt zwei absolute Niveaus
    vergleichen koennen.
    """
    h = dat.ss[(dat.ss.league == dat.league) & (dat.ss.j < j_ziel)
               & (dat.ss.j >= j_ziel - fenster) & dat.ss.avg_start.notna()].copy()
    if not len(h):
        return pd.DataFrame(columns=["player_id", "hist_z", "hist_mw"])
    z = dat.team[["j", "team_id", "z_att"]].dropna(subset=["z_att"])
    h = h.merge(z, on=["j", "team_id"], how="left")
    h["_w"] = h.n_start * (delta ** (j_ziel - h.j - 1))
    h["_wz"] = h._w.where(h.z_att.notna(), 0.0)
    h["_wm"] = h._w.where(h.log2_mw.notna(), 0.0)
    h["_z"] = h.z_att.fillna(0.0) * h._wz
    h["_m"] = h.log2_mw.fillna(0.0) * h._wm
    g = h.groupby("player_id", as_index=False).agg(
        _z=("_z", "sum"), _wz=("_wz", "sum"),
        _m=("_m", "sum"), _wm=("_wm", "sum"))
    g["hist_z"] = g._z / g._wz.replace(0.0, np.nan)
    g["hist_mw"] = g._m / g._wm.replace(0.0, np.nan)
    return g[["player_id", "hist_z", "hist_mw"]]


# ---------------------------------------------------------------------------
# Datenbuendel
# ---------------------------------------------------------------------------

class Daten:
    """Alles, was Backtest und Export gemeinsam brauchen — einmal geladen."""

    def __init__(self, min_starts: int = MIN_STARTS,
                 min_minuten: float = 0.0, liga: str = "1"):
        if liga not in LIGEN:
            raise ValueError(f"Unbekannte Liga {liga!r}; erwartet 1 oder 2")
        self.liga = liga
        self.league = LIGEN[liga]
        self.panel = lade_panel()
        self.tm = lade_tm()
        fein, grob = positions_quelle()
        # Die Kaderpflege beschreibt ausschliesslich die aktuelle Bundesliga.
        # Fuer historische Zweitliga-Folds waere ihre rueckwirkende Anwendung
        # Zukunftswissen; dort gilt deshalb konsequent die saisonale TM-Position.
        if liga == "2":
            fein = {}

        sk = start_kennzahlen(self.panel)
        sk["j"] = _jahr(sk.season)
        sk["pos"] = position_von(sk, fein, grob)
        sk["starter"] = ist_starter(sk, min_starts, min_minuten)
        self.sk_team = sk

        ss = je_spielersaison(sk)
        ss["pos"] = position_von(ss, fein, grob)
        ss["teil"] = ss.pos.map(TEIL)
        ss["starter"] = ist_starter(ss, min_starts, min_minuten)
        tmj = tm_je_spieler(self.tm)[["player_id", "season", "log2_mw", "alter"]]
        self.ss = ss.merge(tmj, on=["player_id", "season"], how="left")

        self.zellen = zell_tabellen(
            sk.merge(tmj, on=["player_id", "season"], how="left"),
            self.league)
        self.team = teamstaerke_tabelle(
            self.panel, self.tm, self.league, self.liga)
        self.zielsaison = str(self.panel.season.max())
        self.j_ziel = _jahr(self.zielsaison)
        erste_ligasaison = int(self.ss.loc[
            self.ss.league == self.league, "j"].min())
        self.erste_zielsaison = max(ERSTE_ZIELSAISON, erste_ligasaison + 1)
        self._zh: dict[int, dict] = {}

    def z_hat(self, j: int) -> dict:
        """Teamstaerke-Prognose je Zielsaison, gecacht — der Fit kostet sonst."""
        if j not in self._zh:
            self._zh[j] = (z_hat_zielsaison(self.team, j, self.liga)
                           if j >= self.j_ziel
                           else z_hat_fuer(self.team, j))
        return self._zh[j]

    def zielgruppe(self, j: int) -> pd.DataFrame:
        """Die Starter einer Saison — Trainingsmenge und Bewertungsmenge zugleich.

        Bewusst dieselbe Menge fuer beides: gelernt wird auf Spielern, die eine
        normale Saison als Gesetzte gespielt haben, und genau fuer solche wird
        prognostiziert. Ein Modell, das auch Randfiguren sortieren muss, misst
        eine Frage, die beim Kaderbau niemand stellt.
        """
        z = self.ss[(self.ss.j == j) & (self.ss.league == self.league)
                    & self.ss.starter]
        return z.dropna(subset=["avg_alle"]).copy()


# ---------------------------------------------------------------------------
# Merkmalszeilen
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Die Vorgaenger-Zelle: was die Startelfspieler dieses Vereins auf dieser
# Position zuletzt geholt haben
# ---------------------------------------------------------------------------

def zell_tabellen(sk_team: pd.DataFrame, league: str = LIGA1) -> dict:
    """Zellen der Vorsaison, als Abweichung vom Ligamittel ihrer Position.

    Warum Abweichung statt rohem Mittelwert: eine Zelle mischt Positionen mit
    verschiedenem Niveau, sobald die Leiter auf Gruppe oder Verein
    zurueckfaellt (ZDM 83,2 gegen ZM 91,5). Der rohe Zellmittelwert gaebe
    beiden dieselbe Zahl; die Abweichung plus das Ligamittel der
    **Zielposition** gibt jedem seinen eigenen Sockel. Auf der feinsten Stufe
    — Verein und Position — ist beides dasselbe.
    """
    s = sk_team[sk_team.starter & (sk_team.league == league)]
    s = s.dropna(subset=["pos", "avg_start"])
    s = s.assign(teil=s.pos.map(TEIL))

    lvl = s.groupby(["j", "pos"], as_index=False).agg(lvl=("avg_start", "mean"))
    liga = s.groupby("j", as_index=False).agg(liga=("avg_start", "mean"))
    lvl = lvl.merge(liga, on="j")
    lvl["off"] = lvl.lvl - lvl.liga
    s = s.merge(lvl[["j", "pos", "lvl"]], on=["j", "pos"])
    s = s.assign(resid=s.avg_start - s.lvl)

    zpos = s.groupby(["j", "team_id", "pos"], as_index=False).agg(
        r=("resid", "mean"), n=("resid", "size"))
    zteil = s.groupby(["j", "team_id", "teil"], as_index=False).agg(
        r=("resid", "mean"))
    zver = s.groupby(["j", "team_id"], as_index=False).agg(r=("resid", "mean"))
    qpos = s.groupby(["j", "team_id", "pos"], as_index=False).agg(
        mw=("log2_mw", "mean"), alter=("alter", "mean"), qn=("player_id", "size"))
    qteil = s.groupby(["j", "team_id", "teil"], as_index=False).agg(
        mw=("log2_mw", "mean"), alter=("alter", "mean"), qn=("player_id", "size"))
    qver = s.groupby(["j", "team_id"], as_index=False).agg(
        mw=("log2_mw", "mean"), alter=("alter", "mean"), qn=("player_id", "size"))
    qliga = s.groupby(["j", "pos"], as_index=False).agg(
        mw=("log2_mw", "mean"), alter=("alter", "mean"), qn=("player_id", "size"))
    return {
        "liga": dict(zip(liga.j, liga.liga)),
        "off": {(j, p): v for j, p, v in zip(lvl.j, lvl.pos, lvl.off)},
        "pos": {(j, t, p): (r, n) for j, t, p, r, n
                in zip(zpos.j, zpos.team_id, zpos.pos, zpos.r, zpos.n)},
        "teil": {(j, t, g): r for j, t, g, r
                 in zip(zteil.j, zteil.team_id, zteil.teil, zteil.r)},
        "ver": {(j, t): r for j, t, r in zip(zver.j, zver.team_id, zver.r)},
        "qpos": {(j, t, p): (mw, a, n) for j, t, p, mw, a, n
                 in zip(qpos.j, qpos.team_id, qpos.pos, qpos.mw,
                        qpos.alter, qpos.qn)},
        "qteil": {(j, t, g): (mw, a, n) for j, t, g, mw, a, n
                  in zip(qteil.j, qteil.team_id, qteil.teil, qteil.mw,
                         qteil.alter, qteil.qn)},
        "qver": {(j, t): (mw, a, n) for j, t, mw, a, n
                 in zip(qver.j, qver.team_id, qver.mw, qver.alter, qver.qn)},
        "qliga": {(j, p): (mw, a, n) for j, p, mw, a, n
                  in zip(qliga.j, qliga.pos, qliga.mw, qliga.alter, qliga.qn)},
    }


def zellen_qualitaet(tab: dict, j_ziel: int, team_id, pos) -> tuple:
    """Marktwert, Alter und Besetzung der Vorgänger derselben Zellstufe."""
    jv = j_ziel - 1
    q = tab["qpos"].get((jv, team_id, pos))
    if q is not None:
        return (*q, "position")
    q = tab["qteil"].get((jv, team_id, TEIL.get(pos)))
    if q is not None:
        return (*q, "teil")
    q = tab["qver"].get((jv, team_id))
    if q is not None:
        return (*q, "verein")
    q = tab["qliga"].get((jv, pos))
    return (*q, "liga") if q is not None else (np.nan, np.nan, 0, None)


def sockel(tab: dict, j_ziel: int, pos) -> float:
    """Das ligaweite Niveau eines Startelfspielers dieser Position.

    Zerlegt in zwei Teile, die verschieden schnell altern: das Ligamittel der
    **Vorsaison** traegt die Epoche (die Punkteniveaus driften spuerbar), der
    Positionsaufschlag wird ueber LVL_FENSTER Saisons gemittelt, weil er in
    einer einzelnen zu duenn steht.
    """
    liga = tab["liga"].get(j_ziel - 1)
    if liga is None or pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return float("nan")
    offs = [tab["off"][(jj, pos)] for jj in range(j_ziel - LVL_FENSTER, j_ziel)
            if (jj, pos) in tab["off"]]
    return liga + float(np.mean(offs)) if offs else float("nan")


def zelle(tab: dict, j_ziel: int, team_id, pos) -> tuple:
    """Was die Startelfspieler dieses Vereins auf dieser Position zuletzt holten.

    Vier Stufen, absteigend fein. Die erste ist die woertliche Antwort auf die
    Frage, fuer die es diese Groesse gibt: Gadou erbt von Anton,
    Schlotterbeck und Bensebaini. Sie deckt zwei Drittel der Faelle c ab; der
    Rest faellt auf Mannschaftsteil, Verein und zuletzt das blosse Ligamittel
    zurueck — letzteres trifft Aufsteiger, deren Verein in der Vorsaison gar
    keine Zeile hat.
    """
    jv = j_ziel - 1
    grund = sockel(tab, j_ziel, pos)
    if not np.isfinite(grund):
        return float("nan"), None
    rn = tab["pos"].get((jv, team_id, pos))
    if rn is not None:
        return grund + rn[0], "position"
    r = tab["teil"].get((jv, team_id, TEIL.get(pos)))
    if r is not None:
        return grund + r, "teil"
    r = tab["ver"].get((jv, team_id))
    if r is not None:
        return grund + r, "verein"
    return grund, "liga"


def merkmalszeilen(dat: Daten, j_ziel: int, kandidaten: pd.DataFrame,
                   delta: float = DELTA, fenster: int = FENSTER) -> pd.DataFrame:
    """Ein Kandidat, eine Zeile: Fall, Historie, Teamstaerke, Mannschaftsteil.

    ``kandidaten`` braucht ``player_id``, ``team_id`` und ``pos``. Alles andere
    stammt aus Saisons vor ``j_ziel`` — die Kaderzugehoerigkeit ist die einzige
    Information der Zielsaison, und sie ist vor dem ersten Spieltag bekannt.
    """
    df = kandidaten.copy()
    df = df.merge(historie(dat.ss, j_ziel, delta, fenster, dat.league),
                  on="player_id", how="left")
    df = df.merge(historie_kontext(dat, j_ziel, delta, fenster),
                  on="player_id", how="left")

    fenster = dat.ss[(dat.ss.league == dat.league) & (dat.ss.j < j_ziel)
                     & (dat.ss.j >= j_ziel - RUECKBLICK) & dat.ss.starter]
    war_starter = set(fenster.player_id)
    hatte_saison = set(dat.ss[dat.ss.j == j_ziel - 1].player_id)
    df["fall"] = ["a" if p in war_starter else
                  ("b" if p in hatte_saison else "c") for p in df.player_id]
    # Fall a ohne Historie gibt es der Definition nach nicht: wer letzte Saison
    # Starter in Liga 1 war, hat sie. Ein Rest aus Datenluecken wandert nach b.
    df.loc[(df.fall == "a") & df.a1.isna(), "fall"] = "b"

    werte = [zelle(dat.zellen, j_ziel, t, p)
             for t, p in zip(df.team_id, df.pos)]
    df["zelle"] = [w for w, _ in werte]
    df["stufe"] = [st for _, st in werte]

    qual = [zellen_qualitaet(dat.zellen, j_ziel, t, p)
            for t, p in zip(df.team_id, df.pos)]
    df["cell_mw"] = [q[0] for q in qual]
    df["cell_age"] = [q[1] for q in qual]
    df["cell_n"] = [q[2] for q in qual]

    df["z_hat"] = df.team_id.map(dat.z_hat(j_ziel))
    # Diagnostische Aufrufer reichen teilweise nur ID, Team und Position ein.
    # Relative Merkmale bleiben dann ehrlich leer.
    if "log2_mw" not in df:
        df["log2_mw"] = np.nan
    if "alter" not in df:
        df["alter"] = np.nan
    alt_z = dict(zip(
        dat.team.loc[dat.team.j == j_ziel - 1, "team_id"],
        dat.team.loc[dat.team.j == j_ziel - 1, "z_att"]))
    df["team_delta"] = df.z_hat - df.team_id.map(alt_z)
    df["hist_team_delta"] = df.z_hat - df.hist_z
    df["mw_delta"] = df.log2_mw - df.hist_mw
    df["rel_mw"] = df.log2_mw - df.cell_mw
    df["rel_age"] = df.alter - df.cell_age
    df["teil"] = df.pos.map(TEIL)
    df["hat_hist"] = df.a1.notna().astype(float)
    # In Fall b ist die eigene Historie ein Neben-, kein Hauptmerkmal: fehlt
    # sie, traegt ``hat_hist`` den Niveauunterschied, und ``a1_ers`` wird mit
    # dem Trainingsmittel gefuellt. Beide zusammen sind die uebliche
    # Fehlend-Markierung — ohne sie waere die Luecke stumm mit dem Mittelwert
    # zugedeckt.
    df["a1_ers"] = df.a1
    df["hist_belegt"] = df.gewicht.fillna(0.0) / (
        df.gewicht.fillna(0.0) + K_HIST_BELEGT)
    df["a1_belegt"] = (df.a1 - 100.0) * df.hist_belegt
    return df


# ---------------------------------------------------------------------------
# Anpassung und Vorhersage
# ---------------------------------------------------------------------------
#
# Eine einzige Regression, kein Fit je Fall: die Grundmerkmale sind mit
# Fall-Dummies verschraenkt (jeder Fall bekommt also seine eigenen
# Koeffizienten und sein eigenes Niveau), der Mannschaftsteil dagegen ist
# **gemeinsam**. Das ist keine Sparsamkeit, sondern die Sache: dass ein Torwart
# von der Staerke seines Vereins nichts hat und ein Stuermer viel, ist ein
# Gesetz des Spiels und nicht Eigenschaft eines Falls. Gemessen ist es
# ausserdem besser — je Fall geschaetzt r 0,681, gemeinsam 0,685, und in Fall c
# (435 Trainingszeilen) 0,517 gegen 0,527. Dort waren acht eigene
# Teil-Parameter schlicht ueberangepasst: sie setzten den Abschlag fuers
# Zentrum auf -22 Punkte, wo die Ligamittel sechs hergeben.

FALL_REF = "a"


def _spalten() -> list:
    """Die Spalten der Entwurfsmatrix, in fester Reihenfolge."""
    cols = []
    for fall, grund in MERKMALE.items():
        if fall != FALL_REF:
            cols.append(f"f_{fall}")
        cols += [f"{fall}:{m}" for m in grund]
    return cols + [f"d_{t}" for t in TEILE] + [f"zx_{t}" for t in TEILE]


def _entwurf(df: pd.DataFrame, mittel: dict) -> np.ndarray:
    """Fall-Dummies mal Grundmerkmale, dazu der gemeinsame Teil-Block.

    Fehlende Nebenmerkmale werden mit dem Trainingsmittel **ihres Falls**
    gefuellt — ein Neuzugang ohne Marktwert bekommt den Schnitt der
    Neuzugaenge, nicht den aller Spieler.
    """
    sp = {}
    z = df.z_hat.astype(float)
    for fall, grund in MERKMALE.items():
        ist = (df.fall == fall).astype(float)
        if fall != FALL_REF:
            sp[f"f_{fall}"] = ist
        for m in grund:
            sp[f"{fall}:{m}"] = (df[m].astype(float)
                                 .fillna(mittel.get((fall, m), 0.0)) * ist)
    for t in TEILE:
        d = (df.teil == t).astype(float)
        sp[f"d_{t}"] = d
        sp[f"zx_{t}"] = z * d
    return pd.DataFrame(sp, index=df.index)[_spalten()].values


def brauchbar(df: pd.DataFrame) -> pd.Series:
    """Zeilen, die eine Vorhersage tragen: Teamstaerke, Teil, Pflichtmerkmale."""
    ok = df.z_hat.notna() & df.teil.notna()
    for fall, pflicht in PFLICHT.items():
        if pflicht:
            fehlt = (df.fall == fall) & df[pflicht].isna().any(axis=1)
            ok &= ~fehlt
    return ok


def anpassen(train: pd.DataFrame, j_ziel: int, ziel: str = "avg_alle",
             rho: float = RHO) -> dict:
    """Eine saisonsgewichtete OLS ueber alle Faelle.

    Gefittet wird nur auf Zeilen, die selbst den Starter-Filter bestanden
    haben — auf Spielern also, die die Frage beantworten, die das Modell
    stellt. Die Gewichte ``rho^Abstand`` holen die juengsten Saisons nach
    vorn, weil das Niveau der Neuzugaenge ab 2020 gefallen ist.
    """
    t = train[train[ziel].notna()]
    t = t[brauchbar(t)]
    mittel = {}
    for fall, grund in MERKMALE.items():
        s = t[t.fall == fall]
        for m in grund:
            v = float(s[m].astype(float).mean()) if len(s) else float("nan")
            mittel[(fall, m)] = v if np.isfinite(v) else 0.0
    coef = _wols(_entwurf(t, mittel), t[ziel].values,
                 rho ** (j_ziel - 1 - t.j.values))
    return {"spalten": _spalten(), "coef": coef, "mittel": mittel,
            "n": int(len(t)),
            "n_fall": {k: int(v) for k, v in t.fall.value_counts().items()}}


def vorhersagen(fit: dict, rows: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=rows.index, dtype=float)
    ok = brauchbar(rows)
    if ok.any():
        out.loc[ok] = _pred(fit["coef"], _entwurf(rows[ok], fit["mittel"]))
    return out


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def zeilen_aller_saisons(dat: Daten, delta: float = DELTA,
                         fenster: int = FENSTER) -> pd.DataFrame:
    """Merkmalszeilen fuer jede Zielsaison, die eine Vorsaison hat."""
    teile = []
    for j in range(dat.erste_zielsaison, dat.j_ziel):
        zg = dat.zielgruppe(j)
        if not len(zg):
            continue
        kand = zg[["player_id", "team_id", "pos", "avg_alle",
                   "log2_mw", "alter"]].copy()
        m = merkmalszeilen(dat, j, kand, delta, fenster)
        m["j"] = j
        teile.append(m)
    return pd.concat(teile, ignore_index=True)


def versatz_tabelle(oof: pd.DataFrame) -> dict:
    """Der Niveauversatz je Fall, der nach der Anpassung uebrig bleibt.

    Gemittelt wird ueber **Saisons**, nicht ueber Zeilen: die Zeilen einer
    Saison teilen sich ihr Kohortenglueck, und wer sie einzeln zaehlt, haelt
    Rauschen fuer Struktur.

    Das Fenster waechst, statt zu rollen — und das ist der ganze Trick. Die
    Saisonverzerrung ist **nicht** autokorreliert (r(t, t-1) = -0,016), die
    letzte Saison sagt also nichts ueber die naechste. Ein rollendes Fenster
    jagt damit Rauschen: mit den letzten vier Saisons faellt r von 0,685 auf
    0,658 und die groesste Saisonverzerrung steigt von 9,5 auf 13,4. Ueber
    alle bisherigen Saisons gemittelt bleibt dagegen nur der strukturelle
    Teil stehen, und der ist echt: Fall b liegt in sechs von sieben Saisons
    zu hoch (+5,3 im Mittel, t = 2,4).
    """
    r = oof.dropna(subset=["pred", "avg_alle"])
    if not len(r):
        return {}
    r = r.assign(res=r.pred - r.avg_alle)
    aus = {}
    for f, s in r.groupby("fall"):
        js = s.groupby("j").res.mean()
        # Unter drei Saisons ist der Mittelwert selbst Kohortenglueck. Mit
        # zweien angewandt macht der Versatz die Zielsaison 2019 von +8,9 auf
        # +13,4 schlechter, weil 2018 zufaellig -12,8 lag.
        aus[f] = float(js.mean()) if len(js) >= MIN_VERSATZ_SAISONS else 0.0
    return aus


def kalibrierung_tabelle(oof: pd.DataFrame) -> dict:
    """Niveau- und Tail-Kalibrierung aus frueheren OOF-Prognosen.

    Das Grundmodell bleibt fuer die breite Zielgruppe zustaendig. Diese Stufe
    korrigiert nur systematische Restfehler, die in Prognosen auf unbekannten
    Saisons entstanden sind. Insbesondere wird der Marktwert eines
    Liganeulings nicht als pauschaler Punkteanker missverstanden: Erst wenn er
    in das historisch klar unterschaetzte Elite-Segment faellt, erhaelt er den
    dort gemessenen, stark geschrumpften Aufschlag.
    """
    versatz = versatz_tabelle(oof)
    r = oof.dropna(subset=["pred", "avg_alle"])
    bonus = {"neuzugang_elite": 0.0}
    n_saisons = {"neuzugang_elite": 0}

    segmente = {
        "neuzugang_elite": (
            "c", np.power(2.0, r.log2_mw.astype(float)) >= SPITZE_MW),
    }
    for name, (fall, maske) in segmente.items():
        s = r[(r.fall == fall) & maske].copy()
        if not len(s):
            continue
        # Rest nach der normalen Fall-Kalibrierung. Das Saisonmittel ist die
        # unabhaengige Beobachtung; Spieler derselben Kohorte sind es nicht.
        s["rest"] = s.avg_alle - (s.pred - versatz.get(fall, 0.0))
        je_saison = s.groupby("j").rest.mean()
        n_saisons[name] = int(len(je_saison))
        if len(je_saison) < MIN_VERSATZ_SAISONS:
            continue
        shrink = len(je_saison) / (len(je_saison) + KALIBRIER_PRIOR_SAISONS)
        bonus[name] = float(je_saison.mean() * shrink)
    return {"versatz": versatz, "bonus": bonus, "n_saisons": n_saisons}


def prognosen_kalibrieren(rows: pd.DataFrame, roh: pd.Series,
                          kalibrierung: dict,
                          mit_segmente: bool = True) -> pd.Series:
    """Rohprognosen ohne Rollen-/Ausfallrisiko auf die Starter-Zielgruppe eichen."""
    versatz = kalibrierung.get("versatz", {})
    out = roh - rows.fall.map(versatz).astype(float).fillna(0.0)
    if not mit_segmente:
        return out

    bonus = kalibrierung.get("bonus", {})
    elite = ((rows.fall == "c")
             & (np.power(2.0, rows.log2_mw.astype(float)) >= SPITZE_MW))
    out.loc[elite] += float(bonus.get("neuzugang_elite", 0.0))
    return out


def backtest(dat: Daten, delta: float = DELTA, fenster: int = FENSTER,
             rho: float = RHO, ab: int | None = None,
             mit_versatz: bool = True,
             mit_segmente: bool = True) -> pd.DataFrame:
    """Je Zielsaison neu anpassen, nur auf frueheren Saisons. Kein Blick nach vorn.

    Der Versatz wird mitgefuehrt, statt hinterher aufgesetzt: fuer Zielsaison
    j speist er sich ausschliesslich aus den Out-of-fold-Resten der Saisons
    davor. Deshalb laeuft die Schleife ab der fruehestmoeglichen Saison und
    gibt erst ab ``ab`` etwas zurueck — die frueheren Faltungen werden
    gebraucht, um den Versatz ueberhaupt schaetzen zu koennen.
    """
    if ab is None:
        ab = (ERSTE_ZIELSAISON + 3 if dat.liga == "1"
              else dat.erste_zielsaison + 1)
    D = zeilen_aller_saisons(dat, delta, fenster)
    frueh, aus = [], []
    for j in sorted(D.j.unique()):
        tr, te = D[D.j < j], D[D.j == j]
        if len(tr) < 200 or not len(te):
            continue
        roh = te.assign(pred=vorhersagen(anpassen(tr, j, rho=rho), te))
        frueh.append(roh)
        if j < ab:
            continue
        if mit_versatz and len(frueh) > MIN_VERSATZ_SAISONS:
            kal = kalibrierung_tabelle(pd.concat(frueh[:-1], ignore_index=True))
            roh = roh.assign(pred=prognosen_kalibrieren(
                roh, roh.pred, kal, mit_segmente=mit_segmente))
        aus.append(roh)
    return pd.concat(aus, ignore_index=True)


def report(bt: pd.DataFrame) -> None:
    """Was das Modell kann, aufgeschluesselt nach Fall und Mannschaftsteil.

    Die Vergleichsmarke ist das stumpfe Fortschreiben des eigenen
    Startelfschnitts. Sie gilt nur fuer Fall a — fuer b und c sagt sie gar
    nichts, und genau das ist der Punkt: dort gibt es ohne Modell keine Zahl.
    """
    b = bt.dropna(subset=["pred", "avg_alle"])
    print("Walk-forward, Ziel = Ø Punkte je Einsatz, Zielsaisons "
          f"{int(b.j.min())}-{int(b.j.max())}")
    print()
    print("  Fall  n      r      MAE   Verzerrung")
    for fall in ("a", "b", "c"):
        s = b[b.fall == fall]
        if not len(s):
            continue
        print(f"  {fall}     {len(s):<6d} {_corr(s.avg_alle, s.pred):.3f}  "
              f"{np.abs(s.avg_alle - s.pred).mean():5.1f}   "
              f"{(s.pred - s.avg_alle).mean():+5.1f}")
    print(f"  alle  {len(b):<6d} {_corr(b.avg_alle, b.pred):.3f}  "
          f"{np.abs(b.avg_alle - b.pred).mean():5.1f}   "
          f"{(b.pred - b.avg_alle).mean():+5.1f}")

    n = b[(b.fall == "a") & b.a1.notna()]
    if len(n):
        print()
        print("  Vergleich Fall a, Vorjahres-Startelfschnitt roh:  "
              f"r={_corr(n.avg_alle, n.a1):.3f}  "
              f"MAE={np.abs(n.avg_alle - n.a1).mean():.1f}  "
              f"Verzerrung={(n.a1 - n.avg_alle).mean():+.1f}")

    print()
    print("  Verzerrung je Mannschaftsteil (das, was ohne den Kanal entgleist):")
    for t in (TEIL_REF,) + TEILE:
        s = b[b.teil == t]
        if len(s) < 20:
            continue
        print(f"    {t:<4s} {(s.pred - s.avg_alle).mean():+5.1f}  (n={len(s)})")

    print()
    print("  Verzerrung je Zielsaison:")
    for j, s in b.groupby("j"):
        print(f"    {int(j)}  {(s.pred - s.avg_alle).mean():+5.1f}  (n={len(s)})")


def gitter(dat: Daten) -> None:
    """Die freien Konstanten gegeneinander messen, statt sie zu setzen."""
    print("delta / fenster / rho, gemessen am Walk-forward")
    print()
    print("  delta  fenster  rho    r      MAE   Verz(a)  Verz(b/c)")
    for delta in (0.5, 0.7, 0.9):
        for fenster in (2, 3, 4):
            for rho in (0.5, 0.6, 1.0):
                bt = backtest(dat, delta=delta, fenster=fenster, rho=rho)
                b = bt.dropna(subset=["pred", "avg_alle"])
                va, vbc = b[b.fall == "a"], b[b.fall != "a"]
                print(f"  {delta:<6.1f} {fenster:<8d} {rho:<6.1f} "
                      f"{_corr(b.avg_alle, b.pred):.3f}  "
                      f"{np.abs(b.avg_alle - b.pred).mean():5.1f}  "
                      f"{(va.pred - va.avg_alle).mean():+6.1f}  "
                      f"{(vbc.pred - vbc.avg_alle).mean():+8.1f}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def verletzte() -> set:
    """Wer heute ausfaellt — aus data/history.json, der einzigen Quelle dafuer.

    Diese Spieler bekommen die **volle** Prognose: das Modell rechnet ohnehin
    unter der Annahme "gesund, spielt seine Rolle", und genau das ist bei einem
    Verletzten die interessante Zahl — was holt er nach der Rueckkehr. Die
    Markierung sorgt nur dafuer, dass scatter.html sie aus der Regression
    heraushaelt: ihr Marktwert ist verletzungsbedingt gedrueckt und wuerde die
    Fair-Value-Linie verziehen.
    """
    pfad = LEGACY_DIR / "history.json"
    if not pfad.exists():
        return set()
    spieler = json.loads(pfad.read_text(encoding="utf-8")).get("players", {})
    aus = set()
    for pid, e in spieler.items():
        st = e.get("st") or []
        if st and st[-1][1] in AUSFALL_CODES:
            aus.add(int(pid))
    return aus


def starter_merkmalszeilen(dat: Daten, j_ziel: int,
                            rows: pd.DataFrame) -> pd.DataFrame:
    """Ex-ante-Merkmale fuer die automatische Zweitliga-Zielgruppe.

    Die Zielvariable des Punkte-Modells bleibt *bedingt auf eine gesetzte
    Saison*. Diese kleine vorgeschaltete Logistik beantwortet nur, fuer wen
    diese Bedingung plausibel ist. Eigene Historie darf aus beiden Ligen
    kommen; Marktwert-Rang und -Niveau werden innerhalb des aktuellen
    Zweitliga-Kaders gebildet und sind deshalb ueber Saisons vergleichbar.
    """
    df = rows.copy()
    h = dat.ss[(dat.ss.j < j_ziel) & (dat.ss.j >= j_ziel - RUECKBLICK)]
    agg = h.groupby("player_id", as_index=False).agg(
        prev_starts=("n_start", "max"), prev_apps=("n_einsaetze", "max"))
    df = df.merge(agg, on="player_id", how="left")
    df["had_prev"] = df.player_id.isin(set(h.player_id)).astype(float)
    df["mw_team_pct"] = df.groupby("team_id").log2_mw.rank(
        pct=True, ascending=True)
    sd = float(df.log2_mw.std())
    df["mw_liga_z"] = ((df.log2_mw - float(df.log2_mw.mean())) / sd
                       if np.isfinite(sd) and sd > 1e-9 else 0.0)
    df["age2"] = ((df.alter - 25.0) / 10.0) ** 2
    return df


def starter_trainingsdaten(dat: Daten) -> pd.DataFrame:
    """Historische Zweitliga-Kader mit ex-ante Merkmalen und Starter-Ziel."""
    if hasattr(dat, "_starter_trainingsdaten"):
        return dat._starter_trainingsdaten
    teile = []
    for j in sorted(dat.ss.loc[
            (dat.ss.league == dat.league) & (dat.ss.j < dat.j_ziel), "j"].unique()):
        ziel = dat.ss[(dat.ss.league == dat.league) & (dat.ss.j == j)][
            ["player_id", "season", "team_id", "starter"]].copy()
        if not len(ziel):
            continue
        tm = dat.tm[(dat.tm.liga == int(dat.liga))
                    & (dat.tm.season == ziel.season.iloc[0])
                    & dat.tm.player_id.notna()].drop_duplicates("player_id")
        ziel = ziel.merge(tm[["player_id", "log2_mw", "alter"]],
                          on="player_id", how="left")
        z = starter_merkmalszeilen(dat, int(j), ziel)
        z["j"] = int(j)
        teile.append(z)
    if not teile:
        raise ValueError(f"Keine Trainingssaisons fuer {dat.league}")
    dat._starter_trainingsdaten = pd.concat(teile, ignore_index=True)
    return dat._starter_trainingsdaten


def _starter_anpassen(train: pd.DataFrame) -> dict:
    """Die kleine regularisierte Logistik auf bereits gebauten Zeilen."""
    mittel = {m: float(train[m].median()) if train[m].notna().any() else 0.0
              for m in STARTER_MERKMALE}
    X = train[list(STARTER_MERKMALE)].fillna(mittel).values
    modell = LogisticRegression(C=0.3, max_iter=1000).fit(
        X, train.starter.astype(int).values)
    return {"modell": modell, "mittel": mittel, "n": int(len(train)),
            "saisons": sorted(int(x) for x in train.j.unique())}


def starter_fit(dat: Daten) -> dict:
    """Auswahlmodell fuer Liga 2 auf allen abgeschlossenen Saisons fitten."""
    if hasattr(dat, "_starter_fit"):
        return dat._starter_fit
    fit = _starter_anpassen(starter_trainingsdaten(dat))
    dat._starter_fit = fit
    return fit


def starter_wahrscheinlichkeit(dat: Daten, rows: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Wahrscheinlichkeit einer normalen Saison als Gesetzter, Liga 2."""
    fit = starter_fit(dat)
    m = starter_merkmalszeilen(dat, dat.j_ziel, rows)
    X = m[list(STARTER_MERKMALE)].fillna(fit["mittel"]).values
    p = pd.Series(fit["modell"].predict_proba(X)[:, 1], index=rows.index)
    return p, fit


def starter_backtest(dat: Daten) -> pd.DataFrame:
    """Walk-forward der automatischen Zielgruppe, immer acht je Verein."""
    D = starter_trainingsdaten(dat)
    aus = []
    for j in sorted(D.j.unique())[1:]:
        tr, te = D[D.j < j], D[D.j == j].copy()
        if not len(tr) or not len(te):
            continue
        fit = _starter_anpassen(tr)
        X = te[list(STARTER_MERKMALE)].fillna(fit["mittel"]).values
        te["starter_wahrscheinlichkeit"] = fit["modell"].predict_proba(X)[:, 1]
        rang = te.groupby("team_id").starter_wahrscheinlichkeit.rank(
            ascending=False, method="first")
        te["ausgewaehlt"] = rang <= STARTER_JE_TEAM
        te["kategorie"] = np.where(rang <= STARTER_KATEGORIE_1_JE_TEAM, 1, 2)
        aus.append(te)
    return pd.concat(aus, ignore_index=True)


def report_starterauswahl(bt: pd.DataFrame) -> None:
    """Trefferquote der ex-ante Zweitliga-Auswahl, getrennt vom Punktemodell."""
    print()
    print(f"Starter-Auswahl, beste {STARTER_JE_TEAM} je Verein")
    print("  Saison  n    Praezision  Recall")
    for j, s in bt.groupby("j"):
        w = s[s.ausgewaehlt]
        praezision = float(w.starter.mean())
        recall = float(s.loc[s.starter, "ausgewaehlt"].mean())
        print(f"  {int(j)}  {int(w.ausgewaehlt.sum()):<4d} "
              f"{praezision:9.3f}  {recall:.3f}")
    w = bt[bt.ausgewaehlt]
    print(f"  alle  {int(w.ausgewaehlt.sum()):<4d} "
          f"{float(w.starter.mean()):9.3f}  "
          f"{float(bt.loc[bt.starter, 'ausgewaehlt'].mean()):.3f}")


def kandidaten(dat: Daten, kategorien=(1, 2)) -> pd.DataFrame:
    """Die Spieler der Zielsaison, fuer die eine Zahl entstehen soll.

    Verein und Marktwert kommen aus ``data/data_{liga}.json``, nicht aus dem Panel:
    Neuzugaenge stehen dort noch nicht (30 von 441 haetten keine team_id). Das
    Kuerzel in ``team_name`` ist bereits dasselbe wie ``verein`` in der
    Kaderpflege.

    Liga 1 behaelt die handgepflegte Kategorie und Feinposition. Fuer Liga 2
    kommt die Feinposition aus dem ligaeigenen TM-Kader; die Kategorien 1/2
    entstehen reproduzierbar aus der vorgeschalteten Starter-Logistik.
    """
    d1 = json.loads((LEGACY_DIR / f"data_{dat.liga}.json").read_text(
        encoding="utf-8"))
    kb = pd.DataFrame([{
        "player_id": int(p["id"]), "name": p["name"],
        "team_id": str(p["team_id"]), "verein": p.get("team_name"),
        "position_kb": p.get("position"),
        "kb_mw": (p.get("market_value") or {}).get("current"),
    } for p in d1["players"]])

    tmj = dat.tm[(dat.tm.season == dat.zielsaison)
                 & (dat.tm.liga == int(dat.liga))
                 & dat.tm.player_id.notna()].drop_duplicates("player_id")
    k = kb.merge(tmj[["player_id", "position_fine", "log2_mw", "alter"]],
                 on="player_id", how="left")
    # Rueckfall auf den Kickbase-Marktwert: beide sind Euro, der Log macht den
    # Rest. Ohne ihn verloeren Spieler ohne TM-Stichtagswert ihr zweitstaerkstes
    # Merkmal.
    k["log2_mw"] = k.log2_mw.fillna(np.log2(k.kb_mw.where(k.kb_mw > 0)))

    if dat.liga == "1":
        fp = pd.read_csv(MANUAL / "fine_positions.csv")
        fp = fp[fp.player_id.notna()].copy()
        fp["player_id"] = fp.player_id.astype("int64")
        k = k.drop(columns=["position_fine"]).merge(
            fp[["player_id", "position_fine", "kategorie"]], on="player_id")
        k = k[k.kategorie.isin(kategorien)]
        k["starter_wahrscheinlichkeit"] = np.nan
    else:
        # Nur vier aktuelle Spieler fehlen in der TM-Bruecke. Ihre grobe
        # Kickbase-Position ist ehrlicher als ein stiller Ausschluss.
        grob = {1: "TW", 2: "IV", 3: "ZM", 4: "ST"}
        k["position_fine"] = k.position_fine.fillna(k.position_kb.map(grob))
        p, _ = starter_wahrscheinlichkeit(dat, k)
        k["starter_wahrscheinlichkeit"] = p
        rang = k.groupby("team_id").starter_wahrscheinlichkeit.rank(
            ascending=False, method="first")
        k["kategorie"] = np.where(rang <= STARTER_KATEGORIE_1_JE_TEAM, 1, 2)
        k = k[rang <= STARTER_JE_TEAM]
        k = k[k.kategorie.isin(kategorien)]

    k = k.rename(columns={"position_fine": "pos"})
    return k.reset_index(drop=True)


def _zahl(x):
    """None statt NaN — JSON kennt kein NaN, und null liest sich als 'fehlt'."""
    return None if x is None or not np.isfinite(x) else float(x)


def export(path: Path | None = None, kategorien=(1, 2),
           liga: str = "1") -> dict:
    """Fallweise Spielerprognose fuer Liga 1 oder 2 schreiben."""
    dat = Daten(liga=liga)
    fits = anpassen(zeilen_aller_saisons(dat), dat.j_ziel)
    # Derselbe Versatz wie im Backtest, aus denselben Out-of-fold-Resten. Er
    # kostet einen vollen Walk-forward — der laeuft in Sekunden, und die
    # Alternative waere eine Konstante im Code, die still veraltet.
    oof = backtest(dat, mit_versatz=False, ab=dat.erste_zielsaison)
    kal = kalibrierung_tabelle(oof)
    vers = kal["versatz"]

    rows = merkmalszeilen(dat, dat.j_ziel, kandidaten(dat, kategorien))
    rows["xp"] = prognosen_kalibrieren(
        rows, vorhersagen(fits, rows), kal, mit_segmente=dat.liga == "1")
    krank = verletzte()

    spieler = {}
    for r in rows.itertuples():
        if not np.isfinite(r.xp):
            continue
        streuung = STREUUNG_LIGA[dat.liga].get(r.fall, 20.0)
        spieler[str(r.player_id)] = {
            "name": r.name, "team_id": r.team_id, "verein": r.verein,
            "pos": r.pos, "teil": r.teil, "kategorie": int(r.kategorie),
            "fall": r.fall,
            "xp": round(float(r.xp), 1),
            "hist": None if _zahl(r.a1) is None else round(float(r.a1), 1),
            "zelle": None if _zahl(r.zelle) is None else round(float(r.zelle), 1),
            "stufe": r.stufe,
            "z_hat": None if _zahl(r.z_hat) is None else round(float(r.z_hat), 2),
            "mw_verhaeltnis": (round(float(2.0 ** r.rel_mw), 2)
                                if r.fall == "c" and _zahl(r.rel_mw) is not None
                                else None),
            "alter_delta": (round(float(r.rel_age), 1)
                             if r.fall == "c" and _zahl(r.rel_age) is not None
                             else None),
            "vorgaenger_n": (int(r.cell_n) if r.fall == "c"
                              and _zahl(r.cell_n) is not None else None),
            "streuung": streuung,
            "sicherheit": round(max(0.05, 1.0 - streuung / 30.0), 3),
            "verletzt": int(r.player_id) in krank,
            "starter_wahrscheinlichkeit": (
                None if _zahl(r.starter_wahrscheinlichkeit) is None
                else round(float(r.starter_wahrscheinlichkeit), 3)),
        }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "avg-je-einsatz-fallweise",
        "season": dat.zielsaison,
        "liga": dat.liga,
        "params": {
            "min_starts": MIN_STARTS,
            "delta": DELTA, "fenster": FENSTER, "rho": RHO,
            "historie_k": K_HIST_BELEGT,
            "versatz": {k: round(v, 2) for k, v in vers.items()},
            "tail_kalibrierung": {
                "neuzugang_mw_ab": int(SPITZE_MW),
                "prior_saisons": KALIBRIER_PRIOR_SAISONS,
                "bonus": {k: round(v, 2) for k, v in kal["bonus"].items()},
                "n_saisons": kal["n_saisons"],
            },
            "teile": {"referenz": TEIL_REF, "uebrige": list(TEILE)},
            "auswahl": ({
                "typ": "manuelle-kategorien",
            } if dat.liga == "1" else {
                "typ": "starter-logistik",
                "je_team": STARTER_JE_TEAM,
                "kategorie_1_je_team": STARTER_KATEGORIE_1_JE_TEAM,
                "merkmale": list(STARTER_MERKMALE),
                "n": starter_fit(dat)["n"],
                "saisons": starter_fit(dat)["saisons"],
            }),
            "koeffizienten": {
                "merkmale": ["const"] + fits["spalten"],
                "werte": [round(float(x), 3) for x in fits["coef"]],
                "n": fits["n"], "n_fall": fits["n_fall"]},
        },
        "faelle": {k: int(v) for k, v in rows.fall.value_counts().items()},
        "kategorien": list(kategorien),
        "players": spieler,
    }
    datei = ("player_projections_avg.json" if dat.liga == "1"
             else f"player_projections_avg_{dat.liga}.json")
    ziel = path or (LEGACY_DIR / datei)
    atomic_write_json(ziel, out)
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backtest", action="store_true",
                    help="walk-forward, aufgeschluesselt nach Fall und Teil")
    ap.add_argument("--gitter", action="store_true",
                    help="delta, fenster und rho gegeneinander messen")
    ap.add_argument("--liga", choices=sorted(LIGEN), default="1",
                    help="Liga 1 oder 2 (Standard: 1)")
    args = ap.parse_args()

    if args.gitter:
        gitter(Daten(liga=args.liga))
    elif args.backtest:
        dat = Daten(liga=args.liga)
        report(backtest(dat))
        if args.liga == "2":
            report_starterauswahl(starter_backtest(dat))
    else:
        out = export(liga=args.liga)
        datei = ("player_projections_avg.json" if args.liga == "1"
                 else f"player_projections_avg_{args.liga}.json")
        print(f"{len(out['players'])} Spieler nach "
              f"{LEGACY_DIR / datei}  ({out['faelle']})")


if __name__ == "__main__":
    _cli()
